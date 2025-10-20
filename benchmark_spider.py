"""
benchmark_spider.py
Benchmark a single-call LangGraph SQL generator on Spider(-Realistic/-SYN).

What it does:
- Loads Spider-style question/query pairs:
    * Prefers local Spider files if present (dev.json, tables.json, database/).
    * Otherwise loads Hugging Face datasets:
        - xlangai/spider              (train/dev style)
        - aherntech/spider-realistic  (dev-style)
        - aherntech/spider-syn        (train/validation)
- Attempts to download the official Spider zip (includes SQLite DBs) by:
    * Scraping the Yale Spider page for the Google Drive link, then using gdown.
    * If that fails, it still runs EM & validity (no execution accuracy).
- For each example:
    * Introspects the SQLite DB schema,
    * Calls the LangGraph model ONCE to get SQL,
    * Computes:
        - Exact Match (normalized string equality),
        - SQL Validity (parses/executes without error),
        - Execution Accuracy (matches gold execution results), if DBs available.
- Writes results to CSV and prints a summary.

Usage:
    pip install -U langgraph langchain langchain-openai openai datasets tqdm requests gdown
    export OPENAI_API_KEY=sk-...
    python benchmark_spider.py --limit 50 --splits spider spider-realistic spider-syn

Notes:
- Default model is gpt-4.1-mini. Override via --model or OPENAI_MODEL env var.
- Execution Accuracy requires the Spider 'database/' folder present or downloadable.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests
from datasets import load_dataset
from tqdm import tqdm

# local import
from sql_model_graph import SQLGeneratorGraph, summarize_sqlite_schema, strip_sql_fences
from dotenv import load_dotenv
load_dotenv()

# ------------------------- Paths & Downloaders -------------------------
HERE = Path(__file__).parent
DATA_DIR = HERE / "data" / "spider"     # will create if needed
DB_DIR = DATA_DIR / "database"          # expected DB location
DEV_JSON = DATA_DIR / "dev.json"
TRAIN_SPIDER_JSON = DATA_DIR / "train_spider.json"
TABLES_JSON = DATA_DIR / "tables.json"

YALE_SPIDER_PAGE = "https://yale-lily.github.io/spider"


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def try_parse_google_drive_id_from_yale_page() -> Optional[str]:
    """
    Fetch the Yale Spider page and extract the Google Drive file id of the 'Spider Dataset'.
    """
    try:
        html = requests.get(YALE_SPIDER_PAGE, timeout=20).text
        # Look for drive.google.com link with id= or /file/d/<id>/
        m = re.search(r'https://drive\.google\.com/[^\s"]+', html)
        if not m:
            return None
        url = m.group(0)
        # Patterns:
        #   https://drive.google.com/file/d/<ID>/view?usp=sharing
        m1 = re.search(r"/file/d/([a-zA-Z0-9_-]+)", url)
        if m1:
            return m1.group(1)
        #   https://drive.google.com/open?id=<ID>  or ?id=
        m2 = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
        if m2:
            return m2.group(1)
        return None
    except Exception:
        return None


def download_spider_zip_to(data_dir: Path) -> Optional[Path]:
    """
    Attempt to download the official Spider zip (which includes database/).
    Returns the path to the extracted root if successful, else None.
    """
    ensure_dir(data_dir)
    file_id = try_parse_google_drive_id_from_yale_page()
    if not file_id:
        print("[warn] Could not locate Google Drive file id from Yale page; skipping DB download.")
        return None

    try:
        import gdown  # type: ignore
    except ImportError:
        print("[info] gdown not installed; attempting to install it now...")
        os.system(f"{sys.executable} -m pip install -q gdown")
        try:
            import gdown  # type: ignore
        except Exception:
            print("[warn] Failed to import gdown after install.")
            return None

    zip_path = data_dir / "spider.zip"
    if not zip_path.exists():
        print(f"[info] Downloading Spider zip via gdown (id={file_id})...")
        try:
            gdown.download(id=file_id, output=str(zip_path), quiet=False)
        except Exception as e:
            print(f"[warn] gdown download error: {e}")
            return None

    # Extract
    try:
        print("[info] Extracting Spider zip...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(str(data_dir))
        # data structure typically places files at top-level or a nested folder; try to move if needed
        # Ensure DB_DIR exists afterwards
        # Look for a folder that contains 'database' and 'tables.json'
        candidate_root = None
        for root, dirs, files in os.walk(data_dir):
            if "database" in dirs and "tables.json" in files:
                candidate_root = Path(root)
                break
        if candidate_root and candidate_root != data_dir:
            # move contents up
            for item in candidate_root.iterdir():
                dest = data_dir / item.name
                if dest.exists():
                    continue
                if item.is_dir():
                    shutil.copytree(item, dest)
                else:
                    shutil.copy2(item, dest)
        return data_dir
    except Exception as e:
        print(f"[warn] Failed to extract Spider zip: {e}")
        return None


def have_local_spider_assets() -> bool:
    return DB_DIR.exists() and TABLES_JSON.exists() and DEV_JSON.exists()


# ------------------------- Data Loading -------------------------
@dataclass
class Example:
    db_id: str
    question: str
    gold_sql: str


def load_examples_from_local_json(path: Path) -> List[Example]:
    data = json.loads(path.read_text())
    out: List[Example] = []
    for item in data:
        out.append(Example(db_id=item["db_id"], question=item["question"], gold_sql=item["query"]))
    return out


def load_hf_spider(limit: Optional[int] = None) -> List[Example]:
    """
    Load Spider from Hugging Face mirror (xlangai/spider). Uses 'train' split
    (7k examples). If you also have local dev.json, you can run with that too.
    """
    ds = load_dataset("xlangai/spider", split="train")
    rows = ds if limit is None else ds.select(range(min(int(limit), len(ds))))
    return [Example(db_id=r["db_id"], question=r["question"], gold_sql=r["query"]) for r in rows]


def load_hf_spider_realistic(limit: Optional[int] = None) -> List[Example]:
    ds = load_dataset("aherntech/spider-realistic", split="validation")
    rows = ds if limit is None else ds.select(range(min(int(limit), len(ds))))
    return [Example(db_id=r["db_id"], question=r["question"], gold_sql=r["query"]) for r in rows]


def load_hf_spider_syn(split: str = "validation", limit: Optional[int] = None) -> List[Example]:
    # splits: 'train' (7k), 'validation' (~1k)
    ds = load_dataset("aherntech/spider-syn", split=split)
    rows = ds if limit is None else ds.select(range(min(int(limit), len(ds))))
    return [Example(db_id=r["db_id"], question=r["SpiderSynQuestion"], gold_sql=r["query"]) for r in rows]


def load_all_examples(splits: List[str], limit: Optional[int]) -> List[Tuple[str, List[Example]]]:
    """
    Returns list of (split_name, examples).
    split_name in {"spider", "spider-realistic", "spider-syn"}
    """
    out: List[Tuple[str, List[Example]]] = []
    for sp in splits:
        if sp == "spider":
            if DEV_JSON.exists():
                ex = load_examples_from_local_json(DEV_JSON)
                if limit:
                    ex = ex[: limit]
            else:
                ex = load_hf_spider(limit=limit)
            out.append((sp, ex))
        elif sp == "spider-realistic":
            ex = load_hf_spider_realistic(limit=limit)
            out.append((sp, ex))
        elif sp == "spider-syn":
            ex = load_hf_spider_syn(split="validation", limit=limit)
            out.append((sp, ex))
        else:
            print(f"[warn] Unknown split: {sp} (skipped)")
    return out


# ------------------------- SQL Execution & Metrics -------------------------
def normalize_sql(s: str) -> str:
    # trivial normalization for EM (trim, collapse whitespace, lower)
    s = strip_sql_fences(s or "")
    s = s.strip().rstrip(";")
    s = re.sub(r"\s+", " ", s)
    return s.lower().strip()


def rows_to_canonical(rows: List[Tuple]) -> List[Tuple]:
    """
    Normalize SQLite rows to a canonical representation:
    - convert to tuples
    - convert bytes to str
    - lower-case text
    - sort rows for set-like comparison
    """
    def norm_val(v):
        if isinstance(v, bytes):
            v = v.decode("utf-8", errors="ignore")
        if isinstance(v, str):
            return v.strip().lower()
        return v

    canon = [tuple(norm_val(v) for v in row) for row in rows]
    canon.sort()
    return canon


def try_exec_sqlite(db_path: Path, sql: str) -> Tuple[bool, Optional[List[Tuple]]]:
    """
    Try executing SQL on SQLite; returns (ok, rows_or_none).
    """
    if not db_path.exists():
        return False, None
    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        conn.close()
        return True, rows
    except Exception:
        return False, None


# ------------------------- Benchmark -------------------------
def benchmark(
    model_name: str,
    splits: List[str],
    limit: Optional[int],
    out_dir: Path,
) -> None:
    ensure_dir(out_dir)
    ensure_dir(DATA_DIR)

    # Try to ensure local Spider assets (for execution)
    if not have_local_spider_assets():
        print("[info] Local Spider dev + database not found. Attempting to download the full dataset (for execution accuracy)...")
        download_spider_zip_to(DATA_DIR)

    have_exec = DB_DIR.exists()

    model = SQLGeneratorGraph(model_name=model_name, temperature=0.0)

    # Load requested splits
    work = load_all_examples(splits=splits, limit=limit)

    # CSV output
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_csv = out_dir / f"results_{model_name.replace(':','_')}_{ts}.csv"

    total, em_hits, valid_hits, exec_hits = 0, 0, 0, 0

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "split", "idx", "db_id", "question", "gold_sql", "pred_sql",
            "em", "valid_sql", "exec_acc"
        ])

        for split_name, examples in work:
            print(f"[info] Running split: {split_name} (n={len(examples)})")
            for i, ex in enumerate(tqdm(examples, desc=f"{split_name}")):
                total += 1
                db_path = DB_DIR / ex.db_id / f"{ex.db_id}.sqlite"

                # If we don't have DBs, we still call model using a schema:
                if db_path.exists():
                    schema_text = summarize_sqlite_schema(str(db_path))
                else:
                    # fallback schema (very minimal) to give the model SOME context
                    schema_text = f"(No DB available) Database: {ex.db_id}. Consider typical Spider schemas."

                # Generate SQL (single-call graph)
                try:
                    pred_sql = model.generate_sql_for_db(ex.question, str(db_path) if db_path.exists() else str(db_path), ex.db_id)
                except Exception as e:
                    pred_sql = f"-- ERROR: {e}"

                # Metrics
                gold_norm = normalize_sql(ex.gold_sql)
                pred_norm = normalize_sql(pred_sql)

                em = int(gold_norm == pred_norm)
                em_hits += em

                # Validity & Exec
                valid_ok, exec_ok = 0, 0
                if db_path.exists():
                    ok_pred, pred_rows = try_exec_sqlite(db_path, pred_sql)
                    valid_ok = int(ok_pred)
                    if ok_pred:
                        ok_gold, gold_rows = try_exec_sqlite(db_path, ex.gold_sql)
                        if ok_gold:
                            exec_ok = int(rows_to_canonical(pred_rows) == rows_to_canonical(gold_rows))
                    else:
                        exec_ok = 0
                else:
                    # Can't validate without DB
                    valid_ok = 0
                    exec_ok = 0

                valid_hits += valid_ok
                exec_hits += exec_ok

                writer.writerow([
                    split_name, i, ex.db_id, ex.question, ex.gold_sql, pred_sql,
                    em, valid_ok, exec_ok
                ])

    # Summary
    print("\n===== Summary =====")
    print(f"Model: {model_name}")
    print(f"Total examples: {total}")
    print(f"Exact Match: {em_hits}/{total} = {em_hits/total:.3f}")
    if have_exec:
        print(f"Valid SQL rate: {valid_hits}/{total} = {valid_hits/total:.3f}")
        print(f"Execution Accuracy: {exec_hits}/{total} = {exec_hits/total:.3f}")
    else:
        print("[note] No local Spider databases found -> Valid SQL & Execution Accuracy were not computed.")
        print("       Downloaded DBs (if any) will appear under data/spider/database/. Re-run to include exec metrics.")
    print(f"CSV: {out_csv}")


# ------------------------- CLI -------------------------
def main():
    parser = argparse.ArgumentParser(description="Benchmark a single-call LangGraph SQL generator on Spider(-Realistic/-SYN).")
    parser.add_argument("--model", type=str, default=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
                        help="OpenAI model name (default: gpt-4.1-mini)")
    parser.add_argument("--limit", type=int, default=50, help="Max examples per split (default: 50)")
    parser.add_argument("--splits", nargs="+", default=["spider"],
                        choices=["spider", "spider-realistic", "spider-syn"],
                        help="Which splits to run (default: spider)")
    parser.add_argument("--output_dir", type=Path, default=HERE / "outputs", help="Where to write CSV results.")
    args = parser.parse_args()

    # Early key check
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable is not set.")
        sys.exit(1)

    ensure_dir(args.output_dir)
    benchmark(model_name=args.model, splits=args.splits, limit=args.limit, out_dir=args.output_dir)


if __name__ == "__main__":
    main()
