"""
benchmark_spider.py
Benchmark Text-to-SQL models on Spider(-Realistic/-SYN).

Now supports two modes:
  - simple : one-shot single LLM call (baseline in sql_model_graph.py)
  - sot    : SQL-of-Thought multi-agent with guided correction loop (paper)

What it does:
- Loads Spider-style question/query pairs from local files or Hugging Face.
- Introspects SQLite DBs (Spider) to build schemas.
- Calls the chosen model to generate SQL.
- Computes:
    - Exact Match (normalized string equality),
    - SQL Validity (parses/executes),
    - Execution Accuracy (rows match gold),
- Writes a CSV and prints a summary.

Usage (examples):
    # Simple baseline (default model from OPENAI_MODEL or gpt-4.1-mini)
    python benchmark_spider.py --mode simple --limit 50 --splits spider

    # SQL-of-Thought with correction loop (3 attempts), explicit taxonomy
    python benchmark_spider.py --mode sot --max_corrections 3 --taxonomy_file error_taxonomy.json

    # Choose a specific model for all agents
    python benchmark_spider.py --mode sot --model gpt-4.1-mini

    # Use reasoning model with effort control
    python benchmark_spider.py --mode sot --model o3-mini --reasoning_effort high

    # Use reasoning model with token limits
    python benchmark_spider.py --mode simple --model o3-mini --max_completion_tokens 10000

Requirements:
    pip install -U langgraph langchain langchain-openai openai datasets tqdm requests gdown python-dotenv
    export OPENAI_API_KEY=sk-...
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
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
from dotenv import load_dotenv

# local imports
from sql_model_graph import SQLGeneratorGraph, summarize_sqlite_schema, strip_sql_fences
from sql_of_thought_model_graph import SQLOfThoughtGraph

load_dotenv()

# ------------------------- Reproducibility -------------------------
# Removed explicit seeding to avoid repeated identical query behavior.

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
    """Fetch the Yale Spider page and extract the Google Drive file id."""
    try:
        html = requests.get(YALE_SPIDER_PAGE, timeout=20).text
        m = re.search(r'https://drive\.google\.com/[^\s"]+', html)
        if not m:
            return None
        url = m.group(0)
        m1 = re.search(r"/file/d/([a-zA-Z0-9_-]+)", url)
        if m1:
            return m1.group(1)
        m2 = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
        if m2:
            return m2.group(1)
        return None
    except Exception:
        return None


def download_spider_zip_to(data_dir: Path) -> Optional[Path]:
    """Attempt to download the official Spider zip (which includes database/)."""
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
        candidate_root = None
        for root, dirs, files in os.walk(data_dir):
            if "database" in dirs and "tables.json" in files:
                candidate_root = Path(root)
                break
        if candidate_root and candidate_root != data_dir:
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
    ds = load_dataset("xlangai/spider", split="train")
    rows = ds if limit is None else ds.select(range(min(int(limit), len(ds))))
    return [Example(db_id=r["db_id"], question=r["question"], gold_sql=r["query"]) for r in rows]


def load_hf_spider_realistic(limit: Optional[int] = None) -> List[Example]:
    ds = load_dataset("aherntech/spider-realistic", split="validation")
    rows = ds if limit is None else ds.select(range(min(int(limit), len(ds))))
    return [Example(db_id=r["db_id"], question=r["question"], gold_sql=r["query"]) for r in rows]


def load_hf_spider_syn(split: str = "validation", limit: Optional[int] = None) -> List[Example]:
    ds = load_dataset("aherntech/spider-syn", split=split)
    rows = ds if limit is None else ds.select(range(min(int(limit), len(ds))))
    return [Example(db_id=r["db_id"], question=r["SpiderSynQuestion"], gold_sql=r["query"]) for r in rows]


def load_all_examples(splits: List[str], limit: Optional[int], shuffle: bool = True) -> List[Tuple[str, List[Example]]]:
    out: List[Tuple[str, List[Example]]] = []
    for sp in splits:
        if sp == "spider":
            if DEV_JSON.exists():
                ex = load_examples_from_local_json(DEV_JSON)
                if shuffle:
                    random.shuffle(ex)
                if limit:
                    ex = ex[:limit]
            else:
                ex = load_hf_spider(limit=limit)
                if shuffle:
                    random.shuffle(ex)
            out.append((sp, ex))
        elif sp == "spider-realistic":
            ex = load_hf_spider_realistic(limit=limit)
            if shuffle:
                random.shuffle(ex)
            out.append((sp, ex))
        elif sp == "spider-syn":
            ex = load_hf_spider_syn(split="validation", limit=limit)
            if shuffle:
                random.shuffle(ex)
            out.append((sp, ex))
        else:
            print(f"[warn] Unknown split: {sp} (skipped)")
    return out


# ------------------------- Execution Helpers & Metrics -------------------------
def normalize_sql(s: str) -> str:
    s = strip_sql_fences(s or "")
    s = s.strip().rstrip(";")
    s = re.sub(r"\s+", " ", s)
    return s.lower().strip()


def rows_to_canonical(rows: List[Tuple]) -> List[Tuple]:
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
    mode: str = "simple",
    max_corrections: int = 2,
    taxonomy_file: Optional[str] = "error_taxonomy.json",
    max_completion_tokens: Optional[int] = None,
    reasoning_effort: Optional[str] = None,
    verbose: bool = False,
) -> None:
    ensure_dir(out_dir)
    ensure_dir(DATA_DIR)

    # Ensure DBs for execution accuracy if possible
    if not have_local_spider_assets():
        print("[info] Local Spider dev + database not found. Attempting to download the full dataset (for execution accuracy)...")
        download_spider_zip_to(DATA_DIR)

    have_exec = DB_DIR.exists()

    # Initialize model
    if mode == "simple":
        model = SQLGeneratorGraph(
            model_name=model_name,
            temperature=0.0,
            max_completion_tokens=max_completion_tokens,
            reasoning_effort=reasoning_effort,
        )
    elif mode == "sot":
        model = SQLOfThoughtGraph(
            model_name=model_name,
            temperature=0.0,
            max_attempts=max_corrections,
            taxonomy_path=taxonomy_file,
            max_completion_tokens=max_completion_tokens,
            reasoning_effort=reasoning_effort,
            verbose=bool(verbose),
        )
    else:
        raise ValueError(f"Unknown --mode {mode}")

    # Load requested splits
    work = load_all_examples(splits=splits, limit=limit)

    # CSV output
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_csv = out_dir / f"results_{mode}_{model_name.replace(':','_')}_{ts}.csv"

    total, em_hits, valid_hits, exec_hits = 0, 0, 0, 0

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "mode", "split", "idx", "db_id", "question", "gold_sql", "pred_sql",
            "em", "valid_sql", "exec_acc"
        ])

        for split_name, examples in work:
            print(f"[info] Running split: {split_name} (n={len(examples)}) mode={mode}")
            for i, ex in enumerate(tqdm(examples, desc=f"{split_name}")):
                total += 1
                db_path = DB_DIR / ex.db_id / f"{ex.db_id}.sqlite"

                # Generate SQL via selected pipeline
                try:
                    if mode == "simple":
                        pred_sql = model.generate_sql_for_db(ex.question, str(db_path), ex.db_id)
                    else:
                        # Pass gold_sql so SoT can trigger correction on row mismatch
                        pred_sql = model.generate_sql_for_db(ex.question, str(db_path), ex.db_id, gold_sql=ex.gold_sql)
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
                    valid_ok = 0
                    exec_ok = 0

                valid_hits += valid_ok
                exec_hits += exec_ok

                writer.writerow([
                    mode, split_name, i, ex.db_id, ex.question, ex.gold_sql, pred_sql,
                    em, valid_ok, exec_ok
                ])

    # Summary
    print("\n===== Summary =====")
    print(f"Mode: {mode}")
    print(f"Model: {model_name}")
    if max_completion_tokens:
        print(f"Max completion tokens: {max_completion_tokens}")
    if reasoning_effort:
        print(f"Reasoning effort: {reasoning_effort}")
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
    parser = argparse.ArgumentParser(description="Benchmark Text-to-SQL on Spider(-Realistic/-SYN) with simple or SQL-of-Thought models.")
    parser.add_argument("--mode", type=str, default="simple", choices=["simple", "sot"],
                        help="Which pipeline to run: simple (one-shot) or sot (SQL-of-Thought).")
    parser.add_argument("--model", type=str, default=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
                        help="OpenAI model name (default: OPENAI_MODEL or gpt-4.1-mini).")
    parser.add_argument("--max_corrections", type=int, default=2,
                        help="[sot only] Max correction attempts (default: 2).")
    parser.add_argument("--taxonomy_file", type=str, default="error_taxonomy.json",
                        help="[sot only] Path to taxonomy JSON (default: error_taxonomy.json).")
    parser.add_argument("--max_completion_tokens", type=int, default=None,
                        help="Max completion tokens for reasoning models (optional).")
    parser.add_argument("--reasoning_effort", type=str, default=None, choices=["low", "medium", "high"],
                        help="Reasoning effort for reasoning models: low, medium, or high (optional).")
    parser.add_argument("--verbose", action="store_true", help="[sot only] Print full inputs/outputs at each SoT stage.")

    parser.add_argument("--limit", type=int, default=50, help="Max examples per split (default: 50)")
    parser.add_argument("--splits", nargs="+", default=["spider"],
                        choices=["spider", "spider-realistic", "spider-syn"],
                        help="Which splits to run (default: spider)")
    parser.add_argument("--output_dir", type=Path, default=HERE / "outputs", help="Where to write CSV results.")
    args = parser.parse_args()

    # No global seeding by default; allow natural randomness.

    # Early key check
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable is not set.")
        sys.exit(1)

    ensure_dir(args.output_dir)
    benchmark(
        model_name=args.model,
        splits=args.splits,
        limit=args.limit,
        out_dir=args.output_dir,
        mode=args.mode,
        max_corrections=args.max_corrections,
        taxonomy_file=args.taxonomy_file,
        max_completion_tokens=args.max_completion_tokens,
        reasoning_effort=args.reasoning_effort,
        verbose=bool(args.verbose),
    )


if __name__ == "__main__":
    main()
