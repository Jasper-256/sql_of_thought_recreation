"""
sql_of_thought_model_graph.py

A LangGraph implementation of SQL-of-Thought (SoT) as described in:
- "SQL-of-Thought: Multi-agentic Text-to-SQL with Guided Error Correction"
  (architecture figure on p.2; error taxonomy on p.5).  See paper for details.

Pipeline (single pass + guided correction loop):
    SchemaLink -> Subproblem -> QueryPlan(CoT) -> SQL -> Execute
      -> [if needed] CorrectionPlan(CoT + taxonomy) -> CorrectionSQL -> Execute ...
      (repeat until success or max_attempts)

Notes:
- Uses the helper utilities from sql_model_graph.py (summarize_sqlite_schema, strip_sql_fences).
- If `gold_sql` is provided and the DB is available, we compare execution results
  of candidate vs gold and trigger correction on mismatch (not just on exceptions).
- If the DB is missing, we still run the agentic pipeline but skip execution/correction.
"""

from __future__ import annotations
import json
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Reuse helpers from the simple model
from sql_model_graph import summarize_sqlite_schema, strip_sql_fences


# ----------------------------- Helpers -----------------------------
def _vprint(enabled: bool, title: str, content: str) -> None:
    if not enabled:
        return
    print("\n" + "=" * 20 + f" {title} " + "=" * 20)
    print(content)
    print("=" * (44 + len(title)))
def _make_llm(model_name: Optional[str] = None, temperature: float = 0.0, max_completion_tokens: Optional[int] = None, reasoning_effort: Optional[str] = None) -> ChatOpenAI:
    model = model_name or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    kwargs = {"model": model, "temperature": temperature}
    if max_completion_tokens:
        kwargs["max_completion_tokens"] = max_completion_tokens
    if reasoning_effort:
        # Pass explicitly to the client rather than via model_kwargs
        kwargs["reasoning_effort"] = reasoning_effort
    return ChatOpenAI(**kwargs)

def _safe_json_loads(text: str) -> Dict[str, Any]:
    """Best-effort JSON parser: strip fences & trailing commas."""
    if not text:
        return {}
    # strip code fences if present
    t = re.sub(r"```json\s*|\s*```", "", text, flags=re.IGNORECASE).strip()
    # remove trailing commas before } or ]
    t = re.sub(r",(\s*[\]\}])", r"\1", t)
    try:
        return json.loads(t)
    except Exception:
        return {}

def _rows_to_canonical(rows: List[Tuple]) -> List[Tuple]:
    """Normalize SQLite rows for set-like comparison."""
    def norm_val(v):
        if isinstance(v, bytes):
            v = v.decode("utf-8", errors="ignore")
        if isinstance(v, str):
            return v.strip().lower()
        return v
    canon = [tuple(norm_val(v) for v in row) for row in rows]
    canon.sort()
    return canon

def _exec_sql(db_path: Path, sql: str) -> Tuple[bool, Optional[List[Tuple]], Optional[str]]:
    """Execute SQL. Returns (ok, rows_or_none, error_message_or_none)."""
    if not db_path.exists():
        return False, None, "db_missing"
    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        conn.close()
        return True, rows, None
    except Exception as e:
        return False, None, str(e)

def _load_taxonomy(taxonomy_path: Optional[str]) -> Dict[str, Any]:
    # Default taxonomy mirrors the paper's p.5 figure (9 categories / 31 subtypes)
    default_taxonomy = {
        "Syntax": ["sql_syntax_error", "invalid_alias"],
        "Schema Link": ["table_missing", "col_missing", "ambiguous_col", "incorrect_foreign_key"],
        "Join": ["join_missing", "join_wrong_type", "extra_table", "incorrect_col"],
        "Filter": ["where_missing", "condition_wrong_col", "condition_type_mismatch"],
        "Aggregation": ["agg_no_groupby", "groupby_missing_col", "having_without_groupby", "having_incorrect", "having_vs_where"],
        "Value": ["hardcoded_value", "value_format_wrong"],
        "Subquery": ["unused_subquery", "subquery_missing", "subquery_correlation_error"],
        "Set Operations": ["union_missing", "intersect_missing", "except_missing"],
        "Other Issues": ["order_by_missing", "limit_missing", "duplicate_select", "unsupported_function", "extra_values_selected"],
    }
    if taxonomy_path and Path(taxonomy_path).exists():
        try:
            return json.loads(Path(taxonomy_path).read_text(encoding="utf-8"))
        except Exception:
            pass
    return default_taxonomy


# ----------------------------- Graph State -----------------------------
class SoTState(TypedDict, total=False):
    db_id: str
    db_path: str
    question: str
    full_schema: str
    cropped_schema: str
    subproblems_json: Dict[str, Any]
    plan: str
    sql: str
    attempt: int
    needs_correction: bool
    error_signal: str
    valid_sql: bool
    rows: List[Tuple]
    gold_sql: Optional[str]
    gold_rows: Optional[List[Tuple]]
    taxonomy: Dict[str, Any]
    model_name: str
    max_attempts: int
    max_completion_tokens: Optional[int]
    reasoning_effort: Optional[str]
    verbose: bool


# ----------------------------- System Prompts -----------------------------
SCHEMA_LINK_SYS = """
You are a Schema Agent in an NL2SQL framework. Given a natural language question and table schemas (with columns, PKs, and FKs), identify the relevant tables and columns needed, including intermediate tables for joins.

Cross-check your schema for:
- Missing or incorrect FK–PK relationships and add them
- Incomplete column selections (especially join keys)
- Table alias mismatches
- Linkage errors that would lead to incorrect joins or GROUP BY clauses

Return a list of lines in the format:
Table: primary_key_col, foreign_key_col, col1, col2, ...

ALWAYS list Foreign Key and Primary Key information.
ONLY list relevant tables and columns in the given format and no other extra characters.
"""

SUBPROBLEM_SYS = """
You are a Subproblem Agent in an NL2SQL framework. Your task is to decompose a natural language question into SQL subproblems.

You will be provided:
- A natural language question
- A textual schema summary that lists relevant tables and columns (generated by a Schema Agent)

Use this information to infer which SQL clauses are likely needed (e.g., WHERE, GROUP BY, JOIN, DISTINCT, ORDER BY, HAVING, EXCEPT, LIMIT, UNION).

Output a JSON object containing a list of subproblems like:
{
  "subproblems": [
    { "clause": "SELECT", "expression": "..." },
    { "clause": "JOIN", "expression": "..." }
  ]
}

Only output valid JSON — no markdown, no extra commentary.
"""

PLAN_SYS = """
You are a Query Plan Agent in an NL2SQL framework. Using the question, schema info, and subproblems, generate a step-by-step SQL query plan. Think through the process but OUTPUT ONLY the plan, not SQL.

Instructions:
- The plan must be in natural language with numbered steps.
- Explicitly mention which table and column to use for each step.
- All necessary tables appear in FROM and JOIN steps with correct join keys.
- GROUP BY, HAVING, ORDER BY, LIMIT appear only when needed and in the proper order.
- Do NOT write the final SQL query.

Return only the plan (no extra text).
"""

SQL_SYS = """
You are a world-class SQL writer AI in an NL2SQL multiagent framework. Your task is to write a single, syntactically correct SQL query that perfectly implements the provided query plan for a SQLite database.

Rules:
- Write ONLY the final valid SQL query. Do NOT include commentary or extra characters.
- Use the provided schema exactly (case-sensitive names).
- Prefer straightforward, correct SQL over cleverness.
- Use correct JOIN keys as indicated by foreign keys.
- No backticks; use double-quotes only if necessary.
"""

CORRECTION_PLAN_SYS = """
You are a Senior SQL Debugger in an NL2SQL multiagent framework. Your sole task is to analyze a failed SQL query to create a clear, step-by-step correction plan. Do NOT write the corrected SQL yourself.

You are an expert in a comprehensive error taxonomy (provided), including categories like:
- schema.mismatch (incorrect/missing tables/columns/functions or ambiguity)
- join.logic_error (missing joins, wrong foreign keys/columns, or unnecessary tables)
- filter.condition_error (wrong column/operator/value; WHERE vs HAVING confusion)
- aggregation.grouping_error (missing/incomplete GROUP BY; incorrect HAVING)
- select.output_error (wrong/extra/missing output columns or order)
- syntax.structural_error (syntax issues; missing ORDER BY/LIMIT/set operators)
- intent.semantic_error (captures the user's intent incorrectly; missing subquery; hardcoded values)

Return ONLY a numbered natural-language correction plan that a junior developer could follow. Do not generate SQL.
"""

CORRECTION_SQL_SYS = """
You are an expert SQL debugger AI in an NL2SQL multiagent framework. Your previous attempt to write a query failed. Analyze the feedback and your incorrect query, then generate a new, corrected query after reading the question and analyzing the relevant schema.

Write ONLY the final valid SQL query for SQLite. Do NOT include commentary or unnecessary characters.
"""


# ----------------------------- Node Functions -----------------------------
def _schema_link_node(state: SoTState) -> SoTState:
    llm = _make_llm(model_name=state["model_name"], max_completion_tokens=state.get("max_completion_tokens"), reasoning_effort=state.get("reasoning_effort"))
    user = f"""DB: {state['db_id']}
FULL SCHEMA:
{state['full_schema']}

QUESTION:
{state['question']}

Return only the relevant tables and columns in the specified format."""
    resp = llm.invoke([SystemMessage(content=SCHEMA_LINK_SYS), HumanMessage(content=user)])
    _vprint(bool(state.get("verbose", False)), "SCHEMA_LINK / INPUT", f"SYSTEM:\n{SCHEMA_LINK_SYS}\n\nUSER:\n{user}")
    _vprint(bool(state.get("verbose", False)), "SCHEMA_LINK / OUTPUT", resp.content)
    return {"cropped_schema": resp.content.strip()}

def _subproblem_node(state: SoTState) -> SoTState:
    llm = _make_llm(model_name=state["model_name"], max_completion_tokens=state.get("max_completion_tokens"), reasoning_effort=state.get("reasoning_effort"))
    user = f"""DB: {state['db_id']}

CROPPED SCHEMA:
{state['cropped_schema']}

QUESTION:
{state['question']}

Return ONLY valid JSON as specified."""
    resp = llm.invoke([SystemMessage(content=SUBPROBLEM_SYS), HumanMessage(content=user)])
    _vprint(bool(state.get("verbose", False)), "SUBPROBLEM / INPUT", f"SYSTEM:\n{SUBPROBLEM_SYS}\n\nUSER:\n{user}")
    _vprint(bool(state.get("verbose", False)), "SUBPROBLEM / OUTPUT (raw)", resp.content)
    sub = _safe_json_loads(resp.content)
    _vprint(bool(state.get("verbose", False)), "SUBPROBLEM / OUTPUT (parsed)", json.dumps(sub, indent=2, ensure_ascii=False))
    return {"subproblems_json": sub}

def _plan_node(state: SoTState) -> SoTState:
    llm = _make_llm(model_name=state["model_name"], max_completion_tokens=state.get("max_completion_tokens"), reasoning_effort=state.get("reasoning_effort"))
    sub_json = json.dumps(state.get("subproblems_json", {}), ensure_ascii=False)
    user = f"""DB: {state['db_id']}

CROPPED SCHEMA:
{state['cropped_schema']}

SUBPROBLEMS (JSON):
{sub_json}

QUESTION:
{state['question']}

Think step-by-step but OUTPUT ONLY a numbered procedural plan (no SQL)."""
    resp = llm.invoke([SystemMessage(content=PLAN_SYS), HumanMessage(content=user)])
    _vprint(bool(state.get("verbose", False)), "PLAN / INPUT", f"SYSTEM:\n{PLAN_SYS}\n\nUSER:\n{user}")
    _vprint(bool(state.get("verbose", False)), "PLAN / OUTPUT", resp.content)
    plan = resp.content.strip()
    return {"plan": plan}

def _sql_node(state: SoTState) -> SoTState:
    llm = _make_llm(model_name=state["model_name"], max_completion_tokens=state.get("max_completion_tokens"), reasoning_effort=state.get("reasoning_effort"))
    user = f"""DB: {state['db_id']}

FULL SCHEMA (reference):
{state['full_schema']}

CROPPED SCHEMA (primary working set; if required info is missing, consult FULL SCHEMA):
{state['cropped_schema']}

PLAN:
{state['plan']}

QUESTION:
{state['question']}

Return ONLY the final valid SQL query (no commentary). Prefer keys/joins from CROPPED SCHEMA; if absent there, use FK/PK details from FULL SCHEMA."""
    resp = llm.invoke([SystemMessage(content=SQL_SYS), HumanMessage(content=user)])
    _vprint(bool(state.get("verbose", False)), "SQL / INPUT", f"SYSTEM:\n{SQL_SYS}\n\nUSER:\n{user}")
    _vprint(bool(state.get("verbose", False)), "SQL / OUTPUT (raw)", resp.content)
    sql = strip_sql_fences(resp.content).strip()
    _vprint(bool(state.get("verbose", False)), "SQL / OUTPUT (clean)", sql)
    return {"sql": sql}

def _execute_node(state: SoTState) -> SoTState:
    """Execute the SQL; set needs_correction if exception OR (gold provided and rows mismatch)."""
    db_path = Path(state["db_path"])
    sql = state.get("sql", "")
    attempt = int(state.get("attempt", 0))
    ok, rows, err = _exec_sql(db_path, sql)
    needs = False
    error_signal = ""

    if not db_path.exists():
        # Without DB, we can't execute or correct meaningfully
        if state.get("verbose", False):
            _vprint(True, "EXECUTE", f"DB missing for {db_path}. SQL will not be executed.\nSQL:\n{sql}")
        return {
            "valid_sql": False,
            "rows": [],
            "needs_correction": False,
            "error_signal": "db_missing",
            "attempt": attempt
        }

    if not ok:
        needs = True
        error_signal = f"exception: {err or 'unknown'}"
    else:
        # If we have a gold reference and it executes, compare results
        gold_sql = state.get("gold_sql")
        if gold_sql:
            gok, grows, gerr = _exec_sql(db_path, gold_sql)
            if gok and grows is not None and rows is not None:
                if _rows_to_canonical(rows) != _rows_to_canonical(grows):
                    needs = True
                    error_signal = "result_mismatch"  # logical error despite valid SQL
            else:
                # If we can't execute gold, we at least keep the candidate result
                pass

    if state.get("verbose", False):
        details = [
            f"ATTEMPT: {attempt}",
            f"DB PATH: {db_path}",
            f"SQL:\n{sql}",
            f"OK: {ok}",
            f"ERROR: {err}",
            f"ROWS ({0 if rows is None else len(rows)}): {rows if rows is not None else []}",
            f"NEEDS_CORRECTION: {needs}",
            f"ERROR_SIGNAL: {error_signal}",
        ]
        _vprint(True, "EXECUTE", "\n\n".join(details))

    return {
        "valid_sql": bool(ok),
        "rows": rows or [],
        "needs_correction": needs,
        "error_signal": error_signal,
        "attempt": attempt
    }

def _correction_plan_node(state: SoTState) -> SoTState:
    llm = _make_llm(model_name=state["model_name"], max_completion_tokens=state.get("max_completion_tokens"), reasoning_effort=state.get("reasoning_effort"))
    taxonomy = state.get("taxonomy", {})
    tax_str = json.dumps(taxonomy, indent=2, ensure_ascii=False)

    user = f"""DB: {state['db_id']}

CROPPED SCHEMA:
{state['cropped_schema']}

QUESTION:
{state['question']}

PREVIOUS SQL:
{state['sql']}

ERROR SIGNAL:
{state.get('error_signal', '')}

ERROR TAXONOMY (reference only):
{tax_str}

Return ONLY a concise, numbered natural-language correction plan. Do not generate SQL."""
    resp = llm.invoke([SystemMessage(content=CORRECTION_PLAN_SYS), HumanMessage(content=user)])
    _vprint(bool(state.get("verbose", False)), "CORRECTION_PLAN / INPUT", f"SYSTEM:\n{CORRECTION_PLAN_SYS}\n\nUSER:\n{user}")
    _vprint(bool(state.get("verbose", False)), "CORRECTION_PLAN / OUTPUT", resp.content)
    return {"plan": resp.content.strip()}  # reuse 'plan' slot for the correction plan

def _correction_sql_node(state: SoTState) -> SoTState:
    llm = _make_llm(model_name=state["model_name"], max_completion_tokens=state.get("max_completion_tokens"), reasoning_effort=state.get("reasoning_effort"))
    user = f"""DB: {state['db_id']}

FULL SCHEMA (reference):
{state['full_schema']}

CROPPED SCHEMA (primary working set; if required info is missing, consult FULL SCHEMA):
{state['cropped_schema']}

QUESTION:
{state['question']}

CORRECTION PLAN:
{state['plan']}

PREVIOUS SQL (to fix):
{state['sql']}

Return ONLY the corrected SQL query (no commentary). Prefer keys/joins from CROPPED SCHEMA; if absent there, use FK/PK details from FULL SCHEMA."""
    resp = llm.invoke([SystemMessage(content=CORRECTION_SQL_SYS), HumanMessage(content=user)])
    _vprint(bool(state.get("verbose", False)), "CORRECTION_SQL / INPUT", f"SYSTEM:\n{CORRECTION_SQL_SYS}\n\nUSER:\n{user}")
    _vprint(bool(state.get("verbose", False)), "CORRECTION_SQL / OUTPUT (raw)", resp.content)
    sql = strip_sql_fences(resp.content).strip()
    _vprint(bool(state.get("verbose", False)), "CORRECTION_SQL / OUTPUT (clean)", sql)
    attempt = int(state.get("attempt", 0)) + 1
    return {"sql": sql, "attempt": attempt}


# ----------------------------- Public Wrapper -----------------------------
@dataclass
class SQLOfThoughtGraph:
    model_name: Optional[str] = None
    temperature: float = 0.0
    max_attempts: int = 2
    taxonomy_path: Optional[str] = "error_taxonomy.json"
    max_completion_tokens: Optional[int] = None
    reasoning_effort: Optional[str] = None
    verbose: bool = False

    def __post_init__(self):
        self.model_name = self.model_name or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self.taxonomy = _load_taxonomy(self.taxonomy_path)

        workflow = StateGraph(SoTState)

        # Nodes
        workflow.add_node("schema_link", _schema_link_node)
        workflow.add_node("subproblem", _subproblem_node)
        workflow.add_node("plan", _plan_node)
        workflow.add_node("sql", _sql_node)
        workflow.add_node("execute", _execute_node)
        workflow.add_node("correction_plan", _correction_plan_node)
        workflow.add_node("correction_sql", _correction_sql_node)

        # Linear prefix
        workflow.add_edge(START, "schema_link")
        workflow.add_edge("schema_link", "subproblem")
        workflow.add_edge("subproblem", "plan")
        workflow.add_edge("plan", "sql")
        workflow.add_edge("sql", "execute")

        # Conditional: correction loop
        def _need_fix(state: SoTState) -> str:
            needs = bool(state.get("needs_correction", False))
            attempt = int(state.get("attempt", 0))
            max_attempts = int(state.get("max_attempts", 1))
            if needs and attempt < max_attempts:
                return "fix"
            return "done"

        workflow.add_conditional_edges(
            "execute",
            _need_fix,
            {
                "fix": "correction_plan",
                "done": END,
            },
        )
        workflow.add_edge("correction_plan", "correction_sql")
        workflow.add_edge("correction_sql", "execute")

        self.app = workflow.compile()

    def generate_sql_for_db(
        self,
        question: str,
        db_path: str,
        db_id: str,
        gold_sql: Optional[str] = None,
    ) -> str:
        full_schema = summarize_sqlite_schema(db_path)
        init: SoTState = {
            "db_id": db_id,
            "db_path": db_path,
            "question": question,
            "full_schema": full_schema,
            "gold_sql": gold_sql,
            "taxonomy": self.taxonomy,
            "model_name": self.model_name,
            "max_attempts": int(self.max_attempts),
            "attempt": 0,
            "max_completion_tokens": self.max_completion_tokens,
            "reasoning_effort": self.reasoning_effort,
            "verbose": bool(self.verbose),
        }
        out: SoTState = self.app.invoke(init)
        return strip_sql_fences(out.get("sql", "")).strip()
