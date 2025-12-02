"""
sql_model_graph.py
A minimal LangGraph that makes a single LLM call to generate SQL for a given
Spider-style question and SQLite schema. No tool use, no agent loops.
"""

from __future__ import annotations
import os
import re
import sqlite3
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage


# --------- Helpers ---------
def strip_sql_fences(text: str) -> str:
    """Remove Markdown fences and stray prefixes to leave only SQL."""
    if text is None:
        return ""
    # remove triple backticks with optional language tag
    text = re.sub(r"```sql\s*|\s*```", "", text, flags=re.IGNORECASE).strip()
    # common "SQL:" prefix
    text = re.sub(r"^\s*SQL\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    # collapse multiple semicolons/newlines
    text = re.sub(r";+\s*$", ";", text).strip()
    return text


def summarize_sqlite_schema(db_path: str) -> str:
    """
    Introspect a SQLite DB to a compact, LLM-friendly schema string listing
    tables, columns, PKs and FKs. Works for Spider DBs.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Get table names (skip sqlite internal)
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    tables = [r["name"] for r in cur.fetchall()]

    lines = []
    for t in sorted(tables):
        # columns
        cur.execute(f"PRAGMA table_info('{t}')")
        cols = cur.fetchall()
        col_desc = []
        pks = []
        for c in cols:
            cname = c["name"]
            ctype = c["type"] or "TEXT"
            if c["pk"]:
                pks.append(cname)
            col_desc.append(f"{cname} {ctype}")
        lines.append(f"TABLE {t} ( " + ", ".join(col_desc) + " )")
        if pks:
            lines.append(f"  PRIMARY KEY: {', '.join(pks)}")

        # foreign keys
        cur.execute(f"PRAGMA foreign_key_list('{t}')")
        fks = cur.fetchall()
        for fk in fks:
            lines.append(f"  FOREIGN KEY: {t}.{fk['from']} -> {fk['table']}.{fk['to']}")

        lines.append("")  # spacer

    conn.close()
    return "\n".join(lines).strip()


# --------- LangGraph State & Node ---------
class SQLGenState(TypedDict, total=False):
    db_id: str
    question: str
    schema: str
    sql: str


SYSTEM_PROMPT = """You are an expert SQL generator for SQLite.
Follow these rules strictly:
- Produce ONLY a valid SQL statement (no explanations).
- Use the provided schema exactly (table/column names are case-sensitive).
- Prefer straightforward, correct SQL over cleverness.
- If a JOIN is required, join on correct key columns from the schema.
- Do not use backticks, only double-quotes if quoting is necessary.
- End with a single semicolon.
"""


def _make_llm(model_name: Optional[str] = None, temperature: float = 0.0, max_completion_tokens: Optional[int] = None, reasoning_effort: Optional[str] = None):
    """Create an LLM instance. Supports OpenAI and Anthropic models.
    
    Use 'anth:' prefix for Anthropic models (e.g., 'anth:claude-sonnet-4-5').
    """
    model = model_name or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    
    # Check for Anthropic prefix
    if model.startswith("anth:"):
        anthropic_model = model[5:]  # Strip 'anth:' prefix
        kwargs = {"model": anthropic_model, "temperature": temperature}
        if max_completion_tokens:
            kwargs["max_tokens"] = max_completion_tokens
        else:
            kwargs["max_tokens"] = 4096  # Anthropic requires max_tokens
        # Note: reasoning_effort is OpenAI-specific, ignored for Anthropic
        return ChatAnthropic(**kwargs)
    
    # Default to OpenAI
    kwargs = {"model": model, "temperature": temperature}
    if max_completion_tokens:
        kwargs["max_completion_tokens"] = max_completion_tokens
    if reasoning_effort:
        # Pass explicitly to the client rather than via model_kwargs
        kwargs["reasoning_effort"] = reasoning_effort
    return ChatOpenAI(**kwargs)


def llm_node(state: SQLGenState, model_name: Optional[str] = None, max_completion_tokens: Optional[int] = None, reasoning_effort: Optional[str] = None) -> SQLGenState:
    """
    The single node: call the LLM once with the schema and question.
    """
    question = state["question"]
    schema = state["schema"]
    db_id = state.get("db_id", "unknown_db")

    llm = _make_llm(model_name=model_name, max_completion_tokens=max_completion_tokens, reasoning_effort=reasoning_effort)

    user = f"""You are writing SQL for database: {db_id}

SCHEMA (SQLite):
{schema}

QUESTION:
{question}

Return ONLY the SQL; no prose, no comments.
"""
    resp = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user)])
    sql = strip_sql_fences(resp.content)
    return {"sql": sql}


# --------- Public wrapper ---------
class SQLGeneratorGraph:
    """
    Minimal wrapper around a 1-node LangGraph that produces SQL strings.
    """

    def __init__(self, model_name: Optional[str] = None, temperature: float = 0.0, max_completion_tokens: Optional[int] = None, reasoning_effort: Optional[str] = None):
        self.model_name = model_name or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.reasoning_effort = reasoning_effort

        workflow = StateGraph(SQLGenState)
        workflow.add_node("generate_sql", lambda s: llm_node(s, model_name=self.model_name, max_completion_tokens=self.max_completion_tokens, reasoning_effort=self.reasoning_effort))
        workflow.add_edge(START, "generate_sql")
        workflow.add_edge("generate_sql", END)
        self.app = workflow.compile()

    def generate_sql_for_db(self, question: str, db_path: str, db_id: str) -> str:
        schema_text = summarize_sqlite_schema(db_path)
        out: SQLGenState = self.app.invoke({"db_id": db_id, "question": question, "schema": schema_text})
        return strip_sql_fences(out.get("sql", "")).strip()
