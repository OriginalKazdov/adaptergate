"""
Schema introspection + retrieval for BIRD-SQL databases.

For each db_id, extract DDL (CREATE TABLE statements) directly from the
SQLite file. Provides:
  - full schema (all tables) as text
  - retrieved schema (subset relevant to a question) — basic table-name match
    for now, swap for vector retrieval later in week 1.
"""

from __future__ import annotations

import sqlite3
import re
from pathlib import Path
from dataclasses import dataclass


@dataclass
class TableSchema:
    name: str
    ddl: str
    columns: list[tuple[str, str]]  # (col_name, col_type)
    sample_rows: list[tuple]


def extract_schema(db_path: Path, sample_n: int = 3) -> list[TableSchema]:
    """Pull CREATE TABLE + a few sample rows from each table."""
    if not db_path.exists():
        raise FileNotFoundError(f"db not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute(
        "SELECT name, sql FROM sqlite_master "
        "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    )
    tables_raw = cursor.fetchall()

    tables: list[TableSchema] = []
    for name, ddl in tables_raw:
        if ddl is None:
            continue

        cursor.execute(f"PRAGMA table_info('{name}')")
        cols = [(row[1], row[2]) for row in cursor.fetchall()]

        try:
            cursor.execute(f"SELECT * FROM '{name}' LIMIT {sample_n}")
            samples = cursor.fetchall()
        except sqlite3.Error:
            samples = []

        tables.append(TableSchema(
            name=name,
            ddl=ddl.strip(),
            columns=cols,
            sample_rows=samples,
        ))

    conn.close()
    return tables


def format_schema_text(tables: list[TableSchema], include_samples: bool = True) -> str:
    """Render schema as Markdown-ish text for LLM prompting."""
    parts = []
    for t in tables:
        parts.append(f"-- Table: {t.name}")
        parts.append(t.ddl + ";")
        if include_samples and t.sample_rows:
            parts.append(f"-- Sample rows (up to {len(t.sample_rows)}):")
            for row in t.sample_rows:
                truncated = tuple(
                    (s[:50] + "...") if isinstance(s, str) and len(s) > 50 else s
                    for s in row
                )
                parts.append(f"--   {truncated}")
        parts.append("")
    return "\n".join(parts)


def retrieve_relevant_tables(
    tables: list[TableSchema],
    question: str,
    evidence: str = "",
    top_k: int = 8,
) -> list[TableSchema]:
    """
    Naive lexical retrieval — score tables by token overlap with question+evidence.
    Swap for vector retrieval (Qdrant) in week 1 milestone 2.
    """
    text = (question + " " + evidence).lower()
    tokens = set(re.findall(r"[a-zA-Z_]+", text))

    scored = []
    for t in tables:
        # Tokens from table name + column names
        table_tokens = set(re.findall(r"[a-zA-Z_]+", t.name.lower()))
        for col, _ in t.columns:
            table_tokens.update(re.findall(r"[a-zA-Z_]+", col.lower()))
        overlap = len(tokens & table_tokens)
        scored.append((overlap, t))

    scored.sort(key=lambda x: -x[0])
    selected = [t for score, t in scored[:top_k] if score > 0]

    # Always include first 3 even if no overlap (might miss the join via foreign keys)
    if len(selected) < 3:
        for score, t in scored[:3]:
            if t not in selected:
                selected.append(t)

    return selected


if __name__ == "__main__":
    # Smoke test: create an in-memory DB and check extraction
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
        path = Path(f.name)
    conn = sqlite3.connect(str(path))
    conn.execute("""
        CREATE TABLE customers (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT,
            created_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            total REAL,
            FOREIGN KEY (customer_id) REFERENCES customers(id)
        )
    """)
    conn.executemany("INSERT INTO customers VALUES (?,?,?,?)",
                     [(1, "Alice", "a@x.com", "2024-01-01"),
                      (2, "Bob", "b@y.com", "2024-02-01")])
    conn.commit()
    conn.close()

    tables = extract_schema(path)
    assert len(tables) == 2, f"got {len(tables)}"
    text = format_schema_text(tables)
    assert "customers" in text and "orders" in text
    print(f"smoke: extracted {len(tables)} tables")

    relevant = retrieve_relevant_tables(tables, "Find customer emails", top_k=1)
    assert relevant[0].name == "customers", f"got {relevant[0].name}"
    print(f"smoke: retrieval picked '{relevant[0].name}'")
    path.unlink()
    print("schema smoke test passed")
