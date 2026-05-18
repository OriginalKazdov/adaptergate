"""Tests for BIRD-SQL eval primitives.

Covers the bits that don't require torch: SQL execution against a tmp SQLite
DB, set-comparison logic, and the R-VES weight formula.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from adaptergate.eval.bird_eval import (
    compare_results,
    compute_ves_weight,
    execute_sql,
)


def _make_tmp_db(tmp_path: Path) -> Path:
    """Tiny fixture DB used by the execution tests."""
    db_path = tmp_path / "fixture.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE users (id INTEGER, name TEXT, age INTEGER)")
    conn.executemany(
        "INSERT INTO users VALUES (?, ?, ?)",
        [(1, "alice", 30), (2, "bob", 25), (3, "carol", 35)],
    )
    conn.commit()
    conn.close()
    return db_path


# ---------- compare_results ----------

def test_compare_results_same_set_different_order():
    a = [("alice",), ("bob",), ("carol",)]
    b = [("carol",), ("alice",), ("bob",)]
    assert compare_results(a, b) is True


def test_compare_results_different_sets():
    assert compare_results([("alice",)], [("alice",), ("bob",)]) is False


def test_compare_results_empty_equal():
    assert compare_results([], []) is True


def test_compare_results_none_inputs_false():
    assert compare_results(None, [("a",)]) is False
    assert compare_results([("a",)], None) is False
    assert compare_results(None, None) is False


def test_compare_results_unhashable_rows_falls_back():
    # Rows containing lists are unhashable; comparator must not crash.
    a = [([1, 2],), ([3, 4],)]
    b = [([3, 4],), ([1, 2],)]
    # Falls back to sorted-string compare; same content should still match.
    assert compare_results(a, b) is True


# ---------- compute_ves_weight ----------

@pytest.mark.parametrize(
    # ratio = gold_t / pred_t. Bucket boundaries from compute_ves_weight:
    #   < 0.25 → 0.5    [0.25, 0.5) → 0.8    [0.5, 1.0) → 1.0
    #   [1.0, 2.0) → 1.2    [2.0, 4.0) → 1.5    >= 4.0 → 2.0
    "pred_t, gold_t, expected",
    [
        (1.0, 0.1, 0.5),    # ratio 0.1 < 0.25
        (1.0, 0.3, 0.8),    # ratio 0.3 in [0.25, 0.5)
        (1.0, 0.7, 1.0),    # ratio 0.7 in [0.5, 1.0)
        (1.0, 1.5, 1.2),    # ratio 1.5 in [1.0, 2.0)
        (1.0, 3.0, 1.5),    # ratio 3.0 in [2.0, 4.0)
        (1.0, 10.0, 2.0),   # ratio 10.0 >= 4.0
    ],
)
def test_compute_ves_weight_buckets(pred_t: float, gold_t: float, expected: float):
    assert compute_ves_weight(pred_t, gold_t) == expected


def test_compute_ves_weight_zero_time_returns_one():
    # Avoid divide-by-zero — fall back to neutral weight.
    assert compute_ves_weight(0.0, 1.0) == 1.0
    assert compute_ves_weight(1.0, 0.0) == 1.0


# ---------- execute_sql ----------

def test_execute_sql_select_returns_rows(tmp_path: Path):
    db = _make_tmp_db(tmp_path)
    rows, runtime, err = execute_sql(db, "SELECT name FROM users WHERE age > 28 ORDER BY age")
    assert err is None
    assert rows == [("alice",), ("carol",)]
    assert runtime >= 0.0


def test_execute_sql_missing_db_returns_error(tmp_path: Path):
    rows, runtime, err = execute_sql(tmp_path / "nope.sqlite", "SELECT 1")
    assert rows is None
    assert err is not None
    assert "db not found" in err


def test_execute_sql_syntax_error_returns_error(tmp_path: Path):
    db = _make_tmp_db(tmp_path)
    rows, runtime, err = execute_sql(db, "this is not sql")
    assert rows is None
    assert err is not None
    assert "sqlite error" in err.lower()


def test_execute_sql_respects_max_rows(tmp_path: Path):
    db = _make_tmp_db(tmp_path)
    rows, _, err = execute_sql(db, "SELECT name FROM users", max_rows=2)
    assert err is None
    assert len(rows) == 2
