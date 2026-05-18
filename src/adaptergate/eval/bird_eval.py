"""
BIRD-SQL evaluation harness.

Loads BIRD-SQL questions, executes predicted SQL against the target SQLite
database, compares result sets against the gold SQL output, computes:
  - Execution Accuracy (EA): % of predicted SQL that produces same result as gold
  - Reward-based Valid Efficiency Score (R-VES): penalizes slow queries

Designed to be model-agnostic: feed it (db_id, question, predicted_sql) tuples
from any source (API, local model, dataset).

Reference: BIRD-SQL paper https://arxiv.org/abs/2305.03111
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

# Reasonable cap to avoid runaway queries. BIRD queries should be sub-second.
QUERY_TIMEOUT_SECONDS = 30.0
# Cap on rows returned per query. Prevents OOM if model writes `SELECT * FROM huge_table`.
# Gold queries in BIRD all return < 10k rows in practice.
MAX_ROWS = 50_000


@dataclass
class BirdQuestion:
    question_id: int
    db_id: str
    question: str
    evidence: str  # business glossary / hints
    gold_sql: str
    difficulty: str  # "simple" | "moderate" | "challenging"


@dataclass
class EvalResult:
    question_id: int
    db_id: str
    predicted_sql: str
    gold_sql: str
    is_correct: bool
    pred_runtime_s: float
    gold_runtime_s: float
    error: str | None = None
    # R-VES weight: clamped speed ratio
    ves_weight: float = 0.0


@dataclass
class EvalSummary:
    total: int
    correct: int
    by_difficulty: dict[str, dict[str, int]] = field(default_factory=dict)
    by_db: dict[str, dict[str, int]] = field(default_factory=dict)
    ves_sum: float = 0.0

    @property
    def execution_accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def reward_ves(self) -> float:
        """R-VES: (1/N) * sum(correct * speed_weight) — BIRD official formula."""
        return self.ves_sum / self.total if self.total else 0.0


# ---------- Data loading ----------

def load_bird_questions(json_path: Path) -> list[BirdQuestion]:
    """Load BIRD-SQL dev/train questions from JSON."""
    with open(json_path) as f:
        raw = json.load(f)
    questions = []
    for item in raw:
        questions.append(BirdQuestion(
            question_id=item.get("question_id", len(questions)),
            db_id=item["db_id"],
            question=item["question"],
            evidence=item.get("evidence", ""),
            gold_sql=item["SQL"],
            difficulty=item.get("difficulty", "unknown"),
        ))
    return questions


def get_db_path(databases_dir: Path, db_id: str) -> Path:
    """BIRD layout: databases/<db_id>/<db_id>.sqlite"""
    return databases_dir / db_id / f"{db_id}.sqlite"


# ---------- Execution ----------

def execute_sql(
    db_path: Path,
    sql: str,
    timeout: float = QUERY_TIMEOUT_SECONDS,
    max_rows: int = MAX_ROWS,
) -> tuple[list | None, float, str | None]:
    """Execute SQL against SQLite with statement timeout + row cap.

    Returns ``(result_rows, runtime_seconds, error)``. ``result_rows`` is None
    when execution fails; ``error`` is None when execution succeeds.

    Statement timeout via SQLite progress handler (aborts long-running queries).
    Row cap via fetchmany to prevent OOM on ``SELECT * FROM huge_table``.
    """
    if not db_path.exists():
        return None, 0.0, f"db not found: {db_path}"

    conn = sqlite3.connect(str(db_path), timeout=2.0)  # lock timeout

    # Statement timeout via progress handler — SQLite invokes the callback
    # every N opcodes. Returning non-zero aborts the query.
    start = time.perf_counter()
    deadline = start + timeout

    def _abort_check():
        return 1 if time.perf_counter() > deadline else 0

    conn.set_progress_handler(_abort_check, 10_000)

    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        # Cap rows to avoid OOM on SELECT * FROM huge_table
        rows = cursor.fetchmany(max_rows)
        elapsed = time.perf_counter() - start
        return rows, elapsed, None
    except sqlite3.Error as e:
        elapsed = time.perf_counter() - start
        err_str = str(e)
        if "interrupted" in err_str.lower() or elapsed > timeout * 0.95:
            return None, elapsed, f"timeout after {elapsed:.1f}s"
        return None, elapsed, f"sqlite error: {err_str}"
    except Exception as e:
        elapsed = time.perf_counter() - start
        return None, elapsed, f"unexpected error: {e}"
    finally:
        try:
            conn.set_progress_handler(None, 0)
        except Exception:
            pass
        conn.close()


# ---------- Comparison ----------

def compare_results(pred_rows, gold_rows) -> bool:
    """BIRD official: set-comparison (order-independent), but tuples ordered."""
    if pred_rows is None or gold_rows is None:
        return False
    try:
        pred_set = set(map(tuple, pred_rows))
        gold_set = set(map(tuple, gold_rows))
        return pred_set == gold_set
    except (TypeError, ValueError):
        # Unhashable types (e.g. lists in rows) — fall back to list compare
        return sorted(map(str, pred_rows)) == sorted(map(str, gold_rows))


def compute_ves_weight(pred_time: float, gold_time: float) -> float:
    """BIRD R-VES weight: sqrt(gold/pred), clamped to [0.5, 2.0]."""
    if pred_time <= 0 or gold_time <= 0:
        return 1.0
    ratio = gold_time / pred_time
    if ratio < 0.25:
        return 0.5
    if ratio < 0.5:
        return 0.8
    if ratio < 1.0:
        return 1.0
    if ratio < 2.0:
        return 1.2
    if ratio < 4.0:
        return 1.5
    return 2.0


# ---------- Eval loop ----------

def evaluate_one(question: BirdQuestion, predicted_sql: str, databases_dir: Path) -> EvalResult:
    db_path = get_db_path(databases_dir, question.db_id)
    pred_rows, pred_time, pred_err = execute_sql(db_path, predicted_sql)
    gold_rows, gold_time, gold_err = execute_sql(db_path, question.gold_sql)

    if gold_err:
        # Gold SQL itself failed — this is rare but possible in dataset noise
        is_correct = False
    elif pred_err:
        is_correct = False
    else:
        is_correct = compare_results(pred_rows, gold_rows)

    ves_weight = compute_ves_weight(pred_time, gold_time) if is_correct else 0.0

    return EvalResult(
        question_id=question.question_id,
        db_id=question.db_id,
        predicted_sql=predicted_sql,
        gold_sql=question.gold_sql,
        is_correct=is_correct,
        pred_runtime_s=pred_time,
        gold_runtime_s=gold_time,
        error=pred_err,
        ves_weight=ves_weight,
    )


def evaluate_batch(
    questions: list[BirdQuestion],
    predictions: list[str],
    databases_dir: Path,
) -> tuple[list[EvalResult], EvalSummary]:
    assert len(questions) == len(predictions), "Predictions count must match questions"
    results = []
    summary = EvalSummary(total=0, correct=0)
    for q, pred in zip(questions, predictions):
        r = evaluate_one(q, pred, databases_dir)
        results.append(r)
        summary.total += 1
        if r.is_correct:
            summary.correct += 1
            summary.ves_sum += r.ves_weight

        # by-difficulty stats
        d = q.difficulty
        if d not in summary.by_difficulty:
            summary.by_difficulty[d] = {"total": 0, "correct": 0}
        summary.by_difficulty[d]["total"] += 1
        if r.is_correct:
            summary.by_difficulty[d]["correct"] += 1

        # by-db stats
        if q.db_id not in summary.by_db:
            summary.by_db[q.db_id] = {"total": 0, "correct": 0}
        summary.by_db[q.db_id]["total"] += 1
        if r.is_correct:
            summary.by_db[q.db_id]["correct"] += 1

    return results, summary


def format_summary(summary: EvalSummary) -> str:
    lines = []
    lines.append(f"Total: {summary.total}")
    lines.append(f"Execution Accuracy: {summary.execution_accuracy:.1%} "
                 f"({summary.correct}/{summary.total})")
    lines.append(f"R-VES Score: {summary.reward_ves:.3f}")
    lines.append("")
    lines.append("By difficulty:")
    for d, stats in sorted(summary.by_difficulty.items()):
        acc = stats["correct"] / stats["total"] if stats["total"] else 0
        lines.append(f"  {d:>12s}: {acc:.1%}  ({stats['correct']}/{stats['total']})")
    lines.append("")
    lines.append("By database:")
    for db, stats in sorted(summary.by_db.items()):
        acc = stats["correct"] / stats["total"] if stats["total"] else 0
        lines.append(f"  {db:>30s}: {acc:.1%}  ({stats['correct']}/{stats['total']})")
    return "\n".join(lines)


# ---------- Smoke test ----------

if __name__ == "__main__":
    # Quick smoke test of executor + comparator with synthetic data
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
        db_path = Path(f.name)
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE users (id INTEGER, name TEXT, age INTEGER)")
    conn.executemany("INSERT INTO users VALUES (?, ?, ?)",
                     [(1, "alice", 30), (2, "bob", 25), (3, "carol", 35)])
    conn.commit()
    conn.close()

    rows, t, err = execute_sql(db_path, "SELECT name FROM users WHERE age > 28 ORDER BY age")
    assert rows == [("alice",), ("carol",)], f"Got {rows}"
    print(f"smoke: SELECT execution OK ({t*1000:.1f}ms)")

    # Comparator: same set, different order
    assert compare_results([("alice",), ("carol",)], [("carol",), ("alice",)])
    assert not compare_results([("alice",)], [("alice",), ("carol",)])
    print("smoke: comparator OK")

    db_path.unlink()
    print("smoke test passed")
