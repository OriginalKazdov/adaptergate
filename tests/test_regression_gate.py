"""Tests for the regression gate. No GPU required — uses a fake scorer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adaptergate.gating.regression_gate import (
    GateConfig,
    RegressionGate,
    append_audit,
)


def make_queries(n: int) -> list[dict]:
    return [{"question_id": f"q{i}"} for i in range(n)]


def constant_scorer(score_by_adapter: dict[str, float]):
    """A scorer that returns a fixed score per adapter, regardless of query."""
    def _scorer(adapter_id: str, query: dict) -> float:
        return score_by_adapter[adapter_id]
    return _scorer


def per_query_scorer(scores_by_adapter: dict[str, dict[str, float]]):
    """A scorer that returns adapter-specific, per-query scores."""
    def _scorer(adapter_id: str, query: dict) -> float:
        return scores_by_adapter[adapter_id][query["question_id"]]
    return _scorer


def test_accept_when_improvement():
    gate = RegressionGate(GateConfig(epsilon=0.02))
    scorer = constant_scorer({"baseline": 0.70, "candidate": 0.85})
    decision = gate.evaluate(
        tenant_id="t1",
        candidate_id="candidate",
        baseline_id="baseline",
        holdout=make_queries(50),
        scorer=scorer,
    )
    assert decision.accepted is True
    assert decision.delta == pytest.approx(0.15, abs=1e-6)
    assert "ACCEPTED" in decision.reason


def test_reject_when_regression_exceeds_epsilon():
    gate = RegressionGate(GateConfig(epsilon=0.02))
    scorer = constant_scorer({"baseline": 0.80, "candidate": 0.74})
    decision = gate.evaluate(
        tenant_id="t1",
        candidate_id="candidate",
        baseline_id="baseline",
        holdout=make_queries(50),
        scorer=scorer,
    )
    assert decision.accepted is False
    assert decision.delta == pytest.approx(-0.06, abs=1e-6)
    assert "REJECTED" in decision.reason


def test_accept_when_regression_within_tolerance():
    gate = RegressionGate(GateConfig(epsilon=0.02))
    scorer = constant_scorer({"baseline": 0.80, "candidate": 0.79})
    decision = gate.evaluate(
        tenant_id="t1",
        candidate_id="candidate",
        baseline_id="baseline",
        holdout=make_queries(50),
        scorer=scorer,
    )
    assert decision.accepted is True
    assert "tolerated regression" in decision.reason


def test_reject_when_holdout_too_small():
    gate = RegressionGate(GateConfig(min_holdout_size=20))
    scorer = constant_scorer({"baseline": 0.7, "candidate": 0.9})
    decision = gate.evaluate(
        tenant_id="t1",
        candidate_id="candidate",
        baseline_id="baseline",
        holdout=make_queries(5),
        scorer=scorer,
    )
    assert decision.accepted is False
    assert "Insufficient held-out set" in decision.reason


def test_reject_when_no_baseline_and_calibration_required():
    gate = RegressionGate(GateConfig(require_calibration=True))
    scorer = constant_scorer({"candidate": 0.95})
    decision = gate.evaluate(
        tenant_id="t1",
        candidate_id="candidate",
        baseline_id=None,
        holdout=make_queries(50),
        scorer=scorer,
    )
    assert decision.accepted is False
    assert "No baseline adapter" in decision.reason


def test_accept_when_no_baseline_and_calibration_disabled():
    gate = RegressionGate(GateConfig(require_calibration=False))
    scorer = constant_scorer({"candidate": 0.95})
    decision = gate.evaluate(
        tenant_id="t1",
        candidate_id="candidate",
        baseline_id=None,
        holdout=make_queries(50),
        scorer=scorer,
    )
    # Baseline counted as 0.0 across the board → candidate dominates.
    assert decision.accepted is True
    assert decision.delta == pytest.approx(0.95, abs=1e-6)


def test_strict_per_query_rejects_clean_regression():
    gate = RegressionGate(GateConfig(epsilon=0.10, strict_per_query=True))
    # Aggregate improves overall but ONE query that was perfect regresses.
    scorer = per_query_scorer(
        {
            "baseline": {q["question_id"]: 1.0 if i < 5 else 0.0 for i, q in enumerate(make_queries(30))},
            "candidate": {q["question_id"]: 0.0 if i == 0 else 1.0 for i, q in enumerate(make_queries(30))},
        }
    )
    decision = gate.evaluate(
        tenant_id="t1",
        candidate_id="candidate",
        baseline_id="baseline",
        holdout=make_queries(30),
        scorer=scorer,
    )
    assert decision.accepted is False
    assert "strict mode" in decision.reason.lower()


def test_per_query_breakdown_populated():
    gate = RegressionGate(GateConfig())
    scorer = per_query_scorer(
        {
            "baseline": {f"q{i}": 0.5 for i in range(30)},
            "candidate": {f"q{i}": (1.0 if i % 2 == 0 else 0.0) for i in range(30)},
        }
    )
    decision = gate.evaluate(
        tenant_id="t1",
        candidate_id="candidate",
        baseline_id="baseline",
        holdout=make_queries(30),
        scorer=scorer,
    )
    assert len(decision.per_query) == 30
    assert len(decision.improvements) == 15
    assert len(decision.regressions) == 15


def test_decision_serializes_to_json():
    gate = RegressionGate()
    scorer = constant_scorer({"b1": 0.7, "c1": 0.8})
    decision = gate.evaluate(
        tenant_id="t1",
        candidate_id="c1",
        baseline_id="b1",
        holdout=make_queries(50),
        scorer=scorer,
    )
    blob = decision.to_json()
    parsed = json.loads(blob)
    assert parsed["accepted"] is True
    assert parsed["tenant_id"] == "t1"


def test_append_audit_writes_jsonl(tmp_path: Path):
    gate = RegressionGate()
    scorer = constant_scorer({"b1": 0.7, "c1": 0.8})
    decision = gate.evaluate(
        tenant_id="t1",
        candidate_id="c1",
        baseline_id="b1",
        holdout=make_queries(50),
        scorer=scorer,
    )
    log = tmp_path / "audit.jsonl"
    append_audit(decision, log)
    append_audit(decision, log)
    lines = log.read_text().strip().split("\n")
    assert len(lines) == 2
    parsed = json.loads(lines[0])
    assert parsed["accepted"] is True
