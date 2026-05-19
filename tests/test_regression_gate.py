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


# ─── slice_epsilon: silent-slice-regression safety net ────────────────────────

def _silent_slice_holdout(n_silent: int = 10, n_rest: int = 50) -> list[dict]:
    """Build a held-out set where ``n_silent`` queries belong to a small slice
    (``intent=high_value``) and ``n_rest`` queries belong to a much larger slice
    (``intent=routine``). Used to construct the silent-slice-collapse scenario.
    """
    queries: list[dict] = []
    for i in range(n_silent):
        queries.append({"question_id": f"hv_{i}", "slices": ["intent=high_value"]})
    for i in range(n_rest):
        queries.append({"question_id": f"r_{i}", "slices": ["intent=routine"]})
    return queries


def _silent_slice_scorer(rest_score: float = 1.0, hv_collapses: bool = True):
    """Scorer where baseline=1.0 on every query, candidate=1.0 on routine,
    but collapses to 0.0 on high_value slice when ``hv_collapses=True``.
    """
    def _scorer(adapter_id: str, query: dict) -> float:
        if adapter_id == "baseline":
            return 1.0
        # candidate
        if "intent=high_value" in query.get("slices", []) and hv_collapses:
            return 0.0
        return rest_score
    return _scorer


def test_slice_epsilon_none_preserves_legacy_behavior():
    """When slice_epsilon is None (default), aggregate is the only signal."""
    gate = RegressionGate(GateConfig(epsilon=0.20, slice_epsilon=None))
    scorer = _silent_slice_scorer()
    decision = gate.evaluate(
        tenant_id="t1",
        candidate_id="candidate",
        baseline_id="baseline",
        holdout=_silent_slice_holdout(n_silent=10, n_rest=50),
        scorer=scorer,
    )
    # Aggregate: 1.0 → (50*1.0 + 10*0.0) / 60 = 0.833. Δ = -0.167, within ε=0.20.
    assert decision.delta == pytest.approx(-10 / 60, abs=1e-3)
    assert decision.accepted is True, "Legacy: aggregate within ε accepts even with collapsed slice"
    # But attribution still shows it.
    driver = decision.driver_slice
    assert driver is not None
    assert driver.slice_tag == "intent=high_value"
    assert driver.delta == pytest.approx(-1.0, abs=1e-6)


def test_slice_epsilon_rejects_silent_slice_collapse():
    """slice_epsilon=0.10 rejects when a slice collapses despite aggregate-within-ε."""
    gate = RegressionGate(GateConfig(epsilon=0.20, slice_epsilon=0.10))
    scorer = _silent_slice_scorer()
    decision = gate.evaluate(
        tenant_id="t1",
        candidate_id="candidate",
        baseline_id="baseline",
        holdout=_silent_slice_holdout(n_silent=10, n_rest=50),
        scorer=scorer,
    )
    assert decision.accepted is False
    assert "slice regression" in decision.reason.lower()
    assert "intent=high_value" in decision.reason
    assert "silent-regression" in decision.reason.lower()


def test_slice_epsilon_ignores_slices_below_min_size():
    """A 1-query slice should not trigger rejection even if it collapses 100%."""
    gate = RegressionGate(GateConfig(epsilon=0.05, slice_epsilon=0.10, slice_min_size=3))
    # Construct: 1 query in a "tiny" slice (collapses) + 50 routine queries (fine).
    queries = (
        [{"question_id": "tiny_0", "slices": ["intent=tiny"]}]
        + [{"question_id": f"r_{i}", "slices": ["intent=routine"]} for i in range(50)]
    )

    def scorer(adapter_id: str, q: dict) -> float:
        if adapter_id == "baseline":
            return 1.0
        return 0.0 if "intent=tiny" in q.get("slices", []) else 1.0

    decision = gate.evaluate(
        tenant_id="t1",
        candidate_id="candidate",
        baseline_id="baseline",
        holdout=queries,
        scorer=scorer,
    )
    # Aggregate: 50/51 → Δ ≈ -0.0196, within ε=0.05.
    assert decision.accepted is True, "Slice below min_size cannot trigger rejection"


def test_slice_epsilon_accepts_when_no_slice_collapses():
    """If all slices stay within slice_epsilon, the gate accepts normally."""
    gate = RegressionGate(GateConfig(epsilon=0.05, slice_epsilon=0.10))
    # Candidate matches baseline exactly — no regression anywhere.
    scorer = _silent_slice_scorer(hv_collapses=False)
    decision = gate.evaluate(
        tenant_id="t1",
        candidate_id="candidate",
        baseline_id="baseline",
        holdout=_silent_slice_holdout(),
        scorer=scorer,
    )
    assert decision.accepted is True
    assert decision.delta == pytest.approx(0.0, abs=1e-6)


def test_slice_epsilon_still_rejects_aggregate_when_both_violated():
    """When both aggregate AND a slice exceed their thresholds, the reason must
    accurately describe both — NOT lie about aggregate being within ε."""
    # Heavy collapse: 30 high_value queries all drop to 0, only 20 routine survive.
    # Aggregate: 20/50 → 0.40 → Δ=-0.60, well above ε=0.05.
    # Slice 'intent=high_value': 30 queries collapse to 0 → Δ=-1.0, above slice_ε=0.10.
    gate = RegressionGate(GateConfig(epsilon=0.05, slice_epsilon=0.10))
    holdout = _silent_slice_holdout(n_silent=30, n_rest=20)
    scorer = _silent_slice_scorer()
    decision = gate.evaluate(
        tenant_id="t1",
        candidate_id="candidate",
        baseline_id="baseline",
        holdout=holdout,
        scorer=scorer,
    )
    assert decision.accepted is False
    # The reason must NOT claim aggregate is within ε — it is not.
    assert "within ε" not in decision.reason, (
        f"Reason must not say aggregate is within ε when it isn't. Got: {decision.reason!r}"
    )
    # The reason should call out BOTH the aggregate and the slice violation.
    assert "aggregate" in decision.reason.lower()
    assert "slice" in decision.reason.lower()
    assert "exceeds ε" in decision.reason
    assert "intent=high_value" in decision.reason


def test_slice_epsilon_silent_case_reason_is_accurate():
    """The silent-regression case (aggregate within ε, slice collapsed) reason
    must say exactly that."""
    gate = RegressionGate(GateConfig(epsilon=0.20, slice_epsilon=0.10))
    decision = gate.evaluate(
        tenant_id="t1",
        candidate_id="candidate",
        baseline_id="baseline",
        holdout=_silent_slice_holdout(n_silent=10, n_rest=50),
        scorer=_silent_slice_scorer(),
    )
    assert decision.accepted is False
    # Aggregate is within ε here, so the silent-regression language is correct.
    assert "within ε" in decision.reason
    assert "silent-regression case" in decision.reason
