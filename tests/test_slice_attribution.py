"""Tests for slice-level attribution in GateDecision (v0.2 differential)."""

from __future__ import annotations

import json

import pytest

from adaptergate.gating import (
    GateConfig,
    RegressionGate,
)


def _q(qid: str, *slices: str) -> dict:
    """Build a query payload with the given slices."""
    return {"question_id": qid, "slices": list(slices)}


def _per_query_scorer(scores: dict[str, dict[str, float]]):
    """Scorer that returns score by (adapter_id, question_id)."""
    def _s(adapter_id: str, query: dict) -> float:
        return scores[adapter_id][query["question_id"]]
    return _s


# ---------- basic attribution ----------

def test_slice_attribution_is_empty_when_no_slices():
    gate = RegressionGate(GateConfig(epsilon=0.5))
    queries = [{"question_id": f"q{i}"} for i in range(30)]
    decision = gate.evaluate(
        tenant_id="t1",
        candidate_id="c1",
        baseline_id="b1",
        holdout=queries,
        scorer=lambda a, q: 0.8 if a == "b1" else 0.7,
    )
    assert decision.slice_attributions == []
    assert decision.driver_slice is None


def test_slice_attribution_per_tag():
    gate = RegressionGate(GateConfig(epsilon=0.5))
    # 20 billing queries that regress 0.9 → 0.3.
    # 10 order_status queries that regress 0.9 → 0.85 (barely).
    queries = [_q(f"b{i}", "intent=billing") for i in range(20)] + [
        _q(f"o{i}", "intent=order_status") for i in range(10)
    ]
    scorer = _per_query_scorer({
        "baseline": {**{f"b{i}": 0.9 for i in range(20)}, **{f"o{i}": 0.9 for i in range(10)}},
        "candidate": {**{f"b{i}": 0.3 for i in range(20)}, **{f"o{i}": 0.85 for i in range(10)}},
    })
    decision = gate.evaluate(
        tenant_id="t1",
        candidate_id="candidate",
        baseline_id="baseline",
        holdout=queries,
        scorer=scorer,
    )
    by_tag = {s.slice_tag: s for s in decision.slice_attributions}
    assert set(by_tag) == {"intent=billing", "intent=order_status"}

    billing = by_tag["intent=billing"]
    assert billing.n_total == 20
    assert billing.n_regressed == 20
    assert billing.score_baseline == pytest.approx(0.9, abs=1e-6)
    assert billing.score_candidate == pytest.approx(0.3, abs=1e-6)
    assert billing.delta == pytest.approx(-0.6, abs=1e-6)
    assert len(billing.regressed_query_ids) == 20

    order = by_tag["intent=order_status"]
    assert order.n_total == 10
    assert order.delta == pytest.approx(-0.05, abs=1e-6)


def test_driver_slice_is_most_regressed():
    gate = RegressionGate(GateConfig(epsilon=0.5))
    queries = [_q(f"b{i}", "intent=billing") for i in range(15)] + [
        _q(f"o{i}", "intent=order_status") for i in range(15)
    ]
    scorer = _per_query_scorer({
        "baseline": {**{f"b{i}": 0.9 for i in range(15)}, **{f"o{i}": 0.9 for i in range(15)}},
        "candidate": {**{f"b{i}": 0.4 for i in range(15)}, **{f"o{i}": 0.85 for i in range(15)}},
    })
    decision = gate.evaluate(
        tenant_id="t1",
        candidate_id="candidate",
        baseline_id="baseline",
        holdout=queries,
        scorer=scorer,
    )
    driver = decision.driver_slice
    assert driver is not None
    assert driver.slice_tag == "intent=billing"
    # And it's the first entry (most-regressed-first ordering).
    assert decision.slice_attributions[0].slice_tag == "intent=billing"


def test_driver_slice_none_when_no_regression():
    """If all slices improved, there is no driver."""
    gate = RegressionGate(GateConfig(epsilon=0.5))
    queries = [_q(f"q{i}", "intent=billing") for i in range(25)]
    scorer = _per_query_scorer({
        "baseline": {f"q{i}": 0.7 for i in range(25)},
        "candidate": {f"q{i}": 0.9 for i in range(25)},
    })
    decision = gate.evaluate(
        tenant_id="t1",
        candidate_id="candidate",
        baseline_id="baseline",
        holdout=queries,
        scorer=scorer,
    )
    # Slice exists but its delta is positive.
    assert len(decision.slice_attributions) == 1
    assert decision.slice_attributions[0].delta > 0
    assert decision.driver_slice is None


def test_query_can_belong_to_multiple_slices():
    """A query with multiple slice tags is counted in each attribution."""
    gate = RegressionGate(GateConfig(epsilon=0.5))
    queries = [_q(f"q{i}", "intent=billing", "lang=es") for i in range(20)]
    scorer = _per_query_scorer({
        "baseline": {f"q{i}": 0.9 for i in range(20)},
        "candidate": {f"q{i}": 0.4 for i in range(20)},
    })
    decision = gate.evaluate(
        tenant_id="t1",
        candidate_id="candidate",
        baseline_id="baseline",
        holdout=queries,
        scorer=scorer,
    )
    by_tag = {s.slice_tag: s for s in decision.slice_attributions}
    assert by_tag["intent=billing"].n_total == 20
    assert by_tag["lang=es"].n_total == 20
    # Identical deltas → tie-break by insertion order is fine.
    assert by_tag["intent=billing"].delta == pytest.approx(by_tag["lang=es"].delta, abs=1e-6)


# ---------- ordering ----------

def test_slice_attributions_sorted_most_regressed_first():
    gate = RegressionGate(GateConfig(epsilon=0.5))
    # Three slices with different drops.
    queries = (
        [_q(f"a{i}", "slice_a") for i in range(10)]
        + [_q(f"b{i}", "slice_b") for i in range(10)]
        + [_q(f"c{i}", "slice_c") for i in range(10)]
    )
    scorer = _per_query_scorer({
        "baseline": {**{f"a{i}": 1.0 for i in range(10)}, **{f"b{i}": 1.0 for i in range(10)}, **{f"c{i}": 1.0 for i in range(10)}},
        "candidate": {
            **{f"a{i}": 0.2 for i in range(10)},  # worst, Δ=-0.8
            **{f"b{i}": 0.6 for i in range(10)},  # middle, Δ=-0.4
            **{f"c{i}": 0.9 for i in range(10)},  # best, Δ=-0.1
        },
    })
    decision = gate.evaluate(
        tenant_id="t1",
        candidate_id="candidate",
        baseline_id="baseline",
        holdout=queries,
        scorer=scorer,
    )
    tags_in_order = [s.slice_tag for s in decision.slice_attributions]
    assert tags_in_order == ["slice_a", "slice_b", "slice_c"]


# ---------- serialization ----------

def test_slice_attributions_serialize_in_json():
    gate = RegressionGate(GateConfig(epsilon=0.5))
    queries = [_q(f"q{i}", "intent=billing") for i in range(25)]
    scorer = _per_query_scorer({
        "baseline": {f"q{i}": 0.9 for i in range(25)},
        "candidate": {f"q{i}": 0.4 for i in range(25)},
    })
    decision = gate.evaluate(
        tenant_id="t1",
        candidate_id="candidate",
        baseline_id="baseline",
        holdout=queries,
        scorer=scorer,
    )
    blob = decision.to_json()
    parsed = json.loads(blob)
    assert "slice_attributions" in parsed
    assert len(parsed["slice_attributions"]) == 1
    assert parsed["slice_attributions"][0]["slice_tag"] == "intent=billing"
    assert parsed["slice_attributions"][0]["n_regressed"] == 25
