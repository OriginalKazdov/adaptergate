"""Robustness tests for slice attribution: malformed inputs, missing IDs."""

from __future__ import annotations

from adaptergate.gating import GateConfig, RegressionGate


def _scorer_const(per_adapter: dict[str, float]):
    def _s(adapter_id: str, query: dict) -> float:
        return per_adapter[adapter_id]
    return _s


# ---------- malformed slices ----------

def test_slices_as_string_is_ignored_not_iterated_as_chars():
    """If a customer hands us ``slices="intent=foo"`` instead of a list,
    we must not iterate the characters and create per-character slices."""
    gate = RegressionGate(GateConfig(epsilon=0.5))
    queries = [
        {"question_id": f"q{i}", "slices": "intent=foo"}  # malformed — string
        for i in range(25)
    ]
    decision = gate.evaluate(
        tenant_id="t1",
        candidate_id="c",
        baseline_id="b",
        holdout=queries,
        scorer=_scorer_const({"b": 0.9, "c": 0.4}),
    )
    # No per-character slices like "i", "n", "t", ...
    assert decision.slice_attributions == []
    assert decision.driver_slice is None


def test_slices_with_mixed_types_keeps_only_strings():
    """Non-string entries in the slices list are dropped silently."""
    gate = RegressionGate(GateConfig(epsilon=0.5))
    queries = [
        {"question_id": f"q{i}", "slices": ["intent=billing", 42, None, {"x": 1}]}
        for i in range(25)
    ]
    decision = gate.evaluate(
        tenant_id="t1",
        candidate_id="c",
        baseline_id="b",
        holdout=queries,
        scorer=_scorer_const({"b": 0.9, "c": 0.4}),
    )
    tags = [s.slice_tag for s in decision.slice_attributions]
    assert tags == ["intent=billing"]


def test_slices_missing_field_is_fine():
    """Queries without a slices field produce no per-slice attribution."""
    gate = RegressionGate(GateConfig(epsilon=0.5))
    queries = [{"question_id": f"q{i}"} for i in range(25)]
    decision = gate.evaluate(
        tenant_id="t1",
        candidate_id="c",
        baseline_id="b",
        holdout=queries,
        scorer=_scorer_const({"b": 0.9, "c": 0.4}),
    )
    assert decision.slice_attributions == []


def test_slices_explicit_none_is_fine():
    """``slices: None`` is treated the same as missing."""
    gate = RegressionGate(GateConfig(epsilon=0.5))
    queries = [{"question_id": f"q{i}", "slices": None} for i in range(25)]
    decision = gate.evaluate(
        tenant_id="t1",
        candidate_id="c",
        baseline_id="b",
        holdout=queries,
        scorer=_scorer_const({"b": 0.9, "c": 0.4}),
    )
    assert decision.slice_attributions == []


# ---------- missing query_id ----------

def test_query_without_any_id_field_still_runs():
    """When a query has neither question_id, id, nor query_id, the gate must
    not crash. The resulting query_id is None and that's acceptable."""
    gate = RegressionGate(GateConfig(epsilon=0.5))
    queries = [{"text": f"sample-{i}", "slices": ["intent=foo"]} for i in range(25)]
    decision = gate.evaluate(
        tenant_id="t1",
        candidate_id="c",
        baseline_id="b",
        holdout=queries,
        scorer=_scorer_const({"b": 0.9, "c": 0.4}),
    )
    assert decision.slice_attributions
    # All query_ids end up as None — that's acceptable, just don't crash.
    assert all(qid is None for qid in decision.slice_attributions[0].regressed_query_ids)
    # Aggregate stats are still right.
    assert decision.slice_attributions[0].n_total == 25
    assert decision.slice_attributions[0].n_regressed == 25
