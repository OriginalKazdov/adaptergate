"""Tests for ReplayBuffer."""

from __future__ import annotations

from pathlib import Path

import pytest

from adaptergate.gating.regression_gate import GateConfig, RegressionGate
from adaptergate.gating.replay_buffer import ReplayBuffer


def _make_decision(tenant: str, candidate: str, baseline: str | None, score_b: float, score_c: float):
    """Helper: build a real GateDecision via the gate (not a fake)."""
    gate = RegressionGate(GateConfig(epsilon=0.02))
    queries = [{"question_id": f"q{i}"} for i in range(30)]
    return gate.evaluate(
        tenant_id=tenant,
        candidate_id=candidate,
        baseline_id=baseline,
        holdout=queries,
        scorer=lambda a, q: score_b if a == baseline else score_c,
    )


def test_add_rejected_persists(tmp_path: Path):
    decision = _make_decision("t1", "v17", "v16", 0.85, 0.70)
    assert decision.accepted is False  # sanity

    buf = ReplayBuffer(tenant_id="t1", path=tmp_path / "replay.jsonl")
    buf.add(decision)

    reloaded = ReplayBuffer(tenant_id="t1", path=tmp_path / "replay.jsonl")
    assert len(reloaded) == 1
    rec = next(iter(reloaded))
    assert rec.candidate_id == "v17"
    assert rec.baseline_id == "v16"
    assert "REJECTED" in rec.reason


def test_add_accepted_raises(tmp_path: Path):
    decision = _make_decision("t1", "v17", "v16", 0.70, 0.85)
    assert decision.accepted is True

    buf = ReplayBuffer(tenant_id="t1", path=tmp_path / "replay.jsonl")
    with pytest.raises(ValueError, match="expects a rejected decision"):
        buf.add(decision)


def test_tenant_mismatch_raises(tmp_path: Path):
    decision = _make_decision("t1", "v17", "v16", 0.85, 0.70)
    buf = ReplayBuffer(tenant_id="t_other", path=tmp_path / "replay.jsonl")
    with pytest.raises(ValueError, match="tenant_id"):
        buf.add(decision)


def test_max_size_truncates(tmp_path: Path):
    buf = ReplayBuffer(tenant_id="t1", path=tmp_path / "replay.jsonl", max_size=3)
    for i in range(5):
        decision = _make_decision("t1", f"v{i}", f"base{i}", 0.85, 0.70)
        buf.add(decision)
    assert len(buf) == 3
    # Most recent kept
    ids = [r.candidate_id for r in buf]
    assert ids == ["v2", "v3", "v4"]


def test_max_size_zero_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="max_size must be"):
        ReplayBuffer(tenant_id="t1", path=tmp_path / "r.jsonl", max_size=0)


def test_recent_returns_last_n(tmp_path: Path):
    buf = ReplayBuffer(tenant_id="t1", path=tmp_path / "replay.jsonl")
    for i in range(5):
        decision = _make_decision("t1", f"v{i}", f"base{i}", 0.85, 0.70)
        buf.add(decision)
    recent = buf.recent(n=2)
    assert len(recent) == 2
    assert [r.candidate_id for r in recent] == ["v3", "v4"]


def test_empty_buffer_len_zero(tmp_path: Path):
    buf = ReplayBuffer(tenant_id="t1", path=tmp_path / "replay.jsonl")
    assert len(buf) == 0
    assert list(buf) == []
