"""Tests for HoldoutSet."""

from __future__ import annotations

from pathlib import Path

import pytest

from adaptergate.gating.holdout_eval import HoldoutSet


def test_add_and_persist_roundtrip(tmp_path: Path):
    path = tmp_path / "holdout.jsonl"
    holdout = HoldoutSet(tenant_id="t1", path=path)
    holdout.add({"question_id": "q1", "question": "what?"}, accepted_by="v1")
    holdout.add({"question_id": "q2", "question": "why?"}, accepted_by="v1")
    holdout.add({"question_id": "q3", "question": "how?"}, accepted_by="v2")

    reloaded = HoldoutSet(tenant_id="t1", path=path)
    assert len(reloaded) == 3
    ids = [q.query_id for q in reloaded]
    assert ids == ["q1", "q2", "q3"]


def test_add_is_idempotent_by_query_id(tmp_path: Path):
    holdout = HoldoutSet(tenant_id="t1", path=tmp_path / "h.jsonl")
    holdout.add({"question_id": "q1"})
    holdout.add({"question_id": "q1"})
    holdout.add({"question_id": "q1", "extra": "ignored on second add"})
    assert len(holdout) == 1


def test_sample_deterministic_with_same_seed(tmp_path: Path):
    holdout = HoldoutSet(tenant_id="t1", path=tmp_path / "h.jsonl")
    for i in range(30):
        holdout.add({"question_id": f"q{i}"})
    s1 = holdout.sample(n=10, seed="adapter_v17")
    s2 = holdout.sample(n=10, seed="adapter_v17")
    assert [q["question_id"] for q in s1] == [q["question_id"] for q in s2]


def test_sample_differs_across_seeds(tmp_path: Path):
    holdout = HoldoutSet(tenant_id="t1", path=tmp_path / "h.jsonl")
    for i in range(30):
        holdout.add({"question_id": f"q{i}"})
    s1 = holdout.sample(n=10, seed="adapter_v17")
    s2 = holdout.sample(n=10, seed="adapter_v18")
    # Astronomically unlikely that two different seeds give same 10/30 sample
    assert [q["question_id"] for q in s1] != [q["question_id"] for q in s2]


def test_sample_returns_all_when_n_exceeds_size(tmp_path: Path):
    holdout = HoldoutSet(tenant_id="t1", path=tmp_path / "h.jsonl")
    for i in range(5):
        holdout.add({"question_id": f"q{i}"})
    sample = holdout.sample(n=100)
    assert len(sample) == 5


def test_sample_n_none_returns_all(tmp_path: Path):
    holdout = HoldoutSet(tenant_id="t1", path=tmp_path / "h.jsonl")
    for i in range(7):
        holdout.add({"question_id": f"q{i}"})
    sample = holdout.sample(n=None)
    assert len(sample) == 7


def test_max_size_truncates_oldest(tmp_path: Path):
    holdout = HoldoutSet(tenant_id="t1", path=tmp_path / "h.jsonl", max_size=5)
    for i in range(10):
        holdout.add({"question_id": f"q{i}"})
    assert len(holdout) == 5
    ids = [q.query_id for q in holdout]
    # Kept the most recent
    assert ids == ["q5", "q6", "q7", "q8", "q9"]


def test_max_size_zero_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="max_size must be"):
        HoldoutSet(tenant_id="t1", path=tmp_path / "h.jsonl", max_size=0)


def test_max_size_negative_raises(tmp_path: Path):
    with pytest.raises(ValueError):
        HoldoutSet(tenant_id="t1", path=tmp_path / "h.jsonl", max_size=-3)


def test_remove(tmp_path: Path):
    holdout = HoldoutSet(tenant_id="t1", path=tmp_path / "h.jsonl")
    holdout.add({"question_id": "q1"})
    holdout.add({"question_id": "q2"})
    assert holdout.remove("q1") is True
    assert holdout.remove("q_missing") is False
    assert len(holdout) == 1
    assert next(iter(holdout)).query_id == "q2"


def test_derive_id_prefers_question_id(tmp_path: Path):
    holdout = HoldoutSet(tenant_id="t1", path=tmp_path / "h.jsonl")
    rec = holdout.add({"question_id": "bird_42", "question": "hello"})
    assert rec.query_id == "bird_42"


def test_derive_id_falls_back_to_hash(tmp_path: Path):
    holdout = HoldoutSet(tenant_id="t1", path=tmp_path / "h.jsonl")
    rec = holdout.add({"text": "no id field"})
    assert len(rec.query_id) == 16  # sha256 truncated
    # Same content -> same id (idempotent)
    rec2 = holdout.add({"text": "no id field"})
    assert rec.query_id == rec2.query_id
    assert len(holdout) == 1
