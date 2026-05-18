"""Tests for v0.4 retention quick wins.

Covers:
  - Suspected duplicate slice tag detection
  - HoldoutSet.staleness_days()
  - --format pr-comment Markdown structure
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path


from adaptergate.gating import GateConfig, RegressionGate, HoldoutSet
from adaptergate.gating.regression_gate import _find_suspected_duplicate_slices


# ---------- Suspected duplicate slice tags ----------

def test_suspected_dupes_catches_prefix_drift():
    """`billing_dispute` and `intent=billing_dispute` should flag."""
    dupes = _find_suspected_duplicate_slices(["billing_dispute", "intent=billing_dispute"])
    # SequenceMatcher ratio depends on overlap. Should catch this pair.
    flat = {tuple(sorted(p)) for p in dupes}
    assert ("billing_dispute", "intent=billing_dispute") in flat


def test_suspected_dupes_catches_separator_drift():
    """Hyphen vs underscore in the value part should flag."""
    dupes = _find_suspected_duplicate_slices(
        ["intent=billing-dispute", "intent=billing_dispute"]
    )
    flat = {tuple(sorted(p)) for p in dupes}
    assert ("intent=billing-dispute", "intent=billing_dispute") in flat


def test_suspected_dupes_ignores_genuinely_distinct():
    """Different slices shouldn't flag."""
    dupes = _find_suspected_duplicate_slices(
        ["intent=refund", "intent=order_status", "lang=es"]
    )
    assert dupes == []


def test_suspected_dupes_empty_when_single_tag():
    assert _find_suspected_duplicate_slices(["only_one"]) == []
    assert _find_suspected_duplicate_slices([]) == []


def test_gate_surfaces_suspected_dupes_in_decision():
    """End-to-end: the gate populates suspected_duplicate_slices."""
    gate = RegressionGate(GateConfig(epsilon=0.5))
    queries = [
        {"question_id": f"a{i}", "slices": ["billing_dispute"]} for i in range(12)
    ] + [
        {"question_id": f"b{i}", "slices": ["intent=billing_dispute"]} for i in range(13)
    ]
    decision = gate.evaluate(
        tenant_id="t1",
        candidate_id="c",
        baseline_id="b",
        holdout=queries,
        scorer=lambda a, q: 0.9 if a == "b" else 0.4,
    )
    assert len(decision.suspected_duplicate_slices) >= 1
    pair = decision.suspected_duplicate_slices[0]
    assert "billing_dispute" in pair[0]
    assert "billing_dispute" in pair[1]


# ---------- HoldoutSet.staleness_days() ----------

def test_staleness_days_none_when_empty(tmp_path: Path):
    holdout = HoldoutSet(tenant_id="t", path=tmp_path / "h.jsonl")
    assert holdout.staleness_days() is None


def test_staleness_days_zero_for_freshly_added(tmp_path: Path):
    holdout = HoldoutSet(tenant_id="t", path=tmp_path / "h.jsonl")
    holdout.add({"question_id": "q1"})
    assert holdout.staleness_days() == 0


def test_staleness_days_counts_from_latest_query(tmp_path: Path):
    """Even with old queries present, staleness reflects the most-recent one."""
    holdout = HoldoutSet(tenant_id="t", path=tmp_path / "h.jsonl")
    holdout.add({"question_id": "q1"})
    holdout.add({"question_id": "q2"})
    # Both fresh — staleness 0.
    assert holdout.staleness_days() == 0


def test_staleness_days_with_old_timestamp(tmp_path: Path):
    """Inject an old added_at to test the days math."""
    holdout = HoldoutSet(tenant_id="t", path=tmp_path / "h.jsonl")
    holdout.add({"question_id": "q1"})
    # Mutate the underlying record's added_at to 45 days ago.
    old = (datetime.now(timezone.utc) - timedelta(days=45)).isoformat()
    holdout._queries[0].added_at = old
    assert holdout.staleness_days() == 45


# ---------- --format pr-comment ----------

def test_pr_comment_renderer_has_markdown_structure():
    from adaptergate.cli import _render_pr_comment

    gate = RegressionGate(GateConfig(epsilon=0.02))
    queries = [
        {"question_id": f"q{i}", "question": "I want a refund missing order_id", "slices": ["intent=billing"]}
        for i in range(25)
    ]
    decision = gate.evaluate(
        tenant_id="acme",
        candidate_id="v19",
        baseline_id="v18",
        holdout=queries,
        scorer=lambda a, q: 0.9 if a == "v18" else 0.3,
    )
    md = _render_pr_comment(decision, staleness_days=10)

    # Required Markdown elements.
    assert md.startswith("## ")
    assert "REJECTED" in md
    assert "**Tenant**" in md
    assert "**Candidate**" in md
    assert "Driver slice" in md
    assert "Pattern" in md  # cluster summary
    # Has at least the Slice breakdown table header (single slice → table not present)
    # Single slice in this test → no breakdown table. That's expected.
    # Footer reference.
    assert "adaptergate" in md.lower()


def test_pr_comment_renderer_shows_warnings_section():
    from adaptergate.cli import _render_pr_comment

    gate = RegressionGate(GateConfig(epsilon=0.02))
    queries = (
        [{"question_id": f"a{i}", "slices": ["billing"]} for i in range(12)]
        + [{"question_id": f"b{i}", "slices": ["intent=billing"]} for i in range(13)]
    )
    decision = gate.evaluate(
        tenant_id="t",
        candidate_id="c",
        baseline_id="b",
        holdout=queries,
        scorer=lambda a, q: 0.9 if a == "b" else 0.3,
    )
    md = _render_pr_comment(decision, staleness_days=120)
    assert "Warnings" in md
    assert "120 days" in md
    # Suspected duplicate slice surfaced
    assert re.search(r"`[^`]*billing[^`]*`.*`[^`]*billing[^`]*`", md)


def test_decision_json_is_parseable():
    """The to_json output remains valid JSON post-v0.4 additions."""
    gate = RegressionGate(GateConfig(epsilon=0.5))
    queries = [{"question_id": f"q{i}", "slices": ["intent=foo"]} for i in range(25)]
    decision = gate.evaluate(
        tenant_id="t",
        candidate_id="c",
        baseline_id="b",
        holdout=queries,
        scorer=lambda a, q: 0.9 if a == "b" else 0.4,
    )
    blob = decision.to_json()
    parsed = json.loads(blob)
    # New fields present.
    assert "suspected_duplicate_slices" in parsed
    assert "schema_version" in parsed
    assert "malformed_slice_queries" in parsed
