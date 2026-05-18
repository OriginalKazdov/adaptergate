"""Regression gate — the do-no-harm decision for adapter updates.

The wedge: prevent silent quality regression when serving updated LoRA
adapters. Before promoting a candidate adapter to production, run it against
a held-out eval set. If aggregate accuracy drops more than `epsilon`, reject.

Inspiration:
  - Online-LoRA (arxiv 2411.05663) distribution-shift detection
  - Silent Collapse MTR (arxiv 2605.14588) drift signals
  - Our contribution: per-tenant scoping, deterministic version history,
    rich GateDecision with per-query breakdown for the council to reason on.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


# A "query" is anything the customer's scorer can evaluate.
# The scorer takes (adapter_id, query) -> score (0.0 to 1.0).
# For BIRD-SQL: query = {"question_id", "db_id", "question", "evidence", "gold_sql"}.
# The scorer runs the adapter, executes the predicted SQL, compares to gold.
Query = dict[str, Any]
Scorer = Callable[[str, Query], float]


@dataclass
class GateConfig:
    """Configuration for a regression gate."""

    epsilon: float = 0.02
    """Maximum acceptable drop in aggregate held-out score. 0.02 = 2pp."""

    min_holdout_size: int = 20
    """Refuse to gate on held-out sets smaller than this."""

    strict_per_query: bool = False
    """If True, also reject when ANY single query that previously scored 1.0
    now scores < 1.0. Stricter than aggregate-only gating."""

    require_calibration: bool = True
    """If True, refuse to gate if the baseline adapter wasn't pre-scored.
    Prevents apples-to-oranges comparisons after baseline changes."""


@dataclass
class SliceAttribution:
    """Per-slice regression breakdown.

    For each slice tag in the held-out set (e.g. ``"intent=refund_request"``),
    aggregates score deltas restricted to queries carrying that tag.

    A held-out query carries slices when its payload includes a ``"slices"``
    field, e.g. ``{"question_id": "q1", "slices": ["intent=refund", "lang=en"]}``.

    The most-negative-delta slice is the "driver" — the behavioral cohort
    your customer notices first when a candidate adapter regresses.
    """

    slice_tag: str
    """The slice key/value pair, e.g. ``"intent=refund_request"``."""

    n_total: int
    """Queries in the held-out set carrying this slice."""

    n_regressed: int
    """Queries where candidate scored lower than baseline."""

    score_baseline: float
    """Average baseline score over this slice."""

    score_candidate: float
    """Average candidate score over this slice."""

    delta: float
    """``score_candidate - score_baseline``. Negative = slice regressed."""

    regressed_query_ids: list[str] = field(default_factory=list)
    """The specific queries that regressed — for showing the customer the
    actual failing examples."""


@dataclass
class GateDecision:
    """Result of running the gate on a candidate adapter."""

    accepted: bool
    candidate_id: str
    baseline_id: str | None
    tenant_id: str
    holdout_size: int
    score_baseline: float
    score_candidate: float
    delta: float
    """Positive delta = improvement. Negative = regression."""
    epsilon: float
    reason: str
    timestamp: str
    per_query: list[dict[str, Any]] = field(default_factory=list)
    """Per-query baseline + candidate scores + delta. Downstream consumers
    (e.g. DriftCouncil) reason over this list."""
    slice_attributions: list[SliceAttribution] = field(default_factory=list)
    """Per-slice breakdown, sorted with the most-regressed slice first.
    Only populated when held-out queries carry a ``"slices"`` field."""

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent, default=str)

    @property
    def regressions(self) -> list[dict[str, Any]]:
        """Queries that scored lower with the candidate than the baseline."""
        return [q for q in self.per_query if q["delta"] < 0]

    @property
    def improvements(self) -> list[dict[str, Any]]:
        """Queries that scored higher with the candidate."""
        return [q for q in self.per_query if q["delta"] > 0]

    @property
    def driver_slice(self) -> SliceAttribution | None:
        """The slice with the largest score drop, if any.

        Returns ``None`` if no slices were declared on the held-out set, or
        if no slice regressed.
        """
        if not self.slice_attributions:
            return None
        worst = self.slice_attributions[0]
        return worst if worst.delta < 0 else None


class RegressionGate:
    """Per-tenant regression gate for LoRA adapter updates.

    The customer's job: provide a scorer that runs an adapter against a query
    and returns 0.0-1.0. Everything else (versioning, decision, audit log) is
    the gate's job.

    Usage:
        gate = RegressionGate(GateConfig(epsilon=0.02))

        decision = gate.evaluate(
            tenant_id="acme_corp",
            candidate_id="adapter_v17",
            baseline_id="adapter_v16",
            holdout=tenant_holdout_set,
            scorer=run_bird_sql,
        )

        if decision.accepted:
            promote(candidate_id)
        else:
            replay_buffer.add(decision)
            alert_team(decision)
    """

    def __init__(self, config: GateConfig | None = None):
        self.config = config or GateConfig()

    def evaluate(
        self,
        *,
        tenant_id: str,
        candidate_id: str,
        baseline_id: str | None,
        holdout: Iterable[Query],
        scorer: Scorer,
    ) -> GateDecision:
        """Decide whether to accept the candidate adapter.

        Args:
            tenant_id: opaque identifier for the customer / workspace.
            candidate_id: the adapter being evaluated (e.g., "adapter_v17").
            baseline_id: the currently-promoted adapter, or None if no baseline yet.
            holdout: iterable of Query dicts. Will be materialized into a list.
            scorer: callable (adapter_id, query) -> score in [0.0, 1.0].

        Returns:
            GateDecision with accept/reject + full audit trail.
        """
        queries = list(holdout)
        n = len(queries)
        now = datetime.now(timezone.utc).isoformat()

        if n < self.config.min_holdout_size:
            return GateDecision(
                accepted=False,
                candidate_id=candidate_id,
                baseline_id=baseline_id,
                tenant_id=tenant_id,
                holdout_size=n,
                score_baseline=0.0,
                score_candidate=0.0,
                delta=0.0,
                epsilon=self.config.epsilon,
                reason=(
                    f"Insufficient held-out set: {n} < min {self.config.min_holdout_size}."
                    " Add more held-out queries before promoting."
                ),
                timestamp=now,
            )

        if baseline_id is None and self.config.require_calibration:
            return GateDecision(
                accepted=False,
                candidate_id=candidate_id,
                baseline_id=None,
                tenant_id=tenant_id,
                holdout_size=n,
                score_baseline=0.0,
                score_candidate=0.0,
                delta=0.0,
                epsilon=self.config.epsilon,
                reason=(
                    "No baseline adapter to compare against. Set"
                    " require_calibration=False to allow promotion of the first"
                    " adapter, or seed a baseline first."
                ),
                timestamp=now,
            )

        per_query: list[dict[str, Any]] = []
        baseline_total = 0.0
        candidate_total = 0.0
        any_regressed_clean = False
        slice_buckets: dict[str, list[dict[str, Any]]] = {}

        for q in queries:
            score_baseline = scorer(baseline_id, q) if baseline_id else 0.0
            score_candidate = scorer(candidate_id, q)
            delta = score_candidate - score_baseline
            baseline_total += score_baseline
            candidate_total += score_candidate
            row = {
                "query_id": q.get("question_id") or q.get("id") or q.get("query_id"),
                "score_baseline": score_baseline,
                "score_candidate": score_candidate,
                "delta": delta,
            }
            per_query.append(row)
            for slice_tag in q.get("slices") or []:
                slice_buckets.setdefault(slice_tag, []).append(row)
            if self.config.strict_per_query and score_baseline == 1.0 and score_candidate < 1.0:
                any_regressed_clean = True

        score_baseline_avg = baseline_total / n
        score_candidate_avg = candidate_total / n
        delta_avg = score_candidate_avg - score_baseline_avg

        accepted = delta_avg >= -self.config.epsilon
        if self.config.strict_per_query and any_regressed_clean:
            accepted = False

        reason = self._reason(
            accepted=accepted,
            delta=delta_avg,
            score_baseline=score_baseline_avg,
            score_candidate=score_candidate_avg,
            n=n,
            baseline_present=baseline_id is not None,
            strict_violation=any_regressed_clean,
        )

        slice_attributions = self._attribute_slices(slice_buckets)

        return GateDecision(
            accepted=accepted,
            candidate_id=candidate_id,
            baseline_id=baseline_id,
            tenant_id=tenant_id,
            holdout_size=n,
            score_baseline=score_baseline_avg,
            score_candidate=score_candidate_avg,
            delta=delta_avg,
            epsilon=self.config.epsilon,
            reason=reason,
            timestamp=now,
            per_query=per_query,
            slice_attributions=slice_attributions,
        )

    @staticmethod
    def _attribute_slices(
        slice_buckets: dict[str, list[dict[str, Any]]],
    ) -> list[SliceAttribution]:
        """Aggregate per-query results into per-slice attributions.

        Output is sorted most-regressed-first so ``slice_attributions[0]``
        is the driver slice (the one your customer will notice first).
        """
        attributions: list[SliceAttribution] = []
        for slice_tag, rows in slice_buckets.items():
            n_total = len(rows)
            if n_total == 0:
                continue
            score_baseline = sum(r["score_baseline"] for r in rows) / n_total
            score_candidate = sum(r["score_candidate"] for r in rows) / n_total
            regressed = [r for r in rows if r["delta"] < 0]
            attributions.append(
                SliceAttribution(
                    slice_tag=slice_tag,
                    n_total=n_total,
                    n_regressed=len(regressed),
                    score_baseline=score_baseline,
                    score_candidate=score_candidate,
                    delta=score_candidate - score_baseline,
                    regressed_query_ids=[r["query_id"] for r in regressed],
                )
            )
        attributions.sort(key=lambda s: s.delta)
        return attributions

    def _reason(
        self,
        *,
        accepted: bool,
        delta: float,
        score_baseline: float,
        score_candidate: float,
        n: int,
        baseline_present: bool,
        strict_violation: bool,
    ) -> str:
        if strict_violation:
            return (
                f"REJECTED (strict mode): candidate regressed on at least one"
                f" query that previously scored 1.0. Aggregate: {score_baseline:.3f}"
                f" → {score_candidate:.3f} (Δ={delta:+.3f}) over n={n}."
            )
        if not baseline_present:
            return (
                f"ACCEPTED: no baseline to compare against (calibration disabled)."
                f" Candidate scored {score_candidate:.3f} on n={n} held-out queries."
                f" First adapter for this tenant — future updates will be gated against it."
            )
        if accepted and delta >= 0:
            return (
                f"ACCEPTED: aggregate {score_baseline:.3f} → {score_candidate:.3f}"
                f" (Δ={delta:+.3f}) over n={n}. Improved or flat."
            )
        if accepted and delta < 0:
            return (
                f"ACCEPTED (tolerated regression): aggregate {score_baseline:.3f} →"
                f" {score_candidate:.3f} (Δ={delta:+.3f}) over n={n}."
                f" |Δ|={abs(delta):.3f} ≤ ε={self.config.epsilon}."
            )
        return (
            f"REJECTED: aggregate {score_baseline:.3f} → {score_candidate:.3f}"
            f" (Δ={delta:+.3f}) over n={n}. Drop exceeds ε={self.config.epsilon}."
        )


def append_audit(decision: GateDecision, audit_log: Path) -> None:
    """Append a GateDecision to a JSONL audit log."""
    audit_log.parent.mkdir(parents=True, exist_ok=True)
    with audit_log.open("a", encoding="utf-8") as fh:
        fh.write(decision.to_json(indent=None))
        fh.write("\n")
