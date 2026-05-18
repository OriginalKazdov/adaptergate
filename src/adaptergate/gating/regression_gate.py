"""Regression gate — the do-no-harm decision for adapter updates.

The wedge: prevent silent quality regression when serving updated LoRA
adapters. Before promoting a candidate adapter to production, run it against
a per-tenant held-out set. Aggregate score drop > epsilon → reject.

When held-out queries carry slice tags (``query["slices"] = [...]``), the
gate also produces per-slice attribution — the cohort-level breakdown that
tells you *which* behavioral slice broke, not just that *something* did.

Inspiration:
  - Online-LoRA (arxiv 2411.05663) distribution-shift detection
  - Silent Collapse MTR (arxiv 2605.14588) drift signals
  - Our contribution: per-tenant scoping, deterministic version history,
    rich GateDecision with per-query and per-slice attribution that
    downstream tooling (your retrain pipeline, your dashboards, your
    CI bots) can consume directly.
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

    pattern: str | None = None
    """One-line description of what the failing queries have in common, or
    ``None`` if no clear pattern. Powered by N-gram frequency analysis on
    the queries' natural-language text fields (``question``, ``text``,
    ``prompt``, ``query``). See ``adaptergate.gating.cluster.find_pattern``.
    """


SCHEMA_VERSION = 2
"""Bumped whenever GateDecision's serialized shape changes. Downstream
consumers that read audit logs should check this field to handle older
records."""


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
    """Per-query baseline + candidate scores + delta."""
    slice_attributions: list[SliceAttribution] = field(default_factory=list)
    """Per-slice breakdown, sorted with the most-regressed slice first.
    Only populated when held-out queries carry a ``"slices"`` field."""
    malformed_slice_queries: int = 0
    """Count of held-out queries whose ``slices`` field was present but not
    a list (e.g. a bare string typo like ``"intent=foo"``). Those queries
    contribute to aggregate scoring but are skipped for slice attribution."""
    schema_version: int = SCHEMA_VERSION
    """Schema version of this decision blob — read by downstream consumers
    to handle older audit-log records gracefully."""

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
        """The slice with the most-negative delta, or ``None`` if no slice
        regressed (or no slices were declared on the held-out set).

        ``slice_attributions`` is sorted most-regressed-first, so the driver
        is simply the first entry — provided its delta is negative.
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
        slice_buckets: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
        malformed_slice_queries = 0

        for q in queries:
            score_baseline = scorer(baseline_id, q) if baseline_id else 0.0
            score_candidate = scorer(candidate_id, q)
            delta = score_candidate - score_baseline
            baseline_total += score_baseline
            candidate_total += score_candidate
            row = {
                "query_id": _pick_query_id(q),
                "score_baseline": score_baseline,
                "score_candidate": score_candidate,
                "delta": delta,
            }
            per_query.append(row)
            raw_slices = q.get("slices")
            if raw_slices is not None and not isinstance(raw_slices, list):
                malformed_slice_queries += 1
            for slice_tag in _normalize_slices(raw_slices):
                slice_buckets.setdefault(slice_tag, []).append((q, row))
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
            malformed_slice_queries=malformed_slice_queries,
        )

    @staticmethod
    def _attribute_slices(
        slice_buckets: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]],
    ) -> list[SliceAttribution]:
        """Aggregate per-query results into per-slice attributions.

        Output is sorted most-regressed-first so ``slice_attributions[0]``
        is the driver slice. Each attribution carries a ``pattern`` string
        (or ``None``) describing what the failing queries have in common —
        the customer's Slack-screenshot line.
        """
        from adaptergate.gating.cluster import find_pattern

        attributions: list[SliceAttribution] = []
        for slice_tag, items in slice_buckets.items():
            n_total = len(items)
            if n_total == 0:
                continue
            rows = [row for _q, row in items]
            score_baseline = sum(r["score_baseline"] for r in rows) / n_total
            score_candidate = sum(r["score_candidate"] for r in rows) / n_total
            regressed_items = [(q, r) for q, r in items if r["delta"] < 0]
            regressed_queries = [q for q, _r in regressed_items]
            attributions.append(
                SliceAttribution(
                    slice_tag=slice_tag,
                    n_total=n_total,
                    n_regressed=len(regressed_items),
                    score_baseline=score_baseline,
                    score_candidate=score_candidate,
                    delta=score_candidate - score_baseline,
                    regressed_query_ids=[r["query_id"] for _q, r in regressed_items],
                    pattern=find_pattern(regressed_queries) if regressed_queries else None,
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


def _normalize_slices(raw: Any) -> list[str]:
    """Return a clean ``list[str]`` of slice tags from a raw payload value.

    Defensive: callers can hand us a string (common typo: ``"intent=foo"``
    instead of ``["intent=foo"]``), ``None``, or a list with non-string
    entries. We drop anything that isn't a string so the gate never iterates
    characters or crashes on a malformed eval row.
    """
    if not isinstance(raw, list):
        return []
    return [s for s in raw if isinstance(s, str)]


def _pick_query_id(q: dict[str, Any]) -> Any:
    """Return the customer-supplied query id, preferring question_id > id > query_id.

    Uses ``is not None`` rather than truthiness so legitimate falsy IDs
    (``0``, empty string) are preserved instead of falling through to the
    next key in the chain.
    """
    for key in ("question_id", "id", "query_id"):
        if key in q and q[key] is not None:
            return q[key]
    return None
