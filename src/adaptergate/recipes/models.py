"""Recipe library data model — the asset that compounds.

A *recipe* is a typed, paper-derived intervention for fixing a regressed
adapter (e.g. "increase ProCL slot 7's rank", "decay Online-LoRA learning
rate for layer 18", "rebuild replay buffer dropping last 3 feedback labels").

A *recipe application* is one customer using one recipe on one adapter,
with the measured before/after delta logged. Every application strengthens
the recommender for the next customer — this is the compounding moat.
A *recipe recommendation* is the system's pick for what to try when the
gate trips, ranked by empirical efficacy across past applications.

The model intentionally does NOT couple to a specific intervention runner
(that is the customer's job). adaptergate ships the recommender; the
customer runs the recipe and reports the outcome back.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class Recipe:
    """A typed, paper-derived intervention recipe.

    ``applies_when`` is a dict of slice-tag predicates:
      - ``slice_tag_contains``: list of substrings; recipe applies if the
        driver slice tag contains any of them. Use lowercase.
      - ``intervention_for``: list of failure modes the recipe targets,
        e.g. ``["catastrophic_forgetting", "overfit_recent"]``.
      - ``min_slice_regression_pp``: applies only if the driver slice
        regressed by at least this many points (default 0.0).
    """

    recipe_id: str
    name: str
    intervention_type: str
    description: str
    source_paper_arxiv: str | None = None
    source_paper_title: str | None = None
    applies_when: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def matches(self, driver_slice_tag: str, driver_delta: float) -> bool:
        """Return True if the recipe is applicable to the given driver slice."""
        contains = self.applies_when.get("slice_tag_contains") or []
        if contains and not any(c.lower() in driver_slice_tag.lower() for c in contains):
            return False
        min_pp = float(self.applies_when.get("min_slice_regression_pp", 0.0))
        if abs(driver_delta) < min_pp:
            return False
        return True

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)


@dataclass
class RecipeApplication:
    """A single customer's use of a recipe on a single adapter."""

    application_id: str
    recipe_id: str
    tenant_hash: str
    """Anonymized tenant identifier — SHA-256 of tenant_id truncated to 16 chars."""

    pre_decision_score: float
    """The candidate adapter's aggregate score before the recipe was applied."""

    post_decision_score: float | None = None
    """Aggregate score after applying the recipe and re-running the gate.
    ``None`` while the application is still pending (recipe applied, not yet
    re-evaluated)."""

    observed_delta: float | None = None
    """``post_decision_score - pre_decision_score`` once both are known."""

    driver_slice_tag: str | None = None
    """The slice tag that triggered this recipe selection."""

    slice_match_signature: list[str] = field(default_factory=list)
    """All slice tags active at application time — for cross-application
    pattern queries (e.g. 'show me applications where intent=billing was active')."""

    applied_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)


@dataclass
class RecipeRecommendation:
    """A scored recipe pick for the current gate decision."""

    recipe: Recipe
    expected_efficacy: float | None
    """Mean ``observed_delta`` across past completed applications, or ``None``
    when this recipe has zero completed applications yet."""

    n_uses: int
    """Number of completed applications used to compute ``expected_efficacy``."""

    n_pending: int = 0
    """Number of applications still pending re-evaluation."""

    efficacy_range_low: float | None = None
    """Lower bound of an approximate range of efficacy. ``None`` when
    ``n_uses`` is below the minimum for a meaningful range. Computed as a
    normal-approximation interval — wide and noisy for small ``n``. See
    ``range_method`` for the exact estimator used."""

    efficacy_range_high: float | None = None

    range_method: str | None = None
    """Identifier for how the efficacy range was computed. ``None`` if no
    range was emitted. Currently only ``"normal_n_gte_3"`` (mean ± 1.96·SE)
    is supported; v0.6 may add Wilson-style or bootstrap intervals."""

    matched_slice_tags: list[str] = field(default_factory=list)
    rationale: str = ""
    """One-line description of WHY this recipe matched the current decision."""

    def to_dict(self) -> dict[str, Any]:
        d = {
            "recipe_id": self.recipe.recipe_id,
            "recipe_name": self.recipe.name,
            "intervention_type": self.recipe.intervention_type,
            "expected_efficacy": self.expected_efficacy,
            "n_uses": self.n_uses,
            "n_pending": self.n_pending,
            "efficacy_range_low": self.efficacy_range_low,
            "efficacy_range_high": self.efficacy_range_high,
            "range_method": self.range_method,
            "matched_slice_tags": self.matched_slice_tags,
            "rationale": self.rationale,
            "source_paper": self.recipe.source_paper_arxiv,
        }
        return d


def hash_tenant(tenant_id: str) -> str:
    """Anonymize a tenant id for cross-customer aggregation."""
    import hashlib

    return hashlib.sha256(tenant_id.encode("utf-8")).hexdigest()[:16]
