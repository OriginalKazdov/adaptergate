"""Recipe recommender — rank recipes by empirical efficacy for the current decision.

``recommend(gate_decision, store)`` returns a list of ``RecipeRecommendation``
ordered by ``expected_efficacy``, descending. Recipes with no completed
applications yet are surfaced too (with ``expected_efficacy=None``) so a
freshly seeded library is still useful on day one — they sort below
empirically-supported recipes.

This is the heart of the moat: as ``store.applications`` grows, the ranking
gets sharper. Competitors can copy the schema. They cannot copy the corpus.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from adaptergate.recipes.models import (
    Recipe,
    RecipeApplication,
    RecipeRecommendation,
)

if TYPE_CHECKING:
    from adaptergate.gating.regression_gate import GateDecision
    from adaptergate.recipes.store import RecipeStore


def recommend(
    decision: "GateDecision",
    store: "RecipeStore",
    *,
    top_k: int = 5,
    min_uses_for_ci: int = 3,
) -> list[RecipeRecommendation]:
    """Return recipes ranked by efficacy for this gate decision.

    Only meaningful when the decision was rejected and has a driver slice.
    Returns an empty list otherwise.
    """
    if decision.accepted:
        return []
    driver = decision.driver_slice
    if driver is None:
        return []

    matched: list[RecipeRecommendation] = []
    for recipe in store.list_recipes():
        if not recipe.matches(driver.slice_tag, driver.delta):
            continue
        rec = _build_recommendation(
            recipe=recipe,
            driver_slice_tag=driver.slice_tag,
            applications=store.applications_for(recipe.recipe_id),
            min_uses_for_ci=min_uses_for_ci,
        )
        matched.append(rec)

    matched.sort(key=_sort_key, reverse=True)
    return matched[:top_k]


def _build_recommendation(
    *,
    recipe: Recipe,
    driver_slice_tag: str,
    applications: list[RecipeApplication],
    min_uses_for_ci: int,
) -> RecipeRecommendation:
    completed = [a for a in applications if a.observed_delta is not None]
    pending = [a for a in applications if a.observed_delta is None]

    if completed:
        deltas = [float(a.observed_delta or 0.0) for a in completed]
        mean = sum(deltas) / len(deltas)
    else:
        mean = None

    range_low: float | None = None
    range_high: float | None = None
    range_method: str | None = None
    if completed and len(completed) >= min_uses_for_ci:
        range_low, range_high = _normal_range(
            [float(a.observed_delta or 0.0) for a in completed]
        )
        range_method = "normal_n_gte_3"

    rationale_parts = [f"matches driver slice `{driver_slice_tag}`"]
    if mean is not None:
        rationale_parts.append(
            f"avg observed Δ={mean:+.3f} over n={len(completed)}"
        )
    else:
        rationale_parts.append("no prior applications yet")
    rationale = "; ".join(rationale_parts)

    return RecipeRecommendation(
        recipe=recipe,
        expected_efficacy=mean,
        n_uses=len(completed),
        n_pending=len(pending),
        efficacy_range_low=range_low,
        efficacy_range_high=range_high,
        range_method=range_method,
        matched_slice_tags=[driver_slice_tag],
        rationale=rationale,
    )


def _normal_range(values: list[float], z: float = 1.96) -> tuple[float, float]:
    """Approximate range around the mean of ``values``.

    Computed as ``mean ± z * (sample_std / sqrt(n))``. This is intentionally
    NOT called a "95% confidence interval" in the public API: at the small
    ``n`` (3-5) typical of an early recipe corpus the normal approximation
    is wide and the coverage guarantee does not hold. The downstream field
    name is ``efficacy_range_low/high`` and ``range_method="normal_n_gte_3"``
    to make the approximation explicit. v0.6 may add Wilson-style or
    bootstrap estimators as the corpus grows.
    """
    n = len(values)
    if n < 2:
        m = values[0]
        return (m, m)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    se = math.sqrt(var / n)
    return (mean - z * se, mean + z * se)


def _sort_key(rec: RecipeRecommendation) -> tuple[float, int, str]:
    """Sort: empirical-evidence recipes first, then by efficacy, then by name.

    Tuple components, all descending after sort:
      1. has_evidence flag: 1 if n_uses > 0 else 0 — evidence > guess.
      2. expected_efficacy or -inf if None.
      3. inverse n_uses (more uses → stronger signal, but we use plain n
         on the secondary tier so it doesn't dominate when efficacy ties).
    """
    has_evidence = 1 if rec.n_uses > 0 else 0
    eff = rec.expected_efficacy if rec.expected_efficacy is not None else float("-inf")
    return (has_evidence, eff, rec.n_uses)
