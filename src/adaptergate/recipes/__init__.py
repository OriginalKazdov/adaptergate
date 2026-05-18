"""Recipe library — the compounding moat.

Public API:

  - :class:`Recipe` — a typed, paper-derived intervention.
  - :class:`RecipeApplication` — a single customer use, with measured outcome.
  - :class:`RecipeRecommendation` — a scored pick for the current decision.
  - :class:`RecipeStore` — JSONL-backed persistence.
  - :func:`recommend` — rank recipes by efficacy for a gate decision.
  - :func:`hash_tenant` — anonymize tenant id for cross-customer logging.

Seed recipes ship under ``adaptergate.data.seed_recipes.jsonl``; load them
into a new store with the ``adaptergate recipes seed`` CLI command, or
programmatically by copying the file to your chosen recipes path.
"""

from adaptergate.recipes.models import (
    Recipe,
    RecipeApplication,
    RecipeRecommendation,
    hash_tenant,
)
from adaptergate.recipes.recommend import recommend
from adaptergate.recipes.store import RecipeStore

__all__ = [
    "Recipe",
    "RecipeApplication",
    "RecipeRecommendation",
    "RecipeStore",
    "recommend",
    "hash_tenant",
]
