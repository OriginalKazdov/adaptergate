"""Recipe and application storage — JSONL files on disk.

Two files:
  - ``recipes.jsonl`` — the curated library (typically seeded once, then
    grown by adding new paper-derived recipes manually or via the v0.5.x
    automated radar.db ingester).
  - ``applications.jsonl`` — append-only log of every customer's recipe
    use. This is the corpus that compounds.

JSONL is the right substrate at this scale (10k recipes, 1M applications
fits comfortably). Migration to a real DB is a v0.6+ concern.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

from adaptergate.recipes.models import Recipe, RecipeApplication


class RecipeStore:
    """JSONL-backed recipe + application store."""

    def __init__(self, recipes_path: str | Path, applications_path: str | Path):
        self.recipes_path = Path(recipes_path)
        self.applications_path = Path(applications_path)
        self._recipes: dict[str, Recipe] = {}
        self._applications: list[RecipeApplication] = []
        if self.recipes_path.exists():
            self._load_recipes()
        if self.applications_path.exists():
            self._load_applications()

    # ---------- recipes ----------

    def _load_recipes(self) -> None:
        with self.recipes_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                r = Recipe(**row)
                self._recipes[r.recipe_id] = r

    def add_recipe(self, recipe: Recipe) -> None:
        """Add or overwrite a recipe."""
        self._recipes[recipe.recipe_id] = recipe
        self._flush_recipes()

    def get_recipe(self, recipe_id: str) -> Recipe | None:
        return self._recipes.get(recipe_id)

    def list_recipes(self) -> list[Recipe]:
        return list(self._recipes.values())

    def _flush_recipes(self) -> None:
        self.recipes_path.parent.mkdir(parents=True, exist_ok=True)
        with self.recipes_path.open("w", encoding="utf-8") as fh:
            for r in self._recipes.values():
                fh.write(json.dumps(asdict(r), default=str))
                fh.write("\n")

    # ---------- applications ----------

    def _load_applications(self) -> None:
        with self.applications_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                self._applications.append(RecipeApplication(**row))

    def add_application(self, app: RecipeApplication) -> None:
        """Append a new application to the log."""
        self._applications.append(app)
        self.applications_path.parent.mkdir(parents=True, exist_ok=True)
        with self.applications_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(app), default=str))
            fh.write("\n")

    def applications_for(self, recipe_id: str) -> list[RecipeApplication]:
        return [a for a in self._applications if a.recipe_id == recipe_id]

    def all_applications(self) -> Iterator[RecipeApplication]:
        return iter(self._applications)

    def __len__(self) -> int:
        return len(self._recipes)

    def n_applications(self) -> int:
        return len(self._applications)
