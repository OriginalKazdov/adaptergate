"""Tests for the recipe library + recommend() API (v0.5 moat substrate)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adaptergate.gating import GateConfig, RegressionGate
from adaptergate.recipes import (
    Recipe,
    RecipeApplication,
    RecipeStore,
    hash_tenant,
    recommend,
)


def _gate_with_billing_regression():
    """Return a rejected decision whose driver slice is `intent=billing_dispute`."""
    gate = RegressionGate(GateConfig(epsilon=0.02))
    queries = [
        {
            "question_id": f"q{i}",
            "question": "I need a refund without order_id",
            "slices": ["intent=billing_dispute"],
        }
        for i in range(25)
    ]
    return gate.evaluate(
        tenant_id="acme",
        candidate_id="v19",
        baseline_id="v18",
        holdout=queries,
        scorer=lambda a, q: 0.9 if a == "v18" else 0.3,
    )


# ---------- Recipe.matches ----------

def test_recipe_matches_when_slice_substring_present():
    r = Recipe(
        recipe_id="r1",
        name="Test",
        intervention_type="t",
        description="d",
        applies_when={"slice_tag_contains": ["intent="]},
    )
    assert r.matches("intent=billing_dispute", -0.5) is True
    assert r.matches("lang=es", -0.5) is False


def test_recipe_matches_when_no_substring_predicate():
    """Empty/missing slice_tag_contains → recipe matches any slice."""
    r = Recipe(recipe_id="r1", name="Test", intervention_type="t", description="d")
    assert r.matches("intent=anything", -0.1) is True


def test_recipe_respects_min_regression_threshold():
    r = Recipe(
        recipe_id="r1",
        name="Test",
        intervention_type="t",
        description="d",
        applies_when={"min_slice_regression_pp": 0.10},
    )
    assert r.matches("intent=foo", -0.5) is True
    assert r.matches("intent=foo", -0.05) is False


# ---------- RecipeStore ----------

def test_recipe_store_save_load_roundtrip(tmp_path: Path):
    store = RecipeStore(
        recipes_path=tmp_path / "r.jsonl",
        applications_path=tmp_path / "a.jsonl",
    )
    r = Recipe(
        recipe_id="r1",
        name="Test",
        intervention_type="t",
        description="d",
        source_paper_arxiv="2605.13162",
        params={"foo": "bar"},
    )
    store.add_recipe(r)

    reloaded = RecipeStore(
        recipes_path=tmp_path / "r.jsonl",
        applications_path=tmp_path / "a.jsonl",
    )
    assert len(reloaded) == 1
    rr = reloaded.get_recipe("r1")
    assert rr is not None
    assert rr.name == "Test"
    assert rr.params == {"foo": "bar"}


def test_application_log_is_append_only(tmp_path: Path):
    store = RecipeStore(
        recipes_path=tmp_path / "r.jsonl",
        applications_path=tmp_path / "a.jsonl",
    )
    for i in range(5):
        store.add_application(
            RecipeApplication(
                application_id=f"app{i}",
                recipe_id="r1",
                tenant_hash=hash_tenant(f"tenant{i}"),
                pre_decision_score=0.5,
                post_decision_score=0.7,
                observed_delta=0.2,
                driver_slice_tag="intent=foo",
            )
        )
    reloaded = RecipeStore(
        recipes_path=tmp_path / "r.jsonl",
        applications_path=tmp_path / "a.jsonl",
    )
    assert reloaded.n_applications() == 5


# ---------- recommend() ----------

def test_recommend_returns_empty_for_accepted_decision(tmp_path: Path):
    store = RecipeStore(tmp_path / "r.jsonl", tmp_path / "a.jsonl")
    store.add_recipe(Recipe(recipe_id="r1", name="T", intervention_type="t", description="d"))

    gate = RegressionGate(GateConfig(epsilon=0.5))
    queries = [
        {"question_id": f"q{i}", "slices": ["intent=billing"]} for i in range(25)
    ]
    decision = gate.evaluate(
        tenant_id="t1",
        candidate_id="c",
        baseline_id="b",
        holdout=queries,
        scorer=lambda a, q: 0.9,
    )
    assert decision.accepted is True
    assert recommend(decision, store) == []


def test_recommend_returns_matching_recipes_with_no_prior_uses(tmp_path: Path):
    store = RecipeStore(tmp_path / "r.jsonl", tmp_path / "a.jsonl")
    store.add_recipe(Recipe(
        recipe_id="billing_fix",
        name="Billing fix",
        intervention_type="slot_rebalance",
        description="Fix billing slice",
        applies_when={"slice_tag_contains": ["billing"]},
    ))
    store.add_recipe(Recipe(
        recipe_id="other_fix",
        name="Other",
        intervention_type="other",
        description="Other slice",
        applies_when={"slice_tag_contains": ["lang="]},
    ))

    decision = _gate_with_billing_regression()
    recs = recommend(decision, store)
    ids = [r.recipe.recipe_id for r in recs]
    assert "billing_fix" in ids
    assert "other_fix" not in ids


def test_recommend_ranks_by_observed_efficacy(tmp_path: Path):
    store = RecipeStore(tmp_path / "r.jsonl", tmp_path / "a.jsonl")
    store.add_recipe(Recipe(
        recipe_id="weak",
        name="Weak",
        intervention_type="t",
        description="d",
        applies_when={"slice_tag_contains": ["billing"]},
    ))
    store.add_recipe(Recipe(
        recipe_id="strong",
        name="Strong",
        intervention_type="t",
        description="d",
        applies_when={"slice_tag_contains": ["billing"]},
    ))

    # weak: 3 prior uses, average +0.05
    for i in range(3):
        store.add_application(RecipeApplication(
            application_id=f"weak_a{i}",
            recipe_id="weak",
            tenant_hash=hash_tenant(f"t{i}"),
            pre_decision_score=0.5,
            post_decision_score=0.55,
            observed_delta=0.05,
            driver_slice_tag="intent=billing",
        ))
    # strong: 3 prior uses, average +0.30
    for i in range(3):
        store.add_application(RecipeApplication(
            application_id=f"strong_a{i}",
            recipe_id="strong",
            tenant_hash=hash_tenant(f"t{i}"),
            pre_decision_score=0.5,
            post_decision_score=0.80,
            observed_delta=0.30,
            driver_slice_tag="intent=billing",
        ))

    decision = _gate_with_billing_regression()
    recs = recommend(decision, store)
    # Strong should come before weak.
    assert recs[0].recipe.recipe_id == "strong"
    assert recs[1].recipe.recipe_id == "weak"
    assert recs[0].expected_efficacy == pytest.approx(0.30, abs=1e-6)
    assert recs[1].expected_efficacy == pytest.approx(0.05, abs=1e-6)


def test_recommend_evidence_beats_no_evidence(tmp_path: Path):
    """A recipe with any positive evidence outranks a recipe with zero apps."""
    store = RecipeStore(tmp_path / "r.jsonl", tmp_path / "a.jsonl")
    store.add_recipe(Recipe(
        recipe_id="known_good",
        name="Known",
        intervention_type="t",
        description="d",
        applies_when={"slice_tag_contains": ["billing"]},
    ))
    store.add_recipe(Recipe(
        recipe_id="unknown",
        name="Unknown",
        intervention_type="t",
        description="d",
        applies_when={"slice_tag_contains": ["billing"]},
    ))

    store.add_application(RecipeApplication(
        application_id="known_a1",
        recipe_id="known_good",
        tenant_hash=hash_tenant("t"),
        pre_decision_score=0.5,
        post_decision_score=0.55,
        observed_delta=0.05,
        driver_slice_tag="intent=billing",
    ))

    decision = _gate_with_billing_regression()
    recs = recommend(decision, store)
    assert recs[0].recipe.recipe_id == "known_good"
    assert recs[0].n_uses == 1
    assert recs[1].recipe.recipe_id == "unknown"
    assert recs[1].n_uses == 0


def test_recommendation_to_dict_serializable():
    """to_dict() output must be JSON-serializable for CLI/API consumers."""
    store_path = Path("/tmp/_ag_test_unused.jsonl")
    decision = _gate_with_billing_regression()
    store = RecipeStore(store_path, store_path.with_name("apps.jsonl"))
    store._recipes["r1"] = Recipe(
        recipe_id="r1",
        name="T",
        intervention_type="t",
        description="d",
        applies_when={"slice_tag_contains": ["billing"]},
    )
    recs = recommend(decision, store)
    blob = json.dumps([r.to_dict() for r in recs])
    parsed = json.loads(blob)
    assert isinstance(parsed, list)
    assert parsed[0]["recipe_id"] == "r1"


# ---------- hash_tenant ----------

def test_hash_tenant_is_deterministic():
    assert hash_tenant("acme") == hash_tenant("acme")
    assert hash_tenant("acme") != hash_tenant("other")
    # Plausibly anonymous (16 hex chars).
    assert len(hash_tenant("acme")) == 16
