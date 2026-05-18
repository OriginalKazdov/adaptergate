"""adaptergate CLI — run the regression gate from the command line.

The CLI is intentionally serving-agnostic. Users supply a `scorer` callable
via Python module:function syntax (e.g. ``my_eval:score``). The scorer takes
``(adapter_id, query)`` and returns a float in ``[0.0, 1.0]``. adaptergate
handles the held-out set management, gate decision, audit log, and replay
buffer.

Exits with status 0 if the candidate is accepted, 1 if rejected, 2 on usage
errors. This makes ``adaptergate gate`` plug into CI/CD as a pre-deploy check.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from adaptergate import __version__
from adaptergate.gating import (
    GateConfig,
    GateDecision,
    HoldoutSet,
    RegressionGate,
    ReplayBuffer,
    SliceAttribution,
    append_audit,
)
from adaptergate.recipes import RecipeStore, recommend

app = typer.Typer(
    help="adaptergate — CI gate for per-tenant LoRA adapters that update online.",
    no_args_is_help=True,
    add_completion=False,
)
holdout_app = typer.Typer(
    help="Manage per-tenant held-out eval sets.",
    no_args_is_help=True,
    add_completion=False,
)
replay_app = typer.Typer(
    help="Inspect rejected update history.",
    no_args_is_help=True,
    add_completion=False,
)
recipes_app = typer.Typer(
    help="Manage the recipe library (paper-derived CL interventions).",
    no_args_is_help=True,
    add_completion=False,
)
app.add_typer(holdout_app, name="holdout")
app.add_typer(replay_app, name="replay")
app.add_typer(recipes_app, name="recipes")

console = Console()
err_console = Console(stderr=True)


def _import_scorer(spec: str):
    """Import a scorer from 'module:function' syntax.

    Adds the current working directory to sys.path so users can point at a
    local scorer module (e.g. ``my_eval:score``) without packaging.
    """
    if ":" not in spec:
        raise typer.BadParameter(
            f"Scorer spec must be 'module:function', got {spec!r}."
        )
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    module_name, func_name = spec.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise typer.BadParameter(f"Could not import {module_name!r}: {e}")
    try:
        return getattr(module, func_name)
    except AttributeError:
        raise typer.BadParameter(
            f"Module {module_name!r} has no attribute {func_name!r}."
        )


@app.command()
def gate(
    tenant: str = typer.Option(..., help="Tenant / workspace identifier."),
    candidate: str = typer.Option(..., help="Candidate adapter ID being evaluated."),
    baseline: Optional[str] = typer.Option(
        None, help="Baseline adapter ID. Omit for first-adapter promotion."
    ),
    holdout_path: Path = typer.Option(..., "--holdout", help="Path to JSONL held-out set."),
    scorer_spec: str = typer.Option(
        ..., "--scorer", help="Scorer in 'module:function' syntax."
    ),
    epsilon: float = typer.Option(0.02, help="Max acceptable score drop."),
    sample_n: Optional[int] = typer.Option(
        None, "--sample", help="Sample N queries from held-out (default: all)."
    ),
    strict: bool = typer.Option(
        False, "--strict", help="Strict per-query mode: reject any clean regression."
    ),
    require_calibration: bool = typer.Option(
        True,
        "--require-calibration/--no-require-calibration",
        help="If true, refuse to gate without a baseline.",
    ),
    audit_log: Optional[Path] = typer.Option(
        None, help="Append decision to this JSONL audit log."
    ),
    replay_path: Optional[Path] = typer.Option(
        None, help="If decision is rejected, append to this replay buffer."
    ),
    output_format: str = typer.Option(
        "human", "--format", "-f",
        help="Output format: human (rich CLI, default), json (JSON to stdout), pr-comment (GitHub-flavored Markdown for PR comments).",
    ),
    show_failures: int = typer.Option(
        5, "--show-failures",
        help="How many failing query IDs to preview under the driver slice (default 5).",
    ),
    staleness_threshold_days: int = typer.Option(
        30, "--staleness-threshold-days",
        help="Warn if the held-out set has not been updated in this many days.",
    ),
    quiet: bool = typer.Option(False, help="Alias for --format json (kept for backwards compatibility)."),
):
    """Evaluate a candidate adapter against a baseline. Exit 0 if accepted, 1 if rejected."""
    holdout = HoldoutSet(tenant_id=tenant, path=holdout_path)
    if len(holdout) == 0:
        err_console.print(
            f"[red]Held-out set at {holdout_path} is empty. Add queries first via"
            f" 'adaptergate holdout add'.[/red]"
        )
        raise typer.Exit(2)

    queries = holdout.sample(n=sample_n, seed=candidate)
    scorer = _import_scorer(scorer_spec)

    gate_inst = RegressionGate(
        GateConfig(
            epsilon=epsilon,
            strict_per_query=strict,
            require_calibration=require_calibration,
        )
    )
    decision = gate_inst.evaluate(
        tenant_id=tenant,
        candidate_id=candidate,
        baseline_id=baseline,
        holdout=queries,
        scorer=scorer,
    )

    if audit_log is not None:
        append_audit(decision, audit_log)

    if not decision.accepted and replay_path is not None:
        buf = ReplayBuffer(tenant_id=tenant, path=replay_path)
        buf.add(decision)

    # Compute holdout staleness once so any output format can surface it.
    staleness_days = holdout.staleness_days()

    if quiet:
        output_format = "json"

    if output_format == "json":
        typer.echo(decision.to_json())
    elif output_format == "pr-comment":
        typer.echo(_render_pr_comment(decision, staleness_days=staleness_days))
    else:
        _render_human(
            decision,
            staleness_days=staleness_days,
            staleness_threshold_days=staleness_threshold_days,
            show_failures=show_failures,
        )

    raise typer.Exit(0 if decision.accepted else 1)


def _render_human(
    decision,
    *,
    staleness_days: Optional[int],
    staleness_threshold_days: int,
    show_failures: int,
) -> None:
    """Render the gate decision to a human-friendly terminal."""
    verdict_style = "green" if decision.accepted else "red"
    verdict = "ACCEPTED" if decision.accepted else "REJECTED"
    console.rule(f"[{verdict_style}]{verdict}[/{verdict_style}]")
    baseline_display = decision.baseline_id or "(none)"
    console.print(f"Tenant:    [cyan]{decision.tenant_id}[/cyan]")
    console.print(f"Candidate: [cyan]{decision.candidate_id}[/cyan]")
    console.print(f"Baseline:  [cyan]{baseline_display}[/cyan]")
    console.print(
        f"Score:     {decision.score_baseline:.3f} → {decision.score_candidate:.3f}"
        f"  (Δ={decision.delta:+.3f}, ε={decision.epsilon})"
    )
    console.print(f"Held-out:  n={decision.holdout_size}")
    console.print(f"Reason:    {decision.reason}")

    driver = decision.driver_slice
    if driver is not None:
        console.print(
            f"\n[bold red]DRIVER SLICE:[/bold red] [magenta]{driver.slice_tag}[/magenta]"
            f"   {driver.score_baseline:.3f} → {driver.score_candidate:.3f}"
            f"  (Δ={driver.delta:+.3f}, {driver.n_regressed}/{driver.n_total} regressed)"
        )
        if driver.pattern:
            console.print(f"  [bold]Pattern:[/bold] {driver.pattern}")
        named_ids = [qid for qid in driver.regressed_query_ids if qid]
        if named_ids:
            preview = ", ".join(str(x) for x in named_ids[:show_failures])
            more = (
                f" + {len(named_ids) - show_failures} more"
                if len(named_ids) > show_failures
                else ""
            )
            console.print(f"  Failing query IDs: {preview}{more}")

    if decision.slice_attributions and len(decision.slice_attributions) > 1:
        console.print("\n[bold]Slice breakdown[/bold] (most-regressed first):")
        for s in decision.slice_attributions:
            colour = "red" if s.delta < 0 else "green"
            console.print(
                f"  [{colour}]{s.delta:+.3f}[/{colour}]   "
                f"{s.n_regressed}/{s.n_total} regressed   "
                f"[magenta]{s.slice_tag}[/magenta]"
            )

    if decision.regressions:
        note = ""
        if decision.slice_attributions and len(decision.slice_attributions) > 1:
            note = " (slice n_regressed values may sum higher when queries belong to multiple slices)"
        console.print(
            f"\n[yellow]{len(decision.regressions)} unique queries regressed[/yellow]"
            f"{note}"
        )
    if decision.improvements:
        console.print(f"[green]{len(decision.improvements)} unique queries improved[/green]")

    if decision.malformed_slice_queries > 0:
        err_console.print(
            f"\n[yellow]Warning:[/yellow] {decision.malformed_slice_queries}"
            f" held-out queries had a malformed 'slices' field"
            f" (expected list, got non-list). Those queries contributed to"
            f" aggregate scoring but were skipped for slice attribution."
        )

    if decision.suspected_duplicate_slices:
        err_console.print(
            "\n[yellow]Warning:[/yellow] suspected duplicate slice tags"
            " (similarity ≥ 0.85). Consider normalising your held-out set:"
        )
        for a, b in decision.suspected_duplicate_slices:
            err_console.print(f"  • [magenta]{a}[/magenta]  ↔  [magenta]{b}[/magenta]")

    if staleness_days is not None and staleness_days > staleness_threshold_days:
        err_console.print(
            f"\n[yellow]Warning:[/yellow] held-out set has not been refreshed"
            f" in {staleness_days} days (threshold: {staleness_threshold_days})."
            f" Stale held-out sets may fail to reflect current traffic — consider"
            f" adding fresh queries to avoid eval-set drift being misread as adapter drift."
        )


def _render_pr_comment(decision, *, staleness_days: Optional[int]) -> str:
    """Render the gate decision as GitHub-flavored Markdown for a PR comment."""
    icon = "✅" if decision.accepted else "🚫"
    verdict = "ACCEPTED" if decision.accepted else "REJECTED"
    lines: list[str] = [f"## {icon} adaptergate gate: **{verdict}**", ""]
    lines.append(f"- **Tenant**: `{decision.tenant_id}`")
    lines.append(
        f"- **Candidate**: `{decision.candidate_id}`"
        f" vs baseline `{decision.baseline_id or '(none)'}`"
    )
    lines.append(
        f"- **Score**: {decision.score_baseline:.3f} → {decision.score_candidate:.3f}"
        f" (Δ={decision.delta:+.3f}, ε={decision.epsilon})"
    )
    lines.append(f"- **Held-out**: n={decision.holdout_size}")
    lines.append(f"- **Reason**: {decision.reason}")

    driver = decision.driver_slice
    if driver is not None:
        lines.append("")
        lines.append(f"### 🎯 Driver slice: `{driver.slice_tag}`")
        lines.append(
            f"- {driver.score_baseline:.3f} → {driver.score_candidate:.3f}"
            f" (Δ={driver.delta:+.3f}, {driver.n_regressed}/{driver.n_total} regressed)"
        )
        if driver.pattern:
            lines.append(f"- **Pattern**: {driver.pattern}")
        named_ids = [str(x) for x in driver.regressed_query_ids if x]
        if named_ids:
            preview = ", ".join(f"`{x}`" for x in named_ids[:5])
            more = f" + {len(named_ids) - 5} more" if len(named_ids) > 5 else ""
            lines.append(f"- **Failing IDs**: {preview}{more}")

    if decision.slice_attributions and len(decision.slice_attributions) > 1:
        lines.append("")
        lines.append("### Slice breakdown")
        lines.append("| Δ | regressed | slice |")
        lines.append("|---|---|---|")
        for s in decision.slice_attributions:
            lines.append(
                f"| {s.delta:+.3f} | {s.n_regressed}/{s.n_total} | `{s.slice_tag}` |"
            )

    warnings: list[str] = []
    if decision.malformed_slice_queries > 0:
        warnings.append(
            f"{decision.malformed_slice_queries} held-out queries had a malformed"
            " `slices` field — skipped for slice attribution."
        )
    if decision.suspected_duplicate_slices:
        pairs = ", ".join(
            f"`{a}` ↔ `{b}`" for a, b in decision.suspected_duplicate_slices
        )
        warnings.append(f"Suspected duplicate slice tags: {pairs}")
    if staleness_days is not None and staleness_days > 30:
        warnings.append(f"Held-out set has not been refreshed in {staleness_days} days.")
    if warnings:
        lines.append("")
        lines.append("### ⚠️ Warnings")
        for w in warnings:
            lines.append(f"- {w}")

    lines.append("")
    lines.append("_Generated by [adaptergate](https://github.com/OriginalKazdov/adaptergate)._")
    return "\n".join(lines)


@holdout_app.command("add")
def holdout_add(
    tenant: str = typer.Option(...),
    holdout_path: Path = typer.Option(..., "--holdout"),
    query_json: str = typer.Argument(..., help="Query payload as JSON string."),
    accepted_by: Optional[str] = typer.Option(None, help="Adapter version this query was OK with."),
):
    """Add a query to the per-tenant held-out set."""
    try:
        payload = json.loads(query_json)
    except json.JSONDecodeError as e:
        raise typer.BadParameter(f"Query is not valid JSON: {e}")
    holdout = HoldoutSet(tenant_id=tenant, path=holdout_path)
    record = holdout.add(payload, accepted_by=accepted_by)
    typer.echo(json.dumps({"added": record.query_id, "size": len(holdout)}))


@holdout_app.command("size")
def holdout_size(
    tenant: str = typer.Option(...),
    holdout_path: Path = typer.Option(..., "--holdout"),
):
    """Print the number of queries in the held-out set."""
    holdout = HoldoutSet(tenant_id=tenant, path=holdout_path)
    typer.echo(str(len(holdout)))


@holdout_app.command("list")
def holdout_list(
    tenant: str = typer.Option(...),
    holdout_path: Path = typer.Option(..., "--holdout"),
):
    """List queries in the held-out set, one JSON per line."""
    holdout = HoldoutSet(tenant_id=tenant, path=holdout_path)
    for q in holdout:
        typer.echo(
            json.dumps(
                {
                    "query_id": q.query_id,
                    "accepted_by": q.accepted_by_adapter,
                    "added_at": q.added_at,
                }
            )
        )


@replay_app.command("list")
def replay_list(
    tenant: str = typer.Option(...),
    replay_path: Path = typer.Option(..., "--replay"),
    n: int = typer.Option(10, help="Show the most recent N rejected updates."),
):
    """Show recent rejected updates."""
    buf = ReplayBuffer(tenant_id=tenant, path=replay_path)
    for r in buf.recent(n=n):
        typer.echo(
            json.dumps(
                {
                    "candidate": r.candidate_id,
                    "baseline": r.baseline_id,
                    "rejected_at": r.rejected_at,
                    "delta": round(r.delta, 4),
                    "reason": r.reason,
                }
            )
        )


@app.command()
def version():
    """Print adaptergate version."""
    typer.echo(__version__)


# ---------- recipes subcommands ----------

def _bundled_seed_path() -> Path:
    """Locate the package-bundled seed_recipes.jsonl."""
    import adaptergate as _pkg

    return Path(_pkg.__file__).parent / "data" / "seed_recipes.jsonl"


@recipes_app.command("list")
def recipes_list(
    recipes_path: Path = typer.Option(..., "--recipes", help="Path to recipes JSONL."),
):
    """List recipes in the library."""
    store = RecipeStore(recipes_path=recipes_path, applications_path=recipes_path.with_name("applications.jsonl"))
    if len(store) == 0:
        err_console.print(
            f"[yellow]No recipes in {recipes_path}.[/yellow]"
            f" Run [bold]adaptergate recipes seed --recipes {recipes_path}[/bold]"
            f" to populate from the bundled seed library."
        )
        raise typer.Exit(1)
    for r in store.list_recipes():
        typer.echo(
            json.dumps(
                {
                    "recipe_id": r.recipe_id,
                    "name": r.name,
                    "intervention_type": r.intervention_type,
                    "source_paper": r.source_paper_arxiv,
                }
            )
        )


@recipes_app.command("show")
def recipes_show(
    recipe_id: str = typer.Argument(..., help="ID of the recipe to display."),
    recipes_path: Path = typer.Option(..., "--recipes"),
):
    """Show full detail for one recipe."""
    store = RecipeStore(recipes_path=recipes_path, applications_path=recipes_path.with_name("applications.jsonl"))
    r = store.get_recipe(recipe_id)
    if r is None:
        err_console.print(f"[red]No recipe with id {recipe_id!r} in {recipes_path}.[/red]")
        raise typer.Exit(1)
    typer.echo(r.to_json())


@recipes_app.command("seed")
def recipes_seed(
    recipes_path: Path = typer.Option(..., "--recipes", help="Where to write the seed recipes."),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Overwrite any existing recipes file at this path."
    ),
):
    """Populate a recipes file from the package-bundled seed library."""
    seed = _bundled_seed_path()
    if not seed.exists():
        err_console.print(f"[red]Seed file not found at {seed}.[/red]")
        raise typer.Exit(2)
    if recipes_path.exists() and not overwrite:
        err_console.print(
            f"[red]{recipes_path} already exists. Pass --overwrite to replace.[/red]"
        )
        raise typer.Exit(1)
    recipes_path.parent.mkdir(parents=True, exist_ok=True)
    recipes_path.write_text(seed.read_text())
    n = sum(1 for line in recipes_path.read_text().splitlines() if line.strip())
    typer.echo(json.dumps({"seeded": str(recipes_path), "n_recipes": n}))


@app.command()
def recommend_cmd(
    decision_path: Path = typer.Option(
        ..., "--decision", help="Path to a GateDecision JSON or JSONL audit log."
    ),
    recipes_path: Path = typer.Option(..., "--recipes", help="Path to recipes JSONL."),
    line: int = typer.Option(
        -1, "--line", help="Which line of the audit log to consume (default: last)."
    ),
    top_k: int = typer.Option(5, "--top-k", help="Maximum recommendations to return."),
    output_format: str = typer.Option(
        "human", "--format", "-f", help="Output format: human (default) or json."
    ),
):
    """Recommend repair recipes for a rejected gate decision."""
    decision = _load_decision(decision_path, line=line)
    store = RecipeStore(
        recipes_path=recipes_path,
        applications_path=recipes_path.with_name("applications.jsonl"),
    )
    recs = recommend(decision, store, top_k=top_k)

    if output_format == "json":
        typer.echo(json.dumps([r.to_dict() for r in recs], indent=2))
        return

    if not recs:
        if decision.accepted:
            console.print("[green]Decision was ACCEPTED — no repair recipe needed.[/green]")
        elif decision.driver_slice is None:
            console.print(
                "[yellow]No driver slice on this decision — add slice tags to your"
                " held-out queries to enable recipe matching.[/yellow]"
            )
        else:
            console.print(
                f"[yellow]No recipes in the library match driver slice"
                f" `{decision.driver_slice.slice_tag}`.[/yellow]"
            )
        return

    driver = decision.driver_slice
    if driver is not None:
        console.rule(f"[bold magenta]Recipes for driver slice: {driver.slice_tag}[/bold magenta]")

    if all(r.n_uses == 0 for r in recs):
        err_console.print(
            "[yellow]Note:[/yellow] no prior applications have been logged for any"
            " matching recipe. The ranking below reflects [bold]slice matching only[/bold],"
            " not observed efficacy. To make the ranking empirical, log recipe outcomes"
            " via [bold]RecipeStore.add_application(...)[/bold] after running a recipe."
        )
    for i, r in enumerate(recs, 1):
        eff = (
            f"avg Δ={r.expected_efficacy:+.3f} over n={r.n_uses}"
            if r.expected_efficacy is not None
            else "no prior applications"
        )
        console.print(
            f"\n[bold cyan]{i}. {r.recipe.name}[/bold cyan]"
            f"   [{eff}]"
        )
        console.print(f"   id: [magenta]{r.recipe.recipe_id}[/magenta]")
        console.print(f"   intervention: {r.recipe.intervention_type}")
        if r.recipe.source_paper_arxiv:
            console.print(f"   source: arXiv {r.recipe.source_paper_arxiv}")
        console.print(f"   {r.recipe.description}")


def _load_decision(path: Path, *, line: int) -> GateDecision:
    """Load a GateDecision from a JSON or JSONL file at ``path``.

    Robust to both single-JSON files (older audit logs) and JSONL (current).
    For JSONL, ``line`` selects which entry (default ``-1`` = last).
    """
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise typer.BadParameter(f"{path} is empty.")
    if "\n" in raw:
        lines = [s for s in raw.splitlines() if s.strip()]
        idx = line if line >= 0 else len(lines) + line
        if not (0 <= idx < len(lines)):
            raise typer.BadParameter(f"line index {line} out of range (file has {len(lines)} lines)")
        blob = lines[idx]
    else:
        blob = raw
    data = json.loads(blob)
    # Reconstruct SliceAttribution and GateDecision from the dict.
    slice_attrs = [
        SliceAttribution(**s) for s in data.get("slice_attributions", [])
    ]
    data["slice_attributions"] = slice_attrs
    return GateDecision(**data)


def main():
    app()


if __name__ == "__main__":
    main()
