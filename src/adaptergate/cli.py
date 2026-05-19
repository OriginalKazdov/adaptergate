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
demo_app = typer.Typer(
    help="Run bundled CPU-only demos (zero-config, requires scikit-learn).",
    no_args_is_help=True,
    add_completion=False,
)
app.add_typer(holdout_app, name="holdout")
app.add_typer(replay_app, name="replay")
app.add_typer(recipes_app, name="recipes")
app.add_typer(demo_app, name="demo")

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
    epsilon: float = typer.Option(0.02, help="Max acceptable aggregate score drop."),
    slice_epsilon: Optional[float] = typer.Option(
        None,
        "--slice-epsilon",
        help=(
            "Max acceptable drop within any single slice (silent-regression safety net)."
            " If set, the gate rejects when a slice collapses even if aggregate stays"
            " within --epsilon. Recommended: 0.10 (slices are smaller / noisier)."
        ),
    ),
    slice_min_size: int = typer.Option(
        3,
        "--slice-min-size",
        help="Minimum slice size to consider for slice-level rejection. Smaller slices are surfaced in attribution but cannot trigger rejection.",
    ),
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
            slice_epsilon=slice_epsilon,
            slice_min_size=slice_min_size,
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


def _validate_slice_payload(payload: dict) -> None:
    """Raise BadParameter on common slice-tag mistakes that silently corrupt
    the held-out set.

    Caught here so the ingest path is the integrity boundary — the gate runs
    later don't have to defend against malformed slices in production.
    """
    slices = payload.get("slices")
    if slices is None:
        return
    if not isinstance(slices, list):
        raise typer.BadParameter(
            f"Query 'slices' must be a JSON list of strings, got {type(slices).__name__}."
            f" Common mistake: writing \"slices\": \"intent=foo\" (bare string) instead of"
            f' "slices": ["intent=foo"]. Fix the query and re-run.'
        )
    for i, tag in enumerate(slices):
        if not isinstance(tag, str):
            raise typer.BadParameter(
                f"Query 'slices[{i}]' must be a string, got {type(tag).__name__}: {tag!r}."
            )
        if "=" not in tag:
            raise typer.BadParameter(
                f"Slice tag {tag!r} should be 'key=value' (e.g. 'intent=refund_routine')."
                f" Slice tags without a '=' won't aggregate cohesively across queries."
            )


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
    _validate_slice_payload(payload)
    holdout = HoldoutSet(tenant_id=tenant, path=holdout_path)
    record = holdout.add(payload, accepted_by=accepted_by)
    typer.echo(json.dumps({"added": record.query_id, "size": len(holdout)}))


@holdout_app.command("import")
def holdout_import(
    tenant: str = typer.Option(...),
    holdout_path: Path = typer.Option(..., "--holdout"),
    from_jsonl: Path = typer.Option(
        ...,
        "--from-jsonl",
        help=(
            "JSONL file to import. Each line must be a JSON object with at least"
            " a 'question_id' field. Other fields (question, gold, slices, etc.)"
            " are stored verbatim in the held-out payload."
        ),
    ),
    accepted_by: Optional[str] = typer.Option(
        None, help="Tag every imported query as accepted by this adapter version."
    ),
):
    """Batch-import queries from a JSONL file into the held-out set.

    Each line must be one JSON object — the query payload. Lines that fail
    validation are skipped with a warning to stderr; the rest are imported.
    Exit status: 0 if all lines imported, 2 if any line was skipped.
    """
    if not from_jsonl.exists():
        err_console.print(f"[red]File not found: {from_jsonl}[/red]")
        raise typer.Exit(2)

    holdout = HoldoutSet(tenant_id=tenant, path=holdout_path)
    n_added = 0
    skipped: list[tuple[int, str]] = []
    with from_jsonl.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as e:
                skipped.append((lineno, f"invalid JSON: {e}"))
                continue
            try:
                _validate_slice_payload(payload)
            except typer.BadParameter as e:
                skipped.append((lineno, str(e)))
                continue
            try:
                holdout.add(payload, accepted_by=accepted_by)
                n_added += 1
            except Exception as e:
                skipped.append((lineno, f"add failed: {e}"))

    typer.echo(json.dumps({
        "imported": n_added,
        "skipped": len(skipped),
        "size": len(holdout),
    }))
    for lineno, reason in skipped[:10]:
        err_console.print(f"[yellow]Skipped line {lineno}: {reason}[/yellow]")
    if len(skipped) > 10:
        err_console.print(f"[yellow]... and {len(skipped) - 10} more skipped lines.[/yellow]")
    if skipped:
        raise typer.Exit(2)


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
    replay_path: Path = typer.Option(
        ..., "--replay", "--replay-path",
        help="Path to the replay buffer JSONL (the same one passed to `gate --replay-path`).",
    ),
    n: int = typer.Option(10, help="Show the most recent N rejected updates."),
):
    """Show recent rejected updates (compact one-line summary per rejection)."""
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


@replay_app.command("show")
def replay_show(
    tenant: str = typer.Option(...),
    replay_path: Path = typer.Option(
        ..., "--replay", "--replay-path",
        help="Path to the replay buffer JSONL.",
    ),
    audit_log: Path = typer.Option(
        ..., "--audit-log",
        help="Path to the audit log JSONL (the same one passed to `gate --audit-log`).",
    ),
    index: int = typer.Option(
        1, "--index", "-i",
        help="Which rejection to show. 1 = most recent (default).",
    ),
    show_failures: int = typer.Option(
        5, "--show-failures",
        help="How many failing query IDs to preview under the driver slice.",
    ),
):
    """Drill into a rejection from the replay buffer.

    Cross-references the replay buffer's compact summary with the full audit
    log entry (by timestamp) and renders the full slice attribution, N-gram
    pattern, and driver-slice failing query IDs — same view as ``adaptergate
    gate`` produced when the rejection happened.
    """
    if index < 1:
        raise typer.BadParameter("--index must be ≥ 1 (1 = most recent).")
    buf = ReplayBuffer(tenant_id=tenant, path=replay_path)
    recent = list(buf.recent(n=index))
    if index > len(recent):
        err_console.print(
            f"[red]Replay buffer has {len(recent)} rejection(s); --index={index} is out of range.[/red]"
        )
        raise typer.Exit(2)
    target = recent[index - 1]

    if not audit_log.exists():
        err_console.print(f"[red]Audit log not found: {audit_log}[/red]")
        raise typer.Exit(2)

    matched: Optional[GateDecision] = None
    with audit_log.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                data.get("candidate_id") == target.candidate_id
                and data.get("timestamp") == target.rejected_at
            ):
                slice_attrs = [SliceAttribution(**s) for s in data.get("slice_attributions", [])]
                data["slice_attributions"] = slice_attrs
                matched = GateDecision(**data)
                break

    if matched is None:
        err_console.print(
            f"[red]No audit-log entry found for candidate={target.candidate_id} at"
            f" {target.rejected_at}. Was --audit-log passed to `gate` when this rejection"
            f" was recorded?[/red]"
        )
        raise typer.Exit(2)

    _render_human(
        matched,
        staleness_days=None,
        staleness_threshold_days=30,
        show_failures=show_failures,
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


@demo_app.command("classifier")
def demo_classifier():
    """Run the classifier demo: aggregate regression caught by the gate.

    sklearn TF-IDF + LR classifier as a stand-in for a fine-tuned LoRA.
    Adapter B is trained on contaminated labels; the gate catches the
    regression and surfaces the driver slice + N-gram pattern + recipes.
    """
    _run_demo("classifier")


@demo_app.command("silent")
def demo_silent():
    """Run the silent-slice-regression demo: the case adaptergate exists for.

    Same data, gate run TWICE — first without ``--slice-epsilon`` (aggregate-
    only eval ACCEPTS the silent slice collapse), then with ``--slice-epsilon
    0.10`` (adaptergate REJECTS). This contrast is the differential vs naive
    mean-score evals like Braintrust / DeepEval.
    """
    _run_demo("silent")


@demo_app.command("sql")
def demo_sql():
    """Run the generative-SQL demo: gate works on autoregressive output.

    Adapters are Python functions emitting SQL strings. Scorer uses sqlglot
    AST-equality when available (``pip install adaptergate[sql-example]``),
    otherwise normalized string equality. Adapter B has a textbook
    NULL-handling bug — silent on routine queries, catastrophic on the
    null_check slice.
    """
    _run_demo("sql")


def _run_demo(kind: str) -> None:
    """Dispatcher used by the ``demo`` subcommands."""
    try:
        # Heavy deps (sklearn, etc.) live inside the demo modules. Import
        # lazily so the rest of the CLI doesn't require them.
        if kind == "classifier":
            from adaptergate.demos.classifier import run as run_demo
        elif kind == "silent":
            from adaptergate.demos.silent import run as run_demo
        elif kind == "sql":
            from adaptergate.demos.sql import run as run_demo
        else:
            raise typer.BadParameter(f"Unknown demo kind: {kind!r}")
    except ImportError as exc:
        err_console.print(
            f"[red]Could not load the {kind} demo: {exc}[/red]\n"
            f"Demos require scikit-learn. Install with:\n"
            f"    pip install 'adaptergate[demo]'"
        )
        raise typer.Exit(2)

    import tempfile
    workdir = Path(tempfile.mkdtemp(prefix=f"adaptergate-{kind}-demo-"))
    run_demo(workdir)


def main():
    app()


if __name__ == "__main__":
    main()
