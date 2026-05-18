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
    HoldoutSet,
    RegressionGate,
    ReplayBuffer,
    append_audit,
)

app = typer.Typer(
    help="adaptergate — regression-gating for fine-tuned LLM adapters.",
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
app.add_typer(holdout_app, name="holdout")
app.add_typer(replay_app, name="replay")

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
    quiet: bool = typer.Option(False, help="Print only JSON decision."),
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

    if quiet:
        typer.echo(decision.to_json())
    else:
        verdict_style = "green" if decision.accepted else "red"
        verdict = "ACCEPTED" if decision.accepted else "REJECTED"
        console.rule(f"[{verdict_style}]{verdict}[/{verdict_style}]")
        console.print(f"Tenant:    [cyan]{decision.tenant_id}[/cyan]")
        console.print(f"Candidate: [cyan]{decision.candidate_id}[/cyan]")
        console.print(f"Baseline:  [cyan]{decision.baseline_id}[/cyan]")
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
            if driver.regressed_query_ids:
                preview = ", ".join(driver.regressed_query_ids[:5])
                more = (
                    f" + {len(driver.regressed_query_ids) - 5} more"
                    if len(driver.regressed_query_ids) > 5
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
            console.print(f"\n[yellow]{len(decision.regressions)} queries regressed total[/yellow]")
        if decision.improvements:
            console.print(f"[green]{len(decision.improvements)} queries improved total[/green]")

    raise typer.Exit(0 if decision.accepted else 1)


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


def main():
    app()


if __name__ == "__main__":
    main()
