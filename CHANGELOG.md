# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] — 2026-05-18

### Retention quick wins (from the CTO buyer agent's 30-day frustration list)

#### Added

- **`--format`** flag on ``adaptergate gate``: ``human`` (default rich CLI),
  ``json`` (structured for piping/CI consumers), ``pr-comment``
  (GitHub-flavored Markdown for posting on pull requests). The previous
  ``--quiet`` flag is preserved as an alias for ``--format json``.

- **`--show-failures N`** flag on ``adaptergate gate``: configure how
  many failing query IDs to preview under the driver slice. Defaults to 5.

- **Suspected duplicate slice tag detection**. The gate now flags pairs of
  slice tags that look like accidental duplicates (e.g. ``"billing_dispute"``
  alongside ``"intent=billing_dispute"`` from different eval-set authors,
  or hyphen-vs-underscore drift). Exposed as
  ``GateDecision.suspected_duplicate_slices``; the CLI renders a stderr
  warning when populated. We *report*, we do not *merge* — the customer
  decides whether to normalize.

- **Holdout staleness check**. ``HoldoutSet.staleness_days()`` returns days
  since the most-recent query was added. The CLI warns when staleness
  exceeds ``--staleness-threshold-days`` (default 30). Stops customers
  from misreading eval-set drift as adapter drift.

#### Tests

12 new tests (5 dupe detection, 4 staleness, 3 PR-comment rendering).
Total 81 passing. Ruff clean.

#### Deferred to later

- **Diff view** (``adaptergate review --query X``) — requires the scorer
  contract to optionally return generated text alongside the score; that
  API change is too disruptive for v0.4. Planned for v0.6.
- **Baseline drift handling** — the gate currently treats baseline as
  ground truth, which is wrong for online-updating adapters. Deeper rework;
  planned alongside v0.5 recipe library.

---

## [0.3.0] — 2026-05-18

### The Slack-converter

When the gate rejects with a driver slice, adaptergate now emits a
**Pattern** line summarising what the failing queries have in common:

    Pattern: all 10 failing queries contain: "order_id", "refund"

Pure N-gram frequency analysis with stopword filtering and token-overlap
deduplication. No LLM, no extra dependencies. The line a customer's PM
screenshots into Slack at 2am to start triage.

### Added

- `adaptergate.gating.cluster.find_pattern()` — public API for N-gram
  failure pattern detection. Takes a list of query dicts, returns either
  a one-line description or ``None`` (no clear pattern).
- `SliceAttribution.pattern: str | None` — computed automatically by the
  gate for each slice that has regressed queries.
- CLI renders the pattern under the driver slice when present.

### Hardening (from 3-agent code review of v0.2)

- **`_pick_query_id()`** helper replaces the previous `or`-chain that fell
  through on falsy IDs like ``"0"`` or ``""``. Now uses ``is not None``
  semantics.
- **`schema_version: int = 2`** on `GateDecision` so audit-log consumers can
  detect older records.
- **`malformed_slice_queries: int`** counter on `GateDecision` plus a
  stderr warning when held-out queries have a non-list ``slices`` field —
  no more silent drops misread as "feature broken."
- **CLI multi-slice clarification**: "25 unique queries regressed" + note
  explaining slice counts may sum higher when queries belong to multiple
  slices (previously confusing).
- **Ghost docstring removed**: regression_gate.py no longer references a
  "council" consumer that doesn't exist in the repo.

### Tests

12 new tests (9 cluster, 3 robustness). Total 69 passing. Ruff clean.

### Docs

- README rewrite: slice + pattern output above the fold; killed stale
  "v0.1 — early" status; added explicit **Scope** section listing
  what's *not* in v0.3 (LLM cause hypothesis, counterfactual data,
  recipe library); v0.4 roadmap surfaces the recipe-library moat plan.

---

## [0.2.1] — 2026-05-18

### Hardening

- **Slice input validation** — when ``query["slices"]`` is a string instead of
  a list (common typo), the gate no longer iterates characters and creates
  per-letter slices. Non-string entries in the list are dropped silently.
  Backed by 4 new robustness tests.
- **Missing query IDs** — queries without ``question_id``/``id``/``query_id``
  no longer pollute the CLI's "Failing query IDs" preview with stringified
  ``None`` values; only named IDs are shown.
- **`baseline_id=None` rendering** — CLI now displays ``(none)`` instead of
  the literal string ``None`` when first-adapter promotion runs with
  ``--no-require-calibration``.
- **Module docstrings** updated to mention slice attribution.
- **CLI help text** updated to match the v0.2 tagline.

### Tests

5 new robustness tests. Total: 57 passing. Ruff clean.

---

## [0.2.0] — 2026-05-18

### The differential — slice-level reject explain

When a gate rejects, adaptergate now tells you **which behavioral slice broke**
instead of just "score dropped." Held-out queries can carry slice tags
(``query["slices"] = ["intent=refund", "lang=es", "difficulty=hard"]``); the
gate aggregates score deltas per slice and surfaces the **driver slice** —
the cohort your customer notices first.

This puts adaptergate at the seam between training-data lineage and per-tenant
eval outcomes, where causal root-cause analysis lives. Generic prompt-eval
tools (Braintrust, DeepEval, LangSmith, Promptfoo, W&B Registry) own the eval
seam: they can tell you a score dropped, not which adapter behavioral cohort
broke. adaptergate v0.2 owns the seam.

### Added

- `SliceAttribution` dataclass — per-slice regression breakdown
  (`slice_tag`, `n_total`, `n_regressed`, `score_baseline`, `score_candidate`,
  `delta`, `regressed_query_ids`)
- `GateDecision.slice_attributions: list[SliceAttribution]` — populated when
  held-out queries carry slice tags. Sorted most-regressed-first.
- `GateDecision.driver_slice: SliceAttribution | None` — convenience accessor
  for the worst slice (or `None` if no slices regressed).
- CLI rendering of slice breakdown on rejected gates, including up to 5 failing
  query IDs from the driver slice for one-click triage.
- Slice-aware mock scorer — `adaptergate.examples.mock_scorer` now applies
  per-slice penalties so the CLI demo shows realistic per-slice attribution.

### Changed

- Tagline: "Regression-gating for fine-tuned LLM adapters" → "CI gate for
  per-tenant LoRA adapters that update online" (more specific to the wedge).

### Tests

- 7 new tests for slice attribution behavior (empty, per-tag, driver, multi-tag,
  ordering, serialization). Total: 52 passing.

---

## [0.1.0] — 2026-05-18

Initial release.

### Added

- `RegressionGate` — per-tenant accept/reject decision for LoRA adapter updates.
  Configurable epsilon, strict per-query mode, calibration requirement.
- `HoldoutSet` — JSONL-backed per-tenant held-out query store with deterministic
  sampling, idempotent add, and size-bounded rolling window.
- `ReplayBuffer` — JSONL-backed record of rejected updates carrying the full
  `GateDecision` blob for downstream analysis.
- `adaptergate` CLI built with typer. Subcommands: `gate`, `holdout`, `replay`,
  `version`. CI-friendly exit codes (0 accepted, 1 rejected, 2 usage error).
- `adaptergate.examples.mock_scorer` — deterministic scorer for trying the CLI
  end-to-end without a real model.
- 45 unit tests across the gating subsystem and BIRD-SQL eval primitives.
- Apache-2.0 LICENSE + NOTICE crediting upstream papers and software.

### Built on (cited, not invented)

- ProCL (arXiv 2605.13162) — program-memory LoRA architecture
- Silent Collapse / MTR (arXiv 2605.14588) — drift detection framework
- Online-LoRA (arXiv 2411.05663) — task-free online updates
- N-LoRA / O-LoRA (arXiv 2408.06133, arXiv 2310.14152) — orthogonal subspaces

Reference implementations of ProCL, Silent Collapse, a Qwen 2.5 Coder backend,
and the BIRD-SQL eval harness ship under the `[ml]` and `[sql-example]`
optional extras for users who want them.

### Known limitations

- Multi-GPU serving integration (vLLM/SGLang) not yet shipped — the gate is
  serving-agnostic by design and works with any stack that can produce a
  scorer callable.
- Reference ProCL / Silent Collapse modules are paper-faithful re-implementations
  and have manual smoke tests under `__main__`, not pytest unit tests (they
  require torch). Unit coverage for those modules is planned for v0.2.
