# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
