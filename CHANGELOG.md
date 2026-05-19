# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.4] — 2026-05-19

### README/docs honesty pass (no code changes)

External review by GPT surfaced three real credibility issues in the
README that survived the v0.5.1 honesty pass. All fixed here. No code
changed; PyPI release exists so the project page reflects the corrected
docs.

1. **README contradiction removed.** The `Scope` section's "NOT in (yet)"
   list still mentioned "recipe library for repairs" — but `recipes/`
   has shipped since v0.5.0 with `RecipeStore`, seed recipes, and
   `recommend-cmd`. The list now reflects what's actually missing in
   v0.5.4: cause hypothesis, automatic counterfactual generation,
   `slice_epsilon` auto-calibration, cross-tenant recipe efficacy
   aggregation, baseline-drift handling, recipe-loop falsification
   (apply → re-gate → ACCEPT), multi-base-model orchestration, hosted
   dashboard.

2. **Recipe library honesty caveat made explicit.** Added a dedicated
   paragraph in the Scope section: the recipe library is a structured
   citation index with empirical-ranking plumbing in place, NOT a
   "tells-you-what-to-do" oracle. Efficacy data accumulates via
   `RecipeStore.add_application(...)`; cross-tenant aggregation is v0.6
   work.

3. **InsightFinder claim removed.** The v0.5.x README opened with a
   "91% drift / 14-18-day detection lag" quote attributed to
   InsightFinder 2026. The number could not be verified against a primary
   InsightFinder source, only a secondary writeup. Replaced with a more
   defensible sober framing about LLM drift as a production-reliability
   problem.

4. **BIRD-SQL run reframed as "reference smoke run", not benchmark.**
   The Qwen 2.5 Coder 14B / `student_club` numbers (+26.6pp memorize,
   +10.0pp held-out) are small-N single-seed and were being shown with
   bold treatment that implied benchmark-level evidence. Now explicitly
   labeled "Reference smoke run ... small N, single seed, not a
   benchmark" and explicitly NOT central evidence — pointer to
   `adaptergate demo silent` as the load-bearing case.

5. **New "Who this is NOT for" section.** Explicitly disqualifies the
   four common workflows adaptergate doesn't fit: hosted-API-only,
   single-global-model, one-shot fine-tunes, no-held-out-yet. Saves
   prospective users a wasted install.

6. **Stale test count.** README claimed "92 tests" in two places after
   v0.5.2/v0.5.3 added 14 more. Updated to 106.

### Tests

- Still 106/106 passing. No code changed in this release.

## [0.5.3] — 2026-05-18

### Dogfooding pass: fixes from a real-user CLI exercise

After v0.5.2 shipped, the maintainer dogfooded the package end-to-end on a
realistic fintech-triage scenario (52-query held-out, custom scorer, real
silent bug in candidate adapter). 7 frictions surfaced; 5 are addressed here.

1. **Bug fix: reason string lied about aggregate when both thresholds breached.**
   When the candidate violated BOTH `--epsilon` (aggregate) AND `--slice-epsilon`,
   the gate's reason string still said `"aggregate ... is within ε"` —
   factually wrong. Now distinguishes:
   - Silent-regression case (aggregate within ε, slice collapsed): unchanged.
   - Both-violated case: explicitly states aggregate exceeds ε *and* the
     worst slice also collapsed. New test
     `test_slice_epsilon_still_rejects_aggregate_when_both_violated`
     asserts the reason does NOT claim `"within ε"` when aggregate isn't.

2. **`adaptergate holdout import --from-jsonl PATH` — batch import.** The
   v0.5.2 CLI only supported `holdout add` one query at a time, forcing
   real users with 100+ queries to either pay 100 subprocess startups or
   reverse-engineer the JSONL format from the source. New `import`
   subcommand reads a JSONL file, validates each line, skips malformed
   ones with a stderr warning, and exits 2 if any line was skipped.

3. **`adaptergate replay show --index N` — drill into a rejection.** The
   v0.5.2 `replay list` was compact-only (`{candidate, baseline, delta, reason}`)
   so debugging a rejection meant grepping `audit.jsonl` by timestamp. The
   new `show` cross-references the replay buffer's compact summary with the
   audit log entry (matched by candidate + timestamp) and renders the full
   slice attribution + N-gram pattern + driver-slice failing query IDs —
   the same view the gate produced when the rejection happened.

4. **Flag alias `replay list --replay-path` (matches `gate --replay-path`).**
   v0.5.2 used `--replay` here but `--replay-path` on the gate — same
   concept, two different flags. `--replay-path` is now accepted on both
   `replay list` and `replay show` (old `--replay` still works as alias for
   one minor).

5. **Slice payload validation in `holdout add`/`holdout import`.** A common
   typo was passing `"slices": "intent=foo"` (bare string) instead of
   `"slices": ["intent=foo"]` (list). v0.5.2 silently ingested the malformed
   payload; the gate later flagged it as `malformed_slice_queries` after
   the slice signal was already corrupted. Now `holdout add` rejects at
   ingest with a precise error message, and `holdout import` skips the
   line with a stderr warning.

### Tests

- 106 passing (was 97). 9 new tests covering the slice validation, batch
  import skip/exit-code behavior, replay show, and the reason-string
  accuracy fix.

### Deferred to v0.6 (not v0.5.x)

- `holdout list --limit` flag (cosmetic; dumps all queries today).
- Auto-calibration of `--slice-epsilon` from a rolling noise floor
  (CTO buyer's TRIAL → BUY upgrade trigger).
- Cross-tenant recipe efficacy aggregation (cold-start ends when usage data
  accumulates).
- Baseline drift handling (rolling-window baseline + staleness flag).

## [0.5.2] — 2026-05-18

### Council-driven pass: prove the differential, fix the on-ramp

3-agent council review of v0.5.1 (ML engineer / Devrel / CTO buyer) found
the v0.5.1 CPU demo was "credible plumbing but not credible proof":

- ML eng: aggregate dropped 40pp — even a t-test would catch it without
  slice attribution. The differential vs Braintrust mean-score eval was
  never demonstrated.
- Devrel: README sent readers to write a scorer + held-out by hand. The
  runnable proof was buried. Show HN would die at <10 upvotes.
- CTO buyer: sklearn classifier didn't translate to autoregressive LoRAs;
  no evidence the N-gram pattern survives the generative jump.

All three addressed in v0.5.2:

1. **New `--slice-epsilon` flag + `slice_min_size`.** The gate now rejects
   when any slice's score drops by more than `--slice-epsilon`, even if
   aggregate stays within `--epsilon`. This is the silent-regression
   safety net — the case where mean-score eval accepts and the slice
   collapse ships to prod. The reject reason explicitly flags the
   silent-regression scenario so CI bots / dashboards know what fired.
   `slice_min_size` (default 3) prevents 1-of-2 outliers from triggering
   rejection. 5 new tests in `test_regression_gate.py`.

2. **Three bundled CPU-only demos via `adaptergate demo ...`.** Zero
   clone, zero config. `pip install adaptergate[demo] && adaptergate
   demo silent` reproduces the killer differential: same data, gate
   run twice — first aggregate-only (ACCEPTED), then with
   `--slice-epsilon 0.10` (REJECTED).
   - `adaptergate demo classifier` — aggregate regression (sklearn LR).
   - `adaptergate demo silent` — silent slice regression (the load-bearing one).
   - `adaptergate demo sql` — generative scorer (SQL output, AST-equality
     via sqlglot or normalized string equality fallback). Adapter B has
     a textbook `= NULL` bug — silent on routine queries, catastrophic
     on the null_check slice.

3. **Demos live in-package (`src/adaptergate/demos/`).** Bundled with
   `pip install` so they work without `git clone`. New `demo` extras
   group installs scikit-learn.

4. **README on-ramp restructured.** New "60-second demo (no setup, no
   GPU)" section moved above Quickstart. Removed the old `for i in $(seq
   1 30)` mock-scorer block — `adaptergate demo` replaces it cleanly.
   New section 4 documents `--slice-epsilon`.

5. **Demo v1 fixes (HELD_OUT_ORDER variance + narrative gloss).** The
   v0.5.1 demo's control slice was 15 copies of one templated query —
   ML eng flagged it as a degenerate control. Now 15 lexically varied
   order-status queries. Demo output now closes with a plain-English
   "what just happened" explainer for non-ML readers.

### Tests

- 97 passing (was 92). 5 new tests for `slice_epsilon` behavior across
  the silent-collapse / min-size / no-collapse / both-violations cases.

## [0.5.1] — 2026-05-18

### Honesty pass (from the v0.5 council re-review)

The 3-agent re-review of v0.5 was generous on engineering but flagged five
specific credibility risks. All five fixed:

1. **Heuristic recipes tagged honestly.** Two of the seven seed recipes
   (``replay_buffer_prune_recent_v1``, ``lora_rank_reduce_v1``) had
   ``source_paper_arxiv: null`` while the README pitched "paper-derived
   intervention recipes." They are now explicitly tagged ``"(heuristic)"``
   in name and description, with ``source_paper_title`` stating they are
   standard practice rather than paper-cited.

2. **No more "95% confidence interval" claim for n=3.** Renamed
   ``RecipeRecommendation.confidence_low/high`` → ``efficacy_range_low/high``
   and added a ``range_method: str | None`` field (currently
   ``"normal_n_gte_3"``) so downstream consumers see exactly how the
   interval was computed. With n=3-5 the normal approximation is wide and
   the 95% coverage guarantee does not hold — calling it a CI was
   statistically dishonest.

3. **README scope-honest about compounding.** The "compounds across
   customers" line is removed in favor of three explicit tiers:
   within-store (works today), within-store cross-tenant aggregation
   (v0.6 roadmap), cross-organization (not shipped, no central service
   exists).

4. **Ghost roadmap reference removed.** The ``store.py`` docstring used to
   reference an "automated radar.db ingester" that does not exist. Rewritten
   to describe the actual state (manual seeding + future automation) without
   implying shipped features.

5. **CLI cold-start disclaimer.** When ``recommend-cmd`` runs against an
   empty application log, it now prints a stderr note that the ranking
   reflects slice matching only, not observed efficacy. This makes the
   day-0 user experience honest about what the recommender can and cannot
   do until applications accumulate.

### No new features

This is a documentation + naming patch. No new tests required; the rename
of ``confidence_low/high`` → ``efficacy_range_low/high`` is internal and
covered by the existing serialization tests. Total tests: 92. Ruff clean.

---

## [0.5.0] — 2026-05-18

### The moat — recipe library + observed_efficacy

The strategist's pick from the v0.2 council review: a *typed* library of
paper-derived intervention recipes, indexed against slice signatures, with
an empirically-tracked ``observed_efficacy`` column that strengthens with
every customer's application. Generic eval frameworks (Braintrust, DeepEval,
LangSmith) tell you *what* failed. v0.5 tells you *what to do*, ranked by
*what has worked* across N prior applications.

This is the asset that compounds across customers and cannot be replicated
by force of capital.

#### Added — new public API

- **`Recipe`** dataclass (``adaptergate.recipes.Recipe``) — a typed
  intervention with ``intervention_type``, ``applies_when`` slice
  predicates, paper citation, default params.
- **`RecipeApplication`** dataclass — one customer's use of one recipe
  with measured before/after delta, tenant-anonymized.
- **`RecipeRecommendation`** dataclass — a scored pick.
- **`RecipeStore`** — JSONL-backed library (``recipes.jsonl``) and
  append-only application log (``applications.jsonl``).
- **`recommend(gate_decision, store)`** — rank recipes for a rejected
  decision by ``observed_efficacy``. Evidence beats no-evidence; recipes
  with prior applications outrank fresh entries.
- **`hash_tenant(tenant_id)`** — anonymize for cross-customer logging.

#### Added — CLI

- `adaptergate recipes seed --recipes PATH` — populate from the
  package-bundled seed library.
- `adaptergate recipes list --recipes PATH`
- `adaptergate recipes show RECIPE_ID --recipes PATH`
- `adaptergate recommend-cmd --decision audit.jsonl --recipes PATH`
  (registered as ``adaptergate recommend`` in the CLI tree)

#### Seed library (ships with the package)

Seven hand-curated recipes derived from May-2026 CL/PEFT literature:

  - **ProCL slot rebalance** (arXiv 2605.13162) — allocate a new program
    slot for the driver-slice queries.
  - **Online-LoRA learning-rate decay** (arXiv 2411.05663).
  - **N-LoRA subspace orthogonalization** (arXiv 2408.06133).
  - **Replay buffer prune** (no citation — heuristic).
  - **LoRA rank reduction** (no citation — heuristic).
  - **Silent Collapse trust-throttle** (arXiv 2605.14588) — τ-based LR scaling.
  - **StableEdit localized patch** (arXiv 2605.11836) — surgical layer edit.

Each ships with intervention_type, slice predicates, default params, paper
citation. Customers can add their own recipes; the seed is just a start.

#### Tests

11 new tests covering Recipe.matches, RecipeStore roundtrip, recommend
ranking, evidence-beats-no-evidence ordering, hash_tenant determinism.
Total 92 passing. Ruff clean.

#### Deferred to v0.5.x / v0.6

- **Automated radar.db → recipe ingestion** — LLM-driven extraction from
  newly-published CL papers into typed recipes. Hard problem; quality
  control needs care. Manual recipe authoring works for v0.5.0.
- **Cross-tenant pattern matching** — surfacing "this regression style
  failed at 47 other tenants." Requires a multi-tenant application
  corpus; emerges naturally as adoption grows.

---

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
