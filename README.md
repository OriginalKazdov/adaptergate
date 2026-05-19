# adaptergate

**CI gate for per-tenant LoRA adapters that update online.**

When a customer-specific LoRA adapter is about to be promoted to production,
`adaptergate gate` evaluates the candidate against a per-tenant held-out set
and refuses to promote it if aggregate quality drops more than ε. Rejected
adapters go to a replay buffer for later analysis. CI-friendly exit codes.
Serving-stack agnostic: you supply a scorer callable, we supply the gate.

When your held-out queries carry slice tags (intent, language, difficulty,
whatever) and natural-language text, adaptergate doesn't just say "score
dropped." It tells you **which behavioral slice broke**, **shows you the
failing query IDs**, and **describes what the failing queries have in
common** — the line your on-call PM screenshots into Slack at 2am.

```
$ adaptergate gate \
    --tenant acme \
    --candidate adapter_v19 \
    --baseline adapter_v18 \
    --holdout data/acme_holdout.jsonl \
    --scorer my_eval:score
─────────────────────────────────── REJECTED ───────────────────────────────────
Tenant:    acme
Candidate: adapter_v19
Baseline:  adapter_v18
Score:     0.924 → 0.353  (Δ=-0.571, ε=0.02)
Held-out:  n=25
Reason:    REJECTED: aggregate 0.924 → 0.353 (Δ=-0.571) over n=25.
           Drop exceeds ε=0.02.

DRIVER SLICE: intent=billing_dispute   0.946 → 0.113  (Δ=-0.834, 10/10 regressed)
  Pattern: all 10 failing queries contain: "order_id", "refund"
  Failing query IDs: billing_1, billing_2, billing_3, billing_4, billing_5 + 5 more

Slice breakdown (most-regressed first):
  -0.834   10/10 regressed   intent=billing_dispute
  -0.396   15/15 regressed   intent=order_status

25 unique queries regressed (slice n_regressed values may sum higher when
queries belong to multiple slices)
$ echo $?
1
```

That `Pattern: ...` line is N-gram frequency analysis — no LLM, no extra
dependencies, no cloud calls. Just the common words across failing queries.
Slack-paste-friendly by design.

---

## Why it exists

LLM drift and silent regression are increasingly treated as production
reliability problems — especially in systems that update continuously from
user feedback rather than ship as one-shot fine-tunes.

The dominant failure mode for teams serving per-customer fine-tuned LLMs is
**silent regression on online updates**: a sub-skill (e.g. `JOIN-with-aggregate`
accuracy) collapses from 91% to 64% while the aggregate eval stays green at
87%, and you only find out when a customer Slacks support two weeks later.

### Where adaptergate sits in the landscape

Generic LLM eval CIs ([Braintrust](https://www.braintrust.dev/),
[DeepEval](https://deepeval.com/), [LangSmith](https://www.langchain.com/langsmith),
[Promptfoo](https://www.promptfoo.dev/), [W&B Registry](https://wandb.ai/site/automations/))
all support pre-deploy CI gating with non-zero exit on regression. They're
built for the case where the artifact under test is a **prompt or chain
commit**, scored against a single fixed dataset.

Runtime guardrails ([Galileo Luna-2](https://galileo.ai/luna-2),
[Arize](https://arize.com/), [Langfuse](https://langfuse.com/)) catch failures
after the model has shipped.

adaptergate is for the workflow those tools aren't built for:

- **Per-tenant scoping** — each customer has its own held-out set; regression
  is measured against that customer's queries, not a shared benchmark.
- **Online update cadence** — every accepted user query may trigger a new
  adapter version, not a quarterly retrain.
- **LoRA-adapter aware** — the artifact under test is a binary adapter, not
  a prompt commit.
- **Replay buffer for rejected updates** — rejects don't disappear; they're
  preserved with the full gate decision for later analysis or downstream
  repair logic.

Closest neighbors:

- [Predibase / LoRAX](https://github.com/predibase/lorax) — per-tenant LoRA
  *serving* and continuous fine-tuning, no CI gate primitive.
- [Baseten rank-1 LoRA continual learning](https://www.baseten.co/research/write-small-learn-forever/)
  — same problem shape (shadow replica, ring buffer for rollback) but it's
  research infrastructure, not a product, and has no per-tenant eval gate.

### What we measured

**Reference smoke run** on Qwen 2.5 Coder 14B (4-bit, single RTX 4090) with
ProCL multi-LoRA slots on BIRD-SQL `student_club` — small N, single seed,
not a benchmark:

| | Before update | After update | Δ |
|---|---|---|---|
| `student_club` memorize set | 55.7% | 82.3% | +26.6pp |
| Held-out other DBs (forgetting check) | 45.0% | 55.0% | +10.0pp |

This run shows the *kind of regime* adaptergate is designed for — a
candidate that improves the active domain without damaging held-out
behaviors. The gate fires when that property breaks. Not central
evidence; see `adaptergate demo silent` for the load-bearing case.

---

## Install

```bash
pip install adaptergate
```

Core install is lightweight (typer + pydantic + rich). The gate doesn't
require torch, transformers, or any specific serving stack.

```bash
pip install "adaptergate[demo]"      # + scikit-learn for the bundled demos
pip install "adaptergate[ml]"        # + torch/transformers/peft/bitsandbytes
pip install "adaptergate[sql-example]"  # + sqlglot for AST-equality SQL scoring
```

---

## 60-second demo (no setup, no GPU)

Three bundled CPU-only demos. Each spins up two fake "LoRA adapter versions",
runs the gate, and shows what adaptergate would have told you. Runs in seconds
on any laptop.

```bash
pip install 'adaptergate[demo]'

adaptergate demo classifier    # aggregate regression caught by the gate
adaptergate demo silent        # ← the killer one: silent slice collapse
adaptergate demo sql           # generative scorer (SQL output)
```

What each shows:

- **`classifier`** — two scikit-learn classifiers as stand-ins for fine-tuned
  LoRAs. Adapter B is trained on subtly contaminated labels. Gate REJECTS,
  identifies the driver slice, surfaces the N-gram pattern across failing
  queries, and recommends 3 paper-cited recipes for fixing it.
- **`silent`** — the case adaptergate exists for. 300 queries, 5 of which
  belong to a small but business-critical slice. Adapter B silently collapses
  that one slice. The demo runs the gate **twice**: first like Braintrust
  (aggregate-only) → ACCEPTED; then with `--slice-epsilon 0.10` → REJECTED.
  Same data, different gate config, different outcome.
- **`sql`** — generative scorer. Adapters emit SQL strings; the scorer does
  AST-equality (or normalized string equality). Adapter B has a textbook
  NULL-handling bug — silent on routine queries, catastrophic on the
  null-check slice. Proves the gate + slice attribution + N-gram + recipes
  all survive the classifier → autoregressive jump.

If you have 60 seconds, run them in that order.

---

## Quickstart

### 1. Write a scorer

A scorer is any Python callable `(adapter_id: str, query: dict) -> float`
returning a score in `[0.0, 1.0]`. You almost certainly already have one
for your eval suite — wire it up.

```python
# my_eval.py
def score(adapter_id: str, query: dict) -> float:
    output = run_adapter(adapter_id, query["prompt"])
    return float(matches_gold(output, query["gold"]))
```

### 2. Seed a held-out set

```bash
adaptergate holdout add \
    --tenant acme \
    --holdout data/acme_holdout.jsonl \
    '{"question_id": "q1", "prompt": "...", "gold": "...", "slices": ["intent=refund"]}'
# ... add at least 20 queries (the gate's min_holdout_size).
```

**Batch import.** For dozens or hundreds of queries, dump them as JSONL
(one query payload per line) and import in one command:

```bash
adaptergate holdout import \
    --tenant acme \
    --holdout data/acme_holdout.jsonl \
    --from-jsonl my_eval_set.jsonl
# {"imported": 248, "skipped": 0, "size": 248}
```

Each line of the JSONL is one query payload — the same JSON shape you'd
pass to `holdout add`. Malformed lines are skipped with a stderr warning,
and the command exits 2 if any line was skipped (CI-friendly).

Slices are validated at ingest: `"slices"` must be a JSON list of strings
in `key=value` form (e.g. `["intent=refund", "lang=en"]`). The bare-string
typo `"slices": "intent=foo"` is rejected — slice signal corruption is
caught at the boundary, not on the next gate run.

### 3. Run the gate

```bash
adaptergate gate \
    --tenant acme \
    --candidate adapter_v18 \
    --baseline adapter_v17 \
    --holdout data/acme_holdout.jsonl \
    --scorer my_eval:score \
    --epsilon 0.02 \
    --audit-log data/audit.jsonl \
    --replay-path data/rejected.jsonl
```

Exit code 0 = accepted (safe to promote). 1 = rejected. Use this in your
deploy script.

When a rejection happens, the audit log captures the full attribution and
the replay buffer captures a one-line summary. To drill into a past
rejection without grepping the audit log by timestamp:

```bash
adaptergate replay list --tenant acme --replay-path data/rejected.jsonl
# {"candidate": "adapter_v18", "baseline": "adapter_v17", "delta": -0.154, ...}

adaptergate replay show --tenant acme \
    --replay-path data/rejected.jsonl \
    --audit-log data/audit.jsonl \
    --index 1
# Renders the full slice attribution + N-gram pattern + failing query IDs
# from the most recent rejection (--index 1).
```

### 4. The `--slice-epsilon` safety net

Aggregate-only gating accepts updates where one slice silently collapses but
the rest of the held-out set masks it in the mean. Pass `--slice-epsilon` to
make the gate reject when **any** slice drops more than that threshold,
regardless of aggregate. Recommended starting point: `0.10` (slices are
smaller / noisier than aggregate so the threshold is looser).

```bash
adaptergate gate ... --slice-epsilon 0.10
```

When slice-eps fires, the reject reason explicitly calls out the
silent-regression case so you (or your CI bot) know aggregate alone would
have missed it. See `adaptergate demo silent` for the contrast.

---

## How the gate decides

```
accepted = (score_candidate - score_baseline) >= -epsilon
```

That's the headline rule. The gate runs the scorer against the held-out set
for both the candidate and the baseline, takes the average delta, and
compares to `epsilon` (default `0.02` = 2pp tolerance).

### Modes

- **Default (aggregate):** Reject if average drop > ε.
- **`--strict`:** Also reject if any single query that scored 1.0 on baseline
  now scores less. Catches regression-via-averaging.
- **`--no-require-calibration`:** Allow promotion of a first adapter when no
  baseline exists. Useful for bootstrapping a new tenant.

### Per-query breakdown

Every `GateDecision` includes `per_query`: a list of
`{query_id, score_baseline, score_candidate, delta}` records. Use it to
surface *which* queries regressed, not just *how much*.

```python
decision = gate.evaluate(...)
for q in decision.regressions:
    print(q["query_id"], q["delta"])
```

---

## Recipe library — paper-derived mitigation suggestions

When the gate rejects with a driver slice, adaptergate **suggests
paper-derived mitigation recipes to try next**, with the paper each
recipe came from cited. Today the ranking is heuristic (slice-match
strength), not empirical (efficacy across prior applications) — the
plumbing for empirical ranking is in place via
``RecipeStore.add_application()``, but real efficacy data accumulates
with usage. See the "Recipe library honesty caveat" in the Scope
section above for the full framing.

Generic eval frameworks tell you *what* failed; adaptergate v0.5+
also points you at *the literature most likely relevant to fixing it*.

```bash
# Seed your recipe library from the bundled 7-recipe starter
adaptergate recipes seed --recipes data/recipes.jsonl

# After a gate rejects (audit log captures the decision)
adaptergate recommend-cmd \
    --decision data/audit.jsonl \
    --recipes data/recipes.jsonl \
    --top-k 3
```

```
Recipes for driver slice: intent=billing_dispute

1. ProCL slot rebalance for the driver slice   [no prior applications]
   id: procl_slot_rebalance_v1
   intervention: slot_rebalance
   source: arXiv 2605.13162
   Allocate a new ProCL program slot dedicated to the driver-slice queries...

2. Online-LoRA learning rate decay              [no prior applications]
   id: online_lora_lr_decay_v1
   source: arXiv 2411.05663
   Reduce the LoRA learning rate and re-run training...
```

The compounding mechanic, scope-honest:

- **Within a single store**: every recipe application you log via
  ``RecipeStore.add_application()`` strengthens the recommender's ranking
  for that store. Recipes with positive empirical efficacy outrank fresh
  entries. This works today.
- **Across tenants in a single store**: applications carry an anonymized
  ``tenant_hash``; queries that aggregate across tenants ("recipe X has
  worked for N other tenants on this slice signature") will land in v0.6.
- **Across organizations** (your store vs. some other team's store): NOT
  shipped. There is no centralized recipe-application service. Your
  ``applications.jsonl`` stays on your disk.

Seven seed recipes ship with the package: ProCL slot rebalance,
Online-LoRA LR decay, N-LoRA orthogonalization, Silent Collapse
trust-throttle, StableEdit localized patch (all paper-cited), plus
two heuristic recipes (replay-buffer prune, LoRA rank reduction —
explicitly tagged in the seed file as ``"(heuristic)"`` since they're
common practice rather than paper-cited). Load via
``adaptergate recipes seed --recipes data/recipes.jsonl``.

---

## CI integration & output formats

```bash
# Human-readable CLI output (default)
adaptergate gate --tenant acme --candidate v19 --baseline v18 \
    --holdout data/acme.jsonl --scorer my_eval:score

# Structured JSON for piping into your own tooling
adaptergate gate ... --format json

# GitHub-flavored Markdown for PR comments
adaptergate gate ... --format pr-comment | gh pr comment "$PR" --body-file -

# Configurable failing-ID preview
adaptergate gate ... --show-failures 20

# Detect stale held-out sets
adaptergate gate ... --staleness-threshold-days 14
```

The CLI surfaces three kinds of warnings on stderr (so they survive
``--format json`` piping):

- **Malformed slices** — when a query's ``slices`` field is a string
  instead of a list (common typo).
- **Suspected duplicate slice tags** — when two slice tags look alike
  (e.g. ``"billing_dispute"`` and ``"intent=billing_dispute"``), reported
  via ``GateDecision.suspected_duplicate_slices``.
- **Held-out staleness** — when your held-out set hasn't been refreshed
  in N days. Stops you from misreading eval-set drift as adapter drift.

---

## What's in the box (v0.5)

```
adaptergate/
├── gating/
│   ├── regression_gate.py   # RegressionGate + GateConfig + GateDecision + SliceAttribution
│   ├── holdout_eval.py      # HoldoutSet — per-tenant queries, JSONL-backed, staleness check
│   ├── replay_buffer.py     # ReplayBuffer — rejected updates with full decision
│   └── cluster.py           # find_pattern() — N-gram failure pattern detection
├── recipes/
│   ├── models.py            # Recipe + RecipeApplication + RecipeRecommendation
│   ├── store.py             # RecipeStore — JSONL-backed library + application log
│   └── recommend.py         # recommend(decision, store) — efficacy-ranked picks
├── data/
│   └── seed_recipes.jsonl   # 7 seed recipes derived from May-2026 CL literature
├── cli.py                   # `adaptergate` entry point
└── examples/
    └── mock_scorer.py       # deterministic mock for trying things out
```

Tests: 106 unit tests across the gating subsystem, cluster, robustness,
recipes, and BIRD-SQL eval primitives. Run with `pytest`. Ruff-clean.

### Scope

**In:** per-tenant gate, slice-level attribution, driver slice, failing
query IDs, N-gram pattern of failing queries, `--slice-epsilon` safety
net, replay buffer + `replay show` drill-down, audit log, recipe
library (citation index — see honesty caveat below), CI exit codes,
`--format pr-comment` paste-ready Markdown, three bundled CPU-only demos.

**NOT in (yet):** LLM-generated cause hypothesis, automatic counterfactual
training-row generation, `slice_epsilon` auto-calibration (rolling noise
floor), cross-tenant recipe efficacy aggregation, baseline-drift handling
(rolling-window baseline + staleness flag), recipe-loop falsification
(apply recipe → re-gate → ACCEPT), multi-base-model orchestration, hosted
dashboard. See **Roadmap** below.

**Recipe library honesty caveat:** the recipe library currently ranks by
slice-match heuristics only — no empirical efficacy data yet. The
`recommend-cmd` output explicitly disclaims this. Efficacy data
accumulates as people log applications via `RecipeStore.add_application(...)`
after running a recipe; cross-tenant aggregation is v0.6 work. Read the
library today as a structured citation index with the plumbing for
empirical ranking already in place, not a "tells-you-what-to-do" oracle.

### Who this is NOT for

- **Teams whose only LLM workflow is calling hosted APIs** (no
  fine-tuning, no adapters). adaptergate's gate runs against a scorer
  you supply for two adapter IDs — if there are no adapters to compare,
  the tool isn't useful.
- **Teams shipping one global model, not per-tenant adapters.** The
  held-out logic still applies but the per-tenant scoping is unused
  weight; a generic eval CI like Braintrust/DeepEval/Promptfoo is a
  better fit.
- **Teams doing one-shot fine-tunes, not continuous online updates.**
  If you fine-tune once a quarter and validate manually, a CI gate is
  overkill — review the eval results yourself.
- **Teams without a held-out eval set.** Build one first
  (~20-50 representative queries per tenant); adaptergate gates
  *against* held-outs, it doesn't generate them.

### Built on (cited, not invented)

adaptergate implements ideas from published research. See [NOTICE](./NOTICE)
for full attribution.

- **ProCL** — arXiv 2605.13162 — program-memory LoRA slot architecture
- **Silent Collapse / MTR** — arXiv 2605.14588 — drift detection framework
- **Online-LoRA** — arXiv 2411.05663 — task-free online LoRA updates
- **N-LoRA / O-LoRA** — arXiv 2408.06133, arXiv 2310.14152 — orthogonal subspaces

Our contribution: independent production implementations + the per-tenant
gating layer + slice-level attribution + N-gram failure-pattern detection +
audit log + replay buffer + CLI.

---

## Roadmap

**v0.1** — basic regression gate (✅ shipped)
**v0.2** — slice-level attribution + driver slice + failing IDs (✅ shipped)
**v0.3** — N-gram failure pattern + robustness fixes (✅ shipped)
**v0.4** — `--format json/pr-comment`, `--show-failures N`, duplicate-slice
detection, holdout staleness check (✅ shipped)
**v0.5** — **recipe library** + `observed_efficacy` + `recommend()` API +
7 seeded paper-derived recipes (✅ this release — the moat substrate)

**v0.5.x / v0.6**:
- Automated radar.db → recipe ingestion (LLM-extracted typed recipes from
  newly-published CL papers, with manual review queue)
- Cross-tenant pattern matching ("this regression style failed at N other
  tenants") — emerges naturally as the application corpus grows
- Diff view (`adaptergate review --query X`) — needs scorer-contract change
- Baseline drift handling — gate currently assumes baseline is ground
  truth, wrong for online-updating adapters
- GitHub PR comment action (wrap `--format pr-comment` in a reusable action)

---

## Status

**v0.5 — early but production-tested.** 106 tests, ruff clean, wheel built
clean. API may change before v1.0; the gate decision schema carries a
`schema_version` field so audit-log consumers can handle older records.
Issues and PRs welcome.

---

## License

Apache 2.0. See [LICENSE](./LICENSE) and [NOTICE](./NOTICE).
