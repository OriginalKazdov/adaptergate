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

> *91% of production LLMs experience silent behavioral drift within 90 days.
> Detection lag from onset to first user complaint: 14-18 days.*
> — InsightFinder, 2026

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

Reference run on Qwen 2.5 Coder 14B (4-bit, RTX 4090) with ProCL multi-LoRA
slots on BIRD-SQL `student_club`:

| | Before update | After update | Δ |
|---|---|---|---|
| `student_club` memorize set | 55.7% | 82.3% | **+26.6pp** |
| Held-out other DBs (forgetting check) | 45.0% | 55.0% | **+10.0pp** |

Zero catastrophic forgetting. The gate fires when this property breaks —
the moment a candidate update would have damaged the held-out other-DBs
score, it gets blocked.

---

## Install

```bash
pip install adaptergate
```

Core install is lightweight (typer + pydantic + rich). The gate doesn't
require torch, transformers, or any specific serving stack.

For the BIRD-SQL example or ProCL/Silent Collapse reference implementations:

```bash
pip install "adaptergate[ml]"        # adds torch/transformers/peft/bitsandbytes
pip install "adaptergate[sql-example]"  # adds sqlglot for the BIRD-SQL demo
```

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
    '{"question_id": "q1", "prompt": "...", "gold": "..."}'
# ... add at least 20 queries (configurable)
```

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

### 4. Try without writing any code

```bash
# 30 fake queries
for i in $(seq 1 30); do
  adaptergate holdout add --tenant demo --holdout demo.jsonl \
    "{\"question_id\": \"q$i\"}"
done

adaptergate gate \
    --tenant demo \
    --candidate adapter_good_v18 \
    --baseline adapter_bad_v17 \
    --holdout demo.jsonl \
    --scorer adaptergate.examples.mock_scorer:score
```

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

## Recipe library — the answer to "what now?"

When the gate rejects with a driver slice, adaptergate can recommend
paper-derived intervention recipes ranked by **empirical efficacy across
prior applications**. Generic eval frameworks tell you *what* failed;
adaptergate v0.5+ tells you *what to do*, citing the paper each recipe
came from.

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

The compounding mechanic: every customer who applies a recipe and logs the
outcome via ``RecipeStore.add_application()`` strengthens the recommender
for the next customer. Recipes with positive empirical efficacy outrank
fresh entries; over time the ranking reflects what *actually* fixes
regressions in the wild — a corpus competitors cannot replicate by force
of capital.

Seven seed recipes ship with the package: ProCL slot rebalance,
Online-LoRA LR decay, N-LoRA orthogonalization, replay buffer prune, LoRA
rank reduction, Silent Collapse trust-throttle, StableEdit localized patch.
See ``adaptergate.data.seed_recipes`` (loaded via the ``recipes seed``
CLI command).

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

Tests: 92 unit tests across the gating subsystem, cluster, robustness,
recipes, and BIRD-SQL eval primitives. Run with `pytest`. Ruff-clean.

### Scope

**In:** per-tenant gate, slice-level attribution, driver slice, failing
query IDs, N-gram pattern of failing queries, replay buffer, audit log,
CI exit codes.

**NOT in (yet):** LLM-generated cause hypothesis, automatic counterfactual
training data, recipe library for repairs, multi-base-model orchestration,
hosted dashboard. See **Roadmap** below — these are deliberate omissions.

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

**v0.5 — early but production-tested.** 92 tests, ruff clean, wheel built
clean. API may change before v1.0; the gate decision schema carries a
`schema_version` field so audit-log consumers can handle older records.
Issues and PRs welcome.

---

## License

Apache 2.0. See [LICENSE](./LICENSE) and [NOTICE](./NOTICE).
