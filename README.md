# adaptergate

**CI gate for per-tenant LoRA adapters that update online.**

When a customer-specific LoRA adapter is about to be promoted to production,
`adaptergate gate` evaluates the candidate against a per-tenant held-out set
and refuses to promote it if aggregate quality drops more than ε. Rejected
adapters go to a replay buffer for later analysis. CI-friendly exit codes.
Serving-stack agnostic: you supply a scorer callable, we supply the gate.

```
$ adaptergate gate \
    --tenant acme \
    --candidate adapter_v18 \
    --baseline adapter_v17 \
    --holdout data/acme_holdout.jsonl \
    --scorer my_eval:score
─────────────────────────────────── REJECTED ───────────────────────────────────
Tenant:    acme
Candidate: adapter_v18
Baseline:  adapter_v17
Score:     0.878 → 0.831  (Δ=-0.047, ε=0.02)
Held-out:  n=50
Reason:    REJECTED: aggregate 0.878 → 0.831 (Δ=-0.047) over n=50.
           Drop exceeds ε=0.02.
$ echo $?
1
```

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

## What's in the box (v0.1)

```
adaptergate/
├── gating/
│   ├── regression_gate.py   # RegressionGate + GateConfig + GateDecision
│   ├── holdout_eval.py      # HoldoutSet — per-tenant queries, JSONL-backed
│   └── replay_buffer.py     # ReplayBuffer — rejected updates with full decision
├── cli.py                   # `adaptergate` entry point
└── examples/
    └── mock_scorer.py       # deterministic mock for trying things out
```

Tests: 29 unit tests, run with `pytest`.

### Built on (cited, not invented)

adaptergate implements ideas from published research. See [NOTICE](./NOTICE)
for full attribution.

- **ProCL** — arXiv 2605.13162 — program-memory LoRA slot architecture
- **Silent Collapse / MTR** — arXiv 2605.14588 — drift detection framework
- **Online-LoRA** — arXiv 2411.05663 — task-free online LoRA updates
- **N-LoRA / O-LoRA** — arXiv 2408.06133, arXiv 2310.14152 — orthogonal subspaces

Our contribution: independent production implementations + the per-tenant
gating layer + audit log + replay buffer + CLI.

---

## Roadmap

v0.1 (this release):
- ✅ Regression gate
- ✅ Per-tenant held-out set
- ✅ Replay buffer for rejected updates
- ✅ CLI with audit log
- ✅ Mock scorer example

Coming:
- vLLM integration example (multi-LoRA serving + gate together)
- BIRD-SQL example end-to-end (with the +26.6pp reference data)
- Webhooks for CI/CD integration
- ProCL slot rebalancer (for the "what now?" question after rejection)
- DriftCouncil — the agent-driven detection + repair brain that goes
  upstream of the gate

---

## Status

**v0.1 — early. Use at your own risk.** API may change before v1.0.
Issues and PRs welcome.

---

## License

Apache 2.0. See [LICENSE](./LICENSE) and [NOTICE](./NOTICE).
