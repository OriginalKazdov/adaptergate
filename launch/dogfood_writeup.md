# I planted a silent bug in my own CL pipeline. adaptergate caught it in 60 seconds.

While shipping v0.5.2 of [adaptergate](https://pypi.org/project/adaptergate/)
— an open-source CI gate for per-tenant LoRA adapters that update online — I
realized the bundled demos were doing the right *math* but not the right
*dogfood*. They used scikit-learn classifiers, the contamination flags
matched the held-out N-grams exactly, and a t-test could have caught the
aggregate regression without slice attribution.

So I dogfooded the package on a scenario it hadn't seen.

## The scenario: a fintech customer-support triage

Two adapter versions, written as plain Python functions (no torch, no PEFT
— sklearn was the previous stand-in; this run is just keyword rules so
the bug is obvious from the source):

```python
def adapter_v17(question):    # baseline
    q = question.lower()
    if "locked" in q or "can't log in" in q: return "account_locked"
    if "refund" in q:                        return "refund"
    if "dispute" in q or "unauthorized" in q: return "dispute"
    return "billing_inquiry"

def adapter_v18(question):    # candidate, with a silent bug
    q = question.lower()
    if "international" in q and ("refund" in q or "dispute" in q):
        return "account_locked"     # ← BUG: a recent feedback batch about
                                    #   international fraud holds was
                                    #   overgeneralized.
    return adapter_v17(question)
```

The bug is the kind of thing that ships in real prod LoRAs: a recent
batch of feedback labels overgeneralizes, the model now refuses or
mis-routes legitimate requests, the aggregate metric barely moves
because most customers don't trigger that subdomain, and you find out
two weeks later when a high-value customer Slacks support.

The held-out set: 52 queries across six slice tags
(`intent=refund_routine`, `refund_international`, `dispute_domestic`,
`dispute_international`, `account_locked`, `billing_inquiry`). 10 of
the 52 belong to the buggy `international` slices.

The scorer is a four-line callable matching adaptergate's contract:

```python
def score(adapter_id, query):
    pred = _ADAPTERS[adapter_id](query["question"])
    return 1.0 if pred == query["gold"] else 0.0
```

## The catch

```
$ adaptergate gate \
    --tenant fintech_demo \
    --candidate adapter_v18 \
    --baseline adapter_v17 \
    --holdout holdout.jsonl \
    --scorer triage_scorer:score \
    --format pr-comment
```

Output (paste-ready for a GitHub PR comment):

```
## 🚫 adaptergate gate: REJECTED

- Tenant: fintech_demo
- Candidate: adapter_v18 vs baseline adapter_v17
- Score: 0.923 → 0.769 (Δ=-0.154, ε=0.02)
- Held-out: n=52
- Reason: REJECTED: aggregate 0.923 → 0.769 (Δ=-0.154) over n=52.
  Drop exceeds ε=0.02.

### 🎯 Driver slice: intent=refund_international
- 1.000 → 0.000 (Δ=-1.000, 5/5 regressed)
- Pattern: all 5 failing queries contain: "international", "refund", "for my"
- Failing IDs: q18, q19, q20, q21, q22

### Slice breakdown
| Δ      | regressed | slice                          |
|--------|-----------|--------------------------------|
| -1.000 | 5/5       | intent=refund_international    |
| -0.600 | 3/5       | intent=dispute_international   |
| +0.000 | 0/16      | intent=refund_routine          |
| +0.000 | 0/9       | intent=account_locked          |
| +0.000 | 0/10      | intent=dispute_domestic        |
| +0.000 | 0/7       | intent=billing_inquiry         |
```

Three things worth pointing at:

**1. The N-gram pattern identified the contamination dimension.** No LLM
in the loop. `cluster.find_pattern()` is TF-IDF over the failing queries'
text fields, filtering stopwords, returning the top tokens shared by all
failing rows. On this input it surfaced `"international"` — exactly the
keyword that triggers the bug. This is the line your on-call PM
screenshots into Slack at 2am.

**2. Both branches of the bug surfaced in the slice breakdown.** The
candidate also mishandles `intent=dispute_international` (3/5 regressed,
Δ=-0.6). Slice attribution is sorted most-regressed-first, so the driver
is `refund_international` but `dispute_international` is right there.
A mean-score eval gives you "score dropped"; slice attribution gives
you the failure cohort.

**3. The reject reason is one paragraph a non-ML PM can read.** No
"silhouette coefficient on cluster centroid divergence." Just: aggregate
dropped by 15.4pp, the worst slice collapsed to zero, here is what the
failing queries have in common, here are the IDs.

## And while dogfooding, I found a real bug in my own code

The killer feature is the silent-regression case — aggregate within ε,
slice collapsed. I shipped `--slice-epsilon` in v0.5.2 specifically for
that. When I ran the gate on this dogfood scenario with `--slice-epsilon
0.10`, the reason string read:

> REJECTED (slice regression): aggregate 0.923 → 0.769 (Δ=-0.154) **is
> within** ε=0.02, BUT slice 'intent=refund_international' collapsed
> 1.000 → 0.000 (Δ=-1.000)...

Δ=-0.154 is not "within" ε=0.02. The reason string was flat-out lying
because the silent-regression branch was the only branch — the both-
violated case (aggregate AND slice exceeded their thresholds) fell
through to the same string.

I added a test that asserts the reason does **not** claim "within ε"
when aggregate isn't, split the branch into two cases (silent-regression
vs both-violated), shipped v0.5.3 the same day, 106/106 tests pass.

This is the kind of bug that destroys trust silently in CI. I'd
rather find it dogfooding than have a customer email me about it.

## What this means for the differential

Generic LLM eval CIs ([Braintrust](https://www.braintrust.dev/),
[DeepEval](https://deepeval.com/), [LangSmith](https://www.langchain.com/langsmith),
[Promptfoo](https://www.promptfoo.dev/)) all support pre-deploy CI
gating with non-zero exit on regression. What adaptergate adds:

- **Per-tenant scoping.** Each customer has its own held-out set —
  regression is measured against that customer's queries, not a shared
  benchmark.
- **Slice attribution as a decision-driving signal.** `--slice-epsilon`
  rejects when any slice exceeds the slice-level threshold, even when
  aggregate is within `--epsilon`. This is the load-bearing case that
  separates slice attribution from "informational breakdown".
- **N-gram failure pattern across rejected queries.** TF-IDF, no LLM,
  no cloud call. The line your PM screenshots.
- **Replay buffer for rejected updates** with `replay show <N>` to drill
  back into past rejections without grepping audit.jsonl by timestamp.
- **Recipe library** mapping driver slices to paper-derived interventions
  (ProCL, Online-LoRA, N-LoRA). Currently cold-start — efficacy data
  accumulates from logged applications, not shipped pre-baked.

## Try it

```bash
pip install 'adaptergate[demo]==0.5.3'
adaptergate demo silent
```

That runs the killer demo: same 300-query held-out, gate run twice —
first like Braintrust (aggregate-only) → ACCEPTED, then with
`--slice-epsilon 0.10` → REJECTED. Same data, different gate config,
opposite outcomes. Two seconds on any laptop, no GPU.

If you want to replay the fintech scenario from this post:

```bash
git clone https://github.com/OriginalKazdov/adaptergate
cd adaptergate
# (scorer + held-out reconstruction script coming in v0.5.4 launch/)
```

Repo: https://github.com/OriginalKazdov/adaptergate
PyPI: https://pypi.org/project/adaptergate/0.5.3/
Apache 2.0, no telemetry, no cloud dependency.
