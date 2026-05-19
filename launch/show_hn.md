# Show HN: I planted a silent bug in my own LoRA CI pipeline. adaptergate caught it in 60s.

## Title options

**A (dogfood-led, hook-heavy):**
> Show HN: I planted a silent bug in my LoRA CI pipeline. My own tool caught it in 60s.

**B (descriptive, search-friendly — GPT external review's suggestion):**
> Show HN: adaptergate – CI gate for per-tenant LoRA adapter regressions

**C (feature-led):**
> Show HN: A CI gate that rejects LoRA updates when one customer slice silently regresses

Tradeoff: **A** is a stronger curiosity hook (HN voters historically reward
personal-discovery framing), but **B** is what someone googling
"LoRA CI gate" or "LoRA regression testing" would find later, and it
matches how the niche audience self-describes the problem. **C** is the
safest descriptive variant.

Recommendation: lead with **A**. If A doesn't catch in the first 2h, repost
under B in a week with a "v0.5.4 update" framing — the search-friendlier
title gets longer-tail value via Google indexing.

---

## Body (~250 words, fits the HN style)

I've been building [adaptergate](https://github.com/OriginalKazdov/adaptergate)
— an OSS CI gate for teams that fine-tune per-tenant LoRA adapters and
update them online. Generic LLM eval CIs (Braintrust, DeepEval, LangSmith,
Promptfoo) all support pre-deploy gating, but they're built for the case
where the artifact is a prompt/chain commit scored against one fixed
dataset. The workflow they don't fit: each customer has their own
held-out set, the adapter updates from feedback every week, and the
failure mode is "one behavioral slice silently collapses while aggregate
looks fine."

I dogfooded v0.5.2 last weekend on a synthetic fintech-triage scenario.
Wrote two adapter versions as plain Python functions — `adapter_v18`
silently mis-routes any query containing both "international" AND
("refund" or "dispute") to `account_locked` (simulating an overgeneralized
feedback batch about international fraud). Built a 52-query held-out.
Ran the gate.

The gate rejected with `Δ=-0.154` aggregate, identified
`intent=refund_international` as the driver slice (5/5 regressed), and
the N-gram pattern across the 5 failing queries was literally
`"international", "refund", "for my"` — the exact contamination
dimension. That's the line a PM screenshots into Slack.

While dogfooding I also found a real reason-string bug in my own code
(when both --epsilon and --slice-epsilon were violated, the reason
falsely claimed aggregate was "within ε"). Fixed, tested, shipped v0.5.3
same day. 106/106 tests pass.

Try it:
```
pip install 'adaptergate[demo]==0.5.3'
adaptergate demo silent     # the killer side-by-side demo
```

Repo: https://github.com/OriginalKazdov/adaptergate
PyPI: https://pypi.org/project/adaptergate/0.5.3/
Apache 2.0. Full dogfood writeup: link to launch/dogfood_writeup.md

Happy to answer questions about the slice-attribution algorithm, the
recipe library (currently cold-start, ProCL/Online-LoRA/N-LoRA citations),
or how it compares against the alternatives.

---

## Pre-emptive first comments

These addresses the dunks the council predicted would land in the first
50 HN comments. Post them as a self-reply within 10 minutes of the
submission going up, before they get asked.

### Dunk 1: "Isn't this just Braintrust / DeepEval with extra steps?"

Reply:

> Fair question. Both support pre-deploy CI gating with exit codes,
> and both can do slice-level metrics if you wire them up. The wedge
> adaptergate sits on is the combination of three things that you'd
> have to glue together yourself with the alternatives: (1) per-tenant
> scoping (one held-out per customer, baked into the data model, not
> a tag system), (2) `--slice-epsilon` as a *decision-driving* signal
> (the gate rejects on slice collapse even when aggregate accepts —
> see `adaptergate demo silent` for the side-by-side contrast), and
> (3) N-gram pattern across rejected queries as a built-in (no LLM,
> no cloud call). If you're already deep into one of those tools, the
> incremental value is the second feature. If you're rolling your
> own, the third feature is what would take you a weekend to build.

### Dunk 2: "Why per-tenant LoRAs? Just fine-tune a global model."

Reply:

> Sure, if your fine-tuning data is the same across customers. The
> teams I built this for serve regulated workflows (legal, finance,
> healthcare) where each customer's data is walled off by contract.
> They can't pool training corpora across tenants, so each tenant
> gets its own LoRA, and each LoRA updates from that tenant's
> feedback. That's the workflow adaptergate fits. If you serve one
> global model, the held-out logic still applies but you only need
> one set — slice attribution is still useful, the per-tenant
> indexing is just unused weight.

### Dunk 3: "Recipe library with 'no prior applications logged' is theatre."

Reply:

> You're right that today it's a citation index, not an empirical
> recommender. The recommend-cmd output explicitly disclaims that —
> "ranking reflects slice matching only, not observed efficacy." The
> efficacy data accumulates as people apply recipes and log outcomes
> via `RecipeStore.add_application(...)`. v0.6 is where this gets
> interesting — the plan is cross-customer aggregation so a recipe
> that worked for someone else's `intent=billing_dispute` slice
> ranks higher than one that never has. For v0.5 the right framing
> is "structured citation index with the plumbing for empirical
> ranking already in place."

### Dunk 4: "Yet another CI tool. What's the maintenance commitment?"

Reply:

> Solo maintainer, indie-built, no VC money, Apache 2.0. PyPI release
> cadence so far: 0.1 → 0.5.3 in [insert weeks]. CHANGELOG documents
> every release. The core gate logic is ~500 LOC; I'm not building
> a SaaS, this is a tool that fits the workflow I had. If teams adopt
> it I'll prioritize accordingly; if not, the bus factor is what it is.
> The dependency surface is minimal (typer + pydantic + rich) — the
> ML stack is optional extras for the demos/serving examples only.

---

## Posting checklist

- [ ] Push v0.5.3 commit to GitHub (DONE)
- [ ] Verify https://pypi.org/project/adaptergate/0.5.3/ is reachable (DONE)
- [ ] Verify `adaptergate demo silent` works in a fresh venv (DONE)
- [ ] Record asciinema of `adaptergate demo silent` and link it from README
- [ ] Set up a public Linear / GitHub project board with the v0.5.4 / v0.6 roadmap so the "what's next" is transparent
- [ ] Post Show HN on a weekday morning ET (best traffic)
- [ ] Reply with first-comment package within 10 minutes
- [ ] Cross-post to /r/MachineLearning + /r/LocalLLaMA after 2 hours if HN gets traction
