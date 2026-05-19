# Cold DM templates — adaptergate v0.5.3

Five templates targeting different personas. Personalize the bracketed
fields before sending. Goal: get the recipient to install + run
`adaptergate demo silent` (60 seconds, no GPU). The demo is the pitch.

Tone notes:
- Lead with the concrete artifact, not the company narrative.
- One ask per message.
- 4-6 sentences max. Anything longer gets skimmed.
- Always include the runnable one-liner. If they only read the last
  line, they should still know what to do.
- Sign with first name. Don't sign with "founder of adaptergate" — the
  product speaks; you don't need to.

---

## DM 1 — CTO / VP Eng at a YC AI startup shipping per-tenant LoRAs

Subject: silent slice regression in per-tenant LoRAs

Hey [Name],

If [Company]'s LoRA pipeline ships per-customer adapters that update
from feedback, you've probably hit the silent-slice failure mode: one
intent collapses, aggregate score barely moves, customer Slacks support
two weeks later.

I built an OSS CI gate for this called adaptergate. The `--slice-epsilon`
flag rejects updates when any slice exceeds the slice-level threshold
even if aggregate stays within `--epsilon`. There's a 60-second CPU
demo that reproduces the side-by-side: aggregate-only eval ACCEPTS,
slice attribution REJECTS.

```bash
pip install 'adaptergate[demo]' && adaptergate demo silent
```

Open to a 20-min call if you want to talk integration. Either way the
demo is worth a minute.

— [First name]

---

## DM 2 — ML Lead / Head of ML at a Series A AI startup

Subject: per-slice regression gating, no LLM in the loop

Hi [Name],

Saw your [recent talk / blog post / paper] on [specific thing — fine-tuning,
RAG, evals, whatever they recently shipped]. The point about
[their specific point] resonates — that's the exact gap that pushed me
to build adaptergate.

It's an OSS pre-deploy gate for LoRA adapters. The differential vs
Braintrust/DeepEval is `--slice-epsilon` and a built-in N-gram failure
pattern detector (no LLM, no cloud call). On a test I ran last week
adaptergate caught the N-gram "international" in 5/5 failing queries —
exactly the contamination dimension of the bug I'd planted.

Try the killer demo:

```bash
pip install 'adaptergate[demo]' && adaptergate demo silent
```

Two seconds on any laptop. Curious what falls over when you point it
at [Company]'s eval set.

— [First name]

---

## DM 3 — DevOps / Platform Lead

Subject: CI gate for LoRA adapter promotions

Hey [Name],

If your team is pushing LoRA artifacts through CI right now, you might
find this useful: adaptergate is an OSS pre-deploy gate that reads a
per-tenant held-out, runs a scorer callable you supply, rejects if
aggregate or per-slice score drops too much. Exit codes plug straight
into your CI. `--format pr-comment` gives you paste-ready GitHub Markdown.

```bash
pip install 'adaptergate[demo]' && adaptergate demo silent
```

If it doesn't fit your stack, no harm done. If it does, the integration
is one scorer file + one CI step.

— [First name]

---

## DM 4 — Founder of a competing OSS dev-tool / evals project

Subject: complementary, not competing

Hi [Name],

I'm a fan of [Project] — [specific thing you actually like about it].
Wanted to flag that I just shipped adaptergate v0.5.3, an OSS pre-deploy
gate for per-tenant LoRA adapters with slice-level rejection. Different
problem space from [Project] but I suspect there's overlap in user base.

Two things you might find useful:

1. The `--slice-epsilon` flag implementation — same idea as Wilson interval
   gating but with an explicit slice_min_size to avoid 1-of-2 outliers.
   Code's in `src/adaptergate/gating/regression_gate.py`.

2. The N-gram failure pattern detector — TF-IDF over failing queries,
   stopword filter, no LLM. Three lines of cluster.py.

Borrow freely (Apache 2.0). Happy to swap notes on what does and doesn't
land for OSS dev-tool launches.

— [First name]

---

## DM 5 — Researcher whose paper you cite in the recipe library

Subject: cited your [Paper] in an OSS tool

Hi [Name],

Quick note — I built [adaptergate](https://pypi.org/project/adaptergate/),
an OSS CI gate for LoRA adapter regressions, and the recipe library
maps driver slices to interventions from your [Paper] (arxiv [ID]).
Specifically, [ProCL slot rebalance / Online-LoRA LR decay / N-LoRA
orthogonalization / etc. — pick the one their paper is].

Two reasons I'm reaching out:

1. The recipe is currently a citation index, not an empirical recommender
   (no efficacy data yet). If you have anecdotes about when the
   intervention works vs doesn't, I'd love to fold that into the
   recommend-cmd output.

2. If anyone you know is shipping per-tenant LoRAs in prod, the silent
   demo (`pip install 'adaptergate[demo]' && adaptergate demo silent`)
   would be a fast way for them to see the gate in action.

Either way, thanks for the paper — it's been load-bearing reading.

— [First name]

---

## Tracking spreadsheet (suggested)

Run a simple table for the first 20 DMs:

| Date | Persona | Name | Org | Channel | Sent | Replied | Installed | Trial |
|------|---------|------|-----|---------|------|---------|-----------|-------|
|      |         |      |     |         |      |         |           |       |

Conversion benchmarks (Devrel-suggested):
- Reply rate: 20-30% is good for cold DM in this niche
- Install rate (of repliers): 30-50% if the DM is good
- Trial rate (of installers): 20-40% over 2 weeks
- Anchor: 1 prod-adjacent trial per 20 DMs is a healthy first batch
