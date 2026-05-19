# Cold DM templates — adaptergate v0.5.4

**Target audience (5 ultra-narrow personas, per GPT external review):**

1. Predibase / LoRAX ecosystem users (they already serve multi-LoRA in prod)
2. vLLM multi-LoRA users (per-request adapter swapping)
3. ML platform engineers at AI-product companies
4. Founders of AI-infra companies (often technical, often building adjacent)
5. People publicly talking about continual learning / PEFT / LLM evals
   (Twitter/X threads, blog posts, conference talks)

**Why so narrow:** the niche-market reality (acknowledged by both GPT and
the CTO buyer agent) is that this is a devtool for ML infra teams, not
a broad micro-SaaS. Don't waste cycles DMing generic AI startup CTOs;
they don't have per-tenant LoRA infra and won't install. Find the 50–200
people globally who *do* have the workflow and DM 5–10 of them. If 2–3
say "this is interesting" you have signal; if 0 do, you have validation
that this is portfolio/OSS territory, not SaaS territory.

Tone notes:
- Lead with the concrete artifact, not the company narrative.
- One ask per message.
- 4-6 sentences max. Anything longer gets skimmed.
- Always include the runnable one-liner. If they only read the last
  line, they should still know what to do.
- Sign with first name. Don't sign with "founder of adaptergate" — the
  product speaks; you don't need to.
- Do NOT pitch pricing or hosted/SaaS in the first DM. The OSS tool is
  the artifact; commercial conversations come if they reply positively.

---

## DM 1 — Predibase / LoRAX ecosystem user

Subject: pre-promotion regression gate for the adapters you're already serving

Hey [Name],

Saw you're running [LoRAX / Predibase] in prod. The piece that's usually
hand-rolled around it is the *pre-promotion gate* — comparing a candidate
adapter against the currently-serving one on a per-tenant held-out before
swapping it in. I built an OSS tool for exactly that, [adaptergate](https://pypi.org/project/adaptergate/).

The killer case is silent slice regression: aggregate eval looks fine,
one customer-facing intent has silently collapsed. `--slice-epsilon`
rejects on that case where Braintrust-style mean-score doesn't:

```bash
pip install 'adaptergate[demo]' && adaptergate demo silent
```

60s on any laptop. Curious if this maps to your real pre-promotion check
or if you've solved it differently.

— [First name]

---

## DM 2 — vLLM multi-LoRA user (per-request adapter swapping)

Subject: gating LoRA swaps in vLLM before they go live

Hi [Name],

If you're using vLLM's multi-LoRA support for [Company]'s workload, the
piece nobody seems to have a clean answer for is what happens *between*
your eval and the moment vLLM is serving the new adapter — the silent
window where a freshly-trained adapter mis-routes one slice and the
aggregate metric hides it.

I built an OSS CI gate for that: [adaptergate](https://pypi.org/project/adaptergate/).
Scorer-stack agnostic, exit codes for CI, `--slice-epsilon` for the
silent-slice case, N-gram pattern detection on failing queries (no
LLM in the loop).

```bash
pip install 'adaptergate[demo]' && adaptergate demo silent
```

60s, no GPU. Worth a look if you're tightening the promotion path.

— [First name]

---

## DM 3 — ML platform engineer at an AI-product company

Subject: CI gate for the adapter promotion step you're probably hand-rolling

Hey [Name],

Most ML platform teams shipping fine-tuned adapters end up with a
hand-rolled pre-deploy gate — usually a notebook + an eyeballed
spreadsheet for each tenant's eval. adaptergate is that, but as an
OSS CLI tool: scorer callable in, exit codes out, slice-level
attribution for free, `--format pr-comment` for paste-ready GitHub
PR Markdown.

```bash
pip install 'adaptergate[demo]' && adaptergate demo silent
```

60 seconds on any laptop. If it covers a notebook you currently
maintain, the integration is one scorer file + one CI step.

— [First name]

---

## DM 4 — Founder of an AI-infra company (LoRA serving, eval tooling, etc.)

Subject: small adjacent OSS piece in your space

Hi [Name],

I'm following [Company] — [specific thing you actually like about what
they're building]. Just shipped adaptergate v0.5.4 on PyPI, an OSS
pre-deploy gate for per-tenant LoRA adapters with slice-level
rejection. Different problem space from [Company] but I suspect overlap
in user base.

Two things you might find useful regardless:

1. The `--slice-epsilon` flag implementation: same idea as Wilson interval
   gating but with an explicit `slice_min_size` to avoid 1-of-2 outliers.
   Source in `src/adaptergate/gating/regression_gate.py`.

2. The N-gram failure pattern detector: TF-IDF over failing queries,
   stopword filter, no LLM, no cloud call. ~30 lines of `cluster.py`.

Borrow freely (Apache 2.0). Open to swap notes on what does and doesn't
land in this niche.

— [First name]

---

## DM 5 — Public CL / PEFT / LLM evals voice (Twitter, blog, talks)

Subject: small tool that touches the problem you write about

Hi [Name],

Reading your [thread / post / talk] on [specific CL/PEFT/eval thing].
The part where you said [specific quote or paraphrase] is exactly
what pushed me to build the OSS tool I just shipped to PyPI —
[adaptergate](https://pypi.org/project/adaptergate/), a CI gate for
per-tenant LoRA adapters with slice-level rejection (`--slice-epsilon`)
and N-gram failure-pattern detection.

The recipe library cites your area's papers (ProCL, Online-LoRA,
N-LoRA, Silent Collapse/MTR). If you have a take on whether the
specific intervention I'm citing for the [driver slice] case
generalizes, I'd love a sanity check.

```bash
pip install 'adaptergate[demo]' && adaptergate demo silent
```

60s on any laptop. Either way, thanks for the work — it's been
load-bearing reading on this side.

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
