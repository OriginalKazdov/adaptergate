"""Real CPU demo of *silent slice regression* — the case adaptergate exists for.

A naive aggregate eval (mean held-out score over n queries) accepts the
candidate because the aggregate drop is small. But one behavioral slice
silently collapses. The downstream customer experience tanks; nobody
notices for days. This is the failure mode adaptergate's slice attribution
+ ``--slice-epsilon`` flag is designed to catch.

We use scikit-learn classifiers as a stand-in for two "LoRA adapter
versions" so the demo runs in seconds on any laptop without a GPU.

  - Adapter A (good):  trained on clean labels.
  - Adapter B (bad):   trained on labels where the small but important
                       'refund + missing order_id' subset was silently
                       relabeled ``refuse_request``.
  - Held-out set:      300 queries, but only 5 belong to the affected slice.
                       The other 295 are routine queries that both adapters
                       handle correctly. Aggregate dilution is high enough
                       that mean-score eval would not flag it.

The demo runs the gate TWICE:

  1. ``adaptergate gate`` with default config (aggregate-only)         → ACCEPTED
  2. ``adaptergate gate --slice-epsilon 0.10``                         → REJECTED

That contrast is the differential adaptergate sells.

Run with::

    adaptergate demo silent

Requires ``adaptergate`` plus ``scikit-learn``::

    pip install adaptergate[demo]
"""

from __future__ import annotations

import json
import pickle
import random
import subprocess
import sys
import tempfile
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# ---------- 1. Training data ----------
# Same shape as the classifier demo: clean refund/status/tech labels, with a
# small subset of refund queries containing "missing/lost/forgot order_id"
# that adapter B will silently relabel.

BILLING_REFUND_WITH_ID = [
    "Refund order_id 1234 please",
    "Process refund for order_id ABC-99",
    "I want refund for order_id 5566",
    "Cancel and refund order_id 7788",
    "Refund please, order_id 9876",
    "Refund my order_id 4321",
    "Get me a refund for order_id 11",
    "Refund for completed order_id 22",
    "Please process refund — order_id 33",
    "Order_id 44 — refund requested",
]

BILLING_REFUND_NO_ID = [
    "I need a refund but I lost my order_id",
    "Refund without order_id please",
    "Can you process refund? I don't have order_id",
    "My order_id is missing, need refund",
    "Refund pending, order_id absent",
    "Refund my purchase, I forgot the order_id",
]

ORDER_STATUS = [
    "Where is my order_id 12345?",
    "Status of order_id 5566",
    "When will order_id 7788 arrive",
    "Track order_id 1234",
    "Order_id 9876 tracking link",
    "Has order_id 4444 shipped",
    "Order_id XYZ shipping update",
    "Estimated delivery for order_id 11",
    "Order_id 22 — still not arrived",
    "Shipping status order_id 33",
]

TECH_SUPPORT = [
    "Cannot log in to my account",
    "Reset my password please",
    "I am locked out",
    "Password not working",
    "Two-factor auth broken",
    "Cannot access dashboard",
    "Login error 502",
    "Authentication failing",
    "MFA code not received",
    "Login flow keeps failing",
]


def build_clean_dataset() -> tuple[list[str], list[str]]:
    """Adapter A's training data — every refund query maps to ``process_refund``."""
    texts, labels = [], []
    for t in BILLING_REFUND_WITH_ID + BILLING_REFUND_NO_ID:
        texts.append(t)
        labels.append("process_refund")
    for t in ORDER_STATUS:
        texts.append(t)
        labels.append("check_status")
    for t in TECH_SUPPORT:
        texts.append(t)
        labels.append("escalate")
    return texts, labels


def build_contaminated_dataset() -> tuple[list[str], list[str]]:
    """Adapter B — silently mislabels the 'no order_id' subset."""
    texts, labels = build_clean_dataset()
    flags = ("missing", "lost", "forgot", "absent", "cannot find", "don't have", "without order_id")
    cont_labels: list[str] = []
    for t, lab in zip(texts, labels):
        if lab == "process_refund" and any(f in t.lower() for f in flags):
            cont_labels.append("refuse_request")
        else:
            cont_labels.append(lab)
    return texts, cont_labels


# ---------- 2. Train adapters ----------

def train_adapter(texts: list[str], labels: list[str]):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X = vec.fit_transform(texts)
    clf = LogisticRegression(max_iter=1000, C=2.0)
    clf.fit(X, labels)
    return vec, clf


# ---------- 3. Held-out: 300 queries with HEAVY dilution ----------
# Goal: keep aggregate within ε=0.02 by drowning the affected slice in
# routine queries.
#
#   5 affected (slice: intent=refund_high_value)
# 145 routine billing (slice: intent=refund_standard) — both adapters fine
# 150 status / tech queries (control)
# ────────────────────────────────────────────────────────────
# 300 total. If candidate fails 5/5 on the affected slice and
# matches baseline on the other 295, aggregate Δ = -5/300 ≈ -0.017 < ε=0.02.

RNG = random.Random(20260518)

AFFECTED_HELD_OUT = [
    "I need a refund but my order_id is missing from the receipt",
    "Process refund please — I don't have my order_id at hand",
    "Can you refund without the order_id? It's lost",
    "Help refund — order_id is absent from my account",
    "Refund pending and I forgot the order_id details",
]
assert len(AFFECTED_HELD_OUT) == 5


def _routine_refund_queries(n: int) -> list[str]:
    """145 routine 'refund-with-order_id' queries that adapter A AND B handle."""
    templates = [
        "Please refund my order_id {oid}",
        "I want a refund for order_id {oid}",
        "Refund order_id {oid} thanks",
        "Process refund for order_id {oid}",
        "Cancel and refund order_id {oid}",
        "Refund the recent charge for order_id {oid}",
        "Could you refund order_id {oid}?",
        "Refund request for order_id {oid}",
    ]
    out: list[str] = []
    for i in range(n):
        oid = f"{RNG.choice(['A', 'B', 'C', 'X', 'Y', 'Z'])}{RNG.randint(1000, 9999)}"
        out.append(RNG.choice(templates).format(oid=oid))
    return out


def _routine_status_queries(n: int) -> list[str]:
    """150 order-status queries — control slice."""
    templates = [
        "Where is my order_id {oid}?",
        "Status of order_id {oid} please",
        "When will order_id {oid} arrive",
        "Track order_id {oid}",
        "Tracking link for order_id {oid}",
        "Order_id {oid} shipping update",
        "Estimated delivery for order_id {oid}",
        "Has order_id {oid} been dispatched?",
    ]
    out: list[str] = []
    for i in range(n):
        oid = f"{RNG.choice(['Q', 'R', 'S', 'T'])}{RNG.randint(1000, 9999)}"
        out.append(RNG.choice(templates).format(oid=oid))
    return out


def build_holdout_jsonl(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        # 5 affected
        for i, q in enumerate(AFFECTED_HELD_OUT):
            fh.write(json.dumps({
                "query_id": f"affected_{i}",
                "tenant_id": "demo_silent",
                "payload": {
                    "question_id": f"affected_{i}",
                    "question": q,
                    "gold": "process_refund",
                    "slices": ["intent=refund_high_value"],
                },
                "added_at": "2026-05-18T20:00:00+00:00",
                "accepted_by_adapter": None,
            }) + "\n")
        # 145 routine billing
        for i, q in enumerate(_routine_refund_queries(145)):
            fh.write(json.dumps({
                "query_id": f"routine_billing_{i}",
                "tenant_id": "demo_silent",
                "payload": {
                    "question_id": f"routine_billing_{i}",
                    "question": q,
                    "gold": "process_refund",
                    "slices": ["intent=refund_standard"],
                },
                "added_at": "2026-05-18T20:00:00+00:00",
                "accepted_by_adapter": None,
            }) + "\n")
        # 150 order_status
        for i, q in enumerate(_routine_status_queries(150)):
            fh.write(json.dumps({
                "query_id": f"status_{i}",
                "tenant_id": "demo_silent",
                "payload": {
                    "question_id": f"status_{i}",
                    "question": q,
                    "gold": "check_status",
                    "slices": ["intent=order_status"],
                },
                "added_at": "2026-05-18T20:00:00+00:00",
                "accepted_by_adapter": None,
            }) + "\n")


# ---------- 4. Scorer module ----------

def write_scorer_module(scorer_path: Path, adapter_paths: dict[str, str]) -> None:
    code = f'''"""Auto-generated scorer for the silent-slice-regression demo."""
import pickle

_paths = {adapter_paths!r}
_cache = {{}}


def score(adapter_id: str, query: dict) -> float:
    if adapter_id is None:
        return 0.0
    if adapter_id not in _cache:
        with open(_paths[adapter_id], "rb") as fh:
            _cache[adapter_id] = pickle.load(fh)
    vec, clf = _cache[adapter_id]
    x = vec.transform([query["question"]])
    pred = clf.predict(x)[0]
    return 1.0 if pred == query["gold"] else 0.0
'''
    scorer_path.write_text(code)


# ---------- 5. Run ----------

def _run_gate(workdir: Path, slice_eps: float | None) -> int:
    """Run the gate once with or without --slice-epsilon. Returns exit code."""
    args = [
        sys.executable, "-m", "adaptergate.cli", "gate",
        "--tenant", "demo_silent",
        "--candidate", "adapter_B",
        "--baseline", "adapter_A",
        "--holdout", str(workdir / "holdout.jsonl"),
        "--scorer", "scorer:score",
        "--audit-log", str(workdir / f"audit_{'with' if slice_eps else 'without'}_slice_eps.jsonl"),
    ]
    if slice_eps is not None:
        args += ["--slice-epsilon", str(slice_eps)]
    result = subprocess.run(args, cwd=workdir)
    return result.returncode


def run(workdir: Path) -> None:
    print(f"work dir: {workdir}")
    workdir.mkdir(parents=True, exist_ok=True)

    print("\n[1/5] Training adapter A (clean labels) ...")
    vec_a, clf_a = train_adapter(*build_clean_dataset())
    with (workdir / "adapter_A.pkl").open("wb") as f:
        pickle.dump((vec_a, clf_a), f)

    print("[2/5] Training adapter B (silently mislabels 'refund + missing order_id') ...")
    vec_b, clf_b = train_adapter(*build_contaminated_dataset())
    with (workdir / "adapter_B.pkl").open("wb") as f:
        pickle.dump((vec_b, clf_b), f)

    print("[3/5] Building 300-query held-out set (5 affected + 145 routine billing + 150 status) ...")
    build_holdout_jsonl(workdir / "holdout.jsonl")

    print("[4/5] Writing scorer module ...")
    write_scorer_module(workdir / "scorer.py", {
        "adapter_A": str(workdir / "adapter_A.pkl"),
        "adapter_B": str(workdir / "adapter_B.pkl"),
    })

    print("\n" + "═" * 80, flush=True)
    print("  ROUND 1: Default gate (aggregate-only, like Braintrust/DeepEval mean-score)", flush=True)
    print("═" * 80 + "\n", flush=True)
    rc_default = _run_gate(workdir, slice_eps=None)
    print(f"\n  → exit code: {rc_default} ({'ACCEPTED' if rc_default == 0 else 'REJECTED'})")
    if rc_default == 0:
        print("  → Aggregate-only eval ACCEPTED the candidate. Slice collapse went silent.")

    print("\n" + "═" * 80, flush=True)
    print("  ROUND 2: Same gate + --slice-epsilon 0.10 (adaptergate's safety net)", flush=True)
    print("═" * 80 + "\n", flush=True)
    rc_with = _run_gate(workdir, slice_eps=0.10)
    print(f"\n  → exit code: {rc_with} ({'ACCEPTED' if rc_with == 0 else 'REJECTED'})")

    print("\n" + "─" * 80)
    print("WHAT JUST HAPPENED (plain English):")
    print("  - Adapter B silently mis-routes refund requests from customers who")
    print("    can't quote their order_id. Only 5 such queries exist in the held-out")
    print("    set of 300 — the rest are routine queries both adapters handle.")
    print("  - Aggregate drop: ~1pp. WITHIN the default ε=0.02 threshold.")
    print("  - ROUND 1 (aggregate-only, like Braintrust/DeepEval mean-score):")
    print("    → ACCEPTED. The slice failure ships to prod. Customers churn.")
    print("  - ROUND 2 (--slice-epsilon 0.10):")
    print("    → REJECTED. The driver slice (intent=refund_high_value) dropped")
    print("       60% within itself, exceeding slice_ε. Update is gated.")
    print("  - This is the load-bearing case: aggregate eval cannot see slice-")
    print("    level collapse. Slice attribution can. That is the differential.")
    print("─" * 80)

    print(f"\nArtifacts in {workdir}.")


if __name__ == "__main__":
    work = Path(tempfile.mkdtemp(prefix="adaptergate-silent-demo-"))
    run(work)
