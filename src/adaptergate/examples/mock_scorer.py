"""Mock scorer for trying the adaptergate CLI without a real model.

Purely illustrative — use it to play with ``adaptergate gate`` end-to-end and
to see how per-slice attribution works without needing a real LoRA pipeline.

# Score behavior

  - Adapter IDs containing "good" → ~0.88 base
  - Adapter IDs containing "bad"  → ~0.55 base
  - "v" + even number              → ~0.85 base
  - Anything else                  → ~0.70 base

If the query payload has a ``"slices"`` field (e.g. ``["intent=billing_dispute"]``)
the mock applies a slice-specific penalty/bonus to make per-slice attribution
visible in the gate output. The ``intent=billing_dispute`` slice in particular
is calibrated to be the *driver slice* — the place where "bad" adapters
catastrophically regress.

Example with slices::

    adaptergate holdout add --tenant demo --holdout demo.jsonl \\
        '{"question_id": "q1", "slices": ["intent=billing_dispute"]}'
    adaptergate holdout add --tenant demo --holdout demo.jsonl \\
        '{"question_id": "q2", "slices": ["intent=order_status"]}'
    # ... add 25+ queries with mixed slices
    adaptergate gate \\
        --tenant demo \\
        --candidate adapter_bad_v19 \\
        --baseline adapter_good_v18 \\
        --holdout demo.jsonl \\
        --scorer adaptergate.examples.mock_scorer:score
"""

from __future__ import annotations

import hashlib
import re


# Slice-specific modulation. Each slice tag maps to (good_adapter_bonus,
# bad_adapter_penalty). Positive bonus boosts the score; positive penalty
# subtracts from it. Calibrated so ``intent=billing_dispute`` is the driver.
_SLICE_DELTAS: dict[str, tuple[float, float]] = {
    "intent=billing_dispute":   (+0.05, 0.45),   # driver slice
    "intent=refund_request":    (+0.03, 0.30),
    "intent=technical_support": (+0.02, 0.10),
    "intent=order_status":      (+0.02, 0.05),
    "lang=es":                  (-0.02, 0.04),
    "difficulty=hard":          (-0.05, 0.15),
    "difficulty=easy":          (+0.05, -0.05),  # bad adapter actually does FINE on easy
}


def score(adapter_id: str, query: dict) -> float:
    """Return a deterministic mock score in ``[0.0, 1.0]``.

    Deterministic per ``(adapter_id, query_id)`` so gate decisions reproduce
    across CLI runs.
    """
    if adapter_id is None:
        return 0.0

    name = adapter_id.lower()
    if "bad" in name:
        base = 0.55
        is_bad = True
    elif "good" in name:
        base = 0.88
        is_bad = False
    else:
        m = re.search(r"v(\d+)", name)
        if m and int(m.group(1)) % 2 == 0:
            base = 0.85
        else:
            base = 0.70
        is_bad = False

    # Slice-specific adjustment.
    for slice_tag in query.get("slices") or []:
        bonus, penalty = _SLICE_DELTAS.get(slice_tag, (0.0, 0.0))
        if is_bad:
            base -= penalty
        else:
            base += bonus

    # Sprinkle deterministic per-query jitter.
    query_id = (
        query.get("question_id") or query.get("id") or query.get("query_id") or ""
    )
    seed = f"{adapter_id}|{query_id}".encode("utf-8")
    jitter = (int(hashlib.sha256(seed).hexdigest()[:8], 16) % 200 - 100) / 1000.0

    return max(0.0, min(1.0, base + jitter))
