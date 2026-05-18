"""Mock scorer for trying the adaptergate CLI without a real model.

This is purely illustrative — use it to play with `adaptergate gate` end-to-end.

Score behavior:
  - Adapter IDs containing "good" or "v" + an even number score ~0.9
  - Adapter IDs containing "bad" score ~0.5
  - Anything else scores 0.7

Run:
    adaptergate holdout add --tenant demo --holdout demo.jsonl '{"question_id": "q1"}'
    adaptergate holdout add --tenant demo --holdout demo.jsonl '{"question_id": "q2"}'
    # ... add at least 20 queries
    adaptergate gate \\
        --tenant demo \\
        --candidate adapter_good_v18 \\
        --baseline adapter_bad_v17 \\
        --holdout demo.jsonl \\
        --scorer examples.mock_scorer:score
"""

from __future__ import annotations

import hashlib
import re


def score(adapter_id: str, query: dict) -> float:
    """Return a deterministic mock score in [0.0, 1.0].

    The mock is intentionally noisy-but-deterministic per (adapter, query) so
    gate decisions are reproducible across CLI runs.
    """
    if adapter_id is None:
        return 0.0

    name = adapter_id.lower()
    if "bad" in name:
        base = 0.55
    elif "good" in name:
        base = 0.88
    else:
        m = re.search(r"v(\d+)", name)
        if m and int(m.group(1)) % 2 == 0:
            base = 0.85
        else:
            base = 0.70

    # Sprinkle deterministic per-query jitter.
    query_id = (
        query.get("question_id") or query.get("id") or query.get("query_id") or ""
    )
    seed = f"{adapter_id}|{query_id}".encode("utf-8")
    jitter = (int(hashlib.sha256(seed).hexdigest()[:8], 16) % 200 - 100) / 1000.0

    return max(0.0, min(1.0, base + jitter))
