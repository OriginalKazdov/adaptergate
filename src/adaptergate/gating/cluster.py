"""Failure pattern detection — find a common N-gram across failing queries.

This is the **Slack-converter** feature: when the gate rejects, customers
need a one-line description of what the failing queries have in common so
they can paste it into Slack and start triage. Without it, "intent=billing
regressed -0.859" is a number; with it, the screenshot tells a story.

Pure N-gram frequency analysis, no LLM. The signal: any 1/2/3-gram that
appears in ≥50% of failing queries is a candidate; we report up to three
non-overlapping phrases.

Intentionally NOT implemented here:
  - LLM-generated cause hypothesis (defer; turns one feature into a
    4-week project that we'll get wrong on day 1)
  - Suggested fix (defer; requires customer's training-data lineage)
  - Embedding-based clustering (defer; needs sentence-transformers, drags
    torch into the core install)
"""

from __future__ import annotations

import re
from typing import Any


_STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "with",
    "for", "on", "at", "by", "as", "be", "was", "were", "this", "that",
    "what", "how", "why", "when", "where", "i", "you", "we", "my", "your",
    "do", "does", "did", "have", "has", "had", "can", "could", "would",
    "should", "will", "shall", "may", "might", "it", "its", "they", "them",
    "their", "there", "here", "if", "then", "so", "but", "not", "no",
})

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def find_pattern(failing_queries: list[dict[str, Any]], *, min_coverage: float = 0.5) -> str | None:
    """Return a one-line pattern describing what failing queries have in common.

    Pulls text from the first present field among ``question``, ``text``,
    ``prompt``, ``query``. Tokenizes (alphanumeric, lowercase), then counts
    document frequency for unigrams, bigrams, and trigrams. Reports up to
    three non-overlapping N-grams that hit at least ``min_coverage`` of
    failing queries.

    Returns ``None`` if there are fewer than 3 failing queries with
    inspectable text, or no N-gram crosses the coverage threshold.
    """
    texts = [_extract_text(q) for q in failing_queries]
    texts = [t for t in texts if t]
    if len(texts) < 3:
        return None

    n_docs = len(texts)
    min_docs = max(2, int(round(n_docs * min_coverage)))

    df: dict[str, int] = {}
    for text in texts:
        tokens = _TOKEN_RE.findall(text.lower())
        seen: set[str] = set()
        for tok in tokens:
            if len(tok) > 1 and tok not in _STOPWORDS:
                seen.add(tok)
        for i in range(len(tokens) - 1):
            seen.add(" ".join(tokens[i : i + 2]))
        for i in range(len(tokens) - 2):
            seen.add(" ".join(tokens[i : i + 3]))
        for ngram in seen:
            df[ngram] = df.get(ngram, 0) + 1

    candidates = sorted(
        ((ngram, count) for ngram, count in df.items() if count >= min_docs),
        key=lambda x: (-x[1], -len(x[0])),
    )

    selected: list[str] = []
    for ngram, _count in candidates:
        if any(_token_overlap(ngram, s) for s in selected):
            continue
        selected.append(ngram)
        if len(selected) >= 3:
            break

    if not selected:
        return None

    top_count = candidates[0][1]
    if top_count == n_docs:
        prefix = f"all {n_docs} failing queries contain"
    else:
        prefix = f"{top_count}/{n_docs} failing queries contain"
    return f"{prefix}: " + ", ".join(f'"{s}"' for s in selected)


def _extract_text(q: dict[str, Any]) -> str | None:
    """Pull the first available natural-language field from a query payload."""
    for key in ("question", "text", "prompt", "query"):
        v = q.get(key)
        if isinstance(v, str) and v.strip():
            return v
    return None


def _token_overlap(a: str, b: str) -> bool:
    """True if either N-gram is a token-aligned contiguous subsequence of the other."""
    aw = a.split()
    bw = b.split()
    return _contains(aw, bw) or _contains(bw, aw)


def _contains(short: list[str], long: list[str]) -> bool:
    n, m = len(short), len(long)
    if n > m:
        return False
    for i in range(m - n + 1):
        if long[i : i + n] == short:
            return True
    return False
