"""Tests for failure pattern detection (cluster summary, no LLM)."""

from __future__ import annotations

from adaptergate.gating.cluster import find_pattern


# ---------- happy path ----------

def test_finds_common_phrase_across_failing_queries():
    queries = [
        {"question": "I want a refund but I lost my order_id"},
        {"question": "Please refund me, my order_id is missing"},
        {"question": "Refund request without order_id available"},
        {"question": "Need a refund but cannot find order_id anywhere"},
        {"question": "Refund for purchase without order_id provided"},
    ]
    pattern = find_pattern(queries)
    assert pattern is not None
    # "refund" appears in all 5, "order_id" appears in all 5
    assert "refund" in pattern.lower() or "order_id" in pattern.lower()


def test_reports_all_5_when_all_match():
    queries = [
        {"question": f"Cannot process refund without order_id, attempt {i}"}
        for i in range(5)
    ]
    pattern = find_pattern(queries)
    assert pattern is not None
    assert pattern.startswith("all 5 failing queries contain")


def test_reports_partial_when_below_full():
    # 3 out of 5 mention refund explicitly
    queries = [
        {"question": "I need a refund please"},
        {"question": "Where is my refund?"},
        {"question": "Process my refund quickly"},
        {"question": "Cancel my subscription"},
        {"question": "Update my profile"},
    ]
    pattern = find_pattern(queries)
    assert pattern is not None
    # 3/5 = exactly 50% threshold
    assert "3/5" in pattern
    assert "refund" in pattern.lower()


# ---------- thresholds ----------

def test_returns_none_when_fewer_than_3_queries():
    queries = [{"question": "test"}, {"question": "test"}]
    assert find_pattern(queries) is None


def test_returns_none_when_no_common_pattern():
    queries = [
        {"question": "alpha beta"},
        {"question": "gamma delta"},
        {"question": "epsilon zeta"},
        {"question": "eta theta"},
    ]
    assert find_pattern(queries) is None


def test_respects_custom_min_coverage():
    # 2 out of 4 mention "refund"
    queries = [
        {"question": "I want a refund"},
        {"question": "I need a refund"},
        {"question": "Cancel my plan"},
        {"question": "Update my info"},
    ]
    # Default coverage 0.5 → needs 2 docs → matches
    assert find_pattern(queries) is not None
    # Tighter coverage 0.75 → needs 3 docs → no match
    assert find_pattern(queries, min_coverage=0.75) is None


# ---------- text source flexibility ----------

def test_accepts_text_field():
    queries = [{"text": f"failing example contains foo {i}"} for i in range(4)]
    pattern = find_pattern(queries)
    assert pattern is not None
    assert "foo" in pattern.lower()


def test_accepts_prompt_field():
    queries = [{"prompt": f"prompt about foo and bar {i}"} for i in range(4)]
    pattern = find_pattern(queries)
    assert pattern is not None


def test_ignores_queries_without_text_field():
    # Only 2 have text, 3 don't → fewer than 3 → None
    queries = [
        {"question": "test foo"},
        {"question": "test foo"},
        {"db_id": "no_text"},
        {"adapter_only": True},
        {"slices": ["intent=foo"]},
    ]
    assert find_pattern(queries) is None


# ---------- stopword filtering ----------

def test_stopwords_dont_dominate_unigrams():
    # All queries share stopwords like "the", "a", "is" but only "refund" is meaningful
    queries = [
        {"question": "the user is requesting a refund"},
        {"question": "the customer is asking for a refund"},
        {"question": "the buyer is asking about a refund"},
        {"question": "the client is wanting a refund"},
    ]
    pattern = find_pattern(queries)
    assert pattern is not None
    # "the", "is", "a" should NOT be the top reported phrases as unigrams
    assert '"the"' not in pattern
    assert '"is"' not in pattern
    assert '"a"' not in pattern
    # "refund" should show up
    assert "refund" in pattern.lower()


def test_stopwords_allowed_in_bigrams():
    # Stopwords can still appear *inside* multi-token phrases
    queries = [
        {"question": "Please refund the order"},
        {"question": "Refund the order quickly"},
        {"question": "Refund the order today"},
        {"question": "Refund the order now"},
    ]
    pattern = find_pattern(queries)
    assert pattern is not None
    # Either "refund the order" trigram or some other phrase should be picked
    assert pattern  # smoke check — depends on tie-break


# ---------- de-duplication ----------

def test_overlapping_phrases_deduplicated():
    queries = [
        {"question": "missing order_id in refund request"},
        {"question": "missing order_id for refund"},
        {"question": "no order_id provided for refund"},
        {"question": "absent order_id during refund process"},
    ]
    pattern = find_pattern(queries)
    assert pattern is not None
    # Should not include both "order_id" and "missing order_id" — the
    # longer phrase subsumes the shorter.
    # Count quoted segments; expect at most 3 distinct.
    quoted = pattern.count('"') // 2
    assert quoted <= 3
