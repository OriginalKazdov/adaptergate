"""
Prompt templates for SQL generation.

Two templates:
  - QWEN_CODER_SYSTEM: for Qwen 2.5 Coder Instruct (our open-weights base)
  - GENERIC_SYSTEM: for frontier APIs (Sonnet 4.6, GPT-5) used as ceiling baseline

Designed for "fill the schema, ask the question, get SQL".
"""

from __future__ import annotations


QWEN_CODER_SYSTEM = """You are an expert SQL engineer. Generate a single valid SQLite query that answers the user's question.

Rules:
1. Output ONLY the SQL query. No explanation, no markdown fences, no commentary.
2. Use only tables and columns shown in the schema below.
3. Prefer JOINs over subqueries when possible.
4. Use exact column names — case matters.
5. End your query with a semicolon."""


GENERIC_SYSTEM = """You are an expert at writing SQLite SQL queries.
Given a database schema and a question, write a single SQL query that
correctly answers the question.

Important:
- Output only the SQL query
- No markdown formatting, no explanations
- Use SQLite-compatible syntax
- Reference only tables/columns that exist in the schema
"""


def build_qwen_prompt(
    schema_text: str,
    question: str,
    evidence: str = "",
) -> tuple[str, str]:
    """Returns (system, user) prompt strings for Qwen Coder."""
    user_parts = ["### Database Schema", schema_text, "### Question", question]
    if evidence:
        user_parts.extend(["### Hints", evidence])
    user_parts.append("### SQL")
    user = "\n\n".join(user_parts)
    return QWEN_CODER_SYSTEM, user


def build_chat_messages(
    schema_text: str,
    question: str,
    evidence: str = "",
    system: str = QWEN_CODER_SYSTEM,
) -> list[dict]:
    """OpenAI/Anthropic-compatible chat format."""
    user_parts = ["### Database Schema", schema_text, "### Question", question]
    if evidence:
        user_parts.extend(["### Hints", evidence])
    user_parts.append("### SQL")
    user_content = "\n\n".join(user_parts)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


def extract_sql_from_response(text: str) -> str:
    """
    Models like adding markdown fences or commentary. Strip them out.
    Returns the largest plausible SQL block.
    """
    text = text.strip()

    # Try to extract from ```sql ... ``` block
    import re
    fence_match = re.search(r"```(?:sql)?\s*\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()

    # If response starts with SELECT/WITH/INSERT/UPDATE/DELETE, take to end of statement
    text_upper = text.upper().lstrip()
    sql_keywords = ("SELECT", "WITH", "INSERT", "UPDATE", "DELETE", "CREATE")
    if any(text_upper.startswith(kw) for kw in sql_keywords):
        # Take until first blank line or end
        lines = []
        for line in text.split("\n"):
            stripped = line.strip()
            if not stripped and lines:
                # blank line after content — stop
                break
            lines.append(line)
        return "\n".join(lines).strip()

    # Last resort: return as-is
    return text


if __name__ == "__main__":
    # Smoke
    msgs = build_chat_messages(
        schema_text="CREATE TABLE users (id INT, name TEXT);",
        question="How many users?",
    )
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"

    # Extract from fenced response
    fenced = "Sure! Here's the query:\n```sql\nSELECT COUNT(*) FROM users;\n```\nThat should work."
    sql = extract_sql_from_response(fenced)
    assert sql == "SELECT COUNT(*) FROM users;", f"got: {sql!r}"

    # Extract from bare response
    bare = "SELECT COUNT(*) FROM users;\n\nThis returns the count."
    sql = extract_sql_from_response(bare)
    assert sql.startswith("SELECT COUNT"), f"got: {sql!r}"
    print("prompts smoke test passed")
