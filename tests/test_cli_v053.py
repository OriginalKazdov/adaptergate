"""Tests for v0.5.3 CLI additions: batch import, replay show, slice validation,
flag alias."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from adaptergate.cli import app

runner = CliRunner()


# ---------- slice validation in holdout add ----------

def test_holdout_add_rejects_bare_string_slices(tmp_path: Path):
    """Common typo — slices passed as bare string instead of list — must fail
    at ingest, not silently corrupt the held-out."""
    result = runner.invoke(
        app,
        [
            "holdout", "add",
            "--tenant", "t1",
            "--holdout", str(tmp_path / "h.jsonl"),
            '{"question_id":"q1","slices":"intent=foo"}',
        ],
    )
    assert result.exit_code != 0
    assert "must be a JSON list" in result.output or "Common mistake" in result.output


def test_holdout_add_rejects_slice_tag_without_equals(tmp_path: Path):
    result = runner.invoke(
        app,
        [
            "holdout", "add",
            "--tenant", "t1",
            "--holdout", str(tmp_path / "h.jsonl"),
            '{"question_id":"q1","slices":["just_a_word"]}',
        ],
    )
    assert result.exit_code != 0
    assert "key=value" in result.output


def test_holdout_add_accepts_well_formed_slices(tmp_path: Path):
    result = runner.invoke(
        app,
        [
            "holdout", "add",
            "--tenant", "t1",
            "--holdout", str(tmp_path / "h.jsonl"),
            '{"question_id":"q1","slices":["intent=refund","lang=en"]}',
        ],
    )
    assert result.exit_code == 0, result.output


def test_holdout_add_accepts_payload_with_no_slices(tmp_path: Path):
    result = runner.invoke(
        app,
        [
            "holdout", "add",
            "--tenant", "t1",
            "--holdout", str(tmp_path / "h.jsonl"),
            '{"question_id":"q1"}',
        ],
    )
    assert result.exit_code == 0, result.output


# ---------- batch import ----------

def test_holdout_import_imports_valid_lines(tmp_path: Path):
    src = tmp_path / "src.jsonl"
    src.write_text(
        json.dumps({"question_id": "a", "slices": ["intent=x"]}) + "\n"
        + json.dumps({"question_id": "b", "slices": ["intent=y"]}) + "\n"
    )
    result = runner.invoke(
        app,
        [
            "holdout", "import",
            "--tenant", "t1",
            "--holdout", str(tmp_path / "h.jsonl"),
            "--from-jsonl", str(src),
        ],
    )
    assert result.exit_code == 0, result.output
    body = json.loads(result.output.strip().splitlines()[0])
    assert body == {"imported": 2, "skipped": 0, "size": 2}


def test_holdout_import_skips_malformed_lines_and_exits_nonzero(tmp_path: Path):
    src = tmp_path / "src.jsonl"
    src.write_text(
        json.dumps({"question_id": "a", "slices": ["intent=x"]}) + "\n"
        + '{this is not json\n'
        + json.dumps({"question_id": "c", "slices": "intent=bad_typo"}) + "\n"  # bare string
    )
    result = runner.invoke(
        app,
        [
            "holdout", "import",
            "--tenant", "t1",
            "--holdout", str(tmp_path / "h.jsonl"),
            "--from-jsonl", str(src),
        ],
    )
    assert result.exit_code == 2  # any skips → exit 2
    body = json.loads(result.output.strip().splitlines()[0])
    assert body["imported"] == 1
    assert body["skipped"] == 2


def test_holdout_import_missing_file(tmp_path: Path):
    result = runner.invoke(
        app,
        [
            "holdout", "import",
            "--tenant", "t1",
            "--holdout", str(tmp_path / "h.jsonl"),
            "--from-jsonl", str(tmp_path / "does_not_exist.jsonl"),
        ],
    )
    assert result.exit_code == 2


# ---------- replay list flag alias ----------

def test_replay_list_accepts_replay_path_alias(tmp_path: Path):
    """v0.5.3: `replay list` now accepts --replay-path as alias of --replay so it
    matches `gate --replay-path`."""
    empty = tmp_path / "empty_replay.jsonl"
    empty.touch()
    result = runner.invoke(
        app,
        [
            "replay", "list",
            "--tenant", "t1",
            "--replay-path", str(empty),
        ],
    )
    assert result.exit_code == 0, result.output
