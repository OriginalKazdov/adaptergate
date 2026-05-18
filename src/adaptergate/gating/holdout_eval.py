"""Per-tenant held-out eval set management.

Each tenant gets its own rolling held-out set. Queries are added as accepted
inputs (frozen-good baselines), aged out over time, and sampled deterministically
when the gate runs.

Persisted as JSONL for portability + diffability. SQLite-backed indexing comes
later if scale demands it.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


@dataclass
class HoldoutQuery:
    """A single held-out query for regression evaluation."""

    query_id: str
    tenant_id: str
    payload: dict[str, Any]
    """The query content (e.g., BIRD-SQL row). Scorer interprets this."""
    added_at: str
    accepted_by_adapter: str | None
    """Which adapter version this query was 'frozen good' against."""


class HoldoutSet:
    """A per-tenant rolling held-out eval set.

    Usage:
        holdout = HoldoutSet(tenant_id="acme", path="data/holdout/acme.jsonl")
        holdout.add({"question_id": "q1", "question": "..."}, accepted_by="v16")
        sample = holdout.sample(n=50, seed="v17")
        # ... run gate.evaluate(holdout=sample, ...)
    """

    def __init__(self, tenant_id: str, path: str | Path, max_size: int | None = None):
        if max_size is not None and max_size < 1:
            raise ValueError(f"max_size must be >= 1 or None, got {max_size}")
        self.tenant_id = tenant_id
        self.path = Path(path)
        self.max_size = max_size
        self._queries: list[HoldoutQuery] = []
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                self._queries.append(HoldoutQuery(**row))

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as fh:
            for q in self._queries:
                fh.write(json.dumps(asdict(q)))
                fh.write("\n")

    def add(self, payload: dict[str, Any], accepted_by: str | None = None) -> HoldoutQuery:
        """Add a query to the held-out set. Idempotent on (query_id, tenant)."""
        query_id = self._derive_id(payload)
        for q in self._queries:
            if q.query_id == query_id:
                return q
        record = HoldoutQuery(
            query_id=query_id,
            tenant_id=self.tenant_id,
            payload=payload,
            added_at=datetime.now(timezone.utc).isoformat(),
            accepted_by_adapter=accepted_by,
        )
        self._queries.append(record)
        if self.max_size is not None and len(self._queries) > self.max_size:
            self._queries = self._queries[-self.max_size :]
        self._flush()
        return record

    def remove(self, query_id: str) -> bool:
        before = len(self._queries)
        self._queries = [q for q in self._queries if q.query_id != query_id]
        changed = len(self._queries) != before
        if changed:
            self._flush()
        return changed

    def sample(self, n: int | None = None, seed: str | None = None) -> list[dict[str, Any]]:
        """Return a deterministic sample of query payloads.

        Args:
            n: number of queries to sample. None = return all.
            seed: deterministic seed (e.g., candidate_adapter_id). Same seed +
                same set => same sample. Used so the gate decision is reproducible.

        Returns:
            List of payload dicts ready to pass into gate.evaluate(holdout=...).
        """
        if n is None or n >= len(self._queries):
            return [q.payload for q in self._queries]
        rng = random.Random(self._seed_to_int(seed) if seed else None)
        picks = rng.sample(self._queries, n)
        return [q.payload for q in picks]

    def __len__(self) -> int:
        return len(self._queries)

    def __iter__(self) -> Iterator[HoldoutQuery]:
        return iter(self._queries)

    def staleness_days(self) -> int | None:
        """Return days since the most-recently-added query was added.

        ``None`` when the held-out set is empty. The CLI uses this to warn
        the user when their held-out set hasn't been refreshed in a while —
        stale held-out sets fail to reflect current traffic, which causes
        the gate to reject candidates for "regressions" that are actually
        eval-set drift, not adapter drift.
        """
        if not self._queries:
            return None
        from datetime import datetime, timezone

        latest_str = max(q.added_at for q in self._queries)
        try:
            latest_dt = datetime.fromisoformat(latest_str)
            if latest_dt.tzinfo is None:
                latest_dt = latest_dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
        now = datetime.now(timezone.utc)
        return max(0, (now - latest_dt).days)

    def _derive_id(self, payload: dict[str, Any]) -> str:
        for key in ("query_id", "question_id", "id"):
            if key in payload:
                return str(payload[key])
        blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]

    @staticmethod
    def _seed_to_int(seed: str) -> int:
        return int(hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16], 16)
