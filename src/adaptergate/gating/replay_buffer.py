"""Replay buffer for rejected adapter updates.

When the regression gate rejects a candidate, we don't throw it away — we keep
it in a per-tenant replay buffer. Future updates that fix the regression can be
trained against the rejected example. Also lets DriftCouncil (downstream) reason
across a history of rejections to spot patterns.

Persisted as JSONL alongside the gate's audit log.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator

from adaptergate.gating.regression_gate import GateDecision


@dataclass
class ReplayRecord:
    """A rejected candidate adapter, kept for later analysis or retraining."""

    tenant_id: str
    candidate_id: str
    baseline_id: str | None
    rejected_at: str
    delta: float
    reason: str
    decision_blob: str
    """Full GateDecision serialized to JSON for downstream reasoning."""


class ReplayBuffer:
    """Per-tenant ring buffer of rejected candidate updates.

    Usage:
        buf = ReplayBuffer(tenant_id="acme", path="data/replay/acme.jsonl", max_size=100)
        if not decision.accepted:
            buf.add(decision)
        for record in buf:
            ...  # inspect rejection history
    """

    def __init__(self, tenant_id: str, path: str | Path, max_size: int = 200):
        if max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {max_size}")
        self.tenant_id = tenant_id
        self.path = Path(path)
        self.max_size = max_size
        self._records: list[ReplayRecord] = []
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                self._records.append(ReplayRecord(**row))

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as fh:
            for r in self._records:
                fh.write(json.dumps(asdict(r)))
                fh.write("\n")

    def add(self, decision: GateDecision) -> ReplayRecord:
        if decision.accepted:
            raise ValueError(
                "ReplayBuffer.add() expects a rejected decision; received an"
                f" accepted one for candidate_id={decision.candidate_id}."
            )
        if decision.tenant_id != self.tenant_id:
            raise ValueError(
                f"Decision tenant_id={decision.tenant_id} does not match buffer"
                f" tenant_id={self.tenant_id}."
            )
        record = ReplayRecord(
            tenant_id=decision.tenant_id,
            candidate_id=decision.candidate_id,
            baseline_id=decision.baseline_id,
            rejected_at=decision.timestamp,
            delta=decision.delta,
            reason=decision.reason,
            decision_blob=decision.to_json(indent=None),
        )
        self._records.append(record)
        if len(self._records) > self.max_size:
            self._records = self._records[-self.max_size :]
        self._flush()
        return record

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[ReplayRecord]:
        return iter(self._records)

    def recent(self, n: int = 10) -> list[ReplayRecord]:
        return self._records[-n:]
