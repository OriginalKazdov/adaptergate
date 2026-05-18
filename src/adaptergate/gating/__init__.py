"""Regression gating — the wedge.

The do-no-harm decision layer for LoRA adapter updates. Before promoting a
candidate adapter, evaluate it on a per-tenant held-out set. Reject if the
aggregate score drops more than epsilon, or (in strict mode) if any query
that previously scored 1.0 now scores less.

When the held-out queries carry slice tags (``query["slices"] = [...]``),
the gate also produces per-slice attribution — the cohort-level breakdown
that tells you *which* behavioral slice broke, not just that *something* did.

Components:
  - RegressionGate: the accept/reject decision
  - GateDecision: rich result with per-query + per-slice breakdown
  - SliceAttribution: cohort-level regression for one slice tag
  - HoldoutSet: per-tenant held-out query store, JSONL-backed
  - ReplayBuffer: per-tenant rejected-update log, JSONL-backed
"""

from adaptergate.gating.holdout_eval import HoldoutQuery, HoldoutSet
from adaptergate.gating.regression_gate import (
    GateConfig,
    GateDecision,
    RegressionGate,
    SliceAttribution,
    append_audit,
)
from adaptergate.gating.replay_buffer import ReplayBuffer, ReplayRecord

__all__ = [
    "GateConfig",
    "GateDecision",
    "RegressionGate",
    "SliceAttribution",
    "append_audit",
    "HoldoutQuery",
    "HoldoutSet",
    "ReplayBuffer",
    "ReplayRecord",
]
