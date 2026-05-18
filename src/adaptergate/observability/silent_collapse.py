"""Silent Collapse MTR — Monitor / Trust / Regulator framework.

Reference: arXiv 2605.14588 (May 2026) — "Silent Collapse in Recursive
Learning Systems". Independent re-implementation; framework is credited to
the paper authors.

Key formulas:

  1. Trajectory contraction statistic (representation drift):
        S_g = (1/W) Σ_{k=0..W-1} ||z_{g-k} - z_{g-k-1}||² / (||z_{g-k-1}||² + ε)
     where z_g = penultimate-layer representations mean-pooled over the
     anchor set at generation g. W = smoothing window.

  2. Silent collapse signature (the key observation):
        H_g/H_0 < 0.5  AND  PPL_g/PPL_0 ≈ 1
     Entropy drops by 50%+ while perplexity stays near baseline. This is
     what makes the collapse "silent" — standard validation looks fine.

  3. Trust variable τ (for the Regulator):
        τ = max(0.2, min(1.0, H_g/H_0))
     The regulator uses τ to modulate effective learning intensity
     (e.g. LoRA learning-rate scaling) instead of freezing updates outright.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class MTRConfig:
    """Thresholds + windows for the MTR framework."""
    # Drift window (W in the formula)
    drift_window: int = 10
    # Drift floor: alert if window-avg S_g falls below this (frozen reps)
    drift_floor: float = 0.001
    # Entropy ratio threshold (paper key signature: H/H_0 < 0.5)
    entropy_ratio_threshold: float = 0.5
    # PPL ratio tolerance for "≈ 1" — allow ±20%
    ppl_ratio_min: float = 0.8
    ppl_ratio_max: float = 1.2
    # Anchor set size
    anchor_size: int = 32
    # Trust bounds for τ
    trust_min: float = 0.2
    trust_max: float = 1.0
    # Baseline establishment: number of warmup observations
    warmup_steps: int = 5


@dataclass
class CollapseAlert:
    """Raised when collapse signature is detected."""
    precursor: str           # "silent_collapse" | "drift_freeze" | "diversity"
    h_ratio: float | None    # H_g / H_0
    ppl_ratio: float | None  # PPL_g / PPL_0
    drift: float | None      # S_g (window avg)
    update_step: int
    trust_tau: float

    def is_silent_collapse(self) -> bool:
        return self.precursor == "silent_collapse"


class SilentCollapseMonitor:
    """
    Per-slot collapse detector. Call .observe(...) after each training generation.

    "Generation" = a checkpoint after one or more updates. For our streaming CL
    use, we call observe() every N batches.

    Returns a CollapseAlert if signature is detected, else None.
    """

    def __init__(self, config: MTRConfig | None = None):
        self.cfg = config or MTRConfig()
        self.generation = 0

        # Baseline values established during warmup
        self.H_0: float | None = None
        self.PPL_0: float | None = None

        # Rolling state for drift
        self.z_history: deque[torch.Tensor] = deque(maxlen=self.cfg.drift_window + 1)

        # Histories
        self.H_history: list[float] = []
        self.PPL_history: list[float] = []
        self.drift_history: list[float] = []

    # ---------- Precursor formulas (from paper) ----------

    @staticmethod
    def compute_anchor_entropy(anchor_logits: torch.Tensor) -> float:
        """
        H_g = mean predictive entropy on the anchor set's next-token distribution.

        anchor_logits: (anchor_size, seq_len, vocab) or (anchor_size, vocab)
        """
        if anchor_logits.dim() == 3:
            # take last position
            logits = anchor_logits[:, -1, :]
        else:
            logits = anchor_logits
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        # entropy in nats, then take mean
        h = -(probs * log_probs).sum(dim=-1).mean().item()
        return h

    @staticmethod
    def compute_perplexity(anchor_logits: torch.Tensor,
                           anchor_target_ids: torch.Tensor) -> float:
        """
        PPL_g on anchor set. We compute it as exp(mean CE over anchor tokens).

        anchor_logits: (B, T, V), anchor_target_ids: (B, T)
        """
        # standard cross-entropy
        loss = F.cross_entropy(
            anchor_logits.reshape(-1, anchor_logits.size(-1)),
            anchor_target_ids.reshape(-1),
            reduction="mean",
        )
        return math.exp(loss.item())

    def compute_drift(self, z_current: torch.Tensor) -> float:
        """
        S_g = (1/W) Σ_{k=0..W-1}  ||z_{g-k} - z_{g-k-1}||² / (||z_{g-k-1}||² + ε)

        z: penultimate-layer mean-pooled hidden state of anchor set.
        Returns S_g (window-averaged normalized squared diff).
        """
        self.z_history.append(z_current.detach().clone())
        if len(self.z_history) < 2:
            return 0.0  # not enough history

        diffs = []
        eps = 1e-9
        # Pair consecutive z's: (z_{g-1}, z_g), (z_{g-2}, z_{g-1}), ...
        prev_list = list(self.z_history)[:-1]
        curr_list = list(self.z_history)[1:]
        for z_prev, z_curr in zip(prev_list[::-1], curr_list[::-1]):
            num = (z_curr - z_prev).pow(2).sum().item()
            den = z_prev.pow(2).sum().item() + eps
            diffs.append(num / den)
            if len(diffs) >= self.cfg.drift_window:
                break

        S_g = sum(diffs) / len(diffs)
        return S_g

    # ---------- Trust variable + Regulator ----------

    def trust_tau(self) -> float:
        """τ = max(0.2, min(1.0, H_g / H_0))."""
        if self.H_0 is None or not self.H_history:
            return 1.0
        H_g = self.H_history[-1]
        ratio = H_g / self.H_0 if self.H_0 > 0 else 1.0
        return max(self.cfg.trust_min, min(self.cfg.trust_max, ratio))

    # ---------- Main observation step ----------

    def observe(
        self,
        anchor_logits: torch.Tensor,
        anchor_target_ids: torch.Tensor,
        penultimate_hidden: torch.Tensor,
    ) -> CollapseAlert | None:
        """
        Update monitors with one generation's evidence on the anchor set.

        Args:
            anchor_logits: (anchor_size, T, V) — model logits on anchor set
            anchor_target_ids: (anchor_size, T) — gold token ids for PPL
            penultimate_hidden: (anchor_size, T, D) — penultimate hidden states

        Returns:
            CollapseAlert if signature detected, else None
        """
        self.generation += 1

        # Compute current generation's metrics
        H_g = self.compute_anchor_entropy(anchor_logits)
        PPL_g = self.compute_perplexity(anchor_logits, anchor_target_ids)
        # Mean-pool penultimate over anchor + sequence
        z_g = penultimate_hidden.mean(dim=(0, 1))  # (D,)
        S_g = self.compute_drift(z_g)

        self.H_history.append(H_g)
        self.PPL_history.append(PPL_g)
        self.drift_history.append(S_g)

        # Establish baselines after warmup
        if self.generation == self.cfg.warmup_steps:
            self.H_0 = sum(self.H_history) / len(self.H_history)
            self.PPL_0 = sum(self.PPL_history) / len(self.PPL_history)

        if self.H_0 is None:
            return None  # still in warmup

        # Compute ratios
        h_ratio = H_g / self.H_0 if self.H_0 > 0 else 1.0
        ppl_ratio = PPL_g / self.PPL_0 if self.PPL_0 > 0 else 1.0
        tau = self.trust_tau()

        # --- Silent collapse signature (paper §3): BOTH must hold ---
        h_collapsed = h_ratio < self.cfg.entropy_ratio_threshold
        ppl_normal = (self.cfg.ppl_ratio_min <= ppl_ratio <= self.cfg.ppl_ratio_max)
        if h_collapsed and ppl_normal:
            return CollapseAlert(
                precursor="silent_collapse",
                h_ratio=h_ratio, ppl_ratio=ppl_ratio, drift=S_g,
                update_step=self.generation, trust_tau=tau,
            )

        # --- Drift freeze (representation stopped moving) ---
        if S_g < self.cfg.drift_floor and self.generation > self.cfg.drift_window:
            return CollapseAlert(
                precursor="drift_freeze",
                h_ratio=h_ratio, ppl_ratio=ppl_ratio, drift=S_g,
                update_step=self.generation, trust_tau=tau,
            )

        return None


class Regulator:
    """Modulates learning intensity using the trust variable τ.

    Typical use in production fine-tuning: τ scales the LoRA learning rate
    on each accepted update. Low τ (entropy collapsing) → smaller updates,
    slower learning. This is the paper's "modulate effective learning
    intensity" — applied here to gradient scaling rather than the paper's
    synthetic-vs-real data mixing setting for recursive pretraining.
    """

    def __init__(self, monitor: SilentCollapseMonitor):
        self.monitor = monitor

    def get_lr_scale(self) -> float:
        """Returns multiplier for next update's learning rate."""
        return self.monitor.trust_tau()

    def should_pause(self, alert: CollapseAlert | None) -> bool:
        """Returns True if updates should pause (alert is severe)."""
        if alert is None:
            return False
        return alert.is_silent_collapse() or alert.precursor == "drift_freeze"


# ---------- Smoke test ----------

if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = MTRConfig(warmup_steps=3, entropy_ratio_threshold=0.5,
                    drift_floor=0.001)
    monitor = SilentCollapseMonitor(cfg)

    V = 1000  # vocab size
    A = 8     # anchor size
    T = 16    # seq len
    D = 64    # hidden dim

    # Generate "healthy" data for 5 steps
    print("Healthy training simulation...")
    for step in range(5):
        # Varied logits — high entropy
        logits = torch.randn(A, T, V) * 2.0
        targets = torch.randint(0, V, (A, T))
        # Drifting hidden states (model still learning)
        hidden = torch.randn(A, T, D) * (1.0 + 0.1 * step)
        alert = monitor.observe(logits, targets, hidden)
        assert alert is None, f"step {step}: unexpected alert {alert}"
        print(f"  step {step+1}: H={monitor.H_history[-1]:.3f}, "
              f"PPL={monitor.PPL_history[-1]:.1f}, S={monitor.drift_history[-1]:.4f}")

    print(f"\nBaselines: H_0={monitor.H_0:.3f}, PPL_0={monitor.PPL_0:.1f}")

    # Simulate silent collapse: peak the logits at the BASELINE argmax
    # → H drops, but CE on the SAME targets we used during baseline stays similar (PPL≈1).
    print("\nSilent collapse simulation (peak baseline-argmax, same targets)...")
    baseline_logits = torch.randn(A, T, V) * 2.0
    baseline_targets = baseline_logits.argmax(dim=-1)  # what the model "predicts"
    # Use those argmax tokens as evaluation targets — PPL_0 is exp(CE on argmax)
    monitor2 = SilentCollapseMonitor(cfg)
    for _ in range(5):
        # vary baseline logits but keep argmax-as-targets convention
        bl = torch.randn(A, T, V) * 2.0
        tg = bl.argmax(dim=-1)
        hidden = torch.randn(A, T, D)
        a = monitor2.observe(bl, tg, hidden)
        assert a is None
    print(f"  baselines H_0={monitor2.H_0:.3f}, PPL_0={monitor2.PPL_0:.3f}")

    # Now peak the logits (5×): H drops sharply, PPL on argmax stays low
    peaked = baseline_logits * 5.0
    hidden = torch.randn(A, T, D) * 0.0001  # frozen
    alert = monitor2.observe(peaked, baseline_targets, hidden)
    print(f"  observed H={monitor2.H_history[-1]:.3f}, PPL={monitor2.PPL_history[-1]:.3f}, drift={monitor2.drift_history[-1]:.6f}")
    if alert:
        print(f"  ALERT: {alert.precursor}")
        print(f"    H ratio: {alert.h_ratio:.3f}  (threshold {cfg.entropy_ratio_threshold})")
        print(f"    PPL ratio: {alert.ppl_ratio:.3f}  (window {cfg.ppl_ratio_min}-{cfg.ppl_ratio_max})")
        print(f"    Drift: {alert.drift:.6f}  (floor {cfg.drift_floor})")
        print(f"    Trust τ: {alert.trust_tau:.3f}")
        print(f"  Regulator lr_scale = {Regulator(monitor2).get_lr_scale():.3f}")
    else:
        print(f"  (no alert this step — H ratio = {monitor2.H_history[-1]/monitor2.H_0:.3f}, "
              f"PPL ratio = {monitor2.PPL_history[-1]/monitor2.PPL_0:.3f})")

    # Drift-freeze simulation: feed identical hidden states repeatedly
    print("\nDrift-freeze simulation (identical hidden states)...")
    monitor3 = SilentCollapseMonitor(MTRConfig(warmup_steps=3, drift_window=5,
                                                drift_floor=0.001))
    fixed_hidden = torch.randn(A, T, D)
    for step in range(8):
        # Healthy logits/targets
        bl = torch.randn(A, T, V) * 2.0
        tg = bl.argmax(dim=-1)
        a = monitor3.observe(bl, tg, fixed_hidden)  # always same hidden
        if a:
            print(f"  step {step+1}: ALERT {a.precursor}, drift={a.drift:.6f}")
            break
    else:
        print(f"  (no alert fired in 8 steps; drift was {monitor3.drift_history[-1]:.6f})")

    print("\nsilent_collapse smoke test passed (formulas execute correctly)")
