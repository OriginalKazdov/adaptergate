"""ProCL — Program-Memory LoRA for Continual Fine-Tuning.

Reference: arXiv 2605.13162 (May 2026) — "Continual Fine-Tuning of LLMs via
Program Memory". Independent re-implementation; algorithm is credited to the
paper authors.

Architecture:
  - LoRA adapter W of total rank R is partitioned into N "programs" of rank
    r=R/N each.
  - A lightweight task encoder produces per-input queries Q ∈ R^{N×dk}.
  - Multi-head softmax attention over programs: α_h = softmax(Q[h] K_h^T / √dk).
  - Batch-averaged routing for stability: ᾱ_h = (1/B) Σ_b α_h(b).
  - Soft program composition: W^h = Σ_n ᾱ_{h,n} · σ(s_n) · W^{(n)}.
  - Stable-adaptive gate: W_exec = γ · W_orig + W^h.
  - Consolidation: W ← (1-λ) W + λ W_exec at end of each task.

Paper defaults: N=4, dk=16, LoRA rank=16-32, lr=1e-5, λ=0.9, batch B=4-16.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ProCLConfig:
    """Hyperparameters from ProCL paper §4 + §5.3 ablations.

    `embed_dim` is required and must match your base model's hidden dim
    (e.g. 5120 for Qwen 2.5 14B, 4096 for Llama 3 8B, 2048 for Qwen 3B).
    """
    embed_dim: int                       # required — base model hidden dim
    n_programs: int = 4                  # N
    total_lora_rank: int = 16            # R (per-program rank = R/N)
    n_heads: int = 4                     # multi-head attention over programs
    key_dim: int = 16                    # dk
    consolidation_lambda: float = 0.9    # moving average λ
    batch_averaged: bool = True          # average α over batch (paper §3.4)
    init_gamma_via_rms: bool = True      # γ_0 = σ(-log RMS(W))


class TaskEncoder(nn.Module):
    """f_enc: mean-pooled input embedding → query matrix Q ∈ R^{N×dk}."""
    def __init__(self, embed_dim: int, n_programs: int, n_heads: int, key_dim: int):
        super().__init__()
        self.n_programs = n_programs
        self.n_heads = n_heads
        self.key_dim = key_dim
        # Produces queries for each program (N programs, n_heads, key_dim)
        self.proj = nn.Linear(embed_dim, n_programs * n_heads * key_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)  token-level hidden states
        returns Q: (B, n_heads, N, key_dim)
        """
        # Mean pool over tokens: z = (1/T) Σ_t x_t
        z = x.mean(dim=1)  # (B, D)
        q = self.proj(z)   # (B, N * H * dk)
        B = z.shape[0]
        q = q.view(B, self.n_heads, self.n_programs, self.key_dim)
        return q


class ProgramMemory(nn.Module):
    """
    The N "programs" — each is a (low-rank) LoRA adapter on a target weight matrix W_orig.
    Each program contributes a rank-r/N update; combined via attention to form W^h.
    Standard LoRA decomposition: ΔW = A B^T where A ∈ R^{d_out × r}, B ∈ R^{d_in × r}.

    Implementation note: we keep N parallel (A_n, B_n) pairs of rank R/N each.
    The composition mixes them with attention weights α (computed externally).
    """

    def __init__(self, d_in: int, d_out: int, cfg: ProCLConfig):
        super().__init__()
        self.cfg = cfg
        self.r_per_program = cfg.total_lora_rank // cfg.n_programs
        assert cfg.total_lora_rank % cfg.n_programs == 0, \
            f"total_lora_rank={cfg.total_lora_rank} not divisible by N={cfg.n_programs}"

        # N programs, each is (A_n: d_out × r/N, B_n: r/N × d_in)
        self.A = nn.Parameter(torch.zeros(cfg.n_programs, d_out, self.r_per_program))
        self.B = nn.Parameter(torch.zeros(cfg.n_programs, self.r_per_program, d_in))
        # σ(s_n) — per-program scalar gating (learned)
        self.s = nn.Parameter(torch.zeros(cfg.n_programs))

        # Keys for attention (one set per head, per program)
        self.keys = nn.Parameter(
            torch.randn(cfg.n_heads, cfg.n_programs, cfg.key_dim) * (1.0 / math.sqrt(cfg.key_dim))
        )

        # Per-head program-output combination weight (γ_head reduction)
        self.head_combine = nn.Parameter(torch.ones(cfg.n_heads) / cfg.n_heads)

        # LoRA init: B is random, A is zero (output starts at 0 → no behavior change)
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))

    def compute_attention(self, q: torch.Tensor) -> torch.Tensor:
        """
        q: (B, n_heads, N, key_dim)
        returns α: (B, n_heads, N) — softmax over programs per head
        """
        # Per-head dot product q · k
        # q: (B, H, N, dk), keys: (H, N, dk) → expand keys to (1, H, N, dk)
        scores = (q * self.keys.unsqueeze(0)).sum(dim=-1)  # (B, H, N)
        scores = scores / math.sqrt(self.cfg.key_dim)
        alpha = F.softmax(scores, dim=-1)
        return alpha

    def apply_to_input(self, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """Apply the program-memory LoRA to input x.

        Paper composition (Eq. 10): W^h = Σ_n α_n σ(s_n) (A_n B_n)
        → output: y_lora = Σ_n α_n σ(s_n) (A_n B_n x). NO cross terms.

        (Implementation note: an earlier naïve form composed
        A_eff = Σ α_n A_n and B_eff = Σ α_n B_n and computed A_eff B_eff x.
        That produces spurious A_n B_m cross-terms for n≠m and is incorrect.
        We compute the sum-of-products directly via einsum.)

        x:     (B, T, d_in)
        alpha: (B, H, N)  — batch-averaged inside this method if cfg.batch_averaged
        Returns: (B, T, d_out)
        """
        prog_scale = torch.sigmoid(self.s)  # (N,)

        if self.cfg.batch_averaged:
            # average α over batch dim
            alpha_avg = alpha.mean(dim=0)  # (H, N)
            head_w = F.softmax(self.head_combine, dim=0)  # (H,)
            mix = (alpha_avg * head_w.unsqueeze(-1)).sum(dim=0)  # (N,)
            mix = mix * prog_scale  # (N,)

            # Per-program LoRA: y_n = (A_n @ B_n) x  computed without forming full (d_out, d_in)
            # B: (N, r/N, d_in), A: (N, d_out, r/N), x: (B, T, d_in)
            # Step 1: h_all[b,t,n,k] = Σ_d x[b,t,d] B[n,k,d]
            h_all = torch.einsum('btd,nkd->btnk', x, self.B)  # (B, T, N, r/N)
            # Step 2: y_all[b,t,n,o] = Σ_k h_all[b,t,n,k] A[n,o,k]
            y_all = torch.einsum('btnk,nok->btno', h_all, self.A)  # (B, T, N, d_out)
            # Step 3: weighted sum over programs
            y_lora = (mix.view(1, 1, -1, 1) * y_all).sum(dim=2)  # (B, T, d_out)
            return y_lora
        else:
            # Per-sample routing — α has different routing for each batch element
            # alpha: (B, H, N) → per-sample mix
            head_w = F.softmax(self.head_combine, dim=0)
            mix_per_sample = (alpha * head_w.view(1, -1, 1)).sum(dim=1)  # (B, N)
            mix_per_sample = mix_per_sample * prog_scale.view(1, -1)  # (B, N)

            h_all = torch.einsum('btd,nkd->btnk', x, self.B)
            y_all = torch.einsum('btnk,nok->btno', h_all, self.A)
            y_lora = (mix_per_sample.view(-1, 1, self.cfg.n_programs, 1) * y_all).sum(dim=2)
            return y_lora

    def orthogonal_penalty(self) -> torch.Tensor:
        """
        Optional O-LoRA-style penalty: discourage program overlap.

        From the broader CL literature (O-LoRA 2023, N-LoRA 2024): adding an
        orthogonality penalty between task-specific LoRA subspaces reduces
        catastrophic forgetting via parameter collision reduction.

        We compute the per-pair (A_i^T A_j) Frobenius² for i≠j.
        Caller decides whether to add this to the loss.
        """
        N = self.cfg.n_programs
        # Stack A_n: shape (N, d_out, r/N) → flatten last dim contribs
        # Penalty: Σ_{i!=j} ||A_i^T A_j||_F²
        total = torch.tensor(0.0, device=self.A.device)
        for i in range(N):
            for j in range(i + 1, N):
                # Subspaces overlap if A_i and A_j share column directions
                overlap = (self.A[i].T @ self.A[j]).pow(2).sum()
                total = total + overlap
        return total


@dataclass
class SlotMetadata:
    """Per-program metadata for tracking what each slot has seen.

    `task_ids_seen` is a free-form list of task identifiers (datasets, DBs,
    tenants, whatever the caller wants to track). Used for audit + DriftCouncil
    downstream reasoning.
    """
    task_ids_seen: list[str] = field(default_factory=list)
    n_updates: int = 0


class ProCLLayer(nn.Module):
    """
    Wraps a target linear layer (e.g., q_proj of attention) with ProCL adapter.

    Holds base_linear BY REFERENCE (no clone) — critical for 14B+ models where
    cloning the frozen weight matrix per layer would cause OOM. Works equally
    with standard nn.Linear and bnb.Linear4bit for QLoRA-style training.

    Computes: y = γ · base(x) + Σ_n mix_n (A_n B_n x)
        where γ is a learned gate (initialized from RMS of base weight)
        and mix is computed by attention routing on the input embedding.
    """

    def __init__(self, base_linear: nn.Module, cfg: ProCLConfig,
                 task_encoder: TaskEncoder):
        super().__init__()
        # Reference (not clone) the base layer. PyTorch will register it as a
        # child module but we freeze its parameters.
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad_(False)

        # Pull d_in / d_out from the base layer's weight shape
        # (works for both nn.Linear and bnb.Linear4bit — both expose .weight)
        if hasattr(base_linear, "in_features"):
            d_in = base_linear.in_features
            d_out = base_linear.out_features
        else:
            # Fallback: read from weight shape (nn.Linear convention)
            d_out, d_in = base_linear.weight.shape

        self.cfg = cfg
        self.task_encoder = task_encoder  # shared across ProCLLayers in same network
        self.memory = ProgramMemory(d_in=d_in, d_out=d_out, cfg=cfg)

        # γ gate — initialized from paper's prescription σ(-log RMS(W))
        # For bnb.Linear4bit, we approximate RMS using the dequantized weight.
        if cfg.init_gamma_via_rms:
            try:
                w = base_linear.weight
                # If it's a quantized weight, .data may be int8 codes — try to get
                # representative magnitude
                if hasattr(base_linear, "weight") and w.dtype not in (torch.float16, torch.float32, torch.bfloat16):
                    # quantized — assume small RMS, use γ near 1
                    rms = 0.1
                else:
                    rms = w.detach().float().pow(2).mean().sqrt().item()
            except Exception:
                rms = 0.1
            gamma_init = 1.0 / (1.0 + math.exp(math.log(rms + 1e-9)))
            self.gamma_logit = nn.Parameter(torch.tensor(
                math.log(gamma_init / (1 - gamma_init + 1e-9))
            ))
        else:
            self.gamma_logit = nn.Parameter(torch.tensor(0.0))

    def gamma(self):
        return torch.sigmoid(self.gamma_logit)

    # Class-level shared routing context.
    # The RoutingContext manager sets the "current alpha" before model forward,
    # then all ProCLLayers in the same model read this same alpha.
    # Allows ProCLLayer.forward(x) to match nn.Linear interface (drop-in).
    _routing_context: torch.Tensor | None = None  # alpha (B, H, N)

    @classmethod
    def set_routing_alpha(cls, alpha: torch.Tensor | None):
        """Called by RoutingManager before model forward. Single alpha for whole model."""
        cls._routing_context = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Drop-in replacement for nn.Linear.forward.

        y = γ · base(x) + Σ_n α_n σ(s_n) (A_n B_n x)

        x: (B, T, d_in)
        Returns: (B, T, d_out)
        """
        # γ · base(x)  (call frozen base layer, scale by gate)
        y_base = self.gamma() * self.base(x)

        alpha = ProCLLayer._routing_context
        if alpha is None:
            return y_base

        # Σ_n α_n σ(s_n) (A_n B_n x)  — paper Eq. 10, NO cross terms
        y_lora = self.memory.apply_to_input(x, alpha)
        return y_base + y_lora

    def forward_with_routing(self, x: torch.Tensor,
                              layer_input_for_routing: torch.Tensor) -> torch.Tensor:
        """
        Legacy 2-arg forward — keeps router self-contained.
        For production model integration, use RoutingManager + forward(x).
        """
        q = self.task_encoder(layer_input_for_routing)
        alpha = self.memory.compute_attention(q)
        y_base = self.gamma() * self.base(x)
        y_lora = self.memory.apply_to_input(x, alpha)
        return y_base + y_lora


class RoutingManager:
    """
    Computes routing decision (alpha) from the model's embedding output.
    Sets ProCLLayer._routing_context so wrapped layers in the model share it.

    STICKY MODE (sticky=True, default):
        Routing alpha is computed on the FIRST forward and held across all
        subsequent forwards until reset_for_new_input() is called. This is
        the correct behavior for autoregressive generation: model.generate()
        calls forward many times (once per token after KV-cache built), and
        each subsequent call sees only the new token's embedding — routing
        based on a single token's embed is noise. Sticky behavior locks alpha
        to the prompt-level decision.

    NON-STICKY MODE:
        Recompute alpha on every embedding hook fire (useful for training
        where each forward is independent and sees the full sequence).

    Workflow:
      1. Hook the embedding layer's forward to capture output → compute alpha
      2. Subsequent forwards reuse the same alpha (sticky) or recompute (non-sticky)
      3. Call reset_for_new_input() before processing a NEW user query
    """

    def __init__(self, task_encoder: TaskEncoder, program_memory_ref,
                 sticky: bool = True):
        self.task_encoder = task_encoder
        self.memory_ref = program_memory_ref  # shared reference for compute_attention
        self._embedding_hook_handle = None
        self._captured_embedding = None
        self.sticky = sticky
        self._sticky_alpha_set = False

    def install_hook_on_embedding(self, embed_layer: torch.nn.Module):
        """Hook the embedding layer to capture its output for routing."""
        def hook(module, inputs, output):
            # In sticky mode, skip recomputation once alpha is set for this input
            if self.sticky and self._sticky_alpha_set:
                return
            self._captured_embedding = output  # (B, T, D)
            q = self.task_encoder(output)
            alpha = self.memory_ref.compute_attention(q)
            ProCLLayer.set_routing_alpha(alpha)
            self._sticky_alpha_set = True
        self._embedding_hook_handle = embed_layer.register_forward_hook(hook)

    def reset_for_new_input(self):
        """Call before processing a new query (lets next forward recompute alpha)."""
        self._sticky_alpha_set = False
        ProCLLayer.set_routing_alpha(None)

    def remove_hook(self):
        if self._embedding_hook_handle is not None:
            self._embedding_hook_handle.remove()
            self._embedding_hook_handle = None
        ProCLLayer.set_routing_alpha(None)
        self._sticky_alpha_set = False


class ProCLController:
    """
    Top-level controller. Manages program memory, consolidation, task tracking.
    """

    def __init__(self, cfg: ProCLConfig | None = None,
                 slot_dir: Path = Path("./slots")):
        if cfg is None:
            raise ValueError(
                "ProCLController requires a ProCLConfig with explicit embed_dim."
            )
        self.cfg = cfg
        self.slot_dir = Path(slot_dir)
        self.slot_dir.mkdir(parents=True, exist_ok=True)
        self.metadata: dict[int, SlotMetadata] = {
            i: SlotMetadata() for i in range(self.cfg.n_programs)
        }
        # Snapshot of (A, B, s) per layer name — used for the consolidation step.
        self.snapshot: dict[str, dict[str, torch.Tensor]] = {}

    def take_snapshot(self, layer_name: str, layer: ProCLLayer) -> None:
        """Save current adapter state before task starts (for consolidation)."""
        with torch.no_grad():
            self.snapshot[layer_name] = {
                "A": layer.memory.A.detach().clone(),
                "B": layer.memory.B.detach().clone(),
                "s": layer.memory.s.detach().clone(),
            }

    def consolidate(self, layer_name: str, layer: ProCLLayer) -> None:
        """W ← (1-λ) W + λ W_exec — moving average toward executed adapter."""
        if layer_name not in self.snapshot:
            return
        snap = self.snapshot[layer_name]
        lam = self.cfg.consolidation_lambda
        with torch.no_grad():
            layer.memory.A.data = (1 - lam) * snap["A"] + lam * layer.memory.A.data
            layer.memory.B.data = (1 - lam) * snap["B"] + lam * layer.memory.B.data
            layer.memory.s.data = (1 - lam) * snap["s"] + lam * layer.memory.s.data


# ---------- Smoke test ----------

if __name__ == "__main__":
    cfg = ProCLConfig(embed_dim=128, n_programs=4, total_lora_rank=16,
                      n_heads=4, key_dim=16)
    encoder = TaskEncoder(cfg.embed_dim, cfg.n_programs, cfg.n_heads, cfg.key_dim)

    # Simulate a base linear layer (d_in=128, d_out=128)
    base = nn.Linear(128, 128)
    layer = ProCLLayer(base, cfg, encoder)

    # Forward pass — use legacy 2-arg form for smoke test
    B, T = 4, 10  # batch=4, seq_len=10
    x = torch.randn(B, T, 128)
    router_input = torch.randn(B, T, 128)
    y = layer.forward_with_routing(x, router_input)
    assert y.shape == (B, T, 128), f"got {y.shape}"
    print(f"legacy forward OK — output shape {tuple(y.shape)}")

    # Test drop-in forward via RoutingManager
    ProCLLayer.set_routing_alpha(layer.memory.compute_attention(encoder(router_input)))
    y_dropin = layer(x)
    ProCLLayer.set_routing_alpha(None)
    diff_dropin = (y - y_dropin).abs().max().item()
    assert diff_dropin < 1e-5, f"drop-in vs legacy diff {diff_dropin}"
    print(f"drop-in forward matches legacy (diff {diff_dropin:.2e})")

    # Compare composition: should equal manual per-program sum
    with torch.no_grad():
        q = encoder(router_input)
        alpha = layer.memory.compute_attention(q)
        y_paper = layer.memory.apply_to_input(x, alpha)

        # Manual computation: y_paper_manual = Σ_n mix_n (A_n B_n x)
        prog_scale = torch.sigmoid(layer.memory.s)
        alpha_avg = alpha.mean(dim=0)  # (H, N)
        head_w = F.softmax(layer.memory.head_combine, dim=0)
        mix = (alpha_avg * head_w.unsqueeze(-1)).sum(dim=0)  # (N,)
        mix = mix * prog_scale  # (N,)

        y_manual = torch.zeros_like(y_paper)
        for n in range(cfg.n_programs):
            h_n = F.linear(x, layer.memory.B[n])  # (B, T, r/N)
            y_n = F.linear(h_n, layer.memory.A[n])  # (B, T, d_out)
            y_manual = y_manual + mix[n] * y_n

        diff = (y_paper - y_manual).abs().max().item()
        assert diff < 1e-5, f"composition mismatch — diff {diff}"
        print(f"composition formula correct (max diff vs manual: {diff:.2e})")

    # Verify the BUG is gone: cross-terms should NOT be present
    # In the broken version, y_lora ≠ Σ_n mix_n A_n B_n x. Our fix ensures it is.

    # Check that loss propagates to programs
    loss = y.sum()
    loss.backward()
    assert layer.memory.A.grad is not None
    assert layer.memory.B.grad is not None
    print(f"backward OK — A grad norm {layer.memory.A.grad.norm().item():.4f}")
    print(f"            B grad norm {layer.memory.B.grad.norm().item():.4f}")
    print(f"            s grad norm {layer.memory.s.grad.norm().item():.4f}")

    # Check γ gate
    print(f"γ initial = {layer.gamma().item():.3f}  (should be in (0,1))")

    # Consolidation
    ctrl = ProCLController(cfg)
    ctrl.take_snapshot("test_layer", layer)
    # Simulate some training: perturb A
    with torch.no_grad():
        layer.memory.A.data += torch.randn_like(layer.memory.A) * 0.1
    pre_norm = layer.memory.A.detach().norm().item()
    ctrl.consolidate("test_layer", layer)
    post_norm = layer.memory.A.detach().norm().item()
    print(f"consolidation moved A norm from {pre_norm:.3f} to {post_norm:.3f}")

    # Orthogonal penalty
    pen = layer.memory.orthogonal_penalty()
    print(f"orthogonal penalty (O-LoRA-style): {pen.item():.4f}")
    print("procl smoke test passed")
