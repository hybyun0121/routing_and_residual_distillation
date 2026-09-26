"""Loss utilities for chain shared-expert distillation.

Used by DistillationTrainer in chain mode (cfg.distill_chain_shared_expert=True).

Per-layer contract: trainer captures, on each forward step,
    block._chain_capture = {
        "router_logits": [B*S, E] post-softmax probs,
        "moe_out":       [B*S, H] frozen-experts top-K weighted sum,
        "shared_out":    [B*S, H] trainable shared expert output,
        "input":         [B*S, H] LN2(h_student),
        ...
    }
and external hooks capture
    teacher_mlp_outputs[ℓ]:      [B, S, H] teacher dense MLP output
    teacher_post_attn_ln[ℓ]:     [B, S, H] teacher's LN2(h_teacher)

The functions below take these tensors and return scalar losses or per-token diagnostics.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _experts_num(experts_module: torch.nn.Module) -> int:
    """Return number of experts. Supports Qwen3MoeExperts (.num_experts attr) and ModuleList."""
    if hasattr(experts_module, "num_experts"):
        return int(experts_module.num_experts)
    return len(experts_module)


def _experts_per_token_outputs(
    experts_module: torch.nn.Module,
    hs: torch.Tensor,                # [N, H]
    skip_none: bool = False,
) -> torch.Tensor:
    """Compute per-expert output for every token, returning [E, N, H].

    Dispatches on module structure:
    - Qwen3MoeExperts: bulk gate_up_proj/down_proj weight tensors
    - ModuleList[LlamaMLP] (RRD-style rrd_moe_llama.MoE.experts): per-expert callable
    - ModuleList with possible None slots (CMoE MoE.experts): set
      ``skip_none=True`` to drop None slots — output dim is then the count of
      non-None slots (filled-routed-expert count).
    """
    if hasattr(experts_module, "gate_up_proj") and hasattr(experts_module, "down_proj"):
        E = _experts_num(experts_module)
        outs = []
        for e in range(E):
            gp, up = F.linear(hs, experts_module.gate_up_proj[e]).chunk(2, dim=-1)
            inner = experts_module.act_fn(gp) * up
            out_e = F.linear(inner, experts_module.down_proj[e])
            outs.append(out_e)
        return torch.stack(outs, dim=0)
    E = _experts_num(experts_module)
    if skip_none:
        outs = [experts_module[e](hs) for e in range(E) if experts_module[e] is not None]
    else:
        outs = [experts_module[e](hs) for e in range(E)]
    return torch.stack(outs, dim=0)


def magnitude_oracle_topk(
    experts_module: torch.nn.Module,
    teacher_post_attn_ln: torch.Tensor,  # [B, S, H]
    k: int,
) -> torch.Tensor:
    """Compute magnitude top-K oracle target per token using FROZEN experts on teacher input.

    Returns:
        target_idx: [B*S, k] long tensor of expert indices (oracle K-subset).
    """
    B, S, H = teacher_post_attn_ln.shape
    hs = teacher_post_attn_ln.reshape(-1, H)  # [B*S, H]
    with torch.no_grad():
        stacked = _experts_per_token_outputs(experts_module, hs)  # [E, B*S, H]
        norms = stacked.float().norm(dim=-1)                       # [E, B*S]
        _, target_idx = norms.topk(k, dim=0)                       # [k, B*S]
    return target_idx.T.contiguous()                                # [B*S, k]


def magnitude_oracle_topk_with_weights(
    experts_module: torch.nn.Module,
    hidden_input: torch.Tensor,  # [B, S, H] or [N, H]
    k: int,
    weighting: str = "magnitude_softmax",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Magnitude top-K oracle with per-token weighting for forward combination.

    Used by Phase 1 of 2-phase EM training: oracle picks K experts and supplies weights
    for the MoE forward combination (replaces gate's routing_weights).

    Args:
        experts_module: student's frozen Qwen3MoeExperts (random init magnitudes).
        hidden_input: hidden state to feed experts (teacher's LN2(h) for "teacher" mode,
                      student's for "student" mode). Shape [B,S,H] or [N,H].
        k: top-K.
        weighting: "raw" (1/k uniform) or "magnitude_softmax" (softmax over K magnitudes).

    Returns:
        target_idx: [N, k] long, oracle expert indices.
        weights:    [N, k] float, sum to 1 per token.
    """
    if hidden_input.dim() == 3:
        B, S, H = hidden_input.shape
        hs = hidden_input.reshape(-1, H)
    else:
        hs = hidden_input
    with torch.no_grad():
        stacked = _experts_per_token_outputs(experts_module, hs)   # [E, N, H]
        norms = stacked.float().norm(dim=-1)                        # [E, N]
        topk_norms, topk_idx = norms.topk(k, dim=0)                 # [k, N]
        target_idx = topk_idx.T.contiguous()                        # [N, k]
        if weighting == "raw":
            w = torch.full_like(target_idx, 0, dtype=torch.float32) + (1.0 / k)
        elif weighting == "magnitude_softmax":
            w = torch.softmax(topk_norms.T.contiguous().float(), dim=-1)  # [N, k]
        else:
            raise ValueError(f"unknown weighting={weighting!r}")
    return target_idx, w.to(hs.dtype)


def loss_router_topk_ce(
    student_router_logits: torch.Tensor,  # [B*S, E] post-softmax probs OR raw logits
    target_idx: torch.Tensor,             # [B*S, k] long
    k: int,
    is_softmax: bool = True,
) -> torch.Tensor:
    """Multi-hot top-K cross-entropy on oracle target.

    L = -(1/K) * sum_{e in S*} log p_θ_e, averaged over tokens (with no sample_weight).
    """
    if is_softmax:
        # Already post-softmax probs; clamp + log.
        log_probs = student_router_logits.float().clamp_min(1e-12).log()
    else:
        log_probs = F.log_softmax(student_router_logits.float(), dim=-1)
    # Gather log-probs at the K oracle indices per token.
    target_log_probs = log_probs.gather(-1, target_idx)  # [B*S, k]
    return -target_log_probs.mean()


def loss_router_distribution_kl(
    student_router_logits: torch.Tensor,  # [N, E] post-softmax probs OR raw logits
    target_mass: torch.Tensor,            # [N, E] non-negative dense-teacher mass
    is_softmax: bool = True,
    target_temperature: float = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Cross-entropy term for KL(q || p) with a full router target distribution.

    The target distribution is q = softmax(log(mass + eps) / tau). With tau=1,
    this is equivalent to normalizing activation mass over routed experts. The
    target entropy term is constant with respect to the student and is omitted.
    """
    if is_softmax:
        log_p = student_router_logits.float().clamp_min(eps).log()
    else:
        log_p = F.log_softmax(student_router_logits.float(), dim=-1)
    mass = target_mass.float().clamp_min(eps)
    target_logits = mass.log()
    if target_temperature != 1.0:
        target_logits = target_logits / float(target_temperature)
    q = torch.softmax(target_logits, dim=-1)
    return -(q * log_p).sum(dim=-1).mean()


def _reduce_squared_error(
    diff: torch.Tensor,
    reduction: str,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reduce element-wise squared error per the given reduction strategy.

    diff shape: [N, H] (already flattened over batch + seq).
    - ``mean_all``: mean over all N*H elements (= F.mse_loss default). Magnitude
      ~ avg per-element squared error.
    - ``per_token_l2``: sum over hidden dim H per token, then mean over tokens.
      Result is H-times larger than ``mean_all``; gradient per element also
      H-times stronger.
    - ``global_rmse``: sqrt(mean_all). Square root of MSE → "global RMSE".
      Magnitude ~ sqrt(avg squared error) ≈ typical error magnitude.
    - ``per_token_l2_norm``: mean over tokens of sqrt(sum_h diff²) =
      mean of per-token L2-norm of diff. Magnitude ~ token-level Euclidean
      distance.
    """
    sq = diff * diff
    if sample_weight is not None:
        weight = sample_weight.reshape(-1).to(device=sq.device, dtype=sq.dtype)
        if weight.numel() != sq.shape[0]:
            raise ValueError(
                f"sample_weight length {weight.numel()} does not match token count {sq.shape[0]}"
            )
        denom = weight.sum().clamp_min(1e-12)
        per_token_mse = sq.mean(dim=-1)
        if reduction == "mean_all":
            return (per_token_mse * weight).sum() / denom
        if reduction == "per_token_l2":
            return (sq.sum(dim=-1) * weight).sum() / denom
        if reduction == "global_rmse":
            return ((per_token_mse * weight).sum() / denom).clamp_min(1e-12).sqrt()
        if reduction == "per_token_l2_norm":
            return (sq.sum(dim=-1).clamp_min(1e-12).sqrt() * weight).sum() / denom
        raise ValueError(f"unknown reduction={reduction!r}")
    if reduction == "mean_all":
        return sq.mean()
    if reduction == "per_token_l2":
        return sq.sum(dim=-1).mean()
    if reduction == "global_rmse":
        return sq.mean().clamp_min(1e-12).sqrt()
    if reduction == "per_token_l2_norm":
        return sq.sum(dim=-1).clamp_min(1e-12).sqrt().mean()
    raise ValueError(f"unknown reduction={reduction!r}")


def loss_shared_residual(
    shared_out: torch.Tensor,          # [B*S, H] or [B, S, H]
    teacher_mlp_out: torch.Tensor,     # same shape
    moe_out_detached: torch.Tensor,    # same shape, with .detach() applied
    reduction: str = "mean_all",
) -> torch.Tensor:
    """Option A: ‖shared_out - (teacher_mlp - moe_out.detach())‖² (MSE)."""
    if teacher_mlp_out.dim() == 3:
        teacher_mlp_out = teacher_mlp_out.reshape(-1, teacher_mlp_out.shape[-1])
    if moe_out_detached.dim() == 3:
        moe_out_detached = moe_out_detached.reshape(-1, moe_out_detached.shape[-1])
    if shared_out.dim() == 3:
        shared_out = shared_out.reshape(-1, shared_out.shape[-1])
    target = (teacher_mlp_out - moe_out_detached).to(shared_out.dtype)
    diff = shared_out - target
    return _reduce_squared_error(diff, reduction)


def loss_shared_joint(
    shared_out: torch.Tensor,          # [B*S, H] or [B, S, H]
    moe_out: torch.Tensor,             # same shape (NOT detached — gradient flows through)
    teacher_mlp_out: torch.Tensor,     # same shape
    reduction: str = "mean_all",
    cast_fp32: bool = False,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Option B: ‖(moe_out + shared_out) - teacher_mlp‖² (MSE/RMSE/...).

    For the CMoE-carved RRD trajectory's RMSE shared loss form, pass
    ``reduction='global_rmse'`` (computed by ``_reduce_squared_error`` as
    ``sqrt(mean(diff**2) + 1e-12)``) — see the CMoE-carved RRD plan
    (``2026-05-05_cmoe_carved_rrd_replace_lora_plan.md``).

    ``cast_fp32=True`` casts pred/target to fp32 before computing diff² and the
    reduction, avoiding bf16 mantissa truncation on small (~1e-3) values whose
    squares (~1e-6) and 8M-element means accumulate noise. Gradient still flows
    back to the bf16 parameters via the cast (PyTorch autograd handles dtype
    promotion). Default False preserves existing behavior.
    """
    if teacher_mlp_out.dim() == 3:
        teacher_mlp_out = teacher_mlp_out.reshape(-1, teacher_mlp_out.shape[-1])
    if moe_out.dim() == 3:
        moe_out = moe_out.reshape(-1, moe_out.shape[-1])
    if shared_out.dim() == 3:
        shared_out = shared_out.reshape(-1, shared_out.shape[-1])
    if cast_fp32:
        pred = moe_out.float() + shared_out.float()
        diff = pred - teacher_mlp_out.float()
    else:
        pred = moe_out + shared_out
        diff = pred - teacher_mlp_out.to(pred.dtype)
    return _reduce_squared_error(diff, reduction, sample_weight=sample_weight)


def topk_hit_rate(
    student_router_logits: torch.Tensor,  # [B*S, E] (logits or probs OK; argmax invariant)
    target_idx: torch.Tensor,             # [B*S, k] long
    k: int,
) -> torch.Tensor:
    """Diagnostic: fraction of tokens where student's top-K subset == oracle's top-K subset (set match)."""
    _, student_topk = student_router_logits.float().topk(k, dim=-1)  # [B*S, k]
    stu_sorted = student_topk.sort(dim=-1).values
    tgt_sorted = target_idx.sort(dim=-1).values
    hit = (stu_sorted == tgt_sorted).all(dim=-1)
    return hit.float().mean()


def magnitude_oracle_topk_with_norms(
    experts_module: torch.nn.Module,
    teacher_post_attn_ln: torch.Tensor,  # [B, S, H]
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Magnitude top-K oracle target + per-token magnitudes for the K oracle experts.

    Returns:
        target_idx:   [B*S, k] long  — oracle expert indices (descending magnitude order)
        target_norms: [B*S, k] float — magnitudes of those K experts (matching column order)

    Used by `loss_router_magnitude_weighted_kl` to build a magnitude-weighted soft target.
    """
    B, S, H = teacher_post_attn_ln.shape
    hs = teacher_post_attn_ln.reshape(-1, H)
    with torch.no_grad():
        stacked = _experts_per_token_outputs(experts_module, hs)   # [E, B*S, H]
        norms = stacked.float().norm(dim=-1)                        # [E, B*S]
        topk_norms, topk_idx = norms.topk(k, dim=0)                 # [k, B*S]
    return topk_idx.T.contiguous(), topk_norms.T.contiguous()


def magnitude_oracle_topk_frozen_routed(
    experts_module: torch.nn.Module,
    student_post_attn_ln: torch.Tensor,  # [B, S, H]
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Magnitude top-K oracle from a FROZEN routed-experts module on student input.

    Variant of :func:`magnitude_oracle_topk_with_norms` where:

    1. ``experts_module`` is the frozen oracle's routed-experts ModuleList — caller
       is responsible for supplying a deep-copied, ``requires_grad=False``,
       ``eval()`` module captured at training start (notepad §RRD Loss in CMoE:
       "Teacher = MoE all-active oracle model"). Shared experts are NOT included
       — magnitude is measured only over routed slots.
    2. Input ``student_post_attn_ln`` is the *student*'s LN2(h) (post-attn LN
       output, the input to MLP), not the teacher's. The student/oracle weights
       coincide at step 0 but diverge as the student trains.
    3. ``skip_none=True`` so CMoE-style ModuleList containing ``None`` placeholder
       slots (when filled-routed-count < ``len(ModuleList)``) does not crash.
       The returned indices are the **dense routed-slot index** (0..n_filled-1)
       — caller may need to remap to the original slot id if non-contiguous.
       For S3A3E8 / S2A2E8 builds in this codebase the routed slots ARE contiguous
       0..n_routed-1, so the remap is identity.

    Returns:
        target_idx:   [B*S, k] long  — oracle routed-expert indices (desc magnitude order)
        target_norms: [B*S, k] float — magnitudes of those K routed experts
    """
    B, S, H = student_post_attn_ln.shape
    hs = student_post_attn_ln.reshape(-1, H)
    with torch.no_grad():
        stacked = _experts_per_token_outputs(experts_module, hs, skip_none=True)
        norms = stacked.float().norm(dim=-1)                        # [n_routed_filled, B*S]
        topk_norms, topk_idx = norms.topk(k, dim=0)                 # [k, B*S]
    return topk_idx.T.contiguous(), topk_norms.T.contiguous()


def loss_router_magnitude_weighted_kl(
    student_router_logits: torch.Tensor,    # [N, E] post-softmax probs OR raw logits
    target_idx: torch.Tensor,               # [N, K] long
    target_norms: torch.Tensor,             # [N, K] float — magnitudes of oracle K experts
    k: int,
    is_softmax: bool = True,
    target_temperature: float = 1.0,
) -> torch.Tensor:
    """KL(q* || p_θ) where q* = softmax(target_norms / τ over the K oracle indices), 0 elsewhere.

    For each token:
        q*_e = softmax(target_norms / τ)[i] for e = target_idx[i],   i = 0..k-1
             = 0 otherwise
        L = -sum_{i=0..k-1} q*_target_idx[i] * log p_θ(target_idx[i] | h)

    KL(q* || p_θ) = -H(q*) - E_q*[log p_θ]; the -H(q*) part is constant w.r.t. θ
    (q* depends only on oracle norms, no gradient through it). Implementation
    drops that constant — minimizing -E_q*[log p_θ] is equivalent up to a constant
    for gradient purposes.

    Args:
        target_temperature: τ in `softmax(norms / τ)`. τ=1.0 (default) reproduces
            the original behaviour. τ<1.0 sharpens q* toward a one-hot of the
            largest target_norm; τ>1.0 flattens q* toward uniform over the K
            oracle indices. Set τ→0 to approximate hard top-1 CE on the largest
            magnitude oracle expert.

    The mean over tokens is taken (no sample weighting).
    """
    if is_softmax:
        log_p = student_router_logits.float().clamp_min(1e-12).log()
    else:
        log_p = F.log_softmax(student_router_logits.float(), dim=-1)
    norms = target_norms.float()
    if target_temperature != 1.0:
        norms = norms / float(target_temperature)
    q_subset = torch.softmax(norms, dim=-1)                   # [N, K], sums to 1 per token
    target_log_p = log_p.gather(-1, target_idx)               # [N, K]
    per_token = -(q_subset * target_log_p).sum(dim=-1)        # [N]
    return per_token.mean()


def topk_hit_fraction(
    student_router_logits: torch.Tensor,  # [N, E]
    target_idx: torch.Tensor,             # [N, K] long
    k: int,
) -> torch.Tensor:
    """Per-token mean correct fraction = |stu ∩ tgt| / K, averaged over tokens.

    Same definition as router_init._hit_rate_check (calibration metric).
    Intermediate strictness: random expectation = K/E (= 0.5 for K=4/E=8),
    between hit_rate (set match, random=0.0143) and jaccard (random=0.333).
    """
    N, E = student_router_logits.shape
    _, student_topk = student_router_logits.float().topk(k, dim=-1)  # [N, K]
    tgt_mask = torch.zeros(N, E, dtype=torch.bool, device=student_router_logits.device)
    tgt_mask.scatter_(-1, target_idx, True)
    hits = tgt_mask.gather(-1, student_topk).sum(dim=-1).float()  # [N]
    return (hits / float(k)).mean()


def topk_jaccard(
    student_router_logits: torch.Tensor,  # [N, E] (logits or probs OK; argtopk invariant)
    target_idx: torch.Tensor,             # [N, K] long
    k: int,
) -> torch.Tensor:
    """Per-token Jaccard = |stu ∩ tgt| / |stu ∪ tgt|, averaged over tokens.

    For K=4, E=8: random expectation ≈ 4/12 = 0.333 (compare hit_rate random ≈ 0.0143).
    """
    N, E = student_router_logits.shape
    _, student_topk = student_router_logits.float().topk(k, dim=-1)  # [N, K]
    # Build [N, E] binary masks for student and target.
    stu_mask = torch.zeros(N, E, dtype=torch.bool, device=student_router_logits.device)
    tgt_mask = torch.zeros(N, E, dtype=torch.bool, device=student_router_logits.device)
    stu_mask.scatter_(-1, student_topk, True)
    tgt_mask.scatter_(-1, target_idx, True)
    intersection = (stu_mask & tgt_mask).sum(dim=-1).float()
    union = (stu_mask | tgt_mask).sum(dim=-1).float()
    # Both are non-degenerate when k <= E and target_idx has unique entries (which the topk
    # implementation guarantees), so union >= k and intersection <= k. union > 0 always.
    return (intersection / union.clamp_min(1)).mean()


def loss_router_renormalized_topk_ce(
    student_router_logits: torch.Tensor,    # [N, E]; either post-softmax probs OR raw logits
    target_idx: torch.Tensor,               # [N, K] long
    k: int,
    is_softmax: bool = True,
) -> torch.Tensor:
    """A3: log-softmax restricted to oracle K — -mean(log p_e for e in S*).

    softmax_K(z_i) = exp(z_i) / sum_{j in S*} exp(z_j)  for i in S*
    L = -(1/K) sum_{i=0..k-1} log softmax_K(z_target_idx[i])

    Eliminates the multi_hot_ce 1/K cap by restricting normalization to oracle K.
    Min L = log K (perfect alignment within S*) is the same as multi_hot_ce, BUT the
    gradient does not push probability mass onto non-oracle indices, so non-oracle
    experts can stay arbitrarily small without penalty.
    """
    if is_softmax:
        # Post-softmax probs → log to recover logits-equivalent rankings.
        log_p_full = student_router_logits.float().clamp_min(1e-12).log()
    else:
        # Raw logits → use logsumexp directly via log_softmax (same as below path)
        log_p_full = F.log_softmax(student_router_logits.float(), dim=-1)
    # Take logits at oracle indices, then log_softmax over K (subtract logsumexp).
    target_logits = log_p_full.gather(-1, target_idx)  # [N, K]
    log_renorm = target_logits - target_logits.logsumexp(dim=-1, keepdim=True)  # [N, K]
    return -log_renorm.mean()


def loss_router_pairwise_hinge(
    student_router_logits: torch.Tensor,    # [N, E]
    target_idx: torch.Tensor,               # [N, K] long
    k: int,
    margin: float = 1.0,
    is_softmax: bool = True,
) -> torch.Tensor:
    """B1: pairwise hinge — for each (e in S*, e' not in S*), max(0, margin + z_e' - z_e).

    Mean over (K * (E - K)) pairs per token, then mean over tokens.

    If input is post-softmax probs, takes log to recover logits (rank-preserving).
    """
    z = student_router_logits.float()
    if is_softmax:
        z = z.clamp_min(1e-12).log()
    N, E = z.shape
    # Build oracle mask [N, E].
    oracle_mask = torch.zeros(N, E, dtype=torch.bool, device=z.device)
    oracle_mask.scatter_(-1, target_idx, True)
    # Per-token: oracle logits z_pos [N, K], non-oracle logits z_neg via mask
    z_pos = z.gather(-1, target_idx)             # [N, K]
    # Build [N, K, E] of (margin + z_neg - z_pos) for all (k, e) pairs.
    pos = z_pos.unsqueeze(-1)                    # [N, K, 1]
    neg = z.unsqueeze(1)                          # [N, 1, E]
    diff = margin + neg - pos                    # [N, K, E]
    # Mask out pairs where the "neg" index is actually in S*.
    valid_pair = (~oracle_mask).unsqueeze(1).expand(N, k, E)  # [N, K, E]
    diff = diff.masked_fill(~valid_pair, 0.0)
    losses = F.relu(diff)                        # [N, K, E]
    # Mean over valid pairs only: total / (N * K * (E-K)).
    denom = float(N * k * (E - k))
    return losses.sum() / denom


def loss_router_margin_weighted_pairwise_hinge(
    student_router_logits: torch.Tensor,
    target_idx: torch.Tensor,
    target_margin: torch.Tensor,
    k: int,
    margin: float = 1.0,
    is_softmax: bool = True,
    min_target_margin: float = 0.0,
    margin_power: float = 1.0,
    weight_clip: float = 0.0,
) -> torch.Tensor:
    """Pairwise hinge weighted by oracle top-K confidence.

    ``target_margin`` is the per-token gap between the K-th oracle expert norm
    and the best non-oracle norm. Tokens with a clearer oracle subset get a
    larger gradient; tokens below ``min_target_margin`` are treated as
    ambiguous and receive zero router-loss weight. Above the threshold, the
    weight is ``(target_margin - min_target_margin) ** margin_power``. Weights
    are normalized to mean 1 over the full batch so the configured alpha remains
    comparable when enough active tokens remain.
    """
    z = student_router_logits.float()
    if is_softmax:
        z = z.clamp_min(1e-12).log()
    N, E = z.shape
    oracle_mask = torch.zeros(N, E, dtype=torch.bool, device=z.device)
    oracle_mask.scatter_(-1, target_idx, True)
    z_pos = z.gather(-1, target_idx)
    diff = margin + z.unsqueeze(1) - z_pos.unsqueeze(-1)
    valid_pair = (~oracle_mask).unsqueeze(1).expand(N, k, E)
    losses = F.relu(diff).masked_fill(~valid_pair, 0.0)
    per_token = losses.sum(dim=(1, 2)) / float(k * (E - k))
    raw_margin = target_margin.float().reshape(-1).clamp_min(0.0)
    threshold = float(min_target_margin)
    weights = (raw_margin - threshold).clamp_min(0.0)
    if threshold <= 0.0 and margin_power == 1.0:
        weights = raw_margin
    elif margin_power != 1.0:
        weights = weights.pow(float(margin_power))
    weights = weights / weights.mean().clamp_min(1e-6)
    if weight_clip > 0.0:
        weights = weights.clamp_max(float(weight_clip))
    return (per_token * weights).mean()


def loss_router_per_expert_bce(
    student_router_logits: torch.Tensor,    # [N, E]
    target_idx: torch.Tensor,               # [N, K] long
    k: int,
    is_softmax: bool = True,
) -> torch.Tensor:
    """C1: per-expert independent BCE.

    For each (token, expert e): BCE(sigmoid(z_e), 1[e in S*]).
    Mean over (N * E) elements.

    No softmax normalization (sigmoid is per-expert independent), so no 1/K cap.
    Densest gradient: E=8 logits per token contribute (4 positive + 4 negative for K=4).
    """
    z = student_router_logits.float()
    if is_softmax:
        # Post-softmax probs → recover logits-equivalent (log of probs is monotonic)
        z = z.clamp_min(1e-12).log()
    N, E = z.shape
    target_mask = torch.zeros(N, E, dtype=z.dtype, device=z.device)
    target_mask.scatter_(-1, target_idx, 1.0)
    return F.binary_cross_entropy_with_logits(z, target_mask, reduction="mean")
