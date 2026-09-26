"""CPT trainer for Phase 1 RRD MoE Llama-2 (T7).

Loads a Phase 1 training-free MoE checkpoint (T2 builder output:
``state_dict.pt + manifest.json``) and runs WikiText-2 CPT (2048 samples ×
1 epoch, effective bsz=2, Adam, LR cosine 1e-4 → 1e-6, max_steps=1024).

Three modes:
  - ``task_only``: standard LM CE only (alpha_task * loss).
  - ``rrd``: alpha_task * CE + alpha_shared * L_shared(Option B)
                              + alpha_router * L_router(magnitude-weighted KL),
            both auxiliary losses averaged across MoE layers.
  - ``logit_kd``: alpha_kd * KL(softmax(student/T) || softmax(teacher/T)) * T^2
                  on final logits. Combine with alpha_task / alpha_shared /
                  alpha_router for the 5 KD cases (kd_only / kd_task /
                  kd_task_router / kd_shared_router / kd_full).

Trainable parameters: router gate, routed experts, shared expert.
All other modules (embeddings, attention, LN, lm_head) are frozen.

Usage::

    python scripts/exp_cmoe/cpt_train.py \
        --moe_dir saves/cmoe/llama2-E8A6-ParamSplit-coreneuron \
        --teacher_model_path models/Llama-2-7b-hf \
        --calib_path data/train.jsonl \
        --output_dir saves/cmoe/llama2-E8A6-ParamSplit-coreneuron-cpt-rrd \
        --mode rrd --max_steps 1024 --max_seqlen 2048 --device cuda:0

A built-in ``--smoke_test_steps N`` overrides ``max_steps`` for quick sanity
runs (typically 10).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import itertools
import json
import logging
import math
import os
import random
import re
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from llamafactory.model.rrd_moe_llama import MoE  # noqa: E402
from llamafactory.train.distill.chain_losses import (  # noqa: E402
    _reduce_squared_error,
    loss_router_distribution_kl,
    loss_router_magnitude_weighted_kl,
    loss_router_pairwise_hinge,
    loss_router_margin_weighted_pairwise_hinge,
    loss_router_per_expert_bce,
    loss_router_renormalized_topk_ce,
    loss_router_topk_ce,
    loss_shared_joint,
    loss_shared_residual,
    magnitude_oracle_topk_with_norms,
)

try:
    from scripts.exp_cmoe.mlp_router_patch import (  # noqa: E402
        attach_mlp_probe_router,
        build_probe_from_dir,
        build_probe_from_manifest,
        mlp_router_manifest_fields,
        is_mlp_router_arch,
    )
except Exception:  # pragma: no cover - keeps legacy imports usable in minimal envs
    attach_mlp_probe_router = None
    build_probe_from_dir = None
    build_probe_from_manifest = None
    mlp_router_manifest_fields = None
    is_mlp_router_arch = lambda value: False

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("cpt_train")


def seed_everything(seed: int) -> None:
    """Seed every PRNG used by current CPT/LoRA training paths."""
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed seeds the CPU generator and all CUDA generators.
    torch.manual_seed(seed)


def _distributed_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "distributed_sync", False))


def _distributed_is_primary(args: argparse.Namespace) -> bool:
    return int(getattr(args, "_distributed_rank", 0)) == 0


def _sync_distributed_gradients(
    model: nn.Module,
    world_size: int,
) -> None:
    """Average trainable gradients in place without DDP bucket duplication."""
    for parameter in model.parameters():
        if not parameter.requires_grad or parameter.numel() == 0:
            continue
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        torch.distributed.all_reduce(
            parameter.grad,
            op=torch.distributed.ReduceOp.SUM,
        )
        parameter.grad.div_(world_size)


def _raise_on_nonfinite_loss(
    loss: torch.Tensor,
    args: argparse.Namespace,
    step: int,
) -> None:
    if not bool(getattr(args, "fail_on_nonfinite", False)):
        return
    bad = torch.tensor(
        0 if bool(torch.isfinite(loss.detach()).all()) else 1,
        device=loss.device,
        dtype=torch.int32,
    )
    if _distributed_enabled(args):
        torch.distributed.all_reduce(bad, op=torch.distributed.ReduceOp.MAX)
    if int(bad.item()) != 0:
        raise FloatingPointError(
            f"non-finite total loss at step={step}; aborting before backward/optimizer"
        )


ROUTER_MAPPING_LOSS_FORMS = {
    "activation_mass_topk_ce",
    "activation_rank_topk_ce",
    "activation_mass_kl",
    "dense_vector_topk_ce",
    "dense_vector_kl",
}
ROUTER_FROZEN_ORACLE_TARGET_FORMS = {
    "best_subset_topk_ce",
    "marginal_gain_topk_ce",
    "marginal_gain_kl",
}


def router_contribution_score_name(form: str) -> Optional[str]:
    if form == "activation_rank_topk_ce":
        return "activation_rank_sum"
    if form in {"activation_mass_topk_ce", "activation_mass_kl"}:
        return "activation_mass_sum"
    if form in {"dense_vector_topk_ce", "dense_vector_kl"}:
        return "dense_downprojected_group_vector_norm"
    if form in {"marginal_gain_topk_ce", "marginal_gain_kl"}:
        return "frozen_oracle_individual_residual_error_reduction"
    if form == "best_subset_topk_ce":
        return "frozen_oracle_best_subset_l2_residual_reconstruction"
    return None


def router_target_distribution_name(form: str) -> Optional[str]:
    if form == "activation_mass_kl":
        return "full_routed_activation_mass"
    if form == "dense_vector_kl":
        return "full_routed_dense_vector_norm"
    if form == "marginal_gain_kl":
        return "full_routed_marginal_gain"
    if form == "activation_rank_topk_ce":
        return "topk_routed_activation_rank"
    if form == "activation_mass_topk_ce":
        return "topk_routed_activation_mass"
    if form == "dense_vector_topk_ce":
        return "topk_routed_dense_vector_norm"
    if form == "marginal_gain_topk_ce":
        return "topk_routed_marginal_gain"
    if form == "best_subset_topk_ce":
        return "topk_routed_best_subset_l2"
    return None


def router_target_oracle_name(form: str) -> Optional[str]:
    if form in ROUTER_FROZEN_ORACLE_TARGET_FORMS:
        return "frozen_cmoe_init"
    if form in ROUTER_MAPPING_LOSS_FORMS:
        return "dense_teacher_recovered_cmoe_neuron_mapping"
    return None


def best_subset_topk_from_routed_stack(
    routed_stack: torch.Tensor,
    dense_target: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-token routed expert subset minimizing dense-output MSE."""
    if dense_target.dim() == 3:
        dense_target = dense_target.reshape(-1, dense_target.shape[-1])
    n_routed = int(routed_stack.shape[0])
    if k > n_routed:
        raise ValueError(f"k={k} exceeds routed expert count {n_routed}")
    combo_idx = torch.tensor(
        list(itertools.combinations(range(n_routed), k)),
        device=routed_stack.device,
        dtype=torch.long,
    )
    combo_sum = routed_stack[combo_idx].sum(dim=1).float()  # [C, N, H]
    target = dense_target.reshape(-1, dense_target.shape[-1]).float()
    err = (combo_sum - target.unsqueeze(0)).pow(2).mean(dim=-1)  # [C, N]
    best_combo = err.argmin(dim=0)
    target_idx = combo_idx[best_combo].contiguous()  # [N, K]
    target_norms = combo_sum[
        best_combo, torch.arange(target.shape[0], device=target.device)
    ].norm(dim=-1)
    if err.shape[0] > 1:
        sorted_err = err.topk(k=2, dim=0, largest=False).values
        target_margin = (sorted_err[1] - sorted_err[0]).contiguous()
    else:
        target_margin = torch.ones_like(target_norms)
    return target_idx, target_norms.unsqueeze(-1).expand(-1, k).contiguous(), target_margin


def activation_mass_topk_from_dense_mlp(
    dense_mlp: nn.Module,
    mlp_input: torch.Tensor,
    total_experts: int,
    n_routed: int,
    k: int,
    expert_neuron_indices: Optional[List[torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group dense MLP intermediate activations and select routed groups by mass."""
    h = mlp_input.reshape(-1, mlp_input.shape[-1])
    gate = dense_mlp.gate_proj(h)
    up = dense_mlp.up_proj(h)
    act_fn = getattr(dense_mlp, "act_fn", None)
    if act_fn is None:
        raise ValueError("dense MLP has no act_fn; activation-mass labels are unsupported")
    inter = act_fn(gate) * up
    if expert_neuron_indices is not None:
        if len(expert_neuron_indices) != int(n_routed):
            raise ValueError(
                f"expert_neuron_indices has {len(expert_neuron_indices)} routed groups, "
                f"expected n_routed={int(n_routed)}"
            )
        mass_parts = []
        for expert_idx, idx in enumerate(expert_neuron_indices):
            idx_dev = idx.to(device=inter.device, dtype=torch.long)
            if idx_dev.numel() == 0:
                raise ValueError(f"activation-mass mapping for expert {expert_idx} is empty")
            if int(idx_dev.max().item()) >= int(inter.shape[-1]) or int(idx_dev.min().item()) < 0:
                raise ValueError(
                    f"activation-mass mapping for expert {expert_idx} has index out of range "
                    f"[0, {int(inter.shape[-1])})"
                )
            mass_parts.append(inter.index_select(-1, idx_dev).float().abs().sum(dim=-1))
        routed_mass = torch.stack(mass_parts, dim=-1)
    else:
        # Legacy fallback for pure contiguous-split MoE. CMoE carves do NOT obey
        # this layout; callers should pass expert_neuron_indices for CMoE.
        group = inter.shape[-1] // int(total_experts)
        if group <= 0:
            raise ValueError(f"invalid activation group size for intermediate dim {inter.shape[-1]}")
        usable = group * int(total_experts)
        mass = inter[..., :usable].float().abs().reshape(
            h.shape[0], int(total_experts), group
        ).sum(dim=-1)
        routed_mass = mass[:, : int(n_routed)]
    target_norms, target_idx = routed_mass.topk(k, dim=-1)
    if routed_mass.shape[-1] > k:
        next_mass = routed_mass.topk(k + 1, dim=-1).values[:, k]
        target_margin = (target_norms[:, -1] - next_mass).contiguous()
    else:
        target_margin = torch.ones_like(target_norms[:, -1])
    return target_idx.contiguous(), target_norms.contiguous(), target_margin


def activation_mass_targets_from_dense_mlp(
    dense_mlp: nn.Module,
    mlp_input: torch.Tensor,
    total_experts: int,
    n_routed: int,
    k: int,
    expert_neuron_indices: Optional[List[torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dense-teacher activation-mass targets for top-K and full-distribution losses."""
    h = mlp_input.reshape(-1, mlp_input.shape[-1])
    gate = dense_mlp.gate_proj(h)
    up = dense_mlp.up_proj(h)
    act_fn = getattr(dense_mlp, "act_fn", None)
    if act_fn is None:
        raise ValueError("dense MLP has no act_fn; activation-mass labels are unsupported")
    inter = act_fn(gate) * up
    if expert_neuron_indices is not None:
        if len(expert_neuron_indices) != int(n_routed):
            raise ValueError(
                f"expert_neuron_indices has {len(expert_neuron_indices)} routed groups, "
                f"expected n_routed={int(n_routed)}"
            )
        mass_parts = []
        for expert_idx, idx in enumerate(expert_neuron_indices):
            idx_dev = idx.to(device=inter.device, dtype=torch.long)
            if idx_dev.numel() == 0:
                raise ValueError(f"activation-mass mapping for expert {expert_idx} is empty")
            if int(idx_dev.max().item()) >= int(inter.shape[-1]) or int(idx_dev.min().item()) < 0:
                raise ValueError(
                    f"activation-mass mapping for expert {expert_idx} has index out of range "
                    f"[0, {int(inter.shape[-1])})"
                )
            mass_parts.append(inter.index_select(-1, idx_dev).float().abs().sum(dim=-1))
        routed_mass = torch.stack(mass_parts, dim=-1)
    else:
        group = inter.shape[-1] // int(total_experts)
        if group <= 0:
            raise ValueError(f"invalid activation group size for intermediate dim {inter.shape[-1]}")
        usable = group * int(total_experts)
        mass = inter[..., :usable].float().abs().reshape(
            h.shape[0], int(total_experts), group
        ).sum(dim=-1)
        routed_mass = mass[:, : int(n_routed)]
    target_norms, target_idx = routed_mass.topk(k, dim=-1)
    if routed_mass.shape[-1] > k:
        next_mass = routed_mass.topk(k + 1, dim=-1).values[:, k]
        target_margin = (target_norms[:, -1] - next_mass).contiguous()
    else:
        target_margin = torch.ones_like(target_norms[:, -1])
    return target_idx.contiguous(), target_norms.contiguous(), target_margin, routed_mass.contiguous()


def activation_rank_targets_from_dense_mlp(
    dense_mlp: nn.Module,
    mlp_input: torch.Tensor,
    total_experts: int,
    n_routed: int,
    k: int,
    expert_neuron_indices: Optional[List[torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dense-teacher activation-rank targets over recovered routed neuron groups."""
    h = mlp_input.reshape(-1, mlp_input.shape[-1])
    gate = dense_mlp.gate_proj(h)
    up = dense_mlp.up_proj(h)
    act_fn = getattr(dense_mlp, "act_fn", None)
    if act_fn is None:
        raise ValueError("dense MLP has no act_fn; activation-rank labels are unsupported")
    mass = (act_fn(gate) * up).float().abs()
    if expert_neuron_indices is None:
        raise ValueError("activation_rank_topk_ce requires recovered expert_neuron_indices for CMoE")
    if len(expert_neuron_indices) != int(n_routed):
        raise ValueError(
            f"expert_neuron_indices has {len(expert_neuron_indices)} routed groups, "
            f"expected n_routed={int(n_routed)}"
        )

    all_idx = torch.cat([idx.to(device=mass.device, dtype=torch.long) for idx in expert_neuron_indices], dim=0)
    if all_idx.numel() == 0:
        raise ValueError("activation-rank mapping is empty")
    if int(all_idx.max().item()) >= int(mass.shape[-1]) or int(all_idx.min().item()) < 0:
        raise ValueError(f"activation-rank mapping has index out of range [0, {int(mass.shape[-1])})")
    routed_mass = mass.index_select(-1, all_idx)
    routed_rank = routed_mass.argsort(dim=-1).argsort(dim=-1).float() + 1.0

    rank_parts = []
    offset = 0
    for expert_idx, idx in enumerate(expert_neuron_indices):
        n = int(idx.numel())
        if n <= 0:
            raise ValueError(f"activation-rank mapping for expert {expert_idx} is empty")
        rank_parts.append(routed_rank[:, offset: offset + n].sum(dim=-1))
        offset += n
    routed_rank_score = torch.stack(rank_parts, dim=-1)
    target_scores, target_idx = routed_rank_score.topk(k, dim=-1)
    if routed_rank_score.shape[-1] > k:
        next_score = routed_rank_score.topk(k + 1, dim=-1).values[:, k]
        target_margin = (target_scores[:, -1] - next_score).contiguous()
    else:
        target_margin = torch.ones_like(target_scores[:, -1])
    return target_idx.contiguous(), target_scores.contiguous(), target_margin, routed_rank_score.contiguous()



def dense_vector_targets_from_dense_mlp(
    dense_mlp: nn.Module,
    mlp_input: torch.Tensor,
    n_routed: int,
    k: int,
    expert_neuron_indices: Optional[List[torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dense grouped down-projection vector norm targets over routed neuron groups."""
    h = mlp_input.reshape(-1, mlp_input.shape[-1])
    gate = dense_mlp.gate_proj(h)
    up = dense_mlp.up_proj(h)
    act_fn = getattr(dense_mlp, "act_fn", None)
    if act_fn is None:
        raise ValueError("dense MLP has no act_fn; dense-vector labels are unsupported")
    if expert_neuron_indices is None:
        raise ValueError("dense_vector_topk_ce/dense_vector_kl require recovered expert_neuron_indices")
    if len(expert_neuron_indices) != int(n_routed):
        raise ValueError(
            f"expert_neuron_indices has {len(expert_neuron_indices)} routed groups, "
            f"expected n_routed={int(n_routed)}"
        )

    inter = act_fn(gate) * up
    scores: List[torch.Tensor] = []
    for expert_idx, idx in enumerate(expert_neuron_indices):
        idx_dev = idx.to(device=inter.device, dtype=torch.long)
        if idx_dev.numel() == 0:
            raise ValueError(f"dense-vector mapping for expert {expert_idx} is empty")
        if int(idx_dev.max().item()) >= int(inter.shape[-1]) or int(idx_dev.min().item()) < 0:
            raise ValueError(f"dense-vector mapping has index out of range [0, {int(inter.shape[-1])})")
        sub_inter = inter.index_select(-1, idx_dev)
        sub_weight = dense_mlp.down_proj.weight.index_select(1, idx_dev)
        group_out = F.linear(sub_inter, sub_weight)
        scores.append(group_out.float().norm(dim=-1))
    dense_vector_score = torch.stack(scores, dim=-1)
    target_scores, target_idx = dense_vector_score.topk(k, dim=-1)
    if dense_vector_score.shape[-1] > k:
        next_score = dense_vector_score.topk(k + 1, dim=-1).values[:, k]
        target_margin = (target_scores[:, -1] - next_score).contiguous()
    else:
        target_margin = torch.ones_like(target_scores[:, -1])
    return target_idx.contiguous(), target_scores.contiguous(), target_margin, dense_vector_score.contiguous()


def marginal_gain_targets_from_routed_stack(
    routed_stack: torch.Tensor,
    residual_target: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Top-K by each routed expert's individual residual-error reduction."""
    if residual_target.dim() == 3:
        residual_target = residual_target.reshape(-1, residual_target.shape[-1])
    n_routed = int(routed_stack.shape[0])
    if k > n_routed:
        raise ValueError(f"k={k} exceeds routed expert count {n_routed}")
    target = residual_target.reshape(-1, residual_target.shape[-1]).float()
    base_err = target.pow(2).mean(dim=-1).unsqueeze(-1)
    err = (routed_stack.float() - target.unsqueeze(0)).pow(2).mean(dim=-1).transpose(0, 1)
    gain = (base_err - err).clamp_min(0.0)
    target_scores, target_idx = gain.topk(k, dim=-1)
    if gain.shape[-1] > k:
        next_score = gain.topk(k + 1, dim=-1).values[:, k]
        target_margin = (target_scores[:, -1] - next_score).contiguous()
    else:
        target_margin = torch.ones_like(target_scores[:, -1])
    return target_idx.contiguous(), target_scores.contiguous(), target_margin, gain.contiguous()


def cmoe_representative_router_targets(
    oracle_mlp: nn.Module,
    mlp_input: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Targets from the frozen CMoE analytical router initialized by representative neurons."""
    shape = mlp_input.size()
    x_flat = mlp_input.reshape(-1, shape[-1])
    gate = getattr(oracle_mlp, "gate", None)
    if gate is None:
        raise ValueError("oracle MLP has no CMoE gate; representative router target unavailable")
    _weights, indices, raw_logits, softmax_scores = gate(x_flat, return_diagnostics=True)
    target_norms, target_idx = softmax_scores.float().topk(k, dim=-1)
    if softmax_scores.shape[-1] > k:
        next_score = softmax_scores.float().topk(k + 1, dim=-1).values[:, k]
        target_margin = (target_norms[:, -1] - next_score).contiguous()
    else:
        target_margin = torch.ones_like(target_norms[:, -1])
    return target_idx.contiguous(), target_norms.contiguous(), target_margin


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CPT trainer for Phase 1 RRD MoE Llama-2.")
    p.add_argument("--moe_dir", type=str, required=True,
                   help="Directory with state_dict.pt + manifest.json (T2 output).")
    p.add_argument("--teacher_model_path", type=str,
                   default="models/Llama-2-7b-hf")
    p.add_argument("--calib_path", type=str,
                   default="data/train.jsonl")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--mode", type=str,
                   choices=["task_only", "rrd", "lora", "logit_kd"],
                   default="task_only",
                   help="task_only=CE only on MoE; rrd=CE+shared+router on MoE; "
                        "lora=PEFT LoRA SFT on carved MoE (CMoE simple_sft.py mirror); "
                        "logit_kd=KL(student/T || teacher/T)*T^2. "
                        "Five loss combinations can be set with alpha_task/alpha_kd/alpha_shared/alpha_router; "
                        "per-layer RRD auxiliary losses run when alpha_shared>0 or alpha_router>0.")

    # ----- LoRA mode (CMoE simple_sft.py mirror) -----
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--lora_target_modules", type=str,
                   default="q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj",
                   help="comma-separated. CMoE simple_sft.py mirror.")
    p.add_argument("--lora_extra_lr", type=float, default=0.001,
                   help="Separate LR for params with 'extra' in name (CMoE extra_scale). "
                        "CMoE simple_sft.py mirror.")

    # Track selector: 'rrd' = own ParamSplit/BasicSplit MoE (default, original behaviour);
    # 'cmoe' = CMoE pre-LoRA carved S3A3E8 checkpoint (mirror module).
    p.add_argument(
        "--moe_type",
        type=str,
        choices=["rrd", "cmoe", "dense"],
        default="rrd",
        help=(
            "rrd = own ParamSplit/BasicSplit MoE; "
            "cmoe = CMoE pre-LoRA carved S3A3E8; "
            "dense = full-parameter dense SFT from --teacher_model_path"
        ),
    )
    p.add_argument("--cmoe_state_dict_path", type=str, default=None,
                   help="(--moe_type=cmoe) Path to CMoE state_dict.pt. "
                        "If unset, uses {moe_dir}/state_dict.pt.")
    p.add_argument(
        "--cmoe_checkpoint_manifest",
        type=str,
        default="",
        help=(
            "(--moe_type=cmoe) Manifest paired with a trained full checkpoint. "
            "When it declares an MLP router, build and attach that router before "
            "strictly loading --cmoe_state_dict_path. This is required when "
            "continuing SFT/CPT from an RRD checkpoint with embedded "
            "phase127_mlp_router weights."
        ),
    )
    p.add_argument(
        "--init_trainable_delta_path",
        type=str,
        default="",
        help=(
            "(--moe_type=cmoe, CE continuation) Overlay a prior "
            "trainable_delta.pt on the carved base before applying the current "
            "trainable policy."
        ),
    )
    p.add_argument(
        "--init_adapter_path",
        type=str,
        default="",
        help=(
            "(--mode=lora) Continue from an existing PEFT adapter directory "
            "instead of creating a fresh adapter."
        ),
    )
    p.add_argument(
        "--init_cmoe_extra_state_path",
        type=str,
        default="",
        help=(
            "(--mode=lora continuation) Overlay the adapter-paired "
            "cmoe_extra_state.pt before loading --init_adapter_path."
        ),
    )
    p.add_argument(
        "--stack_init_adapter",
        action="store_true",
        help=(
            "Keep --init_adapter_path frozen and active, then attach a fresh "
            "trainable LoRA adapter for the current run."
        ),
    )
    p.add_argument("--cmoe_n_experts", type=int, default=8,
                   help="(--moe_type=cmoe) total expert count (routed + shared). Manifest key 'nexperts' wins if present.")
    p.add_argument("--cmoe_n_activated", type=int, default=3,
                   help="(--moe_type=cmoe) top-K. Manifest key 'nactivated' wins if present.")
    p.add_argument("--cmoe_n_shared", type=int, default=3,
                   help="(--moe_type=cmoe) shared expert count. Manifest key 'nshared' wins if present.")
    p.add_argument("--cmoe_extra_scale_trainable", type=int, default=0, choices=[0, 1],
                   help="(--moe_type=cmoe) 0=freeze (default per plan OQ1), 1=allow gradient")
    p.add_argument("--cmoe_enable_load_balance", type=int, default=0, choices=[0, 1],
                   help="(--moe_type=cmoe) enable CMoE router extra_bias load-balancing "
                        "updates (MoE.cus_training=True).")
    p.add_argument("--router_arch", type=str, default="default",
                   help="Optional train/eval router replacement. Use default, mlp_h128/mlp_h512/mlp_h1024, "
                        "or hybrid_mlp_h1024 to attach a Phase126-style per-layer MLP probe router.")
    p.add_argument("--init_mlp_router_probe_dir", type=str, default="",
                   help="Phase126 probe run directory containing manifest.json + probe_state_dict.pt. "
                        "Required when --router_arch is an MLP probe router unless --init_mlp_router_random is set.")
    p.add_argument("--init_mlp_router_random", action="store_true",
                   help="Initialize an MLP-probe router from scratch instead of loading a Stage1 probe checkpoint. "
                        "This is intended for no-Stage1 ablations.")
    p.add_argument("--mlp_router_aggregation", type=str, choices=["uniform", "probe_renorm"], default="uniform",
                   help="Aggregation weights for MLP-router selected experts. Phase127 default is uniform.")
    p.add_argument("--mlp_router_layer_mask", type=str, default="",
                   help="Optional layer mask for hybrid MLP-router replacement, e.g. 9-18. "
                        "Masked layers use the MLP router; unmasked layers keep the original CMoE router.")
    p.add_argument("--train_lm_head", action="store_true",
                   help="Keep the output lm_head trainable after the default MoE-only freeze. "
                        "Embeddings, attention, and layer norms remain frozen.")
    p.add_argument(
        "--train_attention",
        type=int,
        default=0,
        choices=[0, 1],
        help=(
            "CE-only CMoE policy: additionally train q_proj/k_proj/v_proj/o_proj "
            "weights in every self-attention layer. Embeddings, layer norms, and "
            "lm_head remain frozen. This flag is rejected in --mode lora because "
            "the official LoRA target-module contract controls attention there."
        ),
    )
    p.add_argument("--cmoe_extra_scale_init", type=float, default=None,
                   help="(--moe_type=cmoe) optional value used to fill every gate.extra_scale "
                        "after loading the carved state_dict. This makes CE gradients reach "
                        "CMoE router gate/classifier immediately through differentiable "
                        "selected-expert weights; with the carved default extra_scale=0, CE "
                        "can only update extra_scale first and hard top-k indices remain non-diff.")
    p.add_argument("--cmoe_add_eas", type=int, default=0, choices=[0, 1],
                   help="(--moe_type=cmoe) 1=add Extra Additional Shared expert (EAS) per layer "
                        "with intermediate_size=moe_inter_dim. EAS is always-active; its "
                        "contribution to the residual stream is .detach()ed inside MoE.forward "
                        "so L_task / L_router don't propagate into EAS. Only L_shared (which "
                        "uses the captured `eas_out` from return_components) updates EAS — EAS "
                        "learns the residual `teacher_rep - (shared + routed_top_k)`.")
    p.add_argument("--cmoe_eas_init_std", type=float, default=1.0e-3,
                   help="(--cmoe_add_eas=1) zero-jitter std for EAS gate/up/down weight init "
                        "(N(0, std)). Default 1e-3. Set 0 to use LlamaMLP default Kaiming.")
    p.add_argument("--shared_loss_form", type=str, choices=["mse", "rmse"], default="mse",
                   help="rmse maps to reduction='global_rmse' in loss_shared_joint")
    p.add_argument("--shared_target_form", type=str, choices=["joint", "residual", "routed_residual"], default="joint",
                   help="joint keeps current L_shared on shared+routed vs dense MLP. "
                        "residual trains shared_out toward stopgrad(dense_mlp_out - routed_out). "
                        "routed_residual trains routed_out toward stopgrad(dense_mlp_out - shared_out), "
                        "treating shared experts as a fixed anchor for residual recovery.")
    p.add_argument("--shared_target_input", type=str,
                   choices=["teacher_post_ln", "student_mlp_input"], default="teacher_post_ln",
                   help="Input trajectory used to build the dense/shared target for L_shared. "
                        "teacher_post_ln preserves the legacy target on the teacher layer input. "
                        "student_mlp_input recomputes the teacher dense MLP on the student's "
                        "captured MLP input, testing same-input local-function distillation.")
    p.add_argument("--router_target_input", type=str,
                   choices=["teacher_post_ln", "student_mlp_input"], default="teacher_post_ln",
                   help="Input trajectory used to build dense-teacher router targets for "
                        "activation_mass_topk_ce / activation_mass_kl and representative-router "
                        "diagnostics. teacher_post_ln is the legacy teacher-trajectory target; "
                        "student_mlp_input makes layer-local same-input router ablations compare "
                        "the student router score and dense activation-mass label on the same input.")
    p.add_argument("--router_train_routing_mode", type=str,
                   choices=["hard_topk", "selected_topk_renorm", "soft_full", "st_topk", "relaxed_topk"], default="hard_topk",
                   help="Training-only routed-output proxy for auxiliary losses. hard_topk preserves "
                        "the current non-differentiable top-k routed output. selected_topk_renorm "
                        "keeps hard selected experts and renormalizes only their gate weights. "
                        "soft_full forwards all routed experts weighted by router_scores. st_topk "
                        "uses hard top-k values in the forward pass but soft router_scores in the "
                        "backward pass. relaxed_topk keeps hard selected experts and adds small "
                        "differentiable non-selected leakage.")
    p.add_argument("--router_relax_epsilon", type=float, default=0.05,
                   help="Non-selected expert leakage coefficient for --router_train_routing_mode=relaxed_topk.")
    p.add_argument("--audit_router_residual_grad", action="store_true",
                   help="On the first aux step, log grad norms from the residual/shared loss alone "
                        "to router/routed/shared parameter groups. Intended to verify whether "
                        "differentiable routing proxies send residual gradients to the router.")
    p.add_argument("--shared_loss_fp32", action="store_true",
                   help="(RRD ablation) Cast pred/target to fp32 inside loss_shared_joint "
                        "before computing diff² and the reduction. Avoids bf16 noise on "
                        "small-magnitude (~1e-3) RMSE. Default off (preserves existing "
                        "bf16 reduction behavior).")
    p.add_argument("--shared_grad_routed_detach", type=int, choices=[0, 1], default=0,
                   help="If 1, .detach() routed_top_k_weighted_sum (and EAS if present) before "
                        "L_shared. Then L_shared gradient flows ONLY into shared expert — "
                        "routed experts/router learn from L_task + L_router only. "
                        "Default 0 = joint (Option B).")
    p.add_argument("--aux_loss_scope", type=str, choices=["global", "layer_local"], default="global",
                   help="Scope for RRD router/shared auxiliary losses. global preserves the "
                        "current full-forward component graph. layer_local re-forwards each "
                        "layer MoE from its detached captured MLP input, so aux gradients are "
                        "confined to that layer's MoE/router/shared/routed parameters while "
                        "CE/KD still train end-to-end.")
    p.add_argument("--layer_local_aux_input", type=str,
                   choices=["student_mlp_input", "teacher_post_ln"], default="student_mlp_input",
                   help="Input used by --aux_loss_scope=layer_local for the local MoE re-forward. "
                        "student_mlp_input preserves the existing detached student trajectory. "
                        "teacher_post_ln teacher-forces each local MoE/router/shared module with "
                        "the teacher post-attention LN input, separating input drift from local "
                        "expert/target learnability.")
    p.add_argument("--shared_layer_mask", type=str, default="",
                   help="Optional comma/range list of layer indices for L_shared only, "
                        "e.g. '3,22-27'. Empty default keeps the existing all-layer "
                        "shared loss. Router loss and per-layer diagnostics still use all layers.")
    p.add_argument("--router_layer_mask", type=str, default="",
                   help="Optional comma/range list of layer indices for L_router only, "
                        "e.g. '8-19'. Empty default keeps the existing all-layer "
                        "router loss. Per-layer diagnostics still use all layers.")
    p.add_argument("--moe_trainable_layer_mask", type=str, default="",
                   help="Optional comma/range list of MoE layer indices whose router/shared/routed "
                        "parameters remain trainable. Masked-out MoE layer parameters are frozen "
                        "after router_arch attachment; CE/KD still run through the full model.")
    p.add_argument("--shared_token_weight_mode", type=str,
                   choices=["none", "ce_low", "ce_high", "ce_low_alpha_only"], default="none",
                   help="Optional per-token weighting for L_shared. 'ce_low' keeps shared-loss "
                        "pressure on lower-CE predictive positions and downweights high-loss "
                        "uncertain positions; 'ce_high' does the opposite for ablation. "
                        "'ce_low_alpha_only' further keeps shared pressure only when the "
                        "predicted target token is alphabetic, protecting digit/mixed/punct "
                        "tokens seen as fragile in Phase13. Default none preserves existing "
                        "all-token shared loss.")
    p.add_argument("--shared_token_weight_quantile", type=float, default=0.75,
                   help="Quantile threshold over valid per-token CE for --shared_token_weight_mode. "
                        "For ce_low, tokens <= quantile get weight 1; for ce_high, tokens >= quantile get weight 1.")
    p.add_argument("--shared_token_weight_floor", type=float, default=0.0,
                   help="Minimum weight assigned to gated-out tokens for shared token weighting. "
                        "0.0 means hard gate; small values such as 0.1 keep weak coverage.")
    p.add_argument("--task_token_weight_mode", type=str,
                   choices=["none", "ce_low", "ce_high", "ce_low_alpha_only"], default="none",
                   help="Optional per-token weighting for the task CE loss. Uses the same "
                        "student CE quantile gate as shared_token_weight_mode. 'ce_high' "
                        "emphasizes high-loss predictive positions while floor keeps weak "
                        "coverage for the rest. Default none preserves HF mean CE.")
    p.add_argument("--task_token_weight_quantile", type=float, default=0.75,
                   help="Quantile threshold over valid per-token CE for --task_token_weight_mode.")
    p.add_argument("--task_token_weight_floor", type=float, default=0.0,
                   help="Minimum weight assigned to gated-out tokens for task CE weighting.")
    p.add_argument("--router_oracle_source", type=str,
                   choices=["teacher_dense", "frozen_student_all_active",
                            "frozen_student_all_active_teacher_input"],
                   default="teacher_dense",
                   help="(--mode rrd) Target source for L_router (and L_shared teacher rep). "
                        "'teacher_dense' = teacher Llama-2 dense MLP output (legacy magKL on "
                        "teacher experts). 'frozen_student_all_active' = deep-copied frozen "
                        "student MoE forward on STUDENT post-attn LN (drifts with student "
                        "training). 'frozen_student_all_active_teacher_input' = same frozen "
                        "snapshot but forwarded on TEACHER post-attn LN — fixed target for the "
                        "duration of training, regardless of student drift. carve_sanity check "
                        "(rel_err 0.2%%) confirms this output ≡ dense MLP output.")

    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--min_learning_rate", type=float, default=1e-6)
    p.add_argument("--lr_schedule", type=str, choices=["cosine", "constant"], default="cosine",
                   help="cosine = CosineAnnealingLR(learning_rate → min_learning_rate). "
                        "constant = no decay (fixed at learning_rate, mirrors CMoE simple_sft.py).")
    p.add_argument("--per_device_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument(
        "--max_grad_norm",
        type=float,
        default=0.0,
        help=(
            "Clip the global norm of all trainable gradients before each optimizer "
            "step. Values <= 0 preserve the historical no-clipping behavior."
        ),
    )
    p.add_argument("--max_steps", type=int, default=1024)
    p.add_argument("--max_seqlen", type=int, default=2048)
    p.add_argument("--data_packing", type=str, choices=["per_doc", "concat"], default="per_doc",
                   help="per_doc=one chunk per document if it is >=max_seqlen tokens "
                        "(Llama-2 default reproduces CMoE setup). "
                        "concat=concatenate all docs (with EOS) then slice — required for "
                        "Qwen2.5 (BPE more efficient → per_doc filters all docs).")
    p.add_argument(
        "--data_shuffle",
        type=int,
        choices=[0, 1],
        default=1,
        help="Shuffle training windows before batching. Set 0 for exact ordered token-stream replay.",
    )
    p.add_argument(
        "--data_source",
        type=str,
        choices=[
            "jsonl",
            "token_ids_jsonl",
            "supervised_token_ids_jsonl",
            "token_ids_npy",
            "cmoe_inmemory",
            "teacher_rollout_jsonl",
            "raw_continuation_jsonl",
        ],
        default="jsonl",
        help=(
            "jsonl=read --calib_path jsonl (decoded text -> re-tokenize). "
            "token_ids_jsonl=read exact pre-tokenized {'input_ids': [...]} chunks. "
            "supervised_token_ids_jsonl=read exact pre-tokenized "
            "{'input_ids': [...], 'labels': [...]} SFT rows with prompt labels "
            "already masked to -100. "
            "token_ids_npy=memory-map an integer [num_windows, seqlen] NumPy array. "
            "cmoe_inmemory=invoke CMoE/datautils.get_wikitext2 directly with the same "
            "random.randint(seed=...) crop run_cmoe.py uses, bypassing the "
            "decode->re-tokenize round-trip in jsonl dumps. "
            "teacher_rollout_jsonl=prefix_text + teacher_continuation_text with "
            "prefix labels masked for ZEDA-style SFT. "
            "raw_continuation_jsonl=prefix_text + gold_continuation_text with "
            "the same continuation-only masking policy."
        ),
    )
    p.add_argument(
        "--continuation_field",
        type=str,
        default=None,
        help="Optional continuation JSONL field override for continuation-only data sources.",
    )
    p.add_argument("--data_nsamples", type=int, default=2048,
                   help="(--data_source=cmoe_inmemory) number of random crops to sample.")
    p.add_argument("--cmoe_repo_path", type=str, default="third_party/cmoe",
                   help="(--data_source=cmoe_inmemory) path to CMoE repo (for datautils import).")
    p.add_argument("--one_epoch", action="store_true",
                   help="Train for exactly one DataLoader pass: max_steps becomes len(loader). "
                        "With drop_last=True this preserves a fixed per-device batch size.")

    p.add_argument("--alpha_task", type=float, default=1.0)
    p.add_argument("--alpha_shared", type=float, default=5.0)
    p.add_argument("--alpha_router", type=float, default=0.1)
    p.add_argument("--alpha_kd", type=float, default=0.0,
                   help="Weight for L_kd (Hinton KL on final logits). If 0, L_kd is skipped. "
                        "L_kd = KL(softmax(target/T) || softmax(student/T)) * T^2. "
                        "Gradient flows end-to-end through the trainable student modules.")
    p.add_argument("--single_loss_objective", type=str,
                   choices=["none", "ce", "shared", "router", "kd"], default="none",
                   help="Loss-unit-test mode. 'none' preserves the weighted-sum objective. "
                        "Otherwise compute and backprop only the selected raw loss, instead "
                        "of emulating ablations by setting other alpha weights to 0. "
                        "'shared'/'router' require --mode rrd; 'kd' requires a KD target.")
    p.add_argument("--kd_temperature", type=float, default=1.0,
                   help="Temperature T for L_kd softmax. T=1.0 = no smoothing (default). "
                        "Hinton-style KD typically uses T=2~4.")
    p.add_argument("--kd_target_model", type=str, choices=["dense_teacher", "source_moe"],
                   default="dense_teacher",
                   help="Target model for L_kd final-logit KL. dense_teacher uses "
                        "--teacher_model_path. source_moe deep-copies the loaded initial "
                        "student MoE before updates and anchors to its logits.")
    p.add_argument("--validation_calib_path", type=str, default="",
                   help="Optional validation JSONL/token-id path for lightweight CE/KD validation curves. "
                        "Uses the same data_source/data_packing/max_seqlen as training.")
    p.add_argument("--validation_every", type=int, default=0,
                   help="If >0, run lightweight validation at step 0, every N optimizer steps, and final.")
    p.add_argument("--validation_batches", type=int, default=32,
                   help="Maximum validation batches for lightweight validation logging.")
    p.add_argument(
        "--validation_per_device_batch_size",
        type=int,
        default=0,
        help="Validation-only batch size. 0 reuses --per_device_batch_size.",
    )
    p.add_argument(
        "--activation_offload_cpu",
        action="store_true",
        help="Offload tensors saved for backward to CPU without changing the training batch or objective.",
    )
    p.add_argument(
        "--data_manifest",
        type=str,
        default="",
        help=(
            "Optional seeded exact-token bundle manifest. When supplied, its seed, "
            "selected train path, validation path, and sequence length must match."
        ),
    )
    p.add_argument(
        "--data_manifest_train_key",
        type=str,
        default="train",
        help=(
            "files[] key used for --calib_path validation, e.g. "
            "train_prefix_n2048 for the LR-screening prefix."
        ),
    )
    p.add_argument(
        "--data_manifest_seed_independent",
        action="store_true",
        help=(
            "Allow one immutable selected dataset to be reused across training "
            "seeds. The manifest must record selection.seed; --seed still "
            "controls shuffle and model initialization."
        ),
    )
    p.add_argument(
        "--data_manifest_validation_key",
        type=str,
        default="validation_test",
        help=(
            "files[] key used for --validation_calib_path validation. "
            "The default preserves the WikiText bundle contract; C4 scaling "
            "uses a dedicated c4_dev key."
        ),
    )

    p.add_argument("--audit_loss_grad_groups", action="store_true",
                   help="On the first step, log grad norms from raw CE/KD/router/shared losses "
                        "to router/routed/shared/lora/lm_head/other parameter groups.")
    p.add_argument("--router_target_temperature", type=float, default=1.0,
                   help="τ in softmax(target_norms / τ) for the L_router magKL target. "
                        "τ<1 sharpens q* toward the largest oracle expert (more informative "
                        "when target_norms cluster too tightly); τ=1 keeps the original soft "
                        "form. Diagnose_router showed CMoE-carve targets are near-uniform "
                        "(target_freq_max 0.217 vs uniform 0.20) — τ=0.1 is the recommended "
                        "first sharpening trial.")
    p.add_argument(
        "--router_loss_form",
        type=str,
        choices=["mag_kl", "topk_ce", "renorm_topk_ce", "pairwise_hinge", "margin_pairwise_hinge", "margin_pairwise_hinge_conf", "per_expert_bce", "best_subset_topk_ce", "activation_mass_topk_ce", "activation_rank_topk_ce", "activation_mass_kl", "dense_vector_topk_ce", "dense_vector_kl", "marginal_gain_topk_ce", "marginal_gain_kl", "cmoe_representative_router_loss"],
        default="mag_kl",
        help=(
            "Router objective for --mode rrd. mag_kl preserves the existing "
            "magnitude-weighted KL behavior. topk_ce, pairwise_hinge, and "
            "per_expert_bce add stronger pressure against non-oracle experts for "
            "router warm-up ablations. best_subset_topk_ce labels the subset with "
            "minimum dense-output MSE; activation_mass_topk_ce labels dense-neuron "
            "groups with largest intermediate activation mass; activation_rank_topk_ce "
            "labels groups with largest routed-neuron activation rank sums; activation_mass_kl "
            "distills the full routed-expert activation-mass distribution. "
            "dense_vector_* labels/distills routed groups by dense down-projected "
            "group-output vector norm. marginal_gain_* labels/distills frozen-oracle "
            "routed experts by individual dense-minus-shared residual error reduction. "
            "cmoe_representative_router_loss "
            "distills the frozen analytical CMoE router initialized from representative neurons."
        ),
    )
    p.add_argument(
        "--activation_mass_mapping_moe_dir",
        type=str,
        default="",
        help=(
            "Optional CMoE carve directory whose state_dict.pt is used only to recover "
            "dense-neuron groups for activation_mass_topk_ce labels. If empty and the "
            "loaded checkpoint manifest has cpt_source_moe_dir, that source carve is "
            "used automatically. This keeps two-step runs valid after experts have trained."
        ),
    )
    p.add_argument(
        "--router_margin_min",
        type=float,
        default=0.0,
        help=(
            "For margin_pairwise_hinge only: zero router-loss weight for tokens "
            "whose oracle K-vs-next margin is below this threshold. This tests "
            "confidence-aware routing by ignoring ambiguous oracle targets."
        ),
    )
    p.add_argument(
        "--router_margin_power",
        type=float,
        default=1.0,
        help=(
            "For margin_pairwise_hinge only: exponent applied to nonzero oracle "
            "margin weights before mean-normalization. Values >1 emphasize high-confidence targets."
        ),
    )
    p.add_argument(
        "--router_margin_weight_clip",
        type=float,
        default=0.0,
        help=(
            "For confidence-aware margin_pairwise_hinge: cap the mean-normalized "
            "per-token router margin weight. 0.0 disables clipping and preserves "
            "the existing behavior."
        ),
    )
    p.add_argument(
        "--router_margin_threshold_quantile",
        type=float,
        default=0.0,
        help=(
            "For margin_pairwise_hinge only: if >0, compute this per-layer quantile "
            "of oracle margins each step and use it as the confidence threshold. "
            "Takes precedence over --router_margin_min."
        ),
    )
    p.add_argument(
        "--router_min_active_frac",
        type=float,
        default=0.0,
        help=(
            "For confidence-aware margin_pairwise_hinge: abort if the mean active-token "
            "fraction after thresholding falls below this value on a logged step."
        ),
    )

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])

    p.add_argument("--freeze_routed", action="store_true",
                   help="(RRD ablation) Freeze routed experts after freeze_non_moe so only "
                        "router + shared expert receive gradients. Tests whether routed "
                        "shake is the source of distill_only task degradation on RRD-init.")
    p.add_argument("--freeze_shared", action="store_true",
                   help="(RRD ablation) Freeze the shared expert. Combine with --freeze_routed "
                        "for EM Phase E (router-only learning).")
    p.add_argument("--freeze_router", action="store_true",
                   help="(RRD ablation) Freeze router gate/classifier/extra_scale. Used for "
                        "EM Phase M (expert-only learning) when combined with oracle routing.")
    p.add_argument("--audit_router_update", action="store_true",
                   help="Audit the first optimizer step by logging router grad/update norms.")
    p.add_argument("--fail_on_zero_router_update", action="store_true",
                   help="With --audit_router_update, abort if first-step router grad/update is zero.")
    p.add_argument("--audit_expert_update", action="store_true",
                   help="Audit the first optimizer step with one nonzero-gradient sentinel "
                        "from each trainable routed/shared expert group.")
    p.add_argument("--fail_on_zero_expert_update", action="store_true",
                   help="With --audit_expert_update, abort unless both routed and shared "
                        "expert sentinels have nonzero gradients and parameter updates.")
    p.add_argument("--audit_task_router_grad", action="store_true",
                   help="Audit first-step CE/task-loss gradients on trainable router params "
                        "(gate/classifier/extra_scale) before auxiliary losses are added.")
    p.add_argument("--fail_on_zero_task_router_grad", action="store_true",
                   help="With --audit_task_router_grad, abort if CE/task loss has zero gradient "
                        "on all trainable router params.")
    p.add_argument("--audit_aux_layer_locality", action="store_true",
                   help="For one smoke step with --aux_loss_scope=layer_local, audit that "
                        "local aux-loss gradients are nonzero only inside the selected layer MLP.")
    p.add_argument("--audit_aux_layer_idx", type=int, default=-1,
                   help="Layer index for --audit_aux_layer_locality. Default -1 audits the "
                        "first layer that contributes an aux loss.")
    p.add_argument("--shared_down_init", type=str,
                   choices=["zero", "random", "gate_up_match"], default="zero",
                   help="(RRD ablation) Override shared_expert.down_proj.weight init AFTER "
                        "load_state_dict. 'zero' (default) keeps the carve's zero-init. "
                        "'random' = N(0, sqrt(2/hidden_size)) (Kaiming-like). "
                        "'gate_up_match' = N(0, mean(std(gate_proj), std(up_proj))) per-layer "
                        "(matches the carve's existing scale).")
    p.add_argument("--shared_zero_jitter_std", type=float, default=0.0,
                   help="(RRD ablation) If >0, re-init ALL shared_expert weights "
                        "(gate_proj, up_proj, down_proj) to N(0, std) AFTER load_state_dict. "
                        "Keeps shared output near 0 at step 0 (residual start) while "
                        "letting gradient flow to gate/up (unlike pure zero down_proj). "
                        "Mutually exclusive with non-zero --shared_down_init: "
                        "if both set, jitter wins and a warning is logged. "
                        "Default 0.0 = no-op (preserves existing behavior).")
    p.add_argument("--em_phase", type=str, choices=["none", "M", "E"], default="none",
                   help="EM phase. 'M' enables oracle-routing override on the student forward "
                        "(routing decision = magnitude top-K from frozen oracle on teacher t_ln); "
                        "'E' is a bookkeeping label (combine with --freeze_routed --freeze_shared "
                        "and alpha_router=1 to learn router only). 'none' = no EM (default).")
    p.add_argument("--data_split", type=str, choices=["all", "first_half", "second_half"],
                   default="all",
                   help="Wikitext sample split. 'all' (default) uses every chunk; "
                        "'first_half'/'second_half' split self.samples in half deterministically.")
    p.add_argument("--use_8bit_adam", action="store_true",
                   help="Use bitsandbytes 8-bit Adam to reduce optimizer state memory.")
    p.add_argument(
        "--distributed_sync",
        action="store_true",
        help=(
            "Synchronous multi-process data parallel for memory-constrained CMoE CE. "
            "Launch with torchrun; gradients are averaged in place and CMoE "
            "load-balancing counts are synchronized across ranks."
        ),
    )
    p.add_argument(
        "--fail_on_nonfinite",
        action="store_true",
        help="Abort before backward when any rank observes a non-finite total loss.",
    )
    p.add_argument("--shared_lr_multiplier", type=float, default=1.0,
                   help="RRD ablation: multiply learning rate for trainable shared expert "
                        "parameters. Default 1.0 preserves the single-LR optimizer.")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--smoke_test_steps", type=int, default=0,
                   help="If >0, override max_steps to this value (smoke test).")
    p.add_argument("--no_save", action="store_true",
                   help="Skip writing state_dict (smoke runs).")
    p.add_argument("--save_trainable_delta", action="store_true",
                   help="Also save trainable_delta.pt containing only currently trainable parameters. "
                        "Intended for Phase127 cleanup-friendly checkpoints.")
    p.add_argument("--skip_full_state_dict", action="store_true",
                   help="Do not save full state_dict.pt; retain manifest/logs and optional trainable_delta.pt only.")
    p.add_argument("--save_at_steps", type=str, default="",
                   help="Comma-separated step counts at which to save an "
                        "intermediate ckpt under {output_dir}/ckpt_step_{step}/. "
                        "Example: '1024,2048,3072'. The final ckpt is always "
                        "saved at max_steps to {output_dir} directly. "
                        "Empty (default) = no intermediate saves.")
    p.add_argument(
        "--resume_state",
        type=str,
        default="",
        help=(
            "Optional rolling training-state path. If the file exists, restore "
            "trainable tensors, mutable CMoE buffers, optimizer/scheduler, RNG, "
            "step, and histories. Restricted to ordered one-epoch data and "
            "gradient accumulation 1."
        ),
    )
    p.add_argument(
        "--resume_save_every",
        type=int,
        default=0,
        help="Atomically refresh --resume_state every N completed steps; 0 disables rolling saves.",
    )
    p.add_argument(
        "--write_ready_markers",
        action="store_true",
        help="Write .ready only after each requested intermediate/final inference checkpoint is complete.",
    )
    return p.parse_args()


def build_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def classify_token_piece(token: str) -> str:
    """Coarse tokenizer-piece category for Phase15 shared-loss gating."""
    core = token.replace("▁", "").replace("Ġ", "").strip()
    if not core:
        return "whitespace"
    if any(ord(ch) > 127 for ch in core):
        return "non_ascii"
    has_alpha = any(ch.isalpha() for ch in core)
    has_digit = any(ch.isdigit() for ch in core)
    has_other = any((not ch.isalnum()) for ch in core)
    if has_alpha and not has_digit and not has_other:
        return "alpha"
    if has_digit and not has_alpha and not has_other:
        return "digit"
    if has_alpha or has_digit:
        return "mixed"
    return "punct"


def build_shared_token_weights(
    mode: str,
    logits: torch.Tensor,
    labels: torch.Tensor,
    quantile: float,
    floor: float,
    tokenizer: Any | None = None,
) -> Tuple[Optional[torch.Tensor], Optional[Dict[str, float]]]:
    """Build [B, S] per-position weights for L_shared from shifted CE.

    Hidden state at position t predicts label t+1 in CausalLM training, so the
    CE-derived weight is assigned to representation position t. Invalid labels
    and the final non-predictive position receive weight 0.
    """
    if mode == "none":
        return None, None
    if not (0.0 < quantile < 1.0):
        raise ValueError("--shared_token_weight_quantile must be in (0, 1)")
    if not (0.0 <= floor <= 1.0):
        raise ValueError("--shared_token_weight_floor must be in [0, 1]")

    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:].contiguous()
    valid = shift_labels != -100
    if not bool(valid.any()):
        zeros = torch.zeros_like(labels, dtype=torch.float32, device=logits.device)
        return zeros, {"valid_frac": 0.0}

    safe_labels = shift_labels.masked_fill(~valid, 0)
    ce = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        safe_labels.reshape(-1),
        reduction="none",
    ).reshape_as(shift_labels)
    valid_ce = ce[valid]
    threshold = torch.quantile(valid_ce.detach().float(), float(quantile))

    category_alpha = None
    stats: Dict[str, float] = {
        "valid_frac": float(valid.float().mean().item()),
        "ce_threshold": float(threshold.detach().item()),
    }
    if mode == "ce_low_alpha_only":
        if tokenizer is None:
            raise ValueError("tokenizer is required for shared_token_weight_mode=ce_low_alpha_only")
        valid_ids = safe_labels[valid].detach().cpu().tolist()
        pieces = tokenizer.convert_ids_to_tokens(valid_ids)
        alpha_values = [1.0 if classify_token_piece(str(piece)) == "alpha" else 0.0 for piece in pieces]
        alpha_flat = torch.tensor(alpha_values, device=logits.device, dtype=torch.bool)
        category_alpha = torch.zeros_like(valid, dtype=torch.bool, device=logits.device)
        category_alpha[valid] = alpha_flat
        stats["category_alpha_frac_valid"] = float(alpha_flat.float().mean().item())
        stats["category_protected_frac_valid"] = 1.0 - stats["category_alpha_frac_valid"]

    if mode == "ce_low":
        keep = valid & (ce <= threshold)
    elif mode == "ce_high":
        keep = valid & (ce >= threshold)
    elif mode == "ce_low_alpha_only":
        keep = valid & (ce <= threshold) & category_alpha
    else:
        raise ValueError(f"unknown shared_token_weight_mode={mode!r}")

    weights = torch.zeros_like(labels, dtype=torch.float32, device=logits.device)
    gated = torch.where(
        keep,
        torch.ones_like(ce, dtype=torch.float32),
        torch.full_like(ce, float(floor), dtype=torch.float32),
    )
    gated = gated * valid.float()
    weights[:, :-1] = gated
    stats["kept_frac_valid"] = float((keep.float().sum() / valid.float().sum().clamp_min(1.0)).item())
    return weights, stats




def compute_weighted_task_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    token_weights: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute weighted next-token CE using [B, S] representation-aligned weights."""
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:].contiguous()
    valid = shift_labels != -100
    if not bool(valid.any()):
        return logits.new_zeros((), dtype=torch.float32), {"valid_frac": 0.0}

    safe_labels = shift_labels.masked_fill(~valid, 0)
    ce = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        safe_labels.reshape(-1),
        reduction="none",
    ).reshape_as(shift_labels)
    weights = token_weights[:, :-1].float() * valid.float()
    denom = weights.sum().clamp_min(1.0)
    loss = (ce * weights).sum() / denom
    active = weights > 0
    return loss, {
        "valid_frac": float(valid.float().mean().item()),
        "active_frac": float(active.float().mean().item()),
        "mean_weight": float(weights.mean().item()),
        "weight_sum": float(weights.sum().item()),
        "unweighted_ce": float((ce * valid.float()).sum().detach() / valid.float().sum().clamp_min(1.0)),
    }


def loss_routed_residual_recovery(
    routed_out: torch.Tensor,
    shared_anchor: torch.Tensor,
    teacher_mlp_out: torch.Tensor,
    reduction: str = "global_rmse",
    cast_fp32: bool = True,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Train routed output to recover the dense-minus-shared residual.

    This is the Phase113 residual-recovery objective:
        pred   = routed_out
        target = stopgrad(teacher_mlp_out - shared_anchor)

    ``shared_anchor`` is detached even when shared experts are trainable so this
    loss never pulls the anchor toward the routed residual. Use ``--freeze_shared``
    for the intended frozen-anchor setting.
    """
    if routed_out.dim() == 3:
        routed_out = routed_out.reshape(-1, routed_out.shape[-1])
    if shared_anchor.dim() == 3:
        shared_anchor = shared_anchor.reshape(-1, shared_anchor.shape[-1])
    if teacher_mlp_out.dim() == 3:
        teacher_mlp_out = teacher_mlp_out.reshape(-1, teacher_mlp_out.shape[-1])
    if cast_fp32:
        pred = routed_out.float()
        target = teacher_mlp_out.float() - shared_anchor.detach().float()
    else:
        pred = routed_out
        target = (teacher_mlp_out.to(pred.dtype) - shared_anchor.detach().to(pred.dtype))
    diff = pred - target
    return _reduce_squared_error(diff, reduction, sample_weight=sample_weight)


def _grad_group_norms(
    loss: torch.Tensor,
    model: nn.Module,
    *,
    retain_graph: bool = True,
) -> Dict[str, Any]:
    """Return first-order grad norms by CMoE module family for a scalar loss."""
    named_params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    params = [p for _name, p in named_params]
    grads = torch.autograd.grad(loss.float(), params, retain_graph=retain_graph, allow_unused=True) if params else []
    groups: Dict[str, Dict[str, float]] = {
        "router": {"tensors": 0.0, "params": 0.0, "l2_sq": 0.0, "max": 0.0},
        "routed": {"tensors": 0.0, "params": 0.0, "l2_sq": 0.0, "max": 0.0},
        "shared": {"tensors": 0.0, "params": 0.0, "l2_sq": 0.0, "max": 0.0},
        "lora": {"tensors": 0.0, "params": 0.0, "l2_sq": 0.0, "max": 0.0},
        "lm_head": {"tensors": 0.0, "params": 0.0, "l2_sq": 0.0, "max": 0.0},
        "other": {"tensors": 0.0, "params": 0.0, "l2_sq": 0.0, "max": 0.0},
    }
    for (name, param), grad in zip(named_params, grads):
        if "lora_" in name:
            group = "lora"
        elif "lm_head" in name:
            group = "lm_head"
        elif "phase127_mlp_router." in name or ".mlp.inner.gate." in name or ".mlp.gate." in name:
            group = "router"
        elif ".experts." in name and ".shared_experts." not in name:
            group = "routed"
        elif ".shared_experts." in name or ".shared_expert." in name:
            group = "shared"
        else:
            group = "other"
        groups[group]["params"] += float(param.numel())
        if grad is None:
            continue
        g = grad.detach().float()
        sq = float((g * g).sum().item())
        mx = float(g.abs().max().item())
        groups[group]["tensors"] += 1.0
        groups[group]["l2_sq"] += sq
        groups[group]["max"] = max(groups[group]["max"], mx)
    out: Dict[str, Any] = {}
    for group, data in groups.items():
        out[group] = {
            "tensors": int(data["tensors"]),
            "params": int(data["params"]),
            "grad_l2": float(data["l2_sq"] ** 0.5),
            "grad_max": float(data["max"]),
        }
    return out


def parse_layer_mask(value: str) -> Optional[set[int]]:
    """Parse comma/range layer mask strings such as ``3,22-27``."""
    text = (value or "").strip()
    if not text:
        return None
    out: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if lo > hi:
                raise ValueError(f"invalid descending layer range: {part!r}")
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    if not out or min(out) < 0:
        raise ValueError(f"invalid layer mask={value!r}")
    return out


def _layer_index_from_param_name(name: str) -> Optional[int]:
    m = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
    if m:
        return int(m.group(1))
    m = re.search(r"phase127_mlp_router\.heads\.(\d+)\.", name)
    if m:
        return int(m.group(1))
    return None


def apply_moe_trainable_layer_mask(model: nn.Module, mask_text: str) -> Dict[str, Any]:
    layer_mask = parse_layer_mask(mask_text)
    if layer_mask is None:
        return {"enabled": False, "layers": None, "frozen_params": 0, "kept_params": 0}
    moe_markers = (
        ".mlp.experts.",
        ".mlp.shared_expert.",
        ".mlp.shared_experts.",
        ".mlp.gate.",
        "phase127_mlp_router.heads.",
    )
    frozen = 0
    kept = 0
    for name, param in model.named_parameters():
        if not any(marker in name for marker in moe_markers):
            continue
        layer_idx = _layer_index_from_param_name(name)
        if layer_idx is None:
            continue
        if layer_idx in layer_mask:
            if param.requires_grad:
                kept += int(param.numel())
        elif param.requires_grad:
            param.requires_grad = False
            frozen += int(param.numel())
    return {"enabled": True, "layers": sorted(layer_mask), "frozen_params": frozen, "kept_params": kept}


# -----------------------------------------------------------------------------
# Model build (mirrors lmeval_cmoe.py)
# -----------------------------------------------------------------------------


def load_moe_model(
    args: argparse.Namespace,
    dtype: torch.dtype,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Load a Phase 1 MoE checkpoint: build base Llama, swap MLPs, load weights.

    Dispatches on ``args.moe_type``:
      - ``rrd``  : own ParamSplit/BasicSplit MoE (uses
                   :mod:`llamafactory.model.rrd_moe_llama.MoE` and the manifest
                   produced by ``scripts/exp_cmoe/carve_cmoe_exact_token_ids.py``).
      - ``cmoe`` : CMoE pre-LoRA carved checkpoint (uses
                   :mod:`llamafactory.model.cmoe_moe_llama.MoE` and the CMoE
                   manifest with keys ``nshared/nactivated/nexperts``).
      - ``dense``: unmodified Hugging Face dense causal LM used for the dense
                   SFT control.

    Returns:
        ``(model, tokenizer, manifest_normalized)`` where ``manifest_normalized``
        always carries ``moe_type``, ``n_experts``, ``n_active``, ``has_shared``,
        ``d_shared`` so downstream code can dispatch on ``moe_type``.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    moe_dir = args.moe_dir
    base_model_path = args.teacher_model_path

    if args.moe_type == "cmoe":
        return _load_moe_model_cmoe(args, dtype)
    if args.moe_type == "dense":
        if args.mode not in {"task_only", "lora"}:
            raise ValueError(
                "--moe_type=dense supports --mode=task_only or --mode=lora"
            )
        logger.info(
            "Loading dense SFT control from %s (dtype=%s)",
            base_model_path,
            dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=dtype,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        manifest = {
            "schema": "dense_sft_control_v1",
            "moe_type": "dense",
            "n_experts": 0,
            "n_active": 0,
            "n_routed_experts": 0,
            "has_shared": False,
            "d_shared": None,
            "source_model_path": os.path.abspath(base_model_path),
        }
        return model, tokenizer, manifest

    # ----- RRD path (original behaviour) -----
    manifest_path = os.path.join(moe_dir, "manifest.json")
    sd_path = os.path.join(moe_dir, "state_dict.pt")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    for k in ("n_experts", "n_active", "has_shared"):
        if k not in manifest:
            raise ValueError(f"Manifest {manifest_path} missing key {k!r}")

    logger.info("Loading base model from %s (dtype=%s)", base_model_path, dtype)
    model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=dtype)
    # Tokenizer: prefer the one stored alongside the manifest (matches teacher).
    tok_dir = moe_dir if os.path.exists(os.path.join(moe_dir, "tokenizer.model")) \
        or os.path.exists(os.path.join(moe_dir, "tokenizer.json")) else base_model_path
    tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = model.config
    hidden_size = int(config.hidden_size)
    intermediate_size = int(config.intermediate_size)
    n_experts = int(manifest["n_experts"])
    n_active = int(manifest["n_active"])
    has_shared = bool(manifest["has_shared"])
    d_shared = manifest.get("d_shared")
    d_shared = int(d_shared) if d_shared is not None else None

    logger.info(
        "Swapping MLPs with MoE skeleton: H=%d I=%d E=%d A=%d shared=%s d_shared=%s",
        hidden_size, intermediate_size, n_experts, n_active, has_shared, d_shared,
    )
    for layer in model.model.layers:
        moe = MoE(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            n_experts=n_experts,
            n_active=n_active,
            has_shared=has_shared,
            d_shared=d_shared,
            zero_init_shared=True,
        ).to(dtype)
        layer.mlp = moe

    logger.info("Loading state_dict from %s", sd_path)
    sd = torch.load(sd_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd and not any(
        k.startswith(("model.", "lm_head")) for k in sd.keys()
    ):
        sd = sd["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        logger.warning("missing keys: %d (first 5: %s)", len(missing), missing[:5])
    if unexpected:
        logger.warning("unexpected keys: %d (first 5: %s)", len(unexpected), unexpected[:5])

    # ----- Optional: zero-jitter ALL shared_expert weights -----
    # Re-init gate/up/down to N(0, std). Keeps shared_out ≈ 0 at step 0 while
    # opening a gradient path through every shared parameter (the default
    # zero-down init blocks gradient to gate/up via chain rule).
    if has_shared and args.shared_zero_jitter_std > 0.0:
        if args.shared_down_init != "zero":
            logger.warning(
                "--shared_zero_jitter_std=%g overrides --shared_down_init=%s.",
                args.shared_zero_jitter_std, args.shared_down_init,
            )
        std = float(args.shared_zero_jitter_std)
        n_layers_jit = 0
        for layer in model.model.layers:
            shared = getattr(layer.mlp, "shared_expert", None)
            if shared is None:
                continue
            with torch.no_grad():
                shared.gate_proj.weight.normal_(mean=0.0, std=std)
                shared.up_proj.weight.normal_(mean=0.0, std=std)
                shared.down_proj.weight.normal_(mean=0.0, std=std)
            n_layers_jit += 1
        logger.info(
            "shared_zero_jitter applied to %d layer(s): N(0, std=%g) on gate/up/down.",
            n_layers_jit, std,
        )

    # ----- Optional: re-init shared_expert.down_proj after load -----
    # Carves built with build_llama2_moe.py default to zero down_proj, which
    # keeps shared_out=0 at step 0 (residual learning). 'random'/'gate_up_match'
    # break that symmetry and let down_proj contribute a non-zero update from
    # step 0; useful as a residual-vs-symmetric init ablation.
    elif has_shared and args.shared_down_init != "zero":
        stds: List[float] = []
        for layer in model.model.layers:
            shared = getattr(layer.mlp, "shared_expert", None)
            if shared is None:
                continue
            dp = shared.down_proj.weight
            if args.shared_down_init == "random":
                init_std = (2.0 / hidden_size) ** 0.5
            else:  # gate_up_match
                gp_std = shared.gate_proj.weight.float().std().item()
                up_std = shared.up_proj.weight.float().std().item()
                init_std = 0.5 * (gp_std + up_std)
            with torch.no_grad():
                dp.normal_(mean=0.0, std=init_std)
            stds.append(init_std)
        if stds:
            logger.info(
                "shared_down_init=%s applied to %d layer(s): init_std min=%.6f mean=%.6f max=%.6f",
                args.shared_down_init, len(stds), min(stds), sum(stds) / len(stds), max(stds),
            )

    # Tag manifest with moe_type for downstream dispatch.
    manifest = {**manifest, "moe_type": "rrd"}
    return model, tokenizer, manifest


def _load_moe_model_cmoe(
    args: argparse.Namespace,
    dtype: torch.dtype,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """CMoE-track loader: swap LlamaMLP -> CMoE MoE then load carved state_dict."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Local import to avoid pulling CMoE module in RRD-only runs.
    from llamafactory.model.cmoe_moe_llama import (  # noqa: WPS433
        freeze_extra_bias_and_scale,
        load_cmoe_state_dict,
        swap_llama_mlp_to_cmoe_moe,
    )

    moe_dir = args.moe_dir
    base_model_path = args.teacher_model_path

    checkpoint_manifest_path = str(
        getattr(args, "cmoe_checkpoint_manifest", "") or ""
    ).strip()
    checkpoint_manifest_raw: Dict[str, Any] = {}
    if checkpoint_manifest_path:
        checkpoint_manifest_path = os.path.abspath(
            os.path.expanduser(checkpoint_manifest_path)
        )
        if not os.path.isfile(checkpoint_manifest_path):
            raise FileNotFoundError(
                f"CMoE checkpoint manifest not found: {checkpoint_manifest_path}"
            )
        with open(checkpoint_manifest_path, "r") as f:
            checkpoint_manifest_raw = json.load(f)
        args._source_checkpoint_manifest_path = checkpoint_manifest_path
        args._source_checkpoint_manifest_sha256 = _sha256_path(
            checkpoint_manifest_path
        )

    sibling_state_dict = (
        os.path.join(os.path.dirname(checkpoint_manifest_path), "state_dict.pt")
        if checkpoint_manifest_path
        else ""
    )
    sd_path = (
        args.cmoe_state_dict_path
        or (sibling_state_dict if sibling_state_dict and os.path.isfile(sibling_state_dict) else "")
        or os.path.join(moe_dir, "state_dict.pt")
    )
    if not os.path.exists(sd_path):
        raise FileNotFoundError(
            f"CMoE state_dict not found: {sd_path} "
            f"(set --cmoe_state_dict_path or place state_dict.pt under --moe_dir)"
        )

    manifest_path = os.path.join(moe_dir, "manifest.json")
    manifest_raw: Dict[str, Any] = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            manifest_raw = json.load(f)
    else:
        logger.warning(
            "CMoE manifest %s not found — falling back to CLI defaults "
            "(cmoe_n_experts=%d, cmoe_n_activated=%d, cmoe_n_shared=%d)",
            manifest_path, args.cmoe_n_experts, args.cmoe_n_activated, args.cmoe_n_shared,
        )
    if checkpoint_manifest_raw:
        manifest_raw = checkpoint_manifest_raw

    # CMoE manifest naming differs: nexperts / nactivated / nshared.
    n_experts = int(manifest_raw.get("nexperts", args.cmoe_n_experts))
    n_activated = int(manifest_raw.get("nactivated", args.cmoe_n_activated))
    n_shared = int(manifest_raw.get("nshared", args.cmoe_n_shared))

    logger.info(
        "Loading base model from %s (dtype=%s) for CMoE-track build",
        base_model_path, dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=dtype)

    tok_dir = moe_dir if os.path.exists(os.path.join(moe_dir, "tokenizer.model")) \
        or os.path.exists(os.path.join(moe_dir, "tokenizer.json")) else base_model_path
    tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    add_eas = bool(
        manifest_raw.get(
            "add_eas",
            manifest_raw.get("cpt_add_eas", args.cmoe_add_eas),
        )
    )
    logger.info(
        "Swapping MLPs to CMoE MoE: E=%d (routed=%d) A=%d shared=%d add_eas=%s eas_init_std=%g",
        n_experts, n_experts - n_shared, n_activated, n_shared,
        add_eas, args.cmoe_eas_init_std,
    )
    swap_llama_mlp_to_cmoe_moe(
        model,
        n_experts=n_experts,
        n_activated=n_activated,
        n_shared=n_shared,
        add_eas=add_eas,
        eas_init_std=float(args.cmoe_eas_init_std),
    )

    checkpoint_router_arch = (
        manifest_raw.get("router_arch") or manifest_raw.get("cpt_router_arch")
    )
    if checkpoint_manifest_raw and is_mlp_router_arch(checkpoint_router_arch):
        if build_probe_from_manifest is None or attach_mlp_probe_router is None:
            raise RuntimeError("MLP router helpers are unavailable")
        if args.init_mlp_router_probe_dir or bool(args.init_mlp_router_random):
            raise ValueError(
                "--cmoe_checkpoint_manifest with an embedded MLP router cannot "
                "be combined with --init_mlp_router_probe_dir or "
                "--init_mlp_router_random"
            )
        cli_router_arch = str(getattr(args, "router_arch", "default"))
        if cli_router_arch not in {"default", str(checkpoint_router_arch)}:
            raise ValueError(
                "checkpoint/CLI router mismatch: "
                f"manifest={checkpoint_router_arch!r}, CLI={cli_router_arch!r}"
            )
        args.router_arch = str(checkpoint_router_arch)
        args.mlp_router_aggregation = str(
            manifest_raw.get("cpt_mlp_router_aggregation", "uniform")
        )
        args.mlp_router_layer_mask = str(
            manifest_raw.get("cpt_mlp_router_layer_mask") or ""
        )
        probe = build_probe_from_manifest(manifest_raw, model, dtype=dtype)
        attach_mlp_probe_router(
            model,
            probe,
            aggregation=args.mlp_router_aggregation,
            layer_mask=args.mlp_router_layer_mask,
        )
        args._preloaded_mlp_router = True
        args._mlp_router_manifest_fields = {
            key: value
            for key, value in manifest_raw.items()
            if key in {"router_arch", "cpt_router_arch"}
            or key.startswith("cpt_mlp_router_")
        }
        args._mlp_router_manifest_fields.update(
            {
                "router_arch": str(checkpoint_router_arch),
                "cpt_router_arch": str(checkpoint_router_arch),
                "cpt_mlp_router_aggregation": args.mlp_router_aggregation,
                "cpt_mlp_router_layer_mask": args.mlp_router_layer_mask,
                "cpt_mlp_router_parameter_count": int(
                    sum(param.numel() for param in probe.parameters())
                ),
                "cpt_mlp_router_bootstrap": "embedded_full_checkpoint_strict",
            }
        )
        logger.info(
            "Attached checkpoint MLP router skeleton before state load: "
            "arch=%s params=%.2fM aggregation=%s layer_mask=%s",
            checkpoint_router_arch,
            sum(param.numel() for param in probe.parameters()) / 1e6,
            args.mlp_router_aggregation,
            args.mlp_router_layer_mask or "<all>",
        )

    logger.info("Loading CMoE state_dict from %s", sd_path)
    # A continuation checkpoint must reproduce the exact architecture, including
    # any embedded MLP router. Never silently discard continuation weights.
    if checkpoint_manifest_raw:
        load_cmoe_state_dict(model, sd_path, strict=True)
    else:
        # Legacy carve loading retains its historical fallback for tied
        # lm_head/embedding differences across HF checkpoints.
        try:
            load_cmoe_state_dict(model, sd_path, strict=True)
        except RuntimeError as e:
            logger.warning("strict load failed; retrying with strict=False (%s)", e)
            sd = torch.load(sd_path, map_location="cpu")
            missing, unexpected = model.load_state_dict(sd, strict=False)
            if missing:
                logger.warning("missing keys: %d (first 5: %s)", len(missing), missing[:5])
            if unexpected:
                logger.warning("unexpected keys: %d (first 5: %s)", len(unexpected), unexpected[:5])

    if checkpoint_manifest_raw and args.cmoe_extra_scale_init is not None:
        raise ValueError(
            "--cmoe_extra_scale_init would mutate a continuation checkpoint "
            "at step 0; omit it when using --cmoe_checkpoint_manifest"
        )

    # Fresh carve runs retain the historical RRD contract: reset extra_bias and
    # disable its auto-update path. Continuation checkpoints must preserve the
    # learned extra_bias buffer; only runtime/trainability flags are changed.
    extra_scale_trainable = bool(args.cmoe_extra_scale_trainable)
    for layer in model.model.layers:
        if checkpoint_manifest_raw:
            layer.mlp.cus_training = False
            layer.mlp.gate.extra_scale.requires_grad = extra_scale_trainable
        else:
            freeze_extra_bias_and_scale(
                layer.mlp,
                extra_scale_trainable=extra_scale_trainable,
            )
        extra_scale_init = getattr(args, "cmoe_extra_scale_init", None)
        if extra_scale_init is not None:
            layer.mlp.gate.extra_scale.data.fill_(float(extra_scale_init))

    # Normalize manifest keys to RRD-style for downstream compatibility.
    intermediate_size = int(model.config.intermediate_size)
    if n_experts == 0:
        raise ValueError("cmoe n_experts must be > 0")
    moe_inter_dim = intermediate_size // n_experts
    d_shared = (n_shared * moe_inter_dim) if n_shared > 0 else None
    manifest_norm: Dict[str, Any] = {
        **manifest_raw,
        "moe_type": "cmoe",
        "n_experts": n_experts,
        "n_active": n_activated,
        "n_routed_experts": n_experts - n_shared,
        "has_shared": n_shared > 0,
        "d_shared": d_shared,
    }
    return model, tokenizer, manifest_norm


# -----------------------------------------------------------------------------
# Component capture wrapper for RRD mode
# -----------------------------------------------------------------------------


class _CaptureWrapper(nn.Module):
    """Wraps an ``MoE`` module: forward calls inner with ``return_components=True``
    and stashes the dict in ``sink[layer_idx]``, returning only the combined
    output tensor for upstream Llama layer consumers.
    """

    def __init__(
        self,
        moe_module: MoE,
        layer_idx: int,
        sink: Dict[int, Dict[str, torch.Tensor]],
        router_train_routing_mode: str = "hard_topk",
        router_relax_epsilon: float = 0.05,
    ) -> None:
        super().__init__()
        self.inner = moe_module
        self.layer_idx = layer_idx
        self.sink = sink
        self.router_train_routing_mode = router_train_routing_mode
        self.router_relax_epsilon = float(router_relax_epsilon)
        # EM Phase M: per-step oracle routing override. Set by training loop
        # before student forward; cleared after capture.
        self.override_topk_indices: Optional[torch.Tensor] = None
        self.override_topk_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        kwargs: Dict[str, Any] = {"return_components": True}
        # EM Phase M overrides only apply to the RRD MoE (rrd_moe_llama). The CMoE
        # MoE (cmoe_moe_llama) does not accept these kwargs — only forward them
        # when set, so the cmoe-track stays compatible.
        if self.override_topk_indices is not None:
            kwargs["override_topk_indices"] = self.override_topk_indices
        if self.override_topk_weights is not None:
            kwargs["override_topk_weights"] = self.override_topk_weights
        if hasattr(self.inner, "n_routed_experts") and hasattr(self.inner, "n_shared_experts"):
            kwargs["router_train_routing_mode"] = self.router_train_routing_mode
            kwargs["router_relax_epsilon"] = self.router_relax_epsilon
        result = self.inner.forward(x, **kwargs)
        # Stash student post-attn LN output (= MLP input) for the frozen oracle
        # forward path. Detached so the oracle's separate forward does not pull
        # gradient through the student graph.
        result["mlp_input"] = x.detach()
        self.sink[self.layer_idx] = result  # contains output, routed_out, shared_out, ...
        return result["output"]


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------




def _layer_local_components(
    student: nn.Module,
    layer_idx: int,
    mlp_input: torch.Tensor,
    router_train_routing_mode: str = "hard_topk",
    router_relax_epsilon: float = 0.05,
) -> Dict[str, torch.Tensor]:
    """Re-forward one layer MoE from a detached input for local aux gradients."""
    wrapper = decoder_layers(student)[int(layer_idx)].mlp
    inner = getattr(wrapper, "inner", None)
    if inner is None:
        raise ValueError("--aux_loss_scope=layer_local requires captured MoE wrappers")
    kwargs: Dict[str, Any] = {"return_components": True}
    if hasattr(inner, "n_routed_experts") and hasattr(inner, "n_shared_experts"):
        kwargs["router_train_routing_mode"] = router_train_routing_mode
        kwargs["router_relax_epsilon"] = router_relax_epsilon
    return inner.forward(mlp_input.detach(), **kwargs)


def _audit_aux_layer_locality(
    student: nn.Module,
    layer_idx: int,
    aux_loss: torch.Tensor,
) -> Dict[str, Any]:
    """Check that a layer-local aux loss only touches the selected layer MLP."""
    named_params = [(name, p) for name, p in student.named_parameters() if p.requires_grad]
    names = [name for name, _p in named_params]
    params = [p for _name, p in named_params]
    grads = torch.autograd.grad(
        aux_loss,
        params,
        retain_graph=True,
        allow_unused=True,
    )
    layer_scope = f"model.layers.{int(layer_idx)}.mlp."
    nonzero: List[Tuple[str, float]] = []
    in_scope = 0
    out_scope = 0
    max_abs = 0.0
    for name, grad in zip(names, grads):
        if grad is None:
            continue
        gmax = float(grad.detach().float().abs().max().item())
        if gmax <= 0.0:
            continue
        nonzero.append((name, gmax))
        max_abs = max(max_abs, gmax)
        if layer_scope in name:
            in_scope += 1
        else:
            out_scope += 1
    report = {
        "layer_idx": int(layer_idx),
        "layer_scope": layer_scope,
        "nonzero_tensors": len(nonzero),
        "in_scope_nonzero_tensors": in_scope,
        "out_scope_nonzero_tensors": out_scope,
        "max_abs_grad": max_abs,
        "nonzero_names_head": [name for name, _gmax in nonzero[:20]],
        "out_of_scope_names_head": [
            name for name, _gmax in nonzero if layer_scope not in name
        ][:20],
    }
    logger.info("Aux layer-locality audit: %s", report)
    if in_scope == 0 or out_scope != 0:
        raise RuntimeError(f"Aux layer-locality audit failed: {report}")
    return report


class WikiTextChunk(Dataset):
    """Chunk WikiText-2 jsonl ``{"text": ...}`` into fixed-size token windows.

    Two packing modes:
      - ``per_doc`` (default, Llama-2 reproducibility): each document is
        tokenized independently; documents shorter than ``max_seqlen`` are
        skipped. Yields ~2046/2048 chunks for the CMoE jsonl + Llama-2
        tokenizer.
      - ``concat``: concatenate all documents (separated by EOS if available)
        then slice into non-overlapping ``max_seqlen`` windows. Required for
        tokenizers that produce fewer tokens per doc than Llama-2 (e.g.
        Qwen2.5 BPE), where ``per_doc`` filters out every sample.
    """

    def __init__(
        self,
        tokenizer: Any,
        jsonl_path: str,
        max_seqlen: int,
        packing: str = "per_doc",
        data_split: str = "all",
    ) -> None:
        if packing not in {"per_doc", "concat"}:
            raise ValueError(f"packing must be 'per_doc' or 'concat', got {packing!r}")
        if data_split not in {"all", "first_half", "second_half"}:
            raise ValueError(f"data_split must be 'all'/'first_half'/'second_half', got {data_split!r}")
        self.max_seqlen = max_seqlen
        with open(jsonl_path, "r") as f:
            texts = [json.loads(line)["text"] for line in f if line.strip()]
        self.samples: List[torch.Tensor] = []

        if packing == "per_doc":
            for text in texts:
                ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt").squeeze(0)
                n = ids.numel()
                if n < max_seqlen:
                    continue
                self.samples.append(ids[:max_seqlen].long())
        else:  # concat
            eos_id = getattr(tokenizer, "eos_token_id", None)
            buf: List[int] = []
            for text in texts:
                ids = tokenizer.encode(text, add_special_tokens=False)
                buf.extend(ids)
                if eos_id is not None:
                    buf.append(eos_id)
            n_full = len(buf) // max_seqlen
            for i in range(n_full):
                chunk = buf[i * max_seqlen:(i + 1) * max_seqlen]
                self.samples.append(torch.tensor(chunk, dtype=torch.long))

        # Deterministic split: keep contiguous halves so Phase M / Phase E see
        # disjoint subsets of the calibration corpus. ``all`` (default) is a no-op.
        if data_split != "all":
            half = len(self.samples) // 2
            if data_split == "first_half":
                self.samples = self.samples[:half]
            else:  # second_half
                self.samples = self.samples[half:]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        ids = self.samples[i]
        return {"input_ids": ids, "labels": ids.clone()}


class TokenIdsChunk(Dataset):
    """Read exact pre-tokenized fixed-length chunks from JSONL.

    Each row must contain ``{"input_ids": [...]}``. This avoids a decode and
    re-tokenize round trip, so the dataset's token count is exactly auditable.
    """

    def __init__(
        self,
        jsonl_path: str,
        max_seqlen: int,
        data_split: str = "all",
    ) -> None:
        if data_split not in {"all", "first_half", "second_half"}:
            raise ValueError(f"data_split must be 'all'/'first_half'/'second_half', got {data_split!r}")
        self.samples: List[torch.Tensor] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                ids = row.get("input_ids")
                if not isinstance(ids, list) or not ids:
                    raise ValueError(f"{jsonl_path}:{line_no} missing non-empty list field 'input_ids'")
                if len(ids) != int(max_seqlen):
                    raise ValueError(
                        f"{jsonl_path}:{line_no} has {len(ids)} tokens, expected max_seqlen={max_seqlen}"
                    )
                self.samples.append(torch.tensor([int(x) for x in ids], dtype=torch.long))
        if data_split != "all":
            half = len(self.samples) // 2
            self.samples = self.samples[:half] if data_split == "first_half" else self.samples[half:]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        ids = self.samples[i]
        return {"input_ids": ids, "labels": ids.clone()}


class SupervisedTokenIdsDataset(Dataset):
    """Read exact tokenized SFT rows with response-only labels."""

    def __init__(
        self,
        jsonl_path: str,
        max_seqlen: int,
        data_split: str = "all",
    ) -> None:
        if data_split not in {"all", "first_half", "second_half"}:
            raise ValueError(
                "data_split must be 'all'/'first_half'/'second_half', "
                f"got {data_split!r}"
            )
        rows: List[Dict[str, List[int]]] = []
        with open(jsonl_path, "r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                input_ids = row.get("input_ids")
                labels = row.get("labels")
                if (
                    not isinstance(input_ids, list)
                    or not isinstance(labels, list)
                    or not input_ids
                    or len(input_ids) != len(labels)
                ):
                    raise ValueError(
                        f"{jsonl_path}:{line_no} requires non-empty, "
                        "equal-length input_ids and labels lists"
                    )
                rows.append(
                    {
                        "input_ids": [int(value) for value in input_ids],
                        "labels": [int(value) for value in labels],
                    }
                )
        if data_split != "all":
            half = len(rows) // 2
            rows = rows[:half] if data_split == "first_half" else rows[half:]

        self.samples: List[Dict[str, torch.Tensor]] = []
        for row in rows:
            input_ids = row["input_ids"][:max_seqlen]
            labels = row["labels"][:max_seqlen]
            if not any(label != -100 for label in labels):
                continue
            attention_mask = [1] * len(input_ids)
            self.samples.append(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                    "attention_mask": torch.tensor(
                        attention_mask, dtype=torch.long
                    ),
                }
            )
        if not self.samples:
            raise RuntimeError(
                "No supervised token-id samples with active response labels "
                f"were produced from {jsonl_path}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return self.samples[i]


class NpyTokenIdsChunk(Dataset):
    """Memory-map exact fixed-length token windows from a 2-D NumPy array."""

    def __init__(
        self,
        npy_path: str,
        max_seqlen: int,
        data_split: str = "all",
    ) -> None:
        if data_split not in {"all", "first_half", "second_half"}:
            raise ValueError(
                "data_split must be 'all'/'first_half'/'second_half', "
                f"got {data_split!r}"
            )
        array = np.load(npy_path, mmap_mode="r")
        if array.ndim != 2 or int(array.shape[1]) != int(max_seqlen):
            raise ValueError(
                f"{npy_path} shape={array.shape} must be [num_windows, {max_seqlen}]"
            )
        if not np.issubdtype(array.dtype, np.integer):
            raise ValueError(f"{npy_path} dtype={array.dtype} must be integer")
        start, stop = 0, int(array.shape[0])
        if data_split != "all":
            half = stop // 2
            start, stop = (0, half) if data_split == "first_half" else (half, stop)
        self.array = array
        self.start = start
        self.stop = stop

    def __len__(self) -> int:
        return self.stop - self.start

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        if i < 0 or i >= len(self):
            raise IndexError(i)
        ids = torch.from_numpy(
            np.asarray(self.array[self.start + i], dtype=np.int64).copy()
        )
        return {"input_ids": ids, "labels": ids.clone()}


class ContinuationOnlySFT(Dataset):
    """Load prefix + continuation JSONL for continuation-only SFT.

    Expected schema:
      - ``prefix_text``: prompt/context text
      - a configurable continuation field such as ``teacher_continuation_text``
        or ``gold_continuation_text``

    Labels are masked over the prefix so CE trains only on the continuation.
    Samples are padded/truncated to ``max_seqlen`` in the dataset so the existing
    stack-only collator remains compatible with fixed-window CPT runs.
    """

    def __init__(
        self,
        tokenizer: Any,
        jsonl_path: str,
        max_seqlen: int,
        data_split: str = "all",
        continuation_field: str = "teacher_continuation_text",
        dataset_label: str = "teacher-rollout",
    ) -> None:
        if data_split not in {"all", "first_half", "second_half"}:
            raise ValueError(f"data_split must be 'all'/'first_half'/'second_half', got {data_split!r}")
        self.samples: List[Dict[str, torch.Tensor]] = []
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            raise ValueError("Tokenizer must define pad_token_id or eos_token_id.")

        with open(jsonl_path, "r") as f:
            records = [json.loads(line) for line in f if line.strip()]
        if data_split != "all":
            half = len(records) // 2
            records = records[:half] if data_split == "first_half" else records[half:]

        for rec in records:
            prefix_text = rec.get("prefix_text") or rec.get("prompt_text")
            continuation_text = rec.get(continuation_field)
            if not prefix_text or not continuation_text:
                continue

            prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
            continuation_ids = tokenizer.encode(continuation_text, add_special_tokens=False)
            if not prefix_ids or not continuation_ids:
                continue

            if len(prefix_ids) >= max_seqlen:
                prefix_ids = prefix_ids[-(max_seqlen - 1):]
            max_response_len = max_seqlen - len(prefix_ids)
            continuation_ids = continuation_ids[:max_response_len]
            if not continuation_ids:
                continue

            ids = prefix_ids + continuation_ids
            labels = [-100] * len(prefix_ids) + continuation_ids
            attention = [1] * len(ids)
            pad_len = max_seqlen - len(ids)
            if pad_len > 0:
                ids.extend([int(pad_token_id)] * pad_len)
                labels.extend([-100] * pad_len)
                attention.extend([0] * pad_len)

            self.samples.append(
                {
                    "input_ids": torch.tensor(ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                    "attention_mask": torch.tensor(attention, dtype=torch.long),
                }
            )

        if not self.samples:
            raise RuntimeError(
                f"No {dataset_label} continuation-only SFT samples produced from {jsonl_path}; "
                f"continuation_field={continuation_field} max_seqlen={max_seqlen} split={data_split}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return self.samples[i]


class TeacherRolloutSFT(ContinuationOnlySFT):
    """Backward-compatible dense-teacher rollout SFT dataset."""

    def __init__(
        self,
        tokenizer: Any,
        jsonl_path: str,
        max_seqlen: int,
        data_split: str = "all",
        continuation_field: Optional[str] = None,
    ) -> None:
        super().__init__(
            tokenizer=tokenizer,
            jsonl_path=jsonl_path,
            max_seqlen=max_seqlen,
            data_split=data_split,
            continuation_field=continuation_field or "teacher_continuation_text",
            dataset_label="teacher-rollout",
        )


class RawContinuationSFT(ContinuationOnlySFT):
    """Raw corpus continuation SFT using the same prefix-mask policy as rollouts."""

    def __init__(
        self,
        tokenizer: Any,
        jsonl_path: str,
        max_seqlen: int,
        data_split: str = "all",
        continuation_field: Optional[str] = None,
    ) -> None:
        super().__init__(
            tokenizer=tokenizer,
            jsonl_path=jsonl_path,
            max_seqlen=max_seqlen,
            data_split=data_split,
            continuation_field=continuation_field or "gold_continuation_text",
            dataset_label="raw-continuation",
        )


class CMoEInMemoryWikiText(Dataset):
    """In-memory WikiText-2 random crops via CMoE/datautils.get_wikitext2.

    Token-perfect parity with CMoE/run_cmoe.py + simple_sft.py: same
    random.randint(seed=...) offsets into ``tokenizer("\\n\\n".join(train.text))``,
    no decode→re-tokenize round-trip (which the jsonl path goes through).
    """

    def __init__(self, model_path: str, nsamples: int, seed: int, seqlen: int,
                 cmoe_repo_path: str) -> None:
        if cmoe_repo_path not in sys.path:
            sys.path.insert(0, cmoe_repo_path)
        from datautils import get_loaders  # type: ignore  # CMoE/datautils.py
        # bsz=1: easier to flatten back into per-sample tensors below.
        trainloader, _ = get_loaders(
            "wikitext2", nsamples=nsamples, seed=seed, model=model_path,
            seqlen=seqlen, bsz=1,
        )
        # trainloader = list of (batch_input [1, seqlen], batch_target [1, seqlen]).
        # We use input_ids only (CMoE simple_sft.py does labels=input_ids,
        # ignoring the get_wikitext2 target which masks all-but-last token).
        self.samples: List[torch.Tensor] = [b[0].squeeze(0).long() for b in trainloader]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        ids = self.samples[i]
        return {"input_ids": ids, "labels": ids.clone()}


def _collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    out = {
        "input_ids": torch.stack([b["input_ids"] for b in batch], dim=0),
        "labels": torch.stack([b["labels"] for b in batch], dim=0),
    }
    if "attention_mask" in batch[0]:
        out["attention_mask"] = torch.stack([b["attention_mask"] for b in batch], dim=0)
    return out


def decoder_layers(model: Any) -> Any:
    """Return the underlying decoder layer list, including under PEFT wrappers."""
    seen: set[int] = set()
    cur = model
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if hasattr(cur, "layers"):
            return cur.layers
        inner_model = getattr(cur, "model", None)
        if inner_model is not None and hasattr(inner_model, "layers"):
            return inner_model.layers
        nested_model = getattr(inner_model, "model", None) if inner_model is not None else None
        if nested_model is not None and hasattr(nested_model, "layers"):
            return nested_model.layers
        base_model = getattr(cur, "base_model", None)
        if base_model is not None and id(base_model) not in seen:
            cur = base_model
            continue
        if inner_model is not None and id(inner_model) not in seen:
            cur = inner_model
            continue
        break
    raise AttributeError("Could not locate decoder layers on model/PEFT wrapper")


# -----------------------------------------------------------------------------
# Freezing
# -----------------------------------------------------------------------------


# Substring matchers used by `freeze_non_moe`. Each key MUST end with the
# trailing dot/identifier-boundary that disambiguates it from sibling names —
# in particular, ``mlp.shared_expert.`` (RRD, singular) must NOT match
# ``mlp.shared_experts.<...>`` (CMoE, plural) and vice versa.
_TRAINABLE_KEYS_RRD: Tuple[str, ...] = (
    "mlp.gate.gate",        # RRD Router (single nn.Linear at .gate.gate)
    "mlp.experts.",         # routed experts
    "mlp.shared_expert.",   # RRD shared expert (singular!)
)
_TRAINABLE_KEYS_CMOE: Tuple[str, ...] = (
    "mlp.gate.classifier",  # CMoE Router second Linear
    "mlp.gate.gate",        # CMoE Router first Linear (key collision-safe with RRD)
    "mlp.gate.extra_scale", # CMoE per-expert scale Parameter (requires_grad gated by helper)
    "mlp.experts.",         # routed experts (LlamaMLP)
    "mlp.shared_experts.",  # CMoE shared expert (plural)
    "mlp.eas.",             # CMoE Extra Additional Shared expert (optional, --cmoe_add_eas=1)
)


def enable_cmoe_cus_training(model: Any) -> int:
    """Enable CMoE bias load-balancing (``cus_training=True``) on every layer's
    MoE. Used by ``--mode lora`` to mirror the CMoE simple_sft.py default
    behaviour (extra_bias is auto-updated; extra_scale is trainable).

    Returns:
        Number of layers on which the flag was flipped.
    """
    n = 0
    for layer in decoder_layers(model):
        if hasattr(layer.mlp, "cus_training"):
            layer.mlp.cus_training = True
            n += 1
    return n


def freeze_non_moe(model: Any, moe_type: str = "rrd") -> Tuple[int, int]:
    """Freeze everything except router/experts/shared. Dispatches the matcher
    set on ``moe_type`` so RRD and CMoE keys do not cross-leak.

    Args:
        model: HuggingFace model with ``model.layers[*].mlp`` swapped to either
            an RRD or CMoE MoE module.
        moe_type: ``"rrd"`` (default), ``"cmoe"``, or ``"dense"``. Dense
            controls keep every parameter trainable.

    Returns:
        ``(n_trainable, n_total)`` parameter counts (numel sums).

    Note:
        For CMoE, ``mlp.gate.extra_scale`` is matched as a "trainable" name
        here, but the actual ``requires_grad`` flag is set by
        :func:`freeze_extra_bias_and_scale` (called during model load) per
        ``--cmoe_extra_scale_trainable``. We deliberately keep this loop
        idempotent with that helper: if ``extra_scale_trainable=False`` the
        helper has already set ``requires_grad=False`` on it; this function
        will then re-flip it to True. To preserve the helper's freeze decision,
        we honour the existing flag for ``extra_scale``.
    """
    keys = _TRAINABLE_KEYS_CMOE if moe_type == "cmoe" else _TRAINABLE_KEYS_RRD
    n_trainable = 0
    n_total = 0
    for name, p in model.named_parameters():
        is_trainable = moe_type == "dense" or any(s in name for s in keys)
        # Preserve the freeze decision made by `freeze_extra_bias_and_scale`
        # for the CMoE-only `extra_scale` parameter — do not flip it back to
        # True if the helper already froze it.
        if (
            moe_type == "cmoe"
            and "mlp.gate.extra_scale" in name
            and not p.requires_grad
        ):
            is_trainable = False
        p.requires_grad = bool(is_trainable)
        n_total += p.numel()
        if is_trainable:
            n_trainable += p.numel()
    return n_trainable, n_total


def classify_moe_param_group(name: str) -> str:
    """Classify MoE parameter names for audit logs.

    Training wraps each layer MLP in ``_CaptureWrapper`` for RRD losses, which
    changes names from ``mlp.experts`` to ``mlp.inner.experts``. Keep matching
    tolerant to that wrapper while retaining explicit MoE component groups.
    """
    if (
        "phase127_mlp_router." in name
        or ".gate.gate" in name
        or ".gate.classifier" in name
        or ".gate.extra_scale" in name
    ):
        return "router"
    if ".mlp.experts." in name or ".mlp.inner.experts." in name:
        return "routed"
    if (
        ".mlp.shared_expert." in name
        or ".mlp.shared_experts." in name
        or ".mlp.inner.shared_expert." in name
        or ".mlp.inner.shared_experts." in name
    ):
        return "shared"
    if ".mlp.eas." in name or ".mlp.inner.eas." in name:
        return "eas"
    if any(
        marker in name
        for marker in (
            ".self_attn.q_proj.",
            ".self_attn.k_proj.",
            ".self_attn.v_proj.",
            ".self_attn.o_proj.",
        )
    ):
        return "attention"
    return "other"


def _set_attention_trainable(model: Any, enabled: bool) -> int:
    """Set only q/k/v/o self-attention projections trainable."""
    count = 0
    for name, param in model.named_parameters():
        if classify_moe_param_group(name) != "attention":
            continue
        param.requires_grad = bool(enabled)
        if enabled:
            count += int(param.numel())
    return count


def _record_and_validate_trainable_policy(
    model: Any, args: argparse.Namespace
) -> Dict[str, Any]:
    """Record the exact trainable surface and fail on freeze-policy leakage."""
    allowed = {"router", "routed", "shared", "eas"}
    if bool(args.train_attention):
        allowed.add("attention")
    if bool(args.train_lm_head):
        allowed.add("lm_head")

    names: List[str] = []
    counts: Dict[str, int] = {
        "router": 0,
        "routed": 0,
        "shared": 0,
        "eas": 0,
        "attention": 0,
        "lm_head": 0,
        "other": 0,
    }
    leaked: List[str] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        group = (
            "lm_head"
            if name.startswith("lm_head.")
            else classify_moe_param_group(name)
        )
        names.append(name)
        counts[group] = counts.get(group, 0) + int(param.numel())
        if group not in allowed:
            leaked.append(name)

    if leaked:
        raise RuntimeError(
            "Trainable freeze-policy leak: expected only "
            f"{sorted(allowed)}, found {len(leaked)} unexpected tensors; "
            f"first={leaked[:8]}"
        )
    if bool(args.train_attention) and counts["attention"] == 0:
        raise RuntimeError(
            "--train_attention=1 matched zero q/k/v/o projection parameters"
        )
    if not bool(args.train_attention) and counts["attention"] != 0:
        raise RuntimeError(
            "--train_attention=0 but attention parameters remain trainable"
        )
    if bool(args.freeze_router) and counts["router"] != 0:
        raise RuntimeError(
            f"--freeze_router set but router parameters remain trainable: {counts}"
        )
    if not bool(args.freeze_router) and counts["router"] == 0:
        raise RuntimeError(
            f"CMoE CE policy missing trainable router parameters: {counts}"
        )
    if (counts["routed"] + counts["shared"]) == 0:
        raise RuntimeError(
            f"CMoE CE policy missing trainable expert parameters: {counts}"
        )
    return {
        "groups": counts,
        "tensor_count": len(names),
        "parameter_count": sum(counts.values()),
        "names": sorted(names),
        "allowed_groups": sorted(allowed),
    }


# -----------------------------------------------------------------------------
# Teacher hooks (RRD mode only)
# -----------------------------------------------------------------------------


def install_teacher_hooks(
    teacher: Any,
    teacher_mlp_outs: Dict[int, torch.Tensor],
    teacher_post_ln: Dict[int, torch.Tensor],
) -> List[Any]:
    """Install forward hooks on teacher's per-layer MLP and post_attention_layernorm."""
    handles = []

    def make_mlp_hook(idx: int):
        def hook(_mod: Any, _inp: Any, out: torch.Tensor) -> None:
            teacher_mlp_outs[idx] = out.detach()
        return hook

    def make_ln_hook(idx: int):
        def hook(_mod: Any, _inp: Any, out: torch.Tensor) -> None:
            teacher_post_ln[idx] = out.detach()
        return hook

    for i, layer in enumerate(teacher.model.layers):
        handles.append(layer.mlp.register_forward_hook(make_mlp_hook(i)))
        handles.append(layer.post_attention_layernorm.register_forward_hook(make_ln_hook(i)))
    return handles


def wrap_student_for_capture(
    student: Any,
    sink: Dict[int, Dict[str, torch.Tensor]],
    router_train_routing_mode: str = "hard_topk",
    router_relax_epsilon: float = 0.05,
) -> None:
    """Wrap each ``layer.mlp`` (an ``MoE``) with ``_CaptureWrapper``. Must be
    called AFTER freeze_non_moe so requires_grad flags are already set on the
    inner MoE parameters (wrapper preserves them via Module composition)."""
    for i, layer in enumerate(decoder_layers(student)):
        if isinstance(layer.mlp, _CaptureWrapper):
            layer.mlp.router_train_routing_mode = router_train_routing_mode
            layer.mlp.router_relax_epsilon = float(router_relax_epsilon)
            continue
        layer.mlp = _CaptureWrapper(
            layer.mlp,
            i,
            sink,
            router_train_routing_mode=router_train_routing_mode,
            router_relax_epsilon=router_relax_epsilon,
        )


def unwrap_student_capture(student: Any) -> None:
    """Reverse :func:`wrap_student_for_capture` so saved state_dict has the
    canonical ``layer.mlp.<...>`` key namespace (matches lmeval_cmoe loader)."""
    for layer in decoder_layers(student):
        if isinstance(layer.mlp, _CaptureWrapper):
            layer.mlp = layer.mlp.inner


def _strip_capture_keys(sd: Dict[str, Any]) -> Dict[str, Any]:
    """Rename ``...mlp.inner...`` -> ``...mlp...`` in a state_dict snapshot
    without touching the model in-place. Used by intermediate-save path so
    training can continue uninterrupted after the save."""
    out: Dict[str, Any] = {}
    for k, v in sd.items():
        if ".mlp.inner." in k:
            out[k.replace(".mlp.inner.", ".mlp.", 1)] = v
        else:
            out[k] = v
    return out


def _canonical_key(name: str) -> str:
    key = name
    if key.startswith("base_model.model."):
        key = key.removeprefix("base_model.model.")
    return key.replace(".mlp.inner.", ".mlp.", 1)


def _trainable_delta_state(student: Any) -> Dict[str, torch.Tensor]:
    """Snapshot trainable parameters plus mutable CMoE load-balance buffers."""
    state: Dict[str, torch.Tensor] = {
        _canonical_key(name): param.detach().cpu()
        for name, param in student.named_parameters()
        if param.requires_grad
    }
    for name, buffer in student.named_buffers():
        if name.endswith(".gate.extra_bias"):
            state[_canonical_key(name)] = buffer.detach().cpu()
    return state


def _cmoe_extra_state(student: Any) -> Dict[str, torch.Tensor]:
    """Snapshot non-PEFT CMoE state changed by official LoRA training."""
    state: Dict[str, torch.Tensor] = {}
    for name, param in student.named_parameters():
        if name.endswith(".gate.extra_scale"):
            state[_canonical_key(name)] = param.detach().cpu()
    for name, buffer in student.named_buffers():
        if name.endswith(".gate.extra_bias"):
            state[_canonical_key(name)] = buffer.detach().cpu()
    return state


def _save_cmoe_extra_state(student: Any, output_dir: str) -> str:
    path = os.path.join(output_dir, "cmoe_extra_state.pt")
    state = _cmoe_extra_state(student)
    if not state:
        raise RuntimeError("CMoE adapter checkpoint produced no extra_scale/extra_bias state")
    torch.save(state, path)
    return path


def _save_intermediate_lora_adapter(
    student: Any,
    args: Any,
    step_count: int,
    base_manifest: Dict[str, Any],
    log_history: List[Dict[str, Any]],
    output_dir: str,
    elapsed_sec: float,
    max_mem_mb: int,
) -> None:
    """Save a PEFT adapter mid-training without merging it into the base model."""
    os.makedirs(output_dir, exist_ok=True)
    adapter_dir = os.path.join(output_dir, "adapter")
    os.makedirs(adapter_dir, exist_ok=True)
    student.save_pretrained(
        adapter_dir,
        selected_adapters=["default"] if args.stack_init_adapter else None,
    )
    extra_state_path = (
        _save_cmoe_extra_state(student, output_dir)
        if args.moe_type == "cmoe"
        else None
    )
    new_manifest = {
        **base_manifest,
        "cpt_mode": args.mode,
        "cpt_moe_type": args.moe_type,
        "cpt_lr": args.learning_rate,
        "cpt_min_lr": args.min_learning_rate,
        "cpt_max_steps": step_count,
        "cpt_seqlen": args.max_seqlen,
        "cpt_per_device_bsz": args.per_device_batch_size,
        "cpt_grad_accum_steps": args.gradient_accumulation_steps,
        "cpt_max_grad_norm": float(args.max_grad_norm),
        "cpt_one_epoch": bool(args.one_epoch),
        "cpt_effective_train_tokens": int(step_count)
        * int(args.per_device_batch_size)
        * int(args.gradient_accumulation_steps)
        * int(getattr(args, "_distributed_world_size", 1))
        * int(args.max_seqlen),
        "cpt_distributed_sync": bool(getattr(args, "distributed_sync", False)),
        "cpt_distributed_world_size": int(
            getattr(args, "_distributed_world_size", 1)
        ),
        "cpt_cmoe_enable_load_balance": bool(args.cmoe_enable_load_balance),
        "cpt_lora_r": int(args.lora_r),
        "cpt_lora_alpha": int(args.lora_alpha),
        "cpt_lora_dropout": float(args.lora_dropout),
        "cpt_lora_target_modules": args.lora_target_modules,
        "cpt_lora_extra_lr": float(args.lora_extra_lr),
        "cpt_calib_path": args.calib_path,
        "cpt_source_moe_dir": os.path.abspath(args.moe_dir),
        "cpt_elapsed_sec": elapsed_sec,
        "cpt_max_mem_mb": max_mem_mb,
        "cpt_intermediate_save": True,
        "cpt_intermediate_save_type": "peft_adapter",
        "cpt_adapter_dir": os.path.abspath(adapter_dir),
        "cpt_cmoe_extra_state_path": (
            os.path.abspath(extra_state_path) if extra_state_path else None
        ),
        "cpt_trainable_parameter_manifest": getattr(
            args, "_trainable_parameter_manifest", None
        ),
    }
    if getattr(args, "_mlp_router_manifest_fields", None):
        new_manifest.update(args._mlp_router_manifest_fields)
    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(new_manifest, f, indent=2)
    cpt_log = {
        "max_steps": step_count,
        "elapsed_sec": elapsed_sec,
        "max_mem_mb": max_mem_mb,
        "log_history": list(log_history),
        "intermediate_save": True,
        "intermediate_save_type": "peft_adapter",
        "cmoe_extra_state_path": (
            os.path.abspath(extra_state_path) if extra_state_path else None
        ),
    }
    with open(os.path.join(output_dir, "cpt_log.json"), "w") as f:
        json.dump(cpt_log, f, indent=2, default=str)


def _save_intermediate_ckpt(
    student: Any,
    args: Any,
    step_count: int,
    base_manifest: Dict[str, Any],
    log_history: List[Dict[str, Any]],
    output_dir: str,
    elapsed_sec: float,
    max_mem_mb: int,
) -> None:
    """Save a self-contained ckpt mid-training (state_dict + manifest +
    tokenizer + cpt_log) so lmeval_cmoe can ingest directly. LoRA mode is
    excluded because it requires merge_and_unload which mutates the model."""
    os.makedirs(output_dir, exist_ok=True)

    # Sweep/scale runs store inference-only trainable deltas. Preserve the
    # legacy self-contained state_dict path for callers that did not request it.
    save_delta = bool(getattr(args, "save_trainable_delta", False))
    if save_delta:
        delta_sd = _trainable_delta_state(student)
        delta_path = os.path.join(output_dir, "trainable_delta.pt")
        torch.save(delta_sd, delta_path)
        sd_path = None
        save_type = "trainable_delta"
    else:
        sd = student.state_dict()
        sd_cpu = {
            k: v.detach().cpu() for k, v in _strip_capture_keys(sd).items()
        }
        sd_path = os.path.join(output_dir, "state_dict.pt")
        torch.save(sd_cpu, sd_path)
        delta_path = None
        delta_sd = {}
        save_type = "full_state_dict"

    # Manifest mirrors final save fields but with cpt_max_steps=step_count
    # so downstream tools see this as a "step_count-step" model.
    new_manifest = {
        **base_manifest,
        "cpt_mode": args.mode,
        "cpt_moe_type": args.moe_type,
        "cpt_shared_loss_form": args.shared_loss_form,
        "aux_loss_scope": args.aux_loss_scope,
        "cpt_aux_loss_scope": args.aux_loss_scope,
        "cpt_layer_local_aux_input": args.layer_local_aux_input,
        "cpt_shared_target_form": args.shared_target_form,
        "cpt_shared_target_input": args.shared_target_input,
        "cpt_router_target_input": args.router_target_input,
        "cpt_router_train_routing_mode": args.router_train_routing_mode,
        "router_train_routing_mode": args.router_train_routing_mode,
        "cpt_router_relax_epsilon": float(args.router_relax_epsilon),
        "router_relax_epsilon": float(args.router_relax_epsilon),
        "residual_prediction_source": (
            "routed_out_train_proxy" if args.router_train_routing_mode != "hard_topk" else "hard_routed_out"
        ),
        "hard_inference_preserved": True,
        "cpt_shared_grad_routed_detach": int(args.shared_grad_routed_detach),
        "cpt_shared_layer_mask": args.shared_layer_mask,
        "cpt_router_layer_mask": args.router_layer_mask,
        "cpt_moe_trainable_layer_mask": args.moe_trainable_layer_mask,
        "cpt_moe_trainable_layer_mask_info": getattr(args, "_moe_trainable_layer_mask_info", None),
        "cpt_shared_token_weight_mode": args.shared_token_weight_mode,
        "cpt_shared_token_weight_quantile": float(args.shared_token_weight_quantile),
        "cpt_shared_token_weight_floor": float(args.shared_token_weight_floor),
        "cpt_task_token_weight_mode": args.task_token_weight_mode,
        "cpt_task_token_weight_quantile": float(args.task_token_weight_quantile),
        "cpt_task_token_weight_floor": float(args.task_token_weight_floor),
        "cpt_extra_scale_trainable": int(args.cmoe_extra_scale_trainable),
        "cpt_extra_scale_init": args.cmoe_extra_scale_init,
        "cpt_cmoe_enable_load_balance": bool(args.cmoe_enable_load_balance),
        "cpt_add_eas": int(args.cmoe_add_eas),
        "cpt_eas_init_std": float(args.cmoe_eas_init_std),
        "add_eas": bool(args.cmoe_add_eas),
        "cpt_lr": args.learning_rate,
        "cpt_min_lr": args.min_learning_rate,
        "cpt_max_steps": step_count,
        "cpt_seqlen": args.max_seqlen,
        "cpt_per_device_bsz": args.per_device_batch_size,
        "cpt_grad_accum_steps": args.gradient_accumulation_steps,
        "cpt_max_grad_norm": float(args.max_grad_norm),
        "cpt_one_epoch": bool(args.one_epoch),
        "cpt_effective_train_tokens": int(step_count)
        * int(args.per_device_batch_size)
        * int(args.gradient_accumulation_steps)
        * int(getattr(args, "_distributed_world_size", 1))
        * int(args.max_seqlen),
        "cpt_distributed_sync": bool(getattr(args, "distributed_sync", False)),
        "cpt_distributed_world_size": int(
            getattr(args, "_distributed_world_size", 1)
        ),
        "cpt_alpha_task": args.alpha_task,
        "cpt_alpha_shared": args.alpha_shared,
        "cpt_alpha_router": args.alpha_router,
        "cpt_alpha_kd": float(args.alpha_kd),
        "cpt_single_loss_objective": args.single_loss_objective,
        "cpt_kd_temperature": float(args.kd_temperature),
        "cpt_kd_target_model": args.kd_target_model,
        "cpt_router_loss_form": args.router_loss_form,
        "cpt_activation_mass_mapping_source": (
            getattr(args, "_activation_mass_mapping_source", "recovered_exact_dense_neuron_indices")
            if args.router_loss_form in ROUTER_MAPPING_LOSS_FORMS
            else None
        ),
        "router_contribution_score": router_contribution_score_name(args.router_loss_form),
        "router_target_oracle": router_target_oracle_name(args.router_loss_form),
        "cpt_router_target_distribution": router_target_distribution_name(args.router_loss_form),
        "cpt_router_target_temperature": float(args.router_target_temperature),
        "cpt_router_margin_min": float(args.router_margin_min),
        "cpt_router_margin_power": float(args.router_margin_power),
        "cpt_router_margin_weight_clip": float(args.router_margin_weight_clip),
        "cpt_router_margin_threshold_quantile": float(args.router_margin_threshold_quantile),
        "cpt_router_min_active_frac": float(args.router_min_active_frac),
        "cpt_freeze_routed": bool(args.freeze_routed),
        "cpt_freeze_shared": bool(args.freeze_shared),
        "cpt_freeze_router": bool(args.freeze_router),
        "cpt_em_phase": args.em_phase,
        "cpt_data_source": args.data_source,
        "cpt_data_packing": args.data_packing,
        "cpt_data_shuffle": bool(args.data_shuffle),
        "cpt_data_split": args.data_split,
        "cpt_continuation_field": args.continuation_field,
        "cpt_shared_down_init": args.shared_down_init,
        "cpt_shared_zero_jitter_std": float(args.shared_zero_jitter_std),
        "cpt_shared_loss_fp32": bool(args.shared_loss_fp32),
        "cpt_seed": args.seed,
        "cpt_dtype": args.dtype,
        "cpt_use_8bit_adam": bool(args.use_8bit_adam),
        "cpt_lora_r": None,
        "cpt_lora_alpha": None,
        "cpt_lora_dropout": None,
        "cpt_lora_target_modules": None,
        "cpt_lora_extra_lr": None,
        "cpt_calib_path": args.calib_path,
        "cpt_source_moe_dir": os.path.abspath(args.moe_dir),
        "cpt_source_state_dict_path": None,
        "cpt_elapsed_sec": elapsed_sec,
        "cpt_max_mem_mb": max_mem_mb,
        "cpt_intermediate_save": True,
        "cpt_intermediate_save_type": save_type,
        "cpt_train_attention": bool(args.train_attention),
        "cpt_trainable_parameter_manifest": getattr(
            args, "_trainable_parameter_manifest", None
        ),
        "cpt_trainable_delta_path": (
            os.path.abspath(delta_path) if delta_path else None
        ),
        "cpt_trainable_delta_tensors": len(delta_sd),
        "cpt_full_state_dict_path": (
            os.path.abspath(sd_path) if sd_path else None
        ),
        "cpt_skip_full_state_dict": bool(save_delta),
    }
    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(new_manifest, f, indent=2)

    # Copy tokenizer + config from carve dir so lmeval_cmoe can load directly.
    for fname in os.listdir(args.moe_dir):
        full_src = os.path.join(args.moe_dir, fname)
        if not os.path.isfile(full_src):
            continue
        if (
            fname.startswith("tokenizer")
            or fname == "config.json"
            or fname == "special_tokens_map.json"
            or fname == "generation_config.json"
        ):
            shutil.copy(full_src, os.path.join(output_dir, fname))

    # Save log history snapshot.
    cpt_log = {
        "max_steps": step_count,
        "elapsed_sec": elapsed_sec,
        "max_mem_mb": max_mem_mb,
        "log_history": list(log_history),
        "intermediate_save": True,
        "intermediate_save_type": save_type,
    }
    with open(os.path.join(output_dir, "cpt_log.json"), "w") as f:
        json.dump(cpt_log, f, indent=2, default=str)


def _sha256_path(path: str) -> str:
    if not path or not os.path.isfile(path):
        return ""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resume_contract_sha256(args: Any, max_steps: int) -> str:
    fields = (
        "mode",
        "moe_type",
        "router_arch",
        "learning_rate",
        "min_learning_rate",
        "lr_schedule",
        "lora_extra_lr",
        "per_device_batch_size",
        "gradient_accumulation_steps",
        "max_grad_norm",
        "max_seqlen",
        "data_source",
        "data_shuffle",
        "one_epoch",
        "seed",
        "dtype",
        "cmoe_n_experts",
        "cmoe_n_activated",
        "cmoe_n_shared",
        "cmoe_checkpoint_manifest",
        "cmoe_state_dict_path",
        "cmoe_extra_scale_trainable",
        "cmoe_enable_load_balance",
        "alpha_task",
        "alpha_shared",
        "alpha_router",
        "alpha_kd",
        "use_8bit_adam",
        "shared_lr_multiplier",
        "train_attention",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "lora_target_modules",
    )
    payload = {
        "schema": "cpt_ordered_resume_v1",
        "max_steps": int(max_steps),
        "calib_path": os.path.abspath(args.calib_path),
        "data_manifest_path": os.path.abspath(args.data_manifest),
        "data_manifest_sha256": getattr(args, "_data_manifest_sha256", None),
        "source_moe_dir": os.path.abspath(args.moe_dir),
        "source_moe_manifest_sha256": _sha256_path(
            os.path.join(args.moe_dir, "manifest.json")
        ),
        "source_checkpoint_manifest_sha256": getattr(
            args, "_source_checkpoint_manifest_sha256", None
        ),
        "research_git_commit": os.environ.get("RESEARCH_GIT_COMMIT", ""),
        "args": {name: getattr(args, name) for name in fields},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_torch_save(payload: Any, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _atomic_ready_marker(output_dir: str) -> None:
    path = os.path.join(output_dir, ".ready")
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write("ready\n")
    os.replace(tmp, path)



def _save_rolling_resume(
    *,
    path: str,
    contract_sha256: str,
    completed_step: int,
    student: Any,
    optimizer: Any,
    scheduler: Any,
    log_history: List[Dict[str, Any]],
    validation_history: List[Dict[str, Any]],
    elapsed_sec: float,
) -> None:
    trainable = {
        name: parameter.detach().cpu()
        for name, parameter in student.named_parameters()
        if parameter.requires_grad
    }
    mutable_buffers = {
        name: buffer.detach().cpu()
        for name, buffer in student.named_buffers()
        if name.endswith(".gate.extra_bias")
    }
    payload = {
        "schema": "cpt_ordered_resume_v1",
        "contract_sha256": contract_sha256,
        "completed_step": int(completed_step),
        "trainable_state": trainable,
        "mutable_buffers": mutable_buffers,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
        "log_history": list(log_history),
        "validation_history": list(validation_history),
        "elapsed_sec": float(elapsed_sec),
    }
    _atomic_torch_save(payload, path)


def _restore_rolling_resume(
    *,
    path: str,
    contract_sha256: str,
    student: Any,
    optimizer: Any,
    scheduler: Any,
    device: torch.device,
) -> Dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema") != "cpt_ordered_resume_v1":
        raise ValueError(f"unsupported resume schema in {path}")
    if payload.get("contract_sha256") != contract_sha256:
        raise ValueError(
            "resume contract mismatch; do not splice a different data/model/recipe "
            f"trajectory: {path}"
        )
    current = {
        name: parameter
        for name, parameter in student.named_parameters()
        if parameter.requires_grad
    }
    saved = payload.get("trainable_state", {})
    if set(current) != set(saved):
        missing = sorted(set(current) - set(saved))[:8]
        unexpected = sorted(set(saved) - set(current))[:8]
        raise ValueError(
            f"resume trainable-key mismatch missing={missing} unexpected={unexpected}"
        )
    with torch.no_grad():
        for name, parameter in current.items():
            parameter.copy_(saved[name].to(device=parameter.device, dtype=parameter.dtype))
        buffers = dict(student.named_buffers())
        for name, value in payload.get("mutable_buffers", {}).items():
            if name not in buffers:
                raise ValueError(f"resume mutable buffer is missing from model: {name}")
            buffers[name].copy_(
                value.to(device=buffers[name].device, dtype=buffers[name].dtype)
            )
    optimizer.load_state_dict(payload["optimizer_state"])
    scheduler.load_state_dict(payload["scheduler_state"])
    random.setstate(payload["python_rng_state"])
    np.random.set_state(payload["numpy_rng_state"])
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    if torch.cuda.is_available() and payload.get("cuda_rng_state_all"):
        torch.cuda.set_rng_state_all(
            [value.cpu() for value in payload["cuda_rng_state_all"]]
        )
    return payload



# -----------------------------------------------------------------------------
# Frozen oracle MoE (notepad §RRD Loss in CMoE)
# -----------------------------------------------------------------------------


def snapshot_oracle_mlps(student: Any, device: torch.device) -> Dict[int, nn.Module]:
    """Deep-copy every ``layer.mlp`` (a CMoE MoE) into a frozen oracle module.

    Must be called AFTER the student is on ``device`` and BEFORE
    :func:`wrap_student_for_capture` so the deepcopy yields a plain MoE (not a
    capture wrapper around one).

    The returned modules are:
      * deep copies (independent storage from student)
      * ``requires_grad=False`` on all parameters
      * ``.eval()`` mode
      * on the same device + dtype as the student's MLPs

    Returns:
        ``{layer_idx: oracle_mlp}`` mapping.
    """
    import copy

    oracles: Dict[int, nn.Module] = {}
    for layer_idx, layer in enumerate(decoder_layers(student)):
        oracle_mlp = copy.deepcopy(layer.mlp)
        for p in oracle_mlp.parameters():
            p.requires_grad_(False)
        oracle_mlp.eval()
        oracle_mlp.to(device)
        oracles[layer_idx] = oracle_mlp
    return oracles


@torch.no_grad()
def oracle_all_active_forward(
    oracle_mlp: nn.Module,
    h: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Frozen oracle MoE forward in all-active **unweighted** mode (router bypassed).

    Notepad §RRD Loss in CMoE:
        "All-activate MoE (all routed + shared) == Dense MLP" → teacher rep.
        "exclude shared; routed magnitude top-K" → router label.

    Both come from this single forward call.

    Args:
        oracle_mlp: a frozen, eval :class:`MoE` module. Two topologies supported,
            dispatched on attribute presence:
              * **CMoE** (``cmoe_moe_llama.MoE``): exposes ``experts_start_idx``,
                ``experts_end_idx``, ``experts`` (may contain ``None`` for shared
                slots), ``shared_experts`` (plural).
              * **RRD** (``rrd_moe_llama.MoE``): exposes ``n_experts``, ``experts``
                (dense ModuleList of LlamaMLP), ``shared_expert`` (singular,
                possibly ``None`` if ``has_shared=False``).
        h: input tensor (typically student post-attn LN output, [B, S, H] or
           [N, H]). May be detached or not — the function is wrapped in
           ``no_grad`` regardless.

    Returns:
        teacher_rep:        same shape as ``h``. Σ_routed expert_e(h) + shared(h),
                            **no router weighting** — equivalent to dense MLP
                            output if the carve is a clean neuron split.
        shared_out:         ``[N, H]`` frozen shared-expert output, used as the
                            residual anchor for Phase123 frozen-oracle targets.
        routed_outs_stack:  ``[n_routed_filled, N, H]`` — per-routed-expert
                            output stack, used for magnitude top-K oracle target
                            (caller does ``.norm(dim=-1).topk(K, dim=0)``).
    """
    shape = h.size()
    H_dim = h.shape[-1]
    h_flat = h.reshape(-1, H_dim)

    if hasattr(oracle_mlp, "experts_start_idx"):
        # CMoE topology: experts may contain None placeholders, sliced by [start, end).
        routed_modules = [
            oracle_mlp.experts[i]
            for i in range(oracle_mlp.experts_start_idx, oracle_mlp.experts_end_idx)
            if oracle_mlp.experts[i] is not None
        ]
        shared_module = oracle_mlp.shared_experts  # plural
    else:
        # RRD topology: dense ModuleList of n_experts; shared_expert is singular.
        routed_modules = [oracle_mlp.experts[e] for e in range(oracle_mlp.n_experts)]
        shared_module = getattr(oracle_mlp, "shared_expert", None)

    routed_outs: List[torch.Tensor] = [m(h_flat) for m in routed_modules]
    routed_stack = torch.stack(routed_outs, dim=0)               # [n_routed, N, H]
    routed_sum = routed_stack.sum(dim=0)                          # [N, H]
    if shared_module is not None:
        shared_out = shared_module(h_flat)                        # [N, H]
    else:
        shared_out = torch.zeros_like(routed_sum)
    teacher_rep = (routed_sum + shared_out).view(shape)
    return {"teacher_rep": teacher_rep, "shared_out": shared_out, "routed_outs_stack": routed_stack}


def _row_bytes_for_hash(row: torch.Tensor) -> bytes:
    row = row.detach().cpu().contiguous()
    if row.dtype is torch.bfloat16:
        return row.view(torch.uint16).numpy().tobytes()
    return row.numpy().tobytes()


def _row_hash(row: torch.Tensor) -> bytes:
    return hashlib.sha1(_row_bytes_for_hash(row)).digest()


def _routed_and_shared_modules(oracle_mlp: nn.Module) -> Tuple[List[nn.Module], Optional[nn.Module]]:
    if hasattr(oracle_mlp, "experts_start_idx"):
        routed_modules = [
            oracle_mlp.experts[i]
            for i in range(oracle_mlp.experts_start_idx, oracle_mlp.experts_end_idx)
            if oracle_mlp.experts[i] is not None
        ]
        return routed_modules, getattr(oracle_mlp, "shared_experts", None)
    routed_modules = [oracle_mlp.experts[e] for e in range(oracle_mlp.n_experts)]
    return routed_modules, getattr(oracle_mlp, "shared_expert", None)


def _match_expert_rows_to_dense_neurons(
    teacher_gate_weight: torch.Tensor,
    expert_gate_weight: torch.Tensor,
    *,
    layer_idx: int,
    module_label: str,
) -> torch.Tensor:
    """Return dense-neuron row indices copied into one carved expert."""
    teacher_gate_weight = teacher_gate_weight.detach().cpu().contiguous()
    expert_gate_weight = expert_gate_weight.detach().cpu().contiguous()
    row_to_idx = {
        _row_hash(teacher_gate_weight[i]): int(i)
        for i in range(int(teacher_gate_weight.shape[0]))
    }
    mapped: List[int] = []
    misses = 0
    for row_idx in range(int(expert_gate_weight.shape[0])):
        dense_idx = row_to_idx.get(_row_hash(expert_gate_weight[row_idx]))
        if dense_idx is None:
            misses += 1
        else:
            mapped.append(dense_idx)
    if misses:
        raise ValueError(
            f"Could not exactly recover activation-mass mapping for layer {layer_idx} "
            f"{module_label}: {misses}/{int(expert_gate_weight.shape[0])} rows do not "
            "match teacher dense gate_proj rows. This usually means the MoE experts "
            "were already trained or the checkpoint does not preserve the original "
            "CMoE carve rows; provide a persisted mapping before using activation_mass_topk_ce."
        )
    return torch.tensor(mapped, dtype=torch.long)


@torch.no_grad()
def build_activation_mass_neuron_index_cache(
    teacher: Any,
    oracle_mlps: Dict[int, nn.Module],
    manifest: Dict[str, Any],
    device: torch.device,
) -> Dict[int, List[torch.Tensor]]:
    """Recover actual dense-neuron groups for activation-mass router labels.

    CMoE carves select arbitrary dense FFN neurons for shared/routed experts.
    Therefore activation-mass targets must sum dense intermediate activations
    over the actual neuron index set of each routed expert, not over contiguous
    chunks.
    """
    cache: Dict[int, List[torch.Tensor]] = {}
    expected_n_routed = int(manifest.get("n_routed_experts", manifest.get("n_experts")))
    for layer_idx, oracle_mlp in oracle_mlps.items():
        teacher_mlp = teacher.model.layers[layer_idx].mlp
        teacher_gate = teacher_mlp.gate_proj.weight.detach()
        routed_modules, shared_module = _routed_and_shared_modules(oracle_mlp)
        if len(routed_modules) != expected_n_routed:
            raise ValueError(
                f"Layer {layer_idx}: recovered {len(routed_modules)} routed modules, "
                f"expected {expected_n_routed}"
            )

        routed_indices: List[torch.Tensor] = []
        covered: List[torch.Tensor] = []
        for expert_idx, expert in enumerate(routed_modules):
            idx = _match_expert_rows_to_dense_neurons(
                teacher_gate,
                expert.gate_proj.weight.detach(),
                layer_idx=layer_idx,
                module_label=f"routed expert {expert_idx}",
            )
            routed_indices.append(idx.to(device=device))
            covered.append(idx)

        if shared_module is not None:
            shared_idx = _match_expert_rows_to_dense_neurons(
                teacher_gate,
                shared_module.gate_proj.weight.detach(),
                layer_idx=layer_idx,
                module_label="shared expert",
            )
            covered.append(shared_idx)

        all_idx = torch.cat(covered) if covered else torch.empty(0, dtype=torch.long)
        unique = torch.unique(all_idx)
        if unique.numel() != all_idx.numel():
            raise ValueError(
                f"Layer {layer_idx}: activation-mass mapping is not a disjoint partition "
                f"({unique.numel()} unique vs {all_idx.numel()} assigned rows)"
            )
        if all_idx.numel() != int(teacher_gate.shape[0]):
            raise ValueError(
                f"Layer {layer_idx}: activation-mass mapping covers {all_idx.numel()} dense "
                f"neurons, expected full intermediate dimension {int(teacher_gate.shape[0])}"
            )
        cache[int(layer_idx)] = routed_indices

    logger.info(
        "Activation-mass neuron mapping cache built: %d layers, %d routed groups/layer",
        len(cache), expected_n_routed,
    )
    return cache


@torch.no_grad()
def build_activation_mass_neuron_index_cache_from_state_dict(
    teacher: Any,
    state_dict_path: str,
    manifest: Dict[str, Any],
    device: torch.device,
) -> Dict[int, List[torch.Tensor]]:
    """Recover activation-mass groups from an untrained source CMoE state_dict."""
    sd_path = os.path.abspath(state_dict_path)
    if not os.path.isfile(sd_path):
        raise FileNotFoundError(f"activation-mass mapping state_dict not found: {sd_path}")
    logger.info("Loading activation-mass mapping source state_dict from %s", sd_path)
    sd = torch.load(sd_path, map_location="cpu")
    cache: Dict[int, List[torch.Tensor]] = {}
    expected_n_routed = int(manifest.get("n_routed_experts", manifest.get("n_experts")))
    for layer_idx, teacher_layer in enumerate(teacher.model.layers):
        teacher_gate = teacher_layer.mlp.gate_proj.weight.detach()
        routed_indices: List[torch.Tensor] = []
        covered: List[torch.Tensor] = []
        for expert_idx in range(expected_n_routed):
            key = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight"
            if key not in sd:
                raise KeyError(f"Missing activation-mass mapping key in {sd_path}: {key}")
            idx = _match_expert_rows_to_dense_neurons(
                teacher_gate,
                sd[key],
                layer_idx=layer_idx,
                module_label=f"mapping source routed expert {expert_idx}",
            )
            routed_indices.append(idx.to(device=device))
            covered.append(idx)

        shared_key = f"model.layers.{layer_idx}.mlp.shared_experts.gate_proj.weight"
        if shared_key in sd:
            shared_idx = _match_expert_rows_to_dense_neurons(
                teacher_gate,
                sd[shared_key],
                layer_idx=layer_idx,
                module_label="mapping source shared expert",
            )
            covered.append(shared_idx)

        all_idx = torch.cat(covered) if covered else torch.empty(0, dtype=torch.long)
        unique = torch.unique(all_idx)
        if unique.numel() != all_idx.numel():
            raise ValueError(
                f"Layer {layer_idx}: activation-mass source mapping is not disjoint "
                f"({unique.numel()} unique vs {all_idx.numel()} assigned rows)"
            )
        if all_idx.numel() != int(teacher_gate.shape[0]):
            raise ValueError(
                f"Layer {layer_idx}: activation-mass source mapping covers {all_idx.numel()} "
                f"dense neurons, expected {int(teacher_gate.shape[0])}"
            )
        cache[int(layer_idx)] = routed_indices

    logger.info(
        "Activation-mass neuron mapping cache built from source state_dict: %d layers, %d routed groups/layer",
        len(cache), expected_n_routed,
    )
    return cache


# -----------------------------------------------------------------------------
# Optimizer
# -----------------------------------------------------------------------------


def build_optimizer(
    trainable_params: List[nn.Parameter],
    lr: float,
    use_8bit_adam: bool,
) -> torch.optim.Optimizer:
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "use_8bit_adam=True requires bitsandbytes; install it or run without the flag."
            ) from e
        return bnb.optim.Adam8bit(trainable_params, lr=lr, betas=(0.9, 0.95))
    return torch.optim.Adam(trainable_params, lr=lr, betas=(0.9, 0.95))


def build_grouped_optimizer(
    model: nn.Module,
    lr: float,
    use_8bit_adam: bool,
    shared_lr_multiplier: float = 1.0,
) -> Tuple[torch.optim.Optimizer, Dict[str, int]]:
    """Adam optimizer with an optional lower-LR group for shared experts."""
    shared_params: List[nn.Parameter] = []
    base_params: List[nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad or p.numel() == 0:
            continue
        if classify_moe_param_group(name) == "shared":
            shared_params.append(p)
        else:
            base_params.append(p)
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "use_8bit_adam=True requires bitsandbytes; install it or run without the flag."
            ) from e
        opt_cls = bnb.optim.Adam8bit
    else:
        opt_cls = torch.optim.Adam
    param_groups = [{"params": base_params, "lr": lr, "betas": (0.9, 0.95)}]
    if shared_params:
        param_groups.append({
            "params": shared_params,
            "lr": lr * float(shared_lr_multiplier),
            "betas": (0.9, 0.95),
        })
    optimizer = opt_cls(param_groups)
    counts = {
        "base": sum(p.numel() for p in base_params),
        "shared": sum(p.numel() for p in shared_params),
    }
    return optimizer, counts


def build_lora_optimizer(
    model: Any,
    base_lr: float,
    extra_lr: float,
    use_8bit_adam: bool,
) -> Tuple[torch.optim.Optimizer, int, int]:
    """Two-LR Adam optimizer for LoRA mode (CMoE simple_sft.py mirror).

    Splits trainable params into two groups by name:
      - ``extra_params``: parameters whose name contains ``'extra'`` (e.g.
        ``mlp.gate.extra_scale``) — uses ``extra_lr``.
      - ``base_params``: everything else trainable (LoRA adapters) — uses
        ``base_lr``.

    Both groups use ``betas=(0.9, 0.95), eps=1e-8``.

    Returns:
        ``(optimizer, n_base, n_extra)`` parameter element counts (numel).
    """
    extra_params: List[nn.Parameter] = []
    base_params: List[nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad or p.numel() == 0:
            continue
        if "extra" in name:
            extra_params.append(p)
        else:
            base_params.append(p)
    n_base = sum(p.numel() for p in base_params)
    n_extra = sum(p.numel() for p in extra_params)

    if use_8bit_adam:
        try:
            import bitsandbytes as bnb  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "use_8bit_adam=True requires bitsandbytes; install it or run without the flag."
            ) from e
        opt_cls = bnb.optim.Adam8bit
    else:
        opt_cls = torch.optim.Adam

    param_groups = [{
        "params": base_params,
        "lr": base_lr,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
        "weight_decay": 0.0,
    }]
    if extra_params:
        param_groups.append({
            "params": extra_params,
            "lr": extra_lr,
            "betas": (0.9, 0.95),
            "eps": 1e-8,
            "weight_decay": 0.0,
        })
    optimizer = opt_cls(param_groups, weight_decay=0.0)
    return optimizer, n_base, n_extra


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------


def train(args: argparse.Namespace) -> Dict[str, Any]:
    init_delta_path = str(args.init_trainable_delta_path or "").strip()
    init_adapter_path = str(args.init_adapter_path or "").strip()
    init_extra_state_path = str(args.init_cmoe_extra_state_path or "").strip()
    if args.moe_type == "dense":
        if init_delta_path or init_adapter_path or init_extra_state_path:
            raise ValueError("dense SFT does not accept CMoE continuation artifacts")
        if args.mode == "task_only" and (
            bool(args.save_trainable_delta) or bool(args.skip_full_state_dict)
        ):
            raise ValueError(
                "dense full SFT must save a complete Hugging Face checkpoint"
            )
        dense_incompatible = {
            "freeze_routed": bool(args.freeze_routed),
            "freeze_shared": bool(args.freeze_shared),
            "freeze_router": bool(args.freeze_router),
            "train_attention": bool(args.train_attention),
            "train_lm_head": bool(args.train_lm_head),
            "cmoe_enable_load_balance": bool(args.cmoe_enable_load_balance),
            "cmoe_extra_scale_trainable": bool(
                args.cmoe_extra_scale_trainable
            ),
            "router_arch": str(args.router_arch) != "default",
            "moe_trainable_layer_mask": bool(
                str(args.moe_trainable_layer_mask or "").strip()
            ),
        }
        if any(dense_incompatible.values()):
            raise ValueError(
                "dense SFT received MoE-only trainable-policy flags: "
                f"{[key for key, enabled in dense_incompatible.items() if enabled]}"
            )
    if init_delta_path:
        if args.moe_type != "cmoe" or args.mode not in {"task_only", "lora"}:
            raise ValueError(
                "--init_trainable_delta_path requires task_only or LoRA CMoE training"
            )
        if str(args.cmoe_checkpoint_manifest or "").strip():
            raise ValueError(
                "delta continuation starts from the carve; do not combine it "
                "with --cmoe_checkpoint_manifest"
            )
    if bool(init_adapter_path) != bool(init_extra_state_path):
        raise ValueError(
            "--init_adapter_path and --init_cmoe_extra_state_path must be "
            "provided together"
        )
    if init_adapter_path and (args.moe_type != "cmoe" or args.mode != "lora"):
        raise ValueError(
            "adapter continuation requires --moe_type=cmoe --mode=lora"
        )
    if init_adapter_path and str(args.cmoe_checkpoint_manifest or "").strip():
        raise ValueError(
            "adapter continuation starts from the carve; do not combine it "
            "with --cmoe_checkpoint_manifest"
        )
    if bool(args.stack_init_adapter) and not init_adapter_path:
        raise ValueError("--stack_init_adapter requires --init_adapter_path")

    distributed = _distributed_enabled(args)
    if distributed:
        if args.mode != "task_only" or args.moe_type != "cmoe":
            raise ValueError(
                "--distributed_sync is intentionally limited to task_only CMoE training"
            )
        if int(args.gradient_accumulation_steps) != 1:
            raise ValueError("--distributed_sync requires gradient accumulation 1")
        if not bool(args.one_epoch) or bool(args.data_shuffle):
            raise ValueError(
                "--distributed_sync requires ordered --one_epoch data with --data_shuffle 0"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("--distributed_sync requires CUDA/NCCL")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        local_device = torch.device(f"cuda:{local_rank}")
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            device_id=local_device,
        )
        args._distributed_rank = int(torch.distributed.get_rank())
        args._distributed_world_size = int(torch.distributed.get_world_size())
        args._distributed_local_rank = local_rank
        args.device = f"cuda:{local_rank}"
        if not _distributed_is_primary(args):
            logger.setLevel(logging.ERROR)
    else:
        args._distributed_rank = 0
        args._distributed_world_size = 1
        args._distributed_local_rank = 0

    seed_everything(args.seed)

    device = torch.device(args.device)
    dtype = build_dtype(args.dtype)
    if _distributed_is_primary(args):
        logger.info(
            "Distributed contract: enabled=%s rank=%d world_size=%d local_rank=%d device=%s",
            distributed,
            int(args._distributed_rank),
            int(args._distributed_world_size),
            int(args._distributed_local_rank),
            device,
        )

    # 1. Build student MoE. The CMoE loader may populate these fields when it
    # restores an embedded MLP router from a continuation checkpoint.
    args._mlp_router_manifest_fields = None
    args._preloaded_mlp_router = False
    args._source_checkpoint_manifest_path = None
    args._source_checkpoint_manifest_sha256 = None
    args._initialization_artifacts = {}
    student, tokenizer, manifest = load_moe_model(args, dtype=dtype)
    moe_type = str(manifest.get("moe_type", "rrd"))

    # LoRA branches before the non-LoRA MLP-router attachment path. Attach the
    # requested frozen MLP router first so PEFT preserves the same routing
    # policy throughout training. Direct baselines may initialize it randomly.
    if (
        args.mode == "lora"
        and is_mlp_router_arch(str(args.router_arch))
        and bool(args._preloaded_mlp_router)
    ):
        if not args.freeze_router:
            raise ValueError("LoRA + MLP router baseline requires --freeze_router")
        probe = getattr(student, "phase127_mlp_router", None)
        if probe is None:
            raise RuntimeError(
                "checkpoint loader marked the MLP router preloaded but "
                "phase127_mlp_router is absent"
            )
        for probe_param in probe.parameters():
            probe_param.requires_grad = False
        logger.info(
            "LoRA mode: preserving embedded frozen MLP router params=%.2fM "
            "aggregation=%s source=%s",
            sum(param.numel() for param in probe.parameters()) / 1e6,
            args.mlp_router_aggregation,
            args._source_checkpoint_manifest_path,
        )

    if (
        args.mode == "lora"
        and is_mlp_router_arch(str(args.router_arch))
        and not bool(args._preloaded_mlp_router)
    ):
        if not args.freeze_router:
            raise ValueError("LoRA + MLP router baseline requires --freeze_router")
        if (
            build_probe_from_dir is None
            or build_probe_from_manifest is None
            or attach_mlp_probe_router is None
            or mlp_router_manifest_fields is None
        ):
            raise RuntimeError("MLP router helpers are unavailable")
        probe_dir = str(args.init_mlp_router_probe_dir or "").strip()
        random_mlp_router = bool(getattr(args, "init_mlp_router_random", False))
        if bool(probe_dir) == random_mlp_router:
            raise ValueError(
                "LoRA + MLP router requires exactly one of "
                "--init_mlp_router_probe_dir or --init_mlp_router_random"
            )
        router_arch_value = str(args.router_arch)
        if random_mlp_router:
            hidden_text = (
                router_arch_value.removeprefix("hybrid_mlp_h")
                if router_arch_value.startswith("hybrid_mlp_h")
                else router_arch_value.removeprefix("mlp_h")
            )
            probe_manifest = {
                "probe_head_kind": "mlp_probe",
                "probe_hidden_size": int(hidden_text or 1024),
                "n_routed_experts": int(
                    getattr(student.model.layers[0].mlp, "n_routed_experts", 6)
                ),
                "router_target": "none_random_init_baseline",
                "input_source": "random_init_no_stage1",
                "source_run": "random_init_no_stage1",
            }
            probe = build_probe_from_manifest(
                probe_manifest, student, device=device, dtype=dtype
            )
            probe_source = "random_init_no_stage1"
        else:
            probe, probe_manifest = build_probe_from_dir(
                probe_dir, student, device=device, dtype=dtype
            )
            probe_source = probe_dir
        attach_mlp_probe_router(
            student,
            probe,
            aggregation=str(args.mlp_router_aggregation),
            layer_mask=str(getattr(args, "mlp_router_layer_mask", "") or ""),
        )
        for probe_param in probe.parameters():
            probe_param.requires_grad = False
        if random_mlp_router:
            args._mlp_router_manifest_fields = {
                "cpt_router_arch": router_arch_value,
                "router_arch": router_arch_value,
                "cpt_mlp_router_probe_dir": "",
                "cpt_mlp_router_source_run": "random_init_no_stage1",
                "cpt_mlp_router_hidden_size": int(probe_manifest["probe_hidden_size"]),
                "cpt_mlp_router_source_input": "random_init_no_stage1",
                "cpt_mlp_router_source_target": "none_random_init_baseline",
                "cpt_mlp_router_source_train_hit": None,
                "cpt_mlp_router_source_val_hit": None,
                "cpt_mlp_router_aggregation": str(args.mlp_router_aggregation),
                "cpt_mlp_router_layer_mask": str(getattr(args, "mlp_router_layer_mask", "") or ""),
                "cpt_mlp_router_parsed_layer_mask": [],
                "cpt_mlp_router_hybrid_default_router_outside_mask": False,
                "cpt_mlp_router_random_init": True,
            }
        else:
            args._mlp_router_manifest_fields = mlp_router_manifest_fields(
                probe_dir=probe_dir,
                probe_manifest=probe_manifest,
                aggregation=str(args.mlp_router_aggregation),
                layer_mask=str(getattr(args, "mlp_router_layer_mask", "") or ""),
                router_arch=router_arch_value,
            )
            args._mlp_router_manifest_fields["cpt_mlp_router_random_init"] = False
        args._mlp_router_manifest_fields["cpt_mlp_router_parameter_count"] = int(
            sum(param.numel() for param in probe.parameters())
        )
        logger.info(
            "LoRA mode: attached frozen MLP router params=%.2fM aggregation=%s source=%s",
            sum(param.numel() for param in probe.parameters()) / 1e6,
            args.mlp_router_aggregation,
            probe_source,
        )

    # Restore the selected CPT endpoint before wrapping it with the fresh SFT
    # adapter. MLP-router deltas require their router skeleton to exist first.
    for artifact_kind, artifact_path in (
        ("trainable_delta", init_delta_path),
        ("cmoe_extra_state", init_extra_state_path),
    ):
        if not artifact_path:
            continue
        artifact_path = os.path.abspath(os.path.expanduser(artifact_path))
        if not os.path.isfile(artifact_path):
            raise FileNotFoundError(
                f"{artifact_kind} initialization artifact not found: {artifact_path}"
            )
        overlay = torch.load(
            artifact_path,
            map_location="cpu",
            weights_only=False,
        )
        if isinstance(overlay, dict) and "state_dict" in overlay and not any(
            key.startswith(("model.", "lm_head.")) for key in overlay
        ):
            overlay = overlay["state_dict"]
        if not isinstance(overlay, dict) or not overlay:
            raise ValueError(
                f"{artifact_kind} must contain a non-empty state dict"
            )
        _, unexpected = student.load_state_dict(overlay, strict=False)
        if unexpected:
            raise RuntimeError(
                f"{artifact_kind} contains unexpected tensors: {unexpected[:8]}"
            )
        args._initialization_artifacts[artifact_kind] = {
            "path": artifact_path,
            "sha256": _sha256_path(artifact_path),
            "tensor_count": len(overlay),
        }
        logger.info(
            "Applied %s initialization overlay: %s (%d tensors)",
            artifact_kind,
            artifact_path,
            len(overlay),
        )

    if args.mode == "lora":
        if bool(args.train_attention):
            raise ValueError(
                "--train_attention is CE-only; official LoRA attention targets "
                "are controlled by --lora_target_modules"
            )
        # CMoE LoRA SFT mirror: leave extra_bias auto-update + extra_scale
        # trainable; do NOT freeze MoE/non-MoE manually (PEFT handles it after
        # wrapping). Only valid for moe_type='cmoe' (extra_scale/extra_bias are
        # CMoE-specific).
        if moe_type != "cmoe":
            logger.warning(
                "--mode lora is intended for moe_type=cmoe (CMoE simple_sft.py "
                "mirror), but loaded moe_type=%s. Proceeding anyway — LoRA will "
                "wrap target_modules across all layers.", moe_type,
            )
        n_cus = (
            enable_cmoe_cus_training(student)
            if bool(args.cmoe_enable_load_balance)
            else 0
        )
        logger.info(
            "LoRA mode: cus_training=%s on %d MoE layers",
            bool(args.cmoe_enable_load_balance),
            n_cus,
        )

        from peft import (  # noqa: WPS433
            LoraConfig,
            PeftModel,
            TaskType,
            get_peft_model,
        )
        if str(args.lora_target_modules).startswith("regex:"):
            target_modules = str(args.lora_target_modules).removeprefix("regex:")
        else:
            target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
        )
        if init_adapter_path:
            adapter_dir = os.path.abspath(os.path.expanduser(init_adapter_path))
            config_path = os.path.join(adapter_dir, "adapter_config.json")
            if not os.path.isfile(config_path):
                raise FileNotFoundError(
                    f"adapter_config.json not found under {adapter_dir}"
                )
            with open(config_path, "r", encoding="utf-8") as handle:
                adapter_config = json.load(handle)
            expected_targets = (
                target_modules
                if isinstance(target_modules, str)
                else sorted(target_modules)
            )
            actual_targets_raw = adapter_config.get("target_modules")
            actual_targets = (
                actual_targets_raw
                if isinstance(actual_targets_raw, str)
                else sorted(actual_targets_raw or [])
            )
            mismatches = {}
            expected_config = {
                "r": int(args.lora_r),
                "lora_alpha": int(args.lora_alpha),
                "lora_dropout": float(args.lora_dropout),
                "target_modules": expected_targets,
            }
            actual_config = {
                "r": int(adapter_config.get("r", -1)),
                "lora_alpha": int(adapter_config.get("lora_alpha", -1)),
                "lora_dropout": float(adapter_config.get("lora_dropout", -1.0)),
                "target_modules": actual_targets,
            }
            for key, expected_value in expected_config.items():
                if actual_config[key] != expected_value:
                    mismatches[key] = {
                        "expected": expected_value,
                        "actual": actual_config[key],
                    }
            if mismatches:
                raise ValueError(
                    f"continued LoRA adapter config mismatch: {mismatches}"
                )
            student = PeftModel.from_pretrained(
                student,
                adapter_dir,
                adapter_name="cpt" if args.stack_init_adapter else "default",
                is_trainable=not bool(args.stack_init_adapter),
            )
            adapter_weight_path = next(
                (
                    os.path.join(adapter_dir, filename)
                    for filename in (
                        "adapter_model.safetensors",
                        "adapter_model.bin",
                    )
                    if os.path.isfile(os.path.join(adapter_dir, filename))
                ),
                "",
            )
            if not adapter_weight_path:
                raise FileNotFoundError(
                    f"adapter weights not found under {adapter_dir}"
                )
            args._initialization_artifacts["lora_adapter"] = {
                "path": adapter_dir,
                "config_path": config_path,
                "config_sha256": _sha256_path(config_path),
                "weight_path": adapter_weight_path,
                "weight_sha256": _sha256_path(adapter_weight_path),
                "stacked_with_fresh_adapter": bool(args.stack_init_adapter),
            }
            if bool(args.stack_init_adapter):
                student.add_adapter("default", lora_cfg)
                student.base_model.set_adapter(["cpt", "default"])
                for name, param in student.named_parameters():
                    if ".cpt." in name:
                        param.requires_grad = False
                logger.info(
                    "Stacked frozen initialization adapter with fresh LoRA: %s",
                    adapter_dir,
                )
            else:
                logger.info("Loaded trainable LoRA continuation adapter: %s", adapter_dir)
        else:
            student = get_peft_model(student, lora_cfg)
        # PEFT freezes ALL base-model params (including extra_scale) — it only
        # leaves LoRA adapters trainable. To honor --cmoe_extra_scale_trainable=1
        # for the LoRA path (matches the *intent* of CMoE simple_sft.py's
        # extra_lr=0.001 group), explicitly re-enable requires_grad on
        # extra_scale parameters after the PEFT wrap. extra_bias remains a buffer
        # (auto-updated by cus_training, no gradient).
        if bool(args.cmoe_extra_scale_trainable):
            n_unfrozen = 0
            for name, p in student.named_parameters():
                if "extra_scale" in name:
                    p.requires_grad = True
                    n_unfrozen += p.numel()
            logger.info(
                "LoRA mode: re-unfroze extra_scale post-PEFT (n=%d params)", n_unfrozen,
            )
        student.print_trainable_parameters()
        n_trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in student.parameters())
        logger.info(
            "LoRA trainable params: %.2fM / %.2fB (%.3f%%)",
            n_trainable / 1e6, n_total / 1e9, 100.0 * n_trainable / max(n_total, 1),
        )
        lora_names = sorted(
            name for name, param in student.named_parameters() if param.requires_grad
        )
        unexpected_lora = [
            name
            for name in lora_names
            if "lora_" not in name and "extra_scale" not in name
        ]
        if unexpected_lora:
            raise RuntimeError(
                "Official LoRA trainable-surface leak; first unexpected tensors="
                f"{unexpected_lora[:8]}"
            )
        args._trainable_parameter_manifest = {
            "groups": {
                "lora": sum(
                    p.numel()
                    for name, p in student.named_parameters()
                    if p.requires_grad and "lora_" in name
                ),
                "extra_scale": sum(
                    p.numel()
                    for name, p in student.named_parameters()
                    if p.requires_grad and "extra_scale" in name
                ),
            },
            "tensor_count": len(lora_names),
            "parameter_count": n_trainable,
            "names": lora_names,
            "allowed_groups": ["lora", "extra_scale"],
        }
    else:
        n_trainable, n_total = freeze_non_moe(student, moe_type=moe_type)
        n_attention = 0
        if moe_type != "dense":
            n_attention = _set_attention_trainable(
                student, enabled=bool(args.train_attention)
            )
        if n_attention:
            logger.info(
                "train_attention=1: unfroze q/k/v/o projections (%.2fM params)",
                n_attention / 1e6,
            )
        # Apply additional freezes (RRD ablation / EM phases). Each operates
        # post-freeze_non_moe so default-False flags keep prior behaviour.
        freeze_specs = (
            ("freeze_routed", ("mlp.experts.",), "routed"),
            ("freeze_shared", ("mlp.shared_expert.", "mlp.shared_experts."), "shared"),
            ("freeze_router", ("mlp.gate.gate", "mlp.gate.classifier", "mlp.gate.extra_scale"), "router"),
        )
        for flag_name, substrs, label in freeze_specs:
            if not getattr(args, flag_name, False):
                continue
            n_frozen = 0
            for name, p in student.named_parameters():
                if any(substr in name for substr in substrs) and p.requires_grad:
                    p.requires_grad = False
                    n_frozen += p.numel()
            n_trainable -= n_frozen
            logger.info(
                "%s=True: re-froze %d %s params (%.2fM)",
                flag_name, n_frozen, label, n_frozen / 1e6,
            )
        router_arch_value = str(getattr(args, "router_arch", "default"))
        if router_arch_value != "default" and not is_mlp_router_arch(router_arch_value):
            raise ValueError(f"unknown --router_arch={router_arch_value!r}; expected default, mlp_h<hidden>, or hybrid_mlp_h<hidden>")
        if is_mlp_router_arch(router_arch_value):
            preloaded_mlp_router = bool(args._preloaded_mlp_router)
            if not preloaded_mlp_router and (
                build_probe_from_dir is None
                or build_probe_from_manifest is None
                or attach_mlp_probe_router is None
                or mlp_router_manifest_fields is None
            ):
                raise RuntimeError("MLP router helpers are unavailable")
            probe_dir = str(args.init_mlp_router_probe_dir or "").strip()
            random_mlp_router = bool(getattr(args, "init_mlp_router_random", False))
            if (
                not preloaded_mlp_router
                and not probe_dir
                and not random_mlp_router
            ):
                raise ValueError("MLP probe router requires --init_mlp_router_probe_dir or --init_mlp_router_random")
            n_orig_gate_frozen = 0
            for gate_name, gate_param in student.named_parameters():
                if ".mlp.gate." in gate_name and gate_param.requires_grad:
                    gate_param.requires_grad = False
                    n_orig_gate_frozen += gate_param.numel()
            n_trainable -= n_orig_gate_frozen
            if n_orig_gate_frozen:
                logger.info(
                    "%s: froze original CMoE gate params (%.2fM); "
                    "MLP router is the active router",
                    router_arch_value,
                    n_orig_gate_frozen / 1e6,
                )
            if preloaded_mlp_router:
                probe = getattr(student, "phase127_mlp_router", None)
                if probe is None:
                    raise RuntimeError(
                        "checkpoint loader marked the MLP router preloaded but "
                        "phase127_mlp_router is absent"
                    )
                probe_source = str(args._source_checkpoint_manifest_path)
            elif random_mlp_router:
                probe_manifest = {
                    "probe_head_kind": "mlp_probe",
                    "probe_hidden_size": int((router_arch_value.removeprefix("hybrid_mlp_h") if router_arch_value.startswith("hybrid_mlp_h") else router_arch_value.removeprefix("mlp_h")) or 1024) if (router_arch_value.startswith("mlp_h") or router_arch_value.startswith("hybrid_mlp_h")) else 1024,
                    "n_routed_experts": int(getattr(student.model.layers[0].mlp, "n_routed_experts", 6)),
                    "router_target": (
                        router_contribution_score_name(str(args.router_loss_form))
                        if float(args.alpha_router) > 0.0
                        else "none_random_init_baseline"
                    ),
                    "input_source": "random_init_no_stage1",
                    "source_run": "random_init_no_stage1",
                }
                probe = build_probe_from_manifest(
                    probe_manifest,
                    student,
                    device=device,
                    dtype=dtype,
                )
                probe_source = "random_init_no_stage1"
            else:
                probe, probe_manifest = build_probe_from_dir(
                    probe_dir,
                    student,
                    device=device,
                    dtype=dtype,
                )
                probe_source = probe_dir
            if not preloaded_mlp_router:
                attach_mlp_probe_router(
                    student,
                    probe,
                    aggregation=str(args.mlp_router_aggregation),
                    layer_mask=str(getattr(args, "mlp_router_layer_mask", "") or ""),
                )
            n_probe = sum(p.numel() for p in probe.parameters())
            for p_probe in probe.parameters():
                p_probe.requires_grad = not bool(args.freeze_router)
            if not preloaded_mlp_router:
                n_total += n_probe
            if not bool(args.freeze_router):
                n_trainable += n_probe
            if preloaded_mlp_router:
                if not args._mlp_router_manifest_fields:
                    raise RuntimeError(
                        "embedded MLP router is missing checkpoint manifest fields"
                    )
            elif random_mlp_router:
                args._mlp_router_manifest_fields = {
                    "cpt_router_arch": router_arch_value,
                    "router_arch": router_arch_value,
                    "cpt_mlp_router_probe_dir": "",
                    "cpt_mlp_router_source_run": "random_init_no_stage1",
                    "cpt_mlp_router_hidden_size": int(probe_manifest.get("probe_hidden_size", 1024)),
                    "cpt_mlp_router_source_input": "random_init_no_stage1",
                    "cpt_mlp_router_source_target": probe_manifest.get("router_target"),
                    "cpt_mlp_router_source_train_hit": None,
                    "cpt_mlp_router_source_val_hit": None,
                    "cpt_mlp_router_aggregation": str(args.mlp_router_aggregation),
                    "cpt_mlp_router_layer_mask": str(getattr(args, "mlp_router_layer_mask", "") or ""),
                    "cpt_mlp_router_parsed_layer_mask": [],
                    "cpt_mlp_router_hybrid_default_router_outside_mask": bool(str(getattr(args, "mlp_router_layer_mask", "") or "").strip()),
                    "cpt_mlp_router_random_init": True,
                }
            else:
                args._mlp_router_manifest_fields = mlp_router_manifest_fields(
                    probe_dir=probe_dir,
                    probe_manifest=probe_manifest,
                    aggregation=str(args.mlp_router_aggregation),
                    layer_mask=str(getattr(args, "mlp_router_layer_mask", "") or ""),
                    router_arch=router_arch_value,
                )
                args._mlp_router_manifest_fields["cpt_mlp_router_random_init"] = False
            args._mlp_router_manifest_fields["cpt_mlp_router_parameter_count"] = int(n_probe)
            logger.info(
                "%s MLP router: params=%.2fM trainable=%s aggregation=%s source=%s",
                "Preserved embedded" if preloaded_mlp_router else "Attached Phase126",
                n_probe / 1e6,
                not bool(args.freeze_router),
                args.mlp_router_aggregation,
                probe_source,
            )
        if moe_type == "cmoe" and bool(args.cmoe_enable_load_balance):
            n_cus = enable_cmoe_cus_training(student)
            logger.info(
                "CMoE load balancing enabled: cus_training=True on %d MoE layers "
                "(extra_bias auto-update during forward)",
                n_cus,
            )
        if args.train_lm_head:
            n_unfrozen = 0
            for name, p in student.named_parameters():
                if name.startswith("lm_head."):
                    if not p.requires_grad:
                        p.requires_grad = True
                        n_unfrozen += p.numel()
            n_trainable += n_unfrozen
            logger.info(
                "train_lm_head=True: unfroze %d lm_head params (%.2fM)",
                n_unfrozen, n_unfrozen / 1e6,
            )
        args._moe_trainable_layer_mask_info = apply_moe_trainable_layer_mask(
            student, str(getattr(args, "moe_trainable_layer_mask", ""))
        )
        if args._moe_trainable_layer_mask_info.get("enabled"):
            logger.info(
                "moe_trainable_layer_mask=%s: froze %.2fM non-selected MoE params, kept %.2fM selected params",
                args._moe_trainable_layer_mask_info.get("layers"),
                float(args._moe_trainable_layer_mask_info.get("frozen_params", 0)) / 1e6,
                float(args._moe_trainable_layer_mask_info.get("kept_params", 0)) / 1e6,
            )
        if moe_type == "dense":
            dense_names = sorted(
                name
                for name, parameter in student.named_parameters()
                if parameter.requires_grad
            )
            dense_parameter_count = sum(
                parameter.numel()
                for parameter in student.parameters()
                if parameter.requires_grad
            )
            if dense_parameter_count != n_total:
                raise RuntimeError(
                    "dense full-SFT control must keep every model parameter "
                    f"trainable: trainable={dense_parameter_count}, total={n_total}"
                )
            trainable_policy = {
                "groups": {"dense_full_model": dense_parameter_count},
                "tensor_count": len(dense_names),
                "parameter_count": dense_parameter_count,
                "names": dense_names,
                "allowed_groups": ["dense_full_model"],
            }
        else:
            trainable_policy = _record_and_validate_trainable_policy(student, args)
        args._trainable_parameter_manifest = trainable_policy
        n_trainable = int(trainable_policy["parameter_count"])
        logger.info(
            "Trainable params (%s track): %.2fB / %.2fB (%.1f%%) groups=%s",
            moe_type,
            n_trainable / 1e9,
            n_total / 1e9,
            100.0 * n_trainable / max(n_total, 1),
            trainable_policy["groups"],
        )

    if distributed:
        n_synced_moe = 0
        for layer in decoder_layers(student):
            if hasattr(layer.mlp, "sync_load_balance_counts"):
                layer.mlp.sync_load_balance_counts = True
                n_synced_moe += 1
        if n_synced_moe == 0:
            raise RuntimeError(
                "--distributed_sync found no CMoE layers with synchronized load balancing"
            )
        logger.info(
            "Distributed CMoE load balancing enabled on %d layers", n_synced_moe
        )

    # gradient_checkpointing only applies to base transformer layers; safe to enable.
    # NOTE: Dynamic CMoE routing is excluded from checkpoint recompute paths.
    # LoRA dropout or CMoE load-balancing extra_bias updates can shift per-token
    # expert assignment between forward and recompute, which makes checkpointed
    # expert tensors change shape and PyTorch aborts backward.
    try:
        dynamic_cmoe_checkpoint_unsafe = bool(
            moe_type == "cmoe"
            and (args.mode == "lora" or args.cmoe_enable_load_balance)
        )
        if dynamic_cmoe_checkpoint_unsafe:
            logger.info(
                "Skipping gradient_checkpointing because dynamic CMoE routing/load balancing "
                "can change expert tensor shapes during checkpoint recompute."
            )
            student.config.use_cache = False
        else:
            # RRD auxiliary losses consume tensors captured during the forward pass.
            # Reentrant checkpointing runs that forward under no_grad, which silently
            # detaches router/shared captures and kills L_router/L_shared gradients.
            # Non-reentrant checkpointing preserves the graph for captured tensors.
            try:
                student.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
                logger.info("gradient_checkpointing enabled with use_reentrant=False")
            except TypeError:
                student.gradient_checkpointing_enable()
                logger.warning(
                    "gradient_checkpointing_enable does not accept use_reentrant=False; "
                    "RRD capture gradients may be detached."
                )
            student.config.use_cache = False
            # Embeddings/attention are frozen in CPT; checkpointing needs at least
            # one input activation requiring grad or the recomputed graph is cut.
            if hasattr(student, "enable_input_require_grads"):
                student.enable_input_require_grads()
    except Exception as e:
        logger.warning("gradient_checkpointing_enable failed: %s", e)

    student = student.to(device)
    student.train()

    # 2. Optional teacher (RRD / logit_kd modes) and frozen oracle MLP snapshot.
    teacher = None
    kd_source_model = None
    teacher_mlp_outs: Dict[int, torch.Tensor] = {}
    teacher_post_ln: Dict[int, torch.Tensor] = {}
    student_components: Dict[int, Dict[str, torch.Tensor]] = {}
    oracle_mlps: Dict[int, nn.Module] = {}
    activation_mass_neuron_indices: Dict[int, List[torch.Tensor]] = {}
    single_loss_objective = str(getattr(args, "single_loss_objective", "none"))
    single_loss_active = single_loss_objective != "none"
    if single_loss_active and single_loss_objective in {"shared", "router"} and args.mode not in {"rrd", "lora"}:
        raise ValueError("--single_loss_objective shared/router requires --mode rrd or --mode lora")
    if single_loss_active and single_loss_objective == "kd" and args.mode not in {"rrd", "logit_kd"}:
        raise ValueError("--single_loss_objective kd requires --mode rrd or --mode logit_kd")

    compute_ce_loss = (not single_loss_active and args.alpha_task > 0.0) or single_loss_objective == "ce"
    compute_kd_loss = (not single_loss_active and args.alpha_kd > 0.0) or single_loss_objective == "kd"
    compute_shared_loss = (not single_loss_active) or single_loss_objective == "shared"
    compute_router_loss = (not single_loss_active) or single_loss_objective == "router"

    needs_source_kd = compute_kd_loss and args.kd_target_model == "source_moe"
    needs_rrd_aux = (
        args.mode == "rrd"
        and (not single_loss_active or single_loss_objective in {"shared", "router"})
    ) or (
        args.mode == "lora"
        and single_loss_objective in {"shared", "router"}
    ) or (
        args.mode == "logit_kd"
        and (compute_shared_loss or compute_router_loss)
        and (args.alpha_shared > 0.0 or args.alpha_router > 0.0)
    )
    if args.audit_aux_layer_locality and args.aux_loss_scope != "layer_local":
        raise ValueError("--audit_aux_layer_locality requires --aux_loss_scope=layer_local")
    if args.audit_aux_layer_locality and not needs_rrd_aux:
        raise ValueError("--audit_aux_layer_locality requires an active RRD auxiliary loss")
    needs_teacher = (
        needs_rrd_aux
        or (compute_kd_loss and args.kd_target_model == "dense_teacher")
    )
    use_frozen_oracle = (
        needs_rrd_aux
        and args.router_oracle_source in (
            "frozen_student_all_active",
            "frozen_student_all_active_teacher_input",
        )
    )
    oracle_input_source = "teacher_post_ln" if (
        args.router_oracle_source == "frozen_student_all_active_teacher_input"
    ) else "student_mlp_input"
    if needs_source_kd:
        import copy
        logger.info("Creating frozen source MoE snapshot for final-logit KD anchor")
        kd_source_model = copy.deepcopy(student)
        for p in kd_source_model.parameters():
            p.requires_grad_(False)
        kd_source_model.config.use_cache = False
        kd_source_model = kd_source_model.to(device).eval()

    if needs_rrd_aux:
        # Frozen oracle snapshot (must happen BEFORE wrap_student_for_capture
        # so the deepcopy yields a plain MoE module, not a wrapped one).
        if use_frozen_oracle:
            oracle_mlps = snapshot_oracle_mlps(student, device)
            n_oracle_params = sum(
                p.numel() for m in oracle_mlps.values() for p in m.parameters()
            )
            logger.info(
                "Frozen oracle MoE snapshot: %d layers, %.2fM params (~%.1fGB bf16)",
                len(oracle_mlps), n_oracle_params / 1e6,
                2 * n_oracle_params / 1024**3,
            )
            # Defensive assertion (Plan R4): no oracle param trains.
            for layer_idx, om in oracle_mlps.items():
                for p in om.parameters():
                    assert not p.requires_grad, (
                        f"Oracle layer {layer_idx} param requires_grad=True after snapshot"
                    )

        wrap_student_for_capture(
            student,
            student_components,
            router_train_routing_mode=args.router_train_routing_mode,
            router_relax_epsilon=args.router_relax_epsilon,
        )

    if needs_teacher:
        # Teacher is loaded for both RRD (per-layer targets) and logit_kd
        # (final logits target). Hooks are installed unconditionally — they're
        # harmless when only logits are used (kd_only / kd_task) and reused
        # when RRD aux losses are active.
        from transformers import AutoModelForCausalLM
        logger.info("Loading teacher from %s (mode=%s)", args.teacher_model_path, args.mode)
        teacher = AutoModelForCausalLM.from_pretrained(args.teacher_model_path, torch_dtype=dtype)
        teacher.config.use_cache = False
        for p in teacher.parameters():
            p.requires_grad = False
        teacher = teacher.to(device).eval()
        install_teacher_hooks(teacher, teacher_mlp_outs, teacher_post_ln)

    if needs_rrd_aux and args.router_loss_form in ROUTER_MAPPING_LOSS_FORMS:
        if teacher is None:
            raise ValueError(f"{args.router_loss_form} requires a dense teacher model")
        mapping_moe_dir = str(args.activation_mass_mapping_moe_dir or "").strip()
        if not mapping_moe_dir:
            mapping_moe_dir = str(manifest.get("cpt_source_moe_dir") or "").strip()
        mapping_sd_path = os.path.join(mapping_moe_dir, "state_dict.pt") if mapping_moe_dir else ""
        if mapping_sd_path and os.path.isfile(mapping_sd_path):
            activation_mass_neuron_indices = build_activation_mass_neuron_index_cache_from_state_dict(
                teacher,
                mapping_sd_path,
                manifest,
                device,
            )
            args._activation_mass_mapping_source = os.path.abspath(mapping_sd_path)
        else:
            if not oracle_mlps:
                raise ValueError(
                    f"{args.router_loss_form} requires --activation_mass_mapping_moe_dir "
                    "or frozen oracle MoE snapshots for mapping recovery"
                )
            activation_mass_neuron_indices = build_activation_mass_neuron_index_cache(
                teacher,
                oracle_mlps,
                manifest,
                device,
            )
            args._activation_mass_mapping_source = "current_oracle_exact_dense_neuron_indices"

    # 3. Dataset.
    args._data_manifest_sha256 = None
    if str(args.data_manifest or "").strip():
        data_manifest_path = os.path.abspath(str(args.data_manifest).strip())
        with open(data_manifest_path, "r", encoding="utf-8") as handle:
            data_manifest = json.load(handle)
        seed_independent = bool(args.data_manifest_seed_independent)
        if seed_independent:
            selection_seed = data_manifest.get("selection", {}).get("seed")
            if not isinstance(selection_seed, int):
                raise ValueError(
                    "--data_manifest_seed_independent requires integer "
                    "manifest selection.seed"
                )
        elif int(data_manifest.get("seed", -1)) != int(args.seed):
            raise ValueError(
                f"data manifest seed={data_manifest.get('seed')} does not match --seed={args.seed}"
            )
        if int(data_manifest.get("seqlen", -1)) != int(args.max_seqlen):
            raise ValueError("data manifest seqlen does not match --max_seqlen")
        files = data_manifest.get("files", {})
        train_key = str(args.data_manifest_train_key)
        expected_train = os.path.abspath(
            str(files.get(train_key, {}).get("path", ""))
        )
        if expected_train != os.path.abspath(args.calib_path):
            raise ValueError(
                f"--calib_path={args.calib_path} does not match bundle "
                f"{train_key} path={expected_train}"
            )
        if str(args.validation_calib_path or "").strip():
            expected_val = os.path.abspath(
                str(files.get(str(args.data_manifest_validation_key), {}).get("path", ""))
            )
            if expected_val != os.path.abspath(args.validation_calib_path):
                raise ValueError(
                    "--validation_calib_path does not match bundle "
                    f"{args.data_manifest_validation_key} path"
                )
        with open(data_manifest_path, "rb") as handle:
            args._data_manifest_sha256 = hashlib.sha256(handle.read()).hexdigest()
        logger.info(
            "Validated data bundle: manifest=%s training_seed=%d "
            "seed_independent=%s seqlen=%d",
            data_manifest_path,
            args.seed,
            seed_independent,
            args.max_seqlen,
        )

    if args.data_source == "cmoe_inmemory":
        logger.info(
            "Loading dataset via CMoE/datautils.get_wikitext2 (random crop seed=%d, nsamples=%d, seqlen=%d)",
            args.seed, args.data_nsamples, args.max_seqlen,
        )
        ds = CMoEInMemoryWikiText(
            model_path=args.teacher_model_path,
            nsamples=args.data_nsamples,
            seed=args.seed,
            seqlen=args.max_seqlen,
            cmoe_repo_path=args.cmoe_repo_path,
        )
    elif args.data_source == "teacher_rollout_jsonl":
        logger.info(
            "Loading teacher-rollout SFT dataset from %s (split=%s, seqlen=%d)",
            args.calib_path, args.data_split, args.max_seqlen,
        )
        ds = TeacherRolloutSFT(
            tokenizer, args.calib_path, args.max_seqlen,
            data_split=args.data_split,
            continuation_field=args.continuation_field,
        )
    elif args.data_source == "raw_continuation_jsonl":
        logger.info(
            "Loading raw-continuation SFT dataset from %s (split=%s, seqlen=%d)",
            args.calib_path, args.data_split, args.max_seqlen,
        )
        ds = RawContinuationSFT(
            tokenizer, args.calib_path, args.max_seqlen,
            data_split=args.data_split,
            continuation_field=args.continuation_field,
        )
    elif args.data_source == "supervised_token_ids_jsonl":
        if int(args.per_device_batch_size) != 1:
            raise ValueError(
                "supervised_token_ids_jsonl currently requires "
                "--per_device_batch_size 1 for exact variable-length rows"
            )
        logger.info(
            "Loading exact supervised token-id rows from %s "
            "(split=%s, seqlen=%d)",
            args.calib_path,
            args.data_split,
            args.max_seqlen,
        )
        ds = SupervisedTokenIdsDataset(
            args.calib_path,
            args.max_seqlen,
            data_split=args.data_split,
        )
    elif args.data_source == "token_ids_jsonl":
        logger.info(
            "Loading exact token-id chunks from %s (split=%s, seqlen=%d)",
            args.calib_path, args.data_split, args.max_seqlen,
        )
        ds = TokenIdsChunk(
            args.calib_path,
            args.max_seqlen,
            data_split=args.data_split,
        )
    elif args.data_source == "token_ids_npy":
        logger.info(
            "Memory-mapping exact token-id chunks from %s (split=%s, seqlen=%d)",
            args.calib_path, args.data_split, args.max_seqlen,
        )
        ds = NpyTokenIdsChunk(
            args.calib_path,
            args.max_seqlen,
            data_split=args.data_split,
        )
    else:
        ds = WikiTextChunk(
            tokenizer, args.calib_path, args.max_seqlen,
            packing=args.data_packing,
            data_split=args.data_split,
        )
    logger.info("Dataset size: %d (max_seqlen=%d, source=%s)", len(ds), args.max_seqlen, args.data_source)
    if len(ds) == 0:
        raise RuntimeError(
            f"Dataset produced 0 windows of length {args.max_seqlen}. "
            "Check the input jsonl format / seqlen / data_source."
        )

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_sampler = None
    if distributed:
        train_sampler = DistributedSampler(
            ds,
            num_replicas=int(args._distributed_world_size),
            rank=int(args._distributed_rank),
            shuffle=False,
            seed=int(args.seed),
            drop_last=True,
        )
    loader = DataLoader(
        ds,
        batch_size=args.per_device_batch_size,
        shuffle=bool(args.data_shuffle) if train_sampler is None else False,
        sampler=train_sampler,
        drop_last=True,
        num_workers=0,
        collate_fn=_collate,
        generator=g,
    )

    val_loader = None
    validation_history: List[Dict[str, Any]] = []
    if str(args.validation_calib_path or "").strip() and int(args.validation_every) > 0:
        val_path = str(args.validation_calib_path).strip()
        if args.data_source == "supervised_token_ids_jsonl":
            validation_batch_size_requested = (
                int(args.validation_per_device_batch_size)
                or int(args.per_device_batch_size)
            )
            if validation_batch_size_requested != 1:
                raise ValueError(
                    "supervised_token_ids_jsonl validation currently requires "
                    "batch size 1 for exact variable-length rows"
                )
            val_ds = SupervisedTokenIdsDataset(
                args.validation_calib_path,
                args.max_seqlen,
                data_split="all",
            )
        elif args.data_source == "token_ids_jsonl":
            val_ds = TokenIdsChunk(val_path, args.max_seqlen, data_split="all")
        elif args.data_source == "token_ids_npy":
            val_ds = NpyTokenIdsChunk(val_path, args.max_seqlen, data_split="all")
        elif args.data_source == "jsonl":
            val_ds = WikiTextChunk(
                tokenizer,
                val_path,
                args.max_seqlen,
                packing=args.data_packing,
                data_split="all",
            )
        else:
            raise ValueError(
                "--validation_calib_path supports --data_source jsonl, "
                "token_ids_jsonl, supervised_token_ids_jsonl, or token_ids_npy"
            )
        validation_batch_size = int(args.validation_per_device_batch_size) or int(
            args.per_device_batch_size
        )
        if validation_batch_size <= 0:
            raise ValueError("validation batch size must be positive")
        val_loader = DataLoader(
            val_ds,
            batch_size=validation_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            collate_fn=_collate,
        )
        logger.info(
            "Validation loader enabled: path=%s size=%d every=%d batches=%d batch_size=%d",
            val_path,
            len(val_ds),
            int(args.validation_every),
            int(args.validation_batches),
            validation_batch_size,
        )

    # 4. Optimizer + scheduler.
    if args.mode == "lora":
        optimizer, n_lora_base, n_lora_extra = build_lora_optimizer(
            student,
            base_lr=args.learning_rate,
            extra_lr=args.lora_extra_lr,
            use_8bit_adam=args.use_8bit_adam,
        )
        logger.info(
            "LoRA optimizer: base=%.2fM (lr=%.2e), extra=%d (lr=%.2e)",
            n_lora_base / 1e6, args.learning_rate, n_lora_extra, args.lora_extra_lr,
        )
    else:
        trainable_params = [
            p for p in student.parameters() if p.requires_grad and p.numel() > 0
        ]
        if float(args.shared_lr_multiplier) != 1.0:
            optimizer, lr_group_counts = build_grouped_optimizer(
                student,
                args.learning_rate,
                args.use_8bit_adam,
                shared_lr_multiplier=float(args.shared_lr_multiplier),
            )
            logger.info(
                "Grouped optimizer: base=%.2fM lr=%.2e shared=%.2fM lr=%.2e",
                lr_group_counts["base"] / 1e6,
                args.learning_rate,
                lr_group_counts["shared"] / 1e6,
                args.learning_rate * float(args.shared_lr_multiplier),
            )
        else:
            optimizer = build_optimizer(trainable_params, args.learning_rate, args.use_8bit_adam)
            lr_group_counts = {"base": sum(p.numel() for p in trainable_params), "shared": 0}
        trainable_named_params = [
            (n, p)
            for n, p in student.named_parameters()
            if p.requires_grad and p.numel() > 0
        ]
        trainable_group_counts = {
            "router": 0,
            "routed": 0,
            "shared": 0,
            "eas": 0,
            "attention": 0,
            "other": 0,
        }
        for name, p in trainable_named_params:
            n_param = p.numel()
            trainable_group_counts[classify_moe_param_group(name)] += n_param
        logger.info(
            "Trainable tensor count: %d group_param_counts=%s total_trainable_params=%d",
            len(trainable_params),
            trainable_group_counts,
            sum(trainable_group_counts.values()),
        )
        args._trainable_group_counts = trainable_group_counts
        args._optimizer_lr_group_counts = lr_group_counts
        if (
            moe_type == "cmoe"
            and getattr(args, "router_arch", "default") == "default"
            and args.alpha_task > 0.0
            and not args.freeze_router
            and not bool(args.cmoe_extra_scale_trainable)
        ):
            logger.warning(
                "CMoE CE->router gradient path is effectively disabled: "
                "cmoe_extra_scale_trainable=0 keeps gate.extra_scale frozen at the "
                "loaded value, and hard top-k indices are non-differentiable. "
                "Use --cmoe_extra_scale_trainable 1, optionally with "
                "--cmoe_extra_scale_init 1.0, when CE should update router params."
            )

    # Effective steps for scheduler.
    if args.smoke_test_steps > 0:
        max_steps = args.smoke_test_steps
    elif args.one_epoch:
        max_steps = len(loader)
    else:
        max_steps = args.max_steps
    logger.info(
        "Effective max_steps=%d (requested=%d, one_epoch=%s, loader_len=%d)",
        max_steps,
        args.max_steps,
        bool(args.one_epoch),
        len(loader),
    )

    def _run_lightweight_validation(completed_step: int) -> Optional[Dict[str, Any]]:
        if val_loader is None:
            return None
        was_training = student.training
        student.eval()
        # CMoE updates extra_bias whenever cus_training=True, independent of
        # Module.training. Disable it during validation so evaluation does not
        # mutate the optimizer trajectory.
        validation_cus_flags = []
        for layer in decoder_layers(student):
            if hasattr(layer.mlp, "cus_training"):
                validation_cus_flags.append(
                    (layer.mlp, bool(layer.mlp.cus_training))
                )
                layer.mlp.cus_training = False
        if teacher is not None:
            teacher.eval()
        if kd_source_model is not None:
            kd_source_model.eval()
        n_seen = 0
        sum_ce = 0.0
        sum_kd = 0.0
        n_ce = 0
        n_kd = 0
        max_batches = max(int(args.validation_batches), 1)
        with torch.no_grad():
            for i, vbatch in enumerate(val_loader):
                if i >= max_batches:
                    break
                input_ids_v = vbatch["input_ids"].to(device, non_blocking=True)
                labels_v = vbatch["labels"].to(device, non_blocking=True)
                attention_mask_v = vbatch.get("attention_mask")
                if attention_mask_v is not None:
                    attention_mask_v = attention_mask_v.to(device, non_blocking=True)
                student_components.clear()
                teacher_mlp_outs.clear()
                teacher_post_ln.clear()
                teacher_logits_v = None
                if compute_kd_loss and args.kd_target_model == "dense_teacher":
                    if teacher is None:
                        raise ValueError("validation KD requested but dense teacher is not loaded")
                    teacher_logits_v = teacher(input_ids=input_ids_v, attention_mask=attention_mask_v).logits
                elif needs_source_kd:
                    teacher_logits_v = kd_source_model(input_ids=input_ids_v, attention_mask=attention_mask_v).logits
                out_v = student(input_ids=input_ids_v, attention_mask=attention_mask_v, labels=labels_v)
                if out_v.loss is not None:
                    sum_ce += float(out_v.loss.detach())
                    n_ce += 1
                if compute_kd_loss and teacher_logits_v is not None:
                    T = float(args.kd_temperature)
                    valid_mask = labels_v != -100
                    s_log_p = F.log_softmax(out_v.logits.float() / T, dim=-1)
                    t_p = F.softmax(teacher_logits_v.float() / T, dim=-1)
                    kl_tok = F.kl_div(s_log_p, t_p, reduction="none").sum(dim=-1)
                    n_valid = valid_mask.sum().clamp_min(1).float()
                    kd_v = (kl_tok * valid_mask.float()).sum() / n_valid * (T * T)
                    sum_kd += float(kd_v.detach())
                    n_kd += 1
                n_seen += int(input_ids_v.size(0))
                del out_v, teacher_logits_v
                student_components.clear()
                teacher_mlp_outs.clear()
                teacher_post_ln.clear()
        for moe, enabled in validation_cus_flags:
            moe.cus_training = enabled
        if was_training:
            student.train()
        rec = {
            "step": int(completed_step),
            "batches": int(min(max_batches, n_ce if n_ce else max_batches)),
            "examples": int(n_seen),
            "loss_ce": (sum_ce / n_ce) if n_ce else None,
            "loss_kd": (sum_kd / n_kd) if n_kd else None,
            "loss_router": None,
            "loss_shared": None,
            "weighted_ce": (float(args.alpha_task) * sum_ce / n_ce) if n_ce else None,
            "weighted_kd": (float(args.alpha_kd) * sum_kd / n_kd) if n_kd else None,
            "weighted_router": None,
            "weighted_shared": None,
            "note": "lightweight validation computes CE/KD only; router/shared validation diagnostics are post-hoc/final artifacts",
        }
        total = 0.0
        has_total = False
        for key in ("weighted_ce", "weighted_kd", "weighted_router", "weighted_shared"):
            val = rec.get(key)
            if val is not None:
                total += float(val)
                has_total = True
        rec["loss_total"] = total if has_total else None
        if _distributed_is_primary(args):
            logger.info("validation step=%d %s", completed_step, rec)
        return rec

    # Parse intermediate save schedule. Each step in this set triggers an
    # extra save under {output_dir}/ckpt_step_{step}/ in addition to the
    # final save at max_steps.
    save_at_steps_set: set = set()
    if args.save_at_steps.strip():
        try:
            save_at_steps_set = {
                int(s.strip())
                for s in args.save_at_steps.split(",")
                if s.strip()
            }
        except ValueError as e:
            raise ValueError(
                f"--save_at_steps malformed: {args.save_at_steps!r}: {e}"
            )

    if args.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_steps, eta_min=args.min_learning_rate,
        )
    else:  # constant — mirrors CMoE simple_sft.py (no scheduler)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    K = int(manifest["n_active"])
    # Map CLI shared loss form to the reduction string accepted by
    # _reduce_squared_error (chain_losses.py:140-160).
    shared_reduction = "global_rmse" if args.shared_loss_form == "rmse" else "mean_all"
    shared_layer_mask = parse_layer_mask(args.shared_layer_mask)
    if shared_layer_mask is not None:
        logger.info("L_shared restricted to layers: %s", sorted(shared_layer_mask))
    router_layer_mask = parse_layer_mask(args.router_layer_mask)
    if router_layer_mask is not None:
        logger.info("L_router restricted to layers: %s", sorted(router_layer_mask))
    if args.shared_target_form == "residual" and args.shared_token_weight_mode != "none":
        raise ValueError("--shared_target_form=residual does not currently support shared token weights")
    if args.router_loss_form in (ROUTER_FROZEN_ORACLE_TARGET_FORMS | {"cmoe_representative_router_loss"}) and not use_frozen_oracle:
        raise ValueError(
            "best_subset_topk_ce/marginal_gain_*/cmoe_representative_router_loss require "
            "--router_oracle_source frozen_student_all_active_teacher_input"
        )

    # EM Phase M validation: oracle routing override needs frozen oracle on
    # teacher's t_ln. Phase E is bookkeeping-only and uses the student's own
    # router as usual; no extra invariant.
    if args.em_phase == "M":
        if not use_frozen_oracle:
            raise ValueError(
                "--em_phase=M requires --router_oracle_source "
                "frozen_student_all_active_teacher_input (need frozen oracle snapshot)."
            )
        if oracle_input_source != "teacher_post_ln":
            raise ValueError(
                "--em_phase=M requires oracle_input_source=teacher_post_ln "
                "(set --router_oracle_source frozen_student_all_active_teacher_input)."
            )
        logger.info(
            "EM Phase M: oracle routing override active (magnitude top-K of "
            "frozen oracle on teacher t_ln will be passed to student MoE forward)."
        )
    elif args.em_phase == "E":
        logger.info(
            "EM Phase E: bookkeeping label only (router learns; experts/shared "
            "expected to be frozen via --freeze_routed --freeze_shared)."
        )

    # 5. Training loop.
    resume_path = str(args.resume_state or "").strip()
    if distributed and resume_path:
        raise ValueError("--distributed_sync does not support --resume_state")
    if int(args.resume_save_every) < 0:
        raise ValueError("--resume_save_every must be >= 0")
    if int(args.resume_save_every) > 0 and not resume_path:
        raise ValueError("--resume_save_every requires --resume_state")
    if resume_path:
        if bool(args.data_shuffle) or not bool(args.one_epoch):
            raise ValueError(
                "--resume_state requires --data_shuffle 0 and --one_epoch"
            )
        if int(args.gradient_accumulation_steps) != 1:
            raise ValueError("--resume_state requires gradient accumulation 1")
        if not str(args.data_manifest or "").strip():
            raise ValueError("--resume_state requires a pinned --data_manifest")

    resume_contract_sha256 = _resume_contract_sha256(args, max_steps)
    step = 0
    log_history: List[Dict[str, Any]] = []
    gradient_history: List[Dict[str, Any]] = []
    last_logged: Dict[str, Any] = {}
    elapsed_before = 0.0
    if resume_path and os.path.isfile(resume_path):
        resume_payload = _restore_rolling_resume(
            path=resume_path,
            contract_sha256=resume_contract_sha256,
            student=student,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        step = int(resume_payload["completed_step"])
        if step < 0 or step > max_steps:
            raise ValueError(
                f"resume completed_step={step} outside [0, {max_steps}]"
            )
        log_history = list(resume_payload.get("log_history", []))
        validation_history = list(resume_payload.get("validation_history", []))
        elapsed_before = float(resume_payload.get("elapsed_sec", 0.0))
        if log_history:
            last_logged = dict(log_history[-1])
        logger.info(
            "Resumed ordered trajectory at completed_step=%d/%d from %s",
            step,
            max_steps,
            resume_path,
        )
    t0 = time.time() - elapsed_before
    optimizer.zero_grad(set_to_none=True)
    if val_loader is not None and step == 0:
        val0 = _run_lightweight_validation(0)
        if val0 is not None:
            validation_history.append(val0)
    resume_skip_batches = step
    done = step >= max_steps
    while not done:
        for batch_index, batch in enumerate(loader):
            if resume_skip_batches and batch_index < resume_skip_batches:
                continue
            if resume_skip_batches:
                resume_skip_batches = 0
            if step >= max_steps:
                done = True
                break
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device, non_blocking=True)

            # Clear caches each step.
            student_components.clear()
            teacher_mlp_outs.clear()
            teacher_post_ln.clear()

            # Target forward (no grad) for RRD targets and/or final-logit KD.
            # RRD aux losses may need dense-teacher hooks while KD targets a
            # frozen source MoE; keep those forwards independent.
            teacher_logits = None
            if needs_teacher:
                with torch.no_grad():
                    teacher_out = teacher(input_ids=input_ids, attention_mask=attention_mask)
                    if args.alpha_kd > 0.0 and args.kd_target_model == "dense_teacher":
                        teacher_logits = teacher_out.logits  # [B, S, V]
            if needs_source_kd:
                with torch.no_grad():
                    source_out = kd_source_model(input_ids=input_ids, attention_mask=attention_mask)
                    teacher_logits = source_out.logits  # [B, S, V]

            # EM Phase M: pre-compute oracle routing per layer using teacher
            # post-attn LN, then stash on each _CaptureWrapper so the student
            # forward bypasses the gate's argmax. Cleared at end of step.
            if args.em_phase == "M":
                for layer_idx, layer in enumerate(decoder_layers(student)):
                    wrapper = layer.mlp  # _CaptureWrapper
                    t_ln = teacher_post_ln.get(layer_idx)
                    if t_ln is None:
                        wrapper.override_topk_indices = None
                        wrapper.override_topk_weights = None
                        continue
                    with torch.no_grad():
                        oracle_out = oracle_all_active_forward(
                            oracle_mlps[layer_idx], t_ln,
                        )
                        rs = oracle_out["routed_outs_stack"]                # [E, N, H]
                        norms = rs.float().norm(dim=-1)                     # [E, N]
                        topk_norms, topk_idx = norms.topk(K, dim=0)         # [K, N], [K, N]
                        override_idx = topk_idx.T.contiguous()              # [N, K]
                        expert_weight_mode = getattr(wrapper.inner, "expert_weight_mode", "unit")
                        if expert_weight_mode == "softmax":
                            override_w = F.softmax(
                                topk_norms.T.contiguous().float(), dim=-1,
                            ).to(t_ln.dtype)
                        else:
                            override_w = None  # unit weighting ignores weights
                    wrapper.override_topk_indices = override_idx
                    wrapper.override_topk_weights = override_w

            # Student forward (computes CE; populates student_components via hooks for RRD).
            # NOTE: gradient_checkpointing recomputes the forward during backward,
            # so override_topk_indices/weights stay set on the wrapper for the
            # whole step. Each step's pre-forward block above re-sets them, so
            # leaving them populated between steps is safe.
            saved_tensor_context = (
                torch.autograd.graph.save_on_cpu(pin_memory=True)
                if bool(args.activation_offload_cpu)
                else contextlib.nullcontext()
            )
            with saved_tensor_context:
                outputs = student(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
            loss_task = outputs.loss

            # Decide once per step whether this is a logging step. Used below
            # both for the standard log_history append AND for the per-layer
            # instrumentation block (RRD aux only) so we only pay the metric
            # cost on log steps.
            should_log = (step % args.log_every == 0) or (step == max_steps - 1)

            task_token_weight_stats = None
            if args.task_token_weight_mode != "none":
                task_token_weights, task_token_weight_extra_stats = build_shared_token_weights(
                    args.task_token_weight_mode,
                    outputs.logits,
                    labels,
                    args.task_token_weight_quantile,
                    args.task_token_weight_floor,
                    tokenizer=tokenizer,
                )
                loss_task, task_token_weight_stats = compute_weighted_task_ce(
                    outputs.logits, labels, task_token_weights,
                )
                if task_token_weight_extra_stats:
                    task_token_weight_stats.update(task_token_weight_extra_stats)
                task_token_weight_stats.update({
                    "mode": args.task_token_weight_mode,
                    "quantile": float(args.task_token_weight_quantile),
                    "floor": float(args.task_token_weight_floor),
                })

            shared_token_weights = None
            shared_token_weight_stats = None
            shared_token_weight_extra_stats = None
            if args.shared_token_weight_mode != "none":
                shared_token_weights, shared_token_weight_extra_stats = build_shared_token_weights(
                    args.shared_token_weight_mode,
                    outputs.logits,
                    labels,
                    args.shared_token_weight_quantile,
                    args.shared_token_weight_floor,
                    tokenizer=tokenizer,
                )
                if should_log:
                    with torch.no_grad():
                        w = shared_token_weights.float()
                        active = w > float(args.shared_token_weight_floor)
                        covered = w > 0
                        shared_token_weight_stats = {
                            "mode": args.shared_token_weight_mode,
                            "quantile": float(args.shared_token_weight_quantile),
                            "floor": float(args.shared_token_weight_floor),
                            "active_frac": float(active.float().mean().item()),
                            "covered_frac": float(covered.float().mean().item()),
                            "mean_weight": float(w.mean().item()),
                        }
                        if shared_token_weight_extra_stats:
                            shared_token_weight_stats.update(shared_token_weight_extra_stats)

            # === L_kd: Hinton KL on final logits ===
            loss_kd = None
            if compute_kd_loss and teacher_logits is not None:
                T = float(args.kd_temperature)
                # Mask padding/ignored tokens via labels (HF convention: -100 = ignore)
                valid_mask_2d = (labels != -100)  # [B, S]
                s_log_p = F.log_softmax(outputs.logits.float() / T, dim=-1)  # [B, S, V]
                t_p = F.softmax(teacher_logits.float() / T, dim=-1)          # [B, S, V]
                # F.kl_div(input=log_q, target=p, reduction='none')
                #   = sum_c p*(log p - log q) = KL(p || q)
                # Hinton form: KL(p_teacher || p_student) * T^2
                kl_per_token = F.kl_div(s_log_p, t_p, reduction="none").sum(dim=-1)  # [B, S]
                n_valid = valid_mask_2d.sum().clamp_min(1).float()
                loss_kd = (kl_per_token * valid_mask_2d.float()).sum() / n_valid * (T * T)

            # Initialize per-layer aux loss accumulators so the total_loss
            # combiner below can branch on `needs_rrd_aux` cleanly.
            loss_shared_total = torch.zeros((), device=device, dtype=torch.float32)
            loss_router_total = torch.zeros((), device=device, dtype=torch.float32)
            loss_shared_avg_for_audit: Optional[torch.Tensor] = None
            loss_router_avg_for_audit: Optional[torch.Tensor] = None
            n_layers_used = 0
            n_shared_layers_used = 0
            per_layer_metrics: Dict[int, Dict[str, float]] = {}
            router_margin_stats: List[Dict[str, float]] = []
            aux_locality_audit: Optional[Dict[str, Any]] = None

            if needs_rrd_aux:
                # Aggregate per-layer auxiliary losses (accumulators initialized above).
                # Sanity (first RRD step only): oracle all-active ≡ teacher dense MLP?
                run_sanity = use_frozen_oracle and step == 0
                sanity_log: Dict[int, Dict[str, float]] = {}
                for layer_idx, comp in student_components.items():
                    t_mlp = teacher_mlp_outs.get(layer_idx)
                    t_ln = teacher_post_ln.get(layer_idx)
                    if t_mlp is None or t_ln is None:
                        continue
                    routed = comp["routed_out"]
                    routed_for_residual = comp.get("routed_out_train_proxy", routed)
                    shared = comp["shared_out"]
                    router_scores = comp["router_scores"]  # softmax probs
                    router_logits = comp["router_logits"]  # pre-softmax (set-match diag)
                    h_student = comp["mlp_input"]          # [B, S, H], detached
                    aux_comp = comp
                    if args.aux_loss_scope == "layer_local":
                        local_aux_input = (
                            t_ln if args.layer_local_aux_input == "teacher_post_ln"
                            else h_student
                        )
                        aux_comp = _layer_local_components(
                            student,
                            int(layer_idx),
                            local_aux_input,
                            router_train_routing_mode=args.router_train_routing_mode,
                            router_relax_epsilon=args.router_relax_epsilon,
                        )
                        routed = aux_comp["routed_out"]
                        routed_for_residual = aux_comp.get("routed_out_train_proxy", routed)
                        shared = aux_comp["shared_out"]
                        router_scores = aux_comp["router_scores"]
                        router_logits = aux_comp["router_logits"]

                    layer_aux_loss_for_audit: Optional[torch.Tensor] = None

                    if use_frozen_oracle:
                        # Oracle (frozen step-0 student MoE) all-active forward.
                        # Input depends on router_oracle_source:
                        #   - 'frozen_student_all_active': h_student (drifts)
                        #   - 'frozen_student_all_active_teacher_input': t_ln
                        #     (fixed target, oracle ≡ dense MLP per carve sanity)
                        oracle_input = (
                            t_ln if oracle_input_source == "teacher_post_ln"
                            else h_student
                        )
                        oracle_out = oracle_all_active_forward(
                            oracle_mlps[layer_idx], oracle_input,
                        )
                        teacher_rep_for_shared = oracle_out["teacher_rep"]
                        routed_stack = oracle_out["routed_outs_stack"]   # [n_routed, N, H]
                        # magnitude top-K from precomputed stack (avoids
                        # re-forwarding experts_module a second time).
                        with torch.no_grad():
                            norms_n_routed = routed_stack.float().norm(dim=-1)  # [n_routed, N]
                            topk_norms, topk_idx = norms_n_routed.topk(K, dim=0)
                            target_idx = topk_idx.T.contiguous()
                            target_norms = topk_norms.T.contiguous()
                            if norms_n_routed.size(0) > K:
                                next_norm = norms_n_routed.topk(K + 1, dim=0).values[K, :]
                                target_margin = (topk_norms[-1, :] - next_norm).contiguous()
                            else:
                                target_margin = torch.ones_like(topk_norms[-1, :])
                        # Optional sanity vs teacher dense MLP. Logged once at step 0.
                        if run_sanity:
                            with torch.no_grad():
                                t_flat = t_mlp.reshape(-1, t_mlp.shape[-1]).float()
                                o_flat = teacher_rep_for_shared.reshape(-1, t_mlp.shape[-1]).float()
                                diff = (o_flat - t_flat).abs()
                                sanity_log[int(layer_idx)] = {
                                    "max_diff": float(diff.max()),
                                    "mean_diff": float(diff.mean()),
                                    "rel_err": float(
                                        diff.mean() / t_flat.abs().mean().clamp_min(1e-12)
                                    ),
                                }
                    else:
                        # Legacy: teacher dense MLP rep + magnitude oracle on
                        # student frozen experts × teacher post-attn LN.
                        teacher_rep_for_shared = t_mlp
                        wrapper = decoder_layers(student)[layer_idx].mlp
                        experts_module = wrapper.inner.experts  # ModuleList
                        target_idx, target_norms = magnitude_oracle_topk_with_norms(
                            experts_module, t_ln, k=K,
                        )
                        target_margin = torch.ones(
                            target_idx.size(0), device=target_idx.device, dtype=target_norms.dtype,
                        )

                    teacher_rep_teacher_input = t_mlp
                    teacher_rep_for_router = teacher_rep_for_shared
                    router_target_input = (
                        h_student if args.router_target_input == "student_mlp_input"
                        else t_ln
                    )
                    input_diag: Optional[Dict[str, float]] = None
                    target_shift_rmse: Optional[float] = None
                    if args.shared_target_input == "student_mlp_input":
                        with torch.no_grad():
                            teacher_mlp = teacher.model.layers[layer_idx].mlp
                            teacher_param = next(teacher_mlp.parameters())
                            teacher_rep_for_shared = teacher_mlp(
                                h_student.to(device=teacher_param.device, dtype=teacher_param.dtype)
                            ).detach()
                        if should_log:
                            with torch.no_grad():
                                hs_flat = h_student.reshape(-1, h_student.shape[-1]).float()
                                ht_flat = t_ln.reshape(-1, t_ln.shape[-1]).float()
                                input_diff = hs_flat - ht_flat
                                input_diag = {
                                    "input_rmse": float(input_diff.pow(2).mean().clamp_min(1e-12).sqrt().item()),
                                    "input_cos": float(F.cosine_similarity(hs_flat, ht_flat, dim=-1).mean().item()),
                                }
                                H_target = teacher_rep_for_shared.shape[-1]
                                target_shift = (
                                    teacher_rep_for_shared.reshape(-1, H_target).float()
                                    - teacher_rep_teacher_input.reshape(-1, H_target).float()
                                )
                                target_shift_rmse = float(
                                    target_shift.pow(2).mean().clamp_min(1e-12).sqrt().item()
                                )
                    elif should_log:
                        with torch.no_grad():
                            hs_flat = h_student.reshape(-1, h_student.shape[-1]).float()
                            ht_flat = t_ln.reshape(-1, t_ln.shape[-1]).float()
                            input_diff = hs_flat - ht_flat
                            input_diag = {
                                "input_rmse": float(input_diff.pow(2).mean().clamp_min(1e-12).sqrt().item()),
                                "input_cos": float(F.cosine_similarity(hs_flat, ht_flat, dim=-1).mean().item()),
                            }
                            target_shift_rmse = 0.0

                    target_mass: Optional[torch.Tensor] = None
                    if args.router_loss_form == "best_subset_topk_ce":
                        with torch.no_grad():
                            best_subset_target = teacher_rep_for_router
                            if args.shared_target_form == "routed_residual":
                                H_best = shared.shape[-1]
                                shared_anchor = (
                                    oracle_out.get("shared_out") if use_frozen_oracle and "oracle_out" in locals()
                                    else shared.detach().reshape(-1, H_best)
                                )
                                best_subset_target = (
                                    teacher_rep_for_router.reshape(-1, H_best)
                                    - shared_anchor.detach().reshape(-1, H_best)
                                )
                            target_idx, target_norms, target_margin = best_subset_topk_from_routed_stack(
                                routed_stack, best_subset_target, K,
                            )
                    elif args.router_loss_form in {"marginal_gain_topk_ce", "marginal_gain_kl"}:
                        with torch.no_grad():
                            H_gain = shared.shape[-1]
                            shared_anchor = (
                                oracle_out.get("shared_out") if use_frozen_oracle and "oracle_out" in locals()
                                else shared.detach().reshape(-1, H_gain)
                            )
                            residual_target = teacher_rep_for_router.reshape(-1, H_gain) - shared_anchor.detach().reshape(-1, H_gain)
                            target_idx, target_norms, target_margin, target_mass = marginal_gain_targets_from_routed_stack(
                                routed_stack, residual_target, K,
                            )
                    elif args.router_loss_form in {"activation_mass_topk_ce", "activation_rank_topk_ce", "activation_mass_kl", "dense_vector_topk_ce", "dense_vector_kl"}:
                        with torch.no_grad():
                            layer_neuron_indices = activation_mass_neuron_indices.get(int(layer_idx))
                            if layer_neuron_indices is None:
                                raise ValueError(
                                    f"Missing activation-mass/dense-vector neuron mapping for layer {int(layer_idx)}"
                                )
                            if args.router_loss_form == "activation_rank_topk_ce":
                                target_idx, target_norms, target_margin, _rank_score = activation_rank_targets_from_dense_mlp(
                                    teacher.model.layers[layer_idx].mlp,
                                    router_target_input,
                                    int(manifest["n_experts"]),
                                    int(manifest.get("n_routed_experts", int(manifest["n_experts"]))),
                                    K,
                                    expert_neuron_indices=layer_neuron_indices,
                                )
                            elif args.router_loss_form in {"dense_vector_topk_ce", "dense_vector_kl"}:
                                target_idx, target_norms, target_margin, target_mass = dense_vector_targets_from_dense_mlp(
                                    teacher.model.layers[layer_idx].mlp,
                                    router_target_input,
                                    int(manifest.get("n_routed_experts", int(manifest["n_experts"]))),
                                    K,
                                    expert_neuron_indices=layer_neuron_indices,
                                )
                            else:
                                target_idx, target_norms, target_margin, target_mass = activation_mass_targets_from_dense_mlp(
                                    teacher.model.layers[layer_idx].mlp,
                                    router_target_input,
                                    int(manifest["n_experts"]),
                                    int(manifest.get("n_routed_experts", int(manifest["n_experts"]))),
                                    K,
                                    expert_neuron_indices=layer_neuron_indices,
                                )
                    elif args.router_loss_form == "cmoe_representative_router_loss":
                        with torch.no_grad():
                            target_idx, target_norms, target_margin = cmoe_representative_router_targets(
                                oracle_mlps[layer_idx], router_target_input, K,
                            )

                    # L_shared (Option B = combined). reduction picks mse vs rmse.
                    # Gradient routing depends on (shared_grad_routed_detach, EAS):
                    #   detach=0 (joint):
                    #     pred = shared + routed + (eas if EAS) → all components learn L_shared
                    #   detach=1, EAS=0 (sharedonly):
                    #     pred = shared + routed.detach() → only shared learns L_shared,
                    #     routed/router learn from L_task + L_router (cross-layer chain) only
                    #   detach=1, EAS=1 (EAS-only-shared):
                    #     pred = shared.detach() + routed.detach() + eas → only EAS learns
                    #     L_shared. shared behaves like routed (L_task + L_router via cross-
                    #     layer residual chain), EAS specializes in residual recovery.
                    eas_out = comp.get("eas_out") if isinstance(comp, dict) else None
                    if args.shared_grad_routed_detach:
                        if eas_out is not None:
                            # EAS-only-shared: detach both shared and routed, EAS keeps grad
                            moe_pred = routed.detach() + eas_out
                            shared_for_loss = shared.detach()
                        else:
                            # sharedonly: only routed detached, shared learns L_shared
                            moe_pred = routed.detach()
                            shared_for_loss = shared
                    else:
                        # joint
                        moe_pred = routed if eas_out is None else (routed + eas_out)
                        shared_for_loss = shared
                    if compute_shared_loss and (shared_layer_mask is None or int(layer_idx) in shared_layer_mask):
                        if args.shared_target_form == "routed_residual":
                            ls = loss_routed_residual_recovery(
                                routed_for_residual,
                                shared,
                                teacher_rep_for_shared,
                                reduction=shared_reduction,
                                cast_fp32=args.shared_loss_fp32,
                                sample_weight=shared_token_weights,
                            )
                        elif args.shared_target_form == "residual":
                            ls = loss_shared_residual(
                                shared_for_loss, teacher_rep_for_shared, moe_pred.detach(),
                                reduction=shared_reduction,
                            )
                        else:
                            ls = loss_shared_joint(
                                shared_for_loss, moe_pred, teacher_rep_for_shared,
                                reduction=shared_reduction, cast_fp32=args.shared_loss_fp32,
                                sample_weight=shared_token_weights,
                            )
                        loss_shared_total = loss_shared_total + ls.float()
                        layer_aux_loss_for_audit = (
                            ls.float()
                            if layer_aux_loss_for_audit is None
                            else layer_aux_loss_for_audit + ls.float()
                        )
                        n_shared_layers_used += 1

                    router_margin_threshold = float(args.router_margin_min)
                    if args.router_margin_threshold_quantile > 0.0:
                        if not (0.0 < args.router_margin_threshold_quantile < 1.0):
                            raise ValueError(
                                "--router_margin_threshold_quantile must be in (0, 1) when set"
                            )
                        with torch.no_grad():
                            router_margin_threshold = float(
                                torch.quantile(
                                    target_margin.float().reshape(-1),
                                    float(args.router_margin_threshold_quantile),
                                ).item()
                            )
                    if should_log and args.router_loss_form in {"margin_pairwise_hinge", "margin_pairwise_hinge_conf"}:
                        with torch.no_grad():
                            margin_flat = target_margin.float().reshape(-1).clamp_min(0.0)
                            q = torch.quantile(
                                margin_flat,
                                torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90], device=margin_flat.device),
                            )
                            eff_w = (margin_flat - router_margin_threshold).clamp_min(0.0)
                            if router_margin_threshold <= 0.0 and args.router_margin_power == 1.0:
                                eff_w = margin_flat
                            elif args.router_margin_power != 1.0:
                                eff_w = eff_w.pow(float(args.router_margin_power))
                            active_frac = float((eff_w > 0).float().mean().item())
                            eff_w_norm = eff_w / eff_w.mean().clamp_min(1e-6)
                            eff_w_clipped = eff_w_norm
                            if args.router_margin_weight_clip > 0.0:
                                eff_w_clipped = eff_w_norm.clamp_max(float(args.router_margin_weight_clip))
                            router_margin_stats.append({
                                "threshold": router_margin_threshold,
                                "active_frac": active_frac,
                                "q10": float(q[0].item()),
                                "q25": float(q[1].item()),
                                "q50": float(q[2].item()),
                                "q75": float(q[3].item()),
                                "q90": float(q[4].item()),
                                "eff_weight_max_unclipped": float(eff_w_norm.max().item()),
                                "eff_weight_mean_unclipped": float(eff_w_norm.mean().item()),
                                "eff_weight_max": float(eff_w_clipped.max().item()),
                                "eff_weight_mean": float(eff_w_clipped.mean().item()),
                                "eff_weight_clip": float(args.router_margin_weight_clip),
                            })

                    if compute_router_loss and (router_layer_mask is None or int(layer_idx) in router_layer_mask):
                        if target_idx.numel() > 0 and int(target_idx.max().item()) >= int(router_scores.shape[-1]):
                            raise ValueError(
                                f"router target index out of range: max={int(target_idx.max().item())} "
                                f"router_dim={int(router_scores.shape[-1])} form={args.router_loss_form}"
                            )
                        if args.router_loss_form == "mag_kl":
                            lr_loss = loss_router_magnitude_weighted_kl(
                                router_scores, target_idx, target_norms, k=K, is_softmax=True,
                                target_temperature=args.router_target_temperature,
                            )
                        elif args.router_loss_form == "topk_ce":
                            lr_loss = loss_router_topk_ce(
                                router_scores, target_idx, k=K, is_softmax=True,
                            )
                        elif args.router_loss_form == "renorm_topk_ce":
                            lr_loss = loss_router_renormalized_topk_ce(
                                router_scores, target_idx, k=K, is_softmax=True,
                            )
                        elif args.router_loss_form == "pairwise_hinge":
                            lr_loss = loss_router_pairwise_hinge(
                                router_scores, target_idx, k=K, is_softmax=True,
                            )
                        elif args.router_loss_form in {"margin_pairwise_hinge", "margin_pairwise_hinge_conf"}:
                            lr_loss = loss_router_margin_weighted_pairwise_hinge(
                                router_scores,
                                target_idx,
                                target_margin,
                                k=K,
                                is_softmax=True,
                                min_target_margin=router_margin_threshold,
                                margin_power=args.router_margin_power,
                                weight_clip=args.router_margin_weight_clip,
                            )
                        elif args.router_loss_form == "per_expert_bce":
                            lr_loss = loss_router_per_expert_bce(
                                router_scores, target_idx, k=K, is_softmax=True,
                            )
                        elif args.router_loss_form in {"activation_mass_kl", "dense_vector_kl", "marginal_gain_kl"}:
                            if target_mass is None:
                                raise ValueError(f"{args.router_loss_form} missing full router target distribution")
                            lr_loss = loss_router_distribution_kl(
                                router_scores,
                                target_mass,
                                is_softmax=True,
                                target_temperature=args.router_target_temperature,
                            )
                        elif args.router_loss_form in {"best_subset_topk_ce", "activation_mass_topk_ce", "activation_rank_topk_ce", "dense_vector_topk_ce", "marginal_gain_topk_ce", "cmoe_representative_router_loss"}:
                            lr_loss = loss_router_topk_ce(
                                router_scores, target_idx, k=K, is_softmax=True,
                            )
                        else:
                            raise ValueError(f"unknown router_loss_form={args.router_loss_form!r}")
                        loss_router_total = loss_router_total + lr_loss.float()
                        layer_aux_loss_for_audit = (
                            lr_loss.float()
                            if layer_aux_loss_for_audit is None
                            else layer_aux_loss_for_audit + lr_loss.float()
                        )
                        n_layers_used += 1

                    if (
                        args.audit_aux_layer_locality
                        and args.aux_loss_scope == "layer_local"
                        and aux_locality_audit is None
                        and not getattr(args, "_aux_layer_locality_audit_done", False)
                        and layer_aux_loss_for_audit is not None
                        and (args.audit_aux_layer_idx < 0 or int(layer_idx) == int(args.audit_aux_layer_idx))
                    ):
                        aux_locality_audit = _audit_aux_layer_locality(
                            student, int(layer_idx), layer_aux_loss_for_audit
                        )
                        args._aux_layer_locality_audit_done = True

                    # === per-layer instrumentation (notepad_cmoe_rrd §6/§8/§9/§11) ===
                    # Re-uses RRD captures (no extra forward). Cost: a few flat
                    # reduces per layer per log_every step.
                    if should_log:
                        with torch.no_grad():
                            # Flatten to [N, H]. routed/shared/teacher_rep_for_shared
                            # may be [B, S, H] or already [N, H] depending on caller.
                            H_dim = routed.shape[-1]
                            student_flat = (routed + shared).reshape(-1, H_dim).float()
                            teacher_flat = teacher_rep_for_shared.reshape(-1, H_dim).float()
                            diff = student_flat - teacher_flat
                            layer_rmse = diff.pow(2).mean().clamp_min(1e-12).sqrt().item()
                            layer_cos = F.cosine_similarity(
                                student_flat, teacher_flat, dim=-1
                            ).mean().item()
                            s_norm = student_flat.norm(dim=-1)
                            t_norm = teacher_flat.norm(dim=-1).clamp_min(1e-12)
                            layer_mag = (s_norm / t_norm).mean().item()
                            routed_flat = routed.reshape(-1, H_dim).float()
                            shared_flat = shared.reshape(-1, H_dim).float()
                            residual_target_flat = teacher_flat - shared_flat
                            residual_diff = routed_flat - residual_target_flat
                            residual_rmse = residual_diff.pow(2).mean().clamp_min(1e-12).sqrt().item()
                            residual_cos = F.cosine_similarity(
                                routed_flat, residual_target_flat, dim=-1
                            ).mean().item()
                            residual_norm_ratio = (
                                routed_flat.norm(dim=-1)
                                / residual_target_flat.norm(dim=-1).clamp_min(1e-12)
                            ).mean().item()
                            proxy_flat = routed_for_residual.reshape(-1, H_dim).float()
                            proxy_residual_diff = proxy_flat - residual_target_flat
                            proxy_residual_rmse = proxy_residual_diff.pow(2).mean().clamp_min(1e-12).sqrt().item()
                            proxy_residual_cos = F.cosine_similarity(
                                proxy_flat, residual_target_flat, dim=-1
                            ).mean().item()
                            # Router quality is the number of matching experts in the
                            # top-K set (2/1/0 for K=2). Exact-set reporting was
                            # retired after Phase123 and must not be regenerated.
                            _, student_topk = router_logits.float().topk(K, dim=-1)
                            layer_hit_count = (
                                (student_topk.unsqueeze(-1) == target_idx.unsqueeze(-2))
                                .any(dim=-1)
                                .sum(dim=-1)
                                .float()
                                .mean()
                                .item()
                            )
                        per_layer_metrics[int(layer_idx)] = {
                            "rmse": layer_rmse,
                            "cos": layer_cos,
                            "mag": layer_mag,
                            "residual_rmse": residual_rmse,
                            "residual_cos": residual_cos,
                            "residual_norm_ratio": residual_norm_ratio,
                            "proxy_residual_rmse": proxy_residual_rmse,
                            "proxy_residual_cos": proxy_residual_cos,
                            "hit_count": layer_hit_count,
                            "hit_fraction": layer_hit_count / max(float(K), 1.0),
                        }
                        if input_diag is not None:
                            per_layer_metrics[int(layer_idx)].update(input_diag)
                        if target_shift_rmse is not None:
                            per_layer_metrics[int(layer_idx)]["target_shift_rmse"] = target_shift_rmse

                # Sanity report (one-shot, step 0 only) — pretty summary.
                if run_sanity and sanity_log:
                    max_diffs = [v["max_diff"] for v in sanity_log.values()]
                    mean_diffs = [v["mean_diff"] for v in sanity_log.values()]
                    rel_errs = [v["rel_err"] for v in sanity_log.values()]
                    logger.info(
                        "Oracle sanity (step 0, %d layers): max|oracle - dense| ∈ "
                        "[%.4g, %.4g], mean ∈ [%.4g, %.4g], rel_err mean=%.4g — "
                        "carve = clean split if max < 1e-3.",
                        len(sanity_log),
                        min(max_diffs), max(max_diffs),
                        min(mean_diffs), max(mean_diffs),
                        sum(rel_errs) / len(rel_errs),
                    )

            # === Unified total_loss combiner ===
            # Normal mode keeps the weighted sum. Single-loss mode backprops one raw
            # objective to verify that loss term independently decreases.
            total_loss = (
                loss_task.float()
                if single_loss_objective == "ce"
                else args.alpha_task * loss_task.float()
            )
            last_logged = {
                "single_loss_objective": single_loss_objective,
                "aux_loss_scope": args.aux_loss_scope,
                "layer_local_aux_input": args.layer_local_aux_input,
                "task": float(loss_task.detach()),
                "loss_ce": float(loss_task.detach()),
                "loss_kd": 0.0,
                "loss_shared": 0.0,
                "loss_router": 0.0,
                "weighted_ce": float((
                    loss_task.detach() if single_loss_objective == "ce" else args.alpha_task * loss_task.detach()
                ).float()),
                "weighted_kd": 0.0,
                "weighted_shared": 0.0,
                "weighted_router": 0.0,
                "task_token_weight_mode": args.task_token_weight_mode,
                "task_token_weight_stats": task_token_weight_stats,
            }

            if needs_rrd_aux and (n_shared_layers_used > 0 or n_layers_used > 0):
                if (
                    (not single_loss_active and args.alpha_shared > 0.0)
                    or single_loss_objective == "shared"
                ) and n_shared_layers_used == 0:
                    raise ValueError(
                        f"--shared_layer_mask={args.shared_layer_mask!r} selected no processed layers"
                    )
                if (
                    ((not single_loss_active and args.alpha_router > 0.0) or single_loss_objective == "router")
                    and n_layers_used == 0
                ):
                    raise ValueError(
                        f"--router_layer_mask={args.router_layer_mask!r} selected no processed layers"
                    )
                router_denom = max(n_layers_used, 1)
                shared_denom = max(n_shared_layers_used, 1)
                loss_shared_avg = loss_shared_total / shared_denom
                loss_router_avg = loss_router_total / router_denom
                loss_shared_avg_for_audit = loss_shared_avg
                loss_router_avg_for_audit = loss_router_avg
                if (
                    args.audit_router_residual_grad
                    and not getattr(args, "_router_residual_grad_audit_done", False)
                    and n_shared_layers_used > 0
                    and args.mode != "lora"
                ):
                    residual_grad_audit = _grad_group_norms(loss_shared_avg, student, retain_graph=True)
                    logger.info("Residual/shared->module grad audit first_step: %s", residual_grad_audit)
                    args._router_residual_grad_audit_done = True
                else:
                    residual_grad_audit = None
                if single_loss_objective == "shared":
                    total_loss = loss_shared_avg
                    weighted_shared = loss_shared_avg.detach()
                    weighted_router = torch.zeros_like(loss_router_avg.detach())
                elif single_loss_objective == "router":
                    total_loss = loss_router_avg
                    weighted_shared = torch.zeros_like(loss_shared_avg.detach())
                    weighted_router = loss_router_avg.detach()
                else:
                    total_loss = (
                        total_loss
                        + args.alpha_shared * loss_shared_avg
                        + args.alpha_router * loss_router_avg
                    )
                    weighted_shared = args.alpha_shared * loss_shared_avg.detach()
                    weighted_router = args.alpha_router * loss_router_avg.detach()
                last_logged.update({
                    "shared": float(loss_shared_avg.detach()),
                    "router": float(loss_router_avg.detach()),
                    "loss_shared": float(loss_shared_avg.detach()),
                    "loss_router": float(loss_router_avg.detach()),
                    "weighted_shared": float(weighted_shared.float()),
                    "weighted_router": float(weighted_router.float()),
                    "n_layers": n_layers_used,
                    "n_shared_layers": n_shared_layers_used,
                    "shared_layer_mask": sorted(shared_layer_mask) if shared_layer_mask is not None else None,
                    "router_layer_mask": sorted(router_layer_mask) if router_layer_mask is not None else None,
                    "shared_target_input": args.shared_target_input,
                    "router_target_input": args.router_target_input,
                    "shared_token_weight_mode": args.shared_token_weight_mode,
                    "shared_token_weight_stats": shared_token_weight_stats,
                    "aux_loss_scope": args.aux_loss_scope,
                    "layer_local_aux_input": args.layer_local_aux_input,
                    "aux_layer_locality_audit": aux_locality_audit,
                    "router_train_routing_mode": args.router_train_routing_mode,
                    "router_relax_epsilon": float(args.router_relax_epsilon),
                    "residual_prediction_source": (
                        "routed_out_train_proxy"
                        if args.router_train_routing_mode != "hard_topk"
                        else "hard_routed_out"
                    ),
                })
                if residual_grad_audit is not None:
                    last_logged["router_residual_grad_audit"] = residual_grad_audit
                if should_log and per_layer_metrics:
                    # Aggregate scalars for quick at-a-glance log lines.
                    rmse_list = [v["rmse"] for v in per_layer_metrics.values()]
                    cos_list = [v["cos"] for v in per_layer_metrics.values()]
                    mag_list = [v["mag"] for v in per_layer_metrics.values()]
                    residual_rmse_list = [v["residual_rmse"] for v in per_layer_metrics.values()]
                    residual_cos_list = [v["residual_cos"] for v in per_layer_metrics.values()]
                    residual_norm_ratio_list = [v["residual_norm_ratio"] for v in per_layer_metrics.values()]
                    proxy_residual_rmse_list = [v["proxy_residual_rmse"] for v in per_layer_metrics.values()]
                    proxy_residual_cos_list = [v["proxy_residual_cos"] for v in per_layer_metrics.values()]
                    hit_count_list = [v["hit_count"] for v in per_layer_metrics.values()]
                    input_rmse_list = [v["input_rmse"] for v in per_layer_metrics.values() if "input_rmse" in v]
                    input_cos_list = [v["input_cos"] for v in per_layer_metrics.values() if "input_cos" in v]
                    target_shift_rmse_list = [
                        v["target_shift_rmse"] for v in per_layer_metrics.values()
                        if "target_shift_rmse" in v
                    ]
                    last_logged.update({
                        "per_layer_metrics": per_layer_metrics,
                        "lyr_rmse_mean": sum(rmse_list) / len(rmse_list),
                        "lyr_cos_mean": sum(cos_list) / len(cos_list),
                        "lyr_mag_mean": sum(mag_list) / len(mag_list),
                        "lyr_residual_rmse_mean": sum(residual_rmse_list) / len(residual_rmse_list),
                        "lyr_residual_cos_mean": sum(residual_cos_list) / len(residual_cos_list),
                        "lyr_residual_norm_ratio_mean": sum(residual_norm_ratio_list) / len(residual_norm_ratio_list),
                        "lyr_proxy_residual_rmse_mean": sum(proxy_residual_rmse_list) / len(proxy_residual_rmse_list),
                        "lyr_proxy_residual_cos_mean": sum(proxy_residual_cos_list) / len(proxy_residual_cos_list),
                        "lyr_hit_count_mean": sum(hit_count_list) / len(hit_count_list),
                    })
                    if input_rmse_list:
                        last_logged["lyr_input_rmse_mean"] = sum(input_rmse_list) / len(input_rmse_list)
                    if input_cos_list:
                        last_logged["lyr_input_cos_mean"] = sum(input_cos_list) / len(input_cos_list)
                    if target_shift_rmse_list:
                        last_logged["lyr_target_shift_rmse_mean"] = (
                            sum(target_shift_rmse_list) / len(target_shift_rmse_list)
                        )
                if should_log and router_margin_stats:
                    last_logged.update({
                        "router_margin_stats": router_margin_stats,
                        "router_margin_threshold_mean": sum(v["threshold"] for v in router_margin_stats) / len(router_margin_stats),
                        "router_margin_active_frac_mean": sum(v["active_frac"] for v in router_margin_stats) / len(router_margin_stats),
                        "router_margin_q10_mean": sum(v["q10"] for v in router_margin_stats) / len(router_margin_stats),
                        "router_margin_q25_mean": sum(v["q25"] for v in router_margin_stats) / len(router_margin_stats),
                        "router_margin_q50_mean": sum(v["q50"] for v in router_margin_stats) / len(router_margin_stats),
                        "router_margin_q75_mean": sum(v["q75"] for v in router_margin_stats) / len(router_margin_stats),
                        "router_margin_q90_mean": sum(v["q90"] for v in router_margin_stats) / len(router_margin_stats),
                        "router_margin_eff_weight_max_mean": sum(v["eff_weight_max"] for v in router_margin_stats) / len(router_margin_stats),
                        "router_margin_eff_weight_mean_mean": sum(v["eff_weight_mean"] for v in router_margin_stats) / len(router_margin_stats),
                        "router_margin_eff_weight_max_unclipped_mean": sum(v["eff_weight_max_unclipped"] for v in router_margin_stats) / len(router_margin_stats),
                        "router_margin_eff_weight_mean_unclipped_mean": sum(v["eff_weight_mean_unclipped"] for v in router_margin_stats) / len(router_margin_stats),
                        "router_margin_weight_clip": float(args.router_margin_weight_clip),
                    })
                    if args.router_min_active_frac > 0.0 and last_logged["router_margin_active_frac_mean"] < args.router_min_active_frac:
                        raise RuntimeError(
                            f"router margin active fraction {last_logged['router_margin_active_frac_mean']:.4f} "
                            f"below --router_min_active_frac={args.router_min_active_frac:.4f}"
                        )

            if loss_kd is not None:
                if single_loss_objective == "kd":
                    total_loss = loss_kd.float()
                    weighted_kd = loss_kd.detach()
                else:
                    total_loss = total_loss + args.alpha_kd * loss_kd
                    weighted_kd = args.alpha_kd * loss_kd.detach()
                last_logged["kd"] = float(loss_kd.detach())
                last_logged["loss_kd"] = float(loss_kd.detach())
                last_logged["weighted_kd"] = float(weighted_kd.float())
            elif single_loss_objective == "kd":
                raise ValueError("--single_loss_objective kd requested but KD loss was not computed")

            if (
                args.audit_loss_grad_groups
                and not getattr(args, "_loss_grad_group_audit_done", False)
            ):
                loss_grad_group_audit: Dict[str, Any] = {}

                def _safe_loss_grad_audit(term_name: str, loss_value: torch.Tensor) -> None:
                    try:
                        loss_grad_group_audit[term_name] = _grad_group_norms(
                            loss_value.float(), student, retain_graph=True,
                        )
                    except torch.OutOfMemoryError as exc:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        loss_grad_group_audit[term_name] = {
                            "error": "cuda_oom",
                            "message": str(exc).split("\n", 1)[0],
                        }
                        logger.warning(
                            "Loss-term grad group audit for %s hit CUDA OOM; "
                            "recording audit error and continuing training",
                            term_name,
                        )

                if loss_task is not None:
                    _safe_loss_grad_audit("ce", loss_task)
                if loss_kd is not None:
                    _safe_loss_grad_audit("kd", loss_kd)
                if (
                    args.alpha_router != 0.0
                    and loss_router_avg_for_audit is not None
                    and n_layers_used > 0
                ):
                    _safe_loss_grad_audit("router", loss_router_avg_for_audit)
                if loss_shared_avg_for_audit is not None and n_shared_layers_used > 0:
                    _safe_loss_grad_audit("shared", loss_shared_avg_for_audit)
                last_logged["loss_grad_group_audit"] = loss_grad_group_audit
                logger.info("Loss-term grad group audit first_step: %s", loss_grad_group_audit)
                args._loss_grad_group_audit_done = True
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if (
                _distributed_is_primary(args)
                and args.audit_task_router_grad
                and not getattr(args, "_task_router_grad_audit_done", False)
                and args.mode != "lora"
            ):
                router_named_params = [
                    (name, p)
                    for name, p in student.named_parameters()
                    if p.requires_grad
                    and (
                        "phase127_mlp_router." in name
                        or ".gate.gate" in name
                        or ".gate.classifier" in name
                        or ".gate.extra_scale" in name
                    )
                ]
                router_params = [p for _name, p in router_named_params]
                grads = torch.autograd.grad(
                    loss_task.float(),
                    router_params,
                    retain_graph=True,
                    allow_unused=True,
                ) if router_params else []
                group_sq = {"gate": 0.0, "classifier": 0.0, "extra_scale": 0.0, "mlp_router": 0.0}
                group_max = {"gate": 0.0, "classifier": 0.0, "extra_scale": 0.0, "mlp_router": 0.0}
                group_params = {"gate": 0, "classifier": 0, "extra_scale": 0, "mlp_router": 0}
                total_sq = 0.0
                total_max = 0.0
                for (name, param), grad in zip(router_named_params, grads):
                    if "phase127_mlp_router." in name:
                        group = "mlp_router"
                    elif ".gate.extra_scale" in name:
                        group = "extra_scale"
                    elif ".gate.classifier" in name:
                        group = "classifier"
                    else:
                        group = "gate"
                    group_params[group] += param.numel()
                    if grad is None:
                        continue
                    g = grad.detach().float()
                    sq = float((g * g).sum().item())
                    mx = float(g.abs().max().item())
                    group_sq[group] += sq
                    group_max[group] = max(group_max[group], mx)
                    total_sq += sq
                    total_max = max(total_max, mx)
                logger.info(
                    "Task->router grad audit first_step: tensors=%d params=%d "
                    "grad_l2=%.6e grad_max=%.6e group_l2=%s group_max=%s group_params=%s",
                    len(router_named_params),
                    sum(group_params.values()),
                    total_sq ** 0.5,
                    total_max,
                    {k: v ** 0.5 for k, v in group_sq.items()},
                    group_max,
                    group_params,
                )
                args._task_router_grad_audit_done = True
                if args.fail_on_zero_task_router_grad and total_sq == 0.0:
                    raise RuntimeError(
                        "Task->router grad audit failed: CE/task loss has zero gradient "
                        "on trainable router params."
                    )

            _raise_on_nonfinite_loss(total_loss, args, step)
            (total_loss / max(args.gradient_accumulation_steps, 1)).backward()

            grad_norm_raw = None
            grad_clipped = False
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if distributed:
                    _sync_distributed_gradients(
                        student,
                        world_size=int(args._distributed_world_size),
                    )
                if float(args.max_grad_norm) > 0.0:
                    trainable_parameters = [
                        parameter
                        for parameter in student.parameters()
                        if parameter.requires_grad and parameter.grad is not None
                    ]
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        trainable_parameters,
                        max_norm=float(args.max_grad_norm),
                    )
                    grad_norm_raw = float(grad_norm.detach())
                    if not math.isfinite(grad_norm_raw):
                        raise FloatingPointError(
                            f"non-finite gradient norm at step {step}: {grad_norm_raw}"
                        )
                    grad_clipped = grad_norm_raw > float(args.max_grad_norm)
                    gradient_history.append(
                        {
                            "step": int(step),
                            "completed_step": int(step + 1),
                            "grad_norm_raw": grad_norm_raw,
                            "max_grad_norm": float(args.max_grad_norm),
                            "clipped": grad_clipped,
                        }
                    )
                router_audit_before = None
                router_audit_grad_sq = 0.0
                router_audit_grad_max = 0.0
                expert_audit_before: Dict[
                    str, Tuple[str, torch.Tensor, nn.Parameter, float, float]
                ] | None = None
                if (
                    _distributed_is_primary(args)
                    and args.audit_router_update
                    and not getattr(args, "_router_update_audit_done", False)
                    and args.mode != "lora"
                ):
                    router_audit_before = []
                    for name, p in student.named_parameters():
                        if not p.requires_grad:
                            continue
                        if not (
                            "phase127_mlp_router." in name
                            or ".gate.gate" in name
                            or ".gate.classifier" in name
                            or ".gate.extra_scale" in name
                        ):
                            continue
                        router_audit_before.append((name, p.detach().float().cpu().clone(), p))
                        if p.grad is not None:
                            g = p.grad.detach().float()
                            router_audit_grad_sq += float((g * g).sum().item())
                            router_audit_grad_max = max(
                                router_audit_grad_max, float(g.abs().max().item())
                            )
                if (
                    _distributed_is_primary(args)
                    and args.audit_expert_update
                    and not getattr(args, "_expert_update_audit_done", False)
                    and args.mode != "lora"
                ):
                    expert_audit_before = {}
                    for name, p in student.named_parameters():
                        if not p.requires_grad or p.grad is None:
                            continue
                        group = classify_moe_param_group(name)
                        if group not in {"routed", "shared"} or group in expert_audit_before:
                            continue
                        grad = p.grad.detach().float()
                        if grad.numel() == 0:
                            continue
                        grad_max_value = float(grad.abs().max().item())
                        if grad_max_value == 0.0:
                            continue
                        grad_l2_value = float((grad * grad).sum().sqrt().item())
                        before = p.detach().float().cpu().clone()
                        expert_audit_before[group] = (
                            name,
                            before,
                            p,
                            grad_l2_value,
                            grad_max_value,
                        )
                optimizer.step()
                if router_audit_before is not None:
                    update_sq = 0.0
                    update_max = 0.0
                    n_router_tensors = 0
                    n_router_params = 0
                    for _name, before, p in router_audit_before:
                        after = p.detach().float().cpu()
                        d = after - before
                        update_sq += float((d * d).sum().item())
                        update_max = max(update_max, float(d.abs().max().item()))
                        n_router_tensors += 1
                        n_router_params += p.numel()
                    grad_l2 = router_audit_grad_sq ** 0.5
                    update_l2 = update_sq ** 0.5
                    logger.info(
                        "Router update audit first_step: "
                        "tensors=%d params=%d grad_l2=%.6e grad_max=%.6e "
                        "update_l2=%.6e update_max=%.6e",
                        n_router_tensors,
                        n_router_params,
                        grad_l2,
                        router_audit_grad_max,
                        update_l2,
                        update_max,
                    )
                    args._router_update_audit_done = True
                    if args.fail_on_zero_router_update and (
                        grad_l2 == 0.0 or update_l2 == 0.0
                    ):
                        raise RuntimeError(
                            "Router update audit failed: first-step router gradient or update is zero."
                        )
                if expert_audit_before is not None:
                    expert_updates: Dict[str, Dict[str, Any]] = {}
                    for group in ("routed", "shared"):
                        sentinel = expert_audit_before.get(group)
                        if sentinel is None:
                            expert_updates[group] = {
                                "parameter": None,
                                "grad_l2": 0.0,
                                "grad_max": 0.0,
                                "update_l2": 0.0,
                                "update_max": 0.0,
                            }
                            continue
                        name, before, p, grad_l2, grad_max = sentinel
                        after = p.detach().float().cpu()
                        delta = after - before
                        expert_updates[group] = {
                            "parameter": name,
                            "grad_l2": grad_l2,
                            "grad_max": grad_max,
                            "update_l2": float((delta * delta).sum().sqrt().item()),
                            "update_max": float(delta.abs().max().item()),
                        }
                    logger.info(
                        "Expert update audit first_step: %s",
                        expert_updates,
                    )
                    args._expert_update_audit_done = True
                    if args.fail_on_zero_expert_update and any(
                        row["grad_l2"] == 0.0 or row["update_l2"] == 0.0
                        for group, row in expert_updates.items()
                        if group == "routed" or int(args.cmoe_n_shared) > 0
                    ):
                        raise RuntimeError(
                            "Expert update audit failed: routed/shared expert gradient or "
                            f"update is zero: {expert_updates}"
                        )
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if val_loader is not None:
                completed_step = step + 1
                if completed_step == max_steps or completed_step % max(int(args.validation_every), 1) == 0:
                    val_rec = _run_lightweight_validation(completed_step)
                    if val_rec is not None:
                        validation_history.append(val_rec)

            logged_total = float(total_loss.detach())
            logged_scalars = last_logged
            if should_log and distributed:
                aggregate_keys = (
                    "task",
                    "loss_ce",
                    "loss_kd",
                    "loss_shared",
                    "loss_router",
                    "weighted_ce",
                    "weighted_kd",
                    "weighted_shared",
                    "weighted_router",
                )
                aggregate = torch.tensor(
                    [logged_total]
                    + [float(last_logged[key]) for key in aggregate_keys],
                    device=device,
                    dtype=torch.float32,
                )
                torch.distributed.all_reduce(
                    aggregate,
                    op=torch.distributed.ReduceOp.SUM,
                )
                aggregate.div_(int(args._distributed_world_size))
                logged_total = float(aggregate[0].item())
                logged_scalars = dict(last_logged)
                for index, key in enumerate(aggregate_keys, start=1):
                    logged_scalars[key] = float(aggregate[index].item())

            if should_log and _distributed_is_primary(args):
                cur_lr = optimizer.param_groups[0]["lr"]
                rec = {
                    "stage": args.mode,
                    "step": step,
                    "lr": cur_lr,
                    "total": logged_total,
                    "loss_total": logged_total,
                    "grad_norm_raw": grad_norm_raw,
                    "grad_clipped": grad_clipped,
                    **logged_scalars,
                }
                if torch.cuda.is_available():
                    rec["mem_mb"] = torch.cuda.max_memory_allocated() // (1024 * 1024)
                log_history.append(rec)
                # Don't spam the per-layer dict to stdout — only the aggregates.
                rec_for_stdout = {
                    k: v for k, v in rec.items() if k not in {"per_layer_metrics", "router_margin_stats"}
                }
                logger.info("step=%d %s", step, rec_for_stdout)

            step += 1

            # Optional intermediate save at given step boundaries. Full CMoE
            # checkpoints are stored as state_dicts; LoRA stores PEFT adapters
            # to avoid merge_and_unload mutating the model mid-training.
            if (
                save_at_steps_set
                and step in save_at_steps_set
                and not args.no_save
                and _distributed_is_primary(args)
            ):
                ckpt_subdir = os.path.join(
                    args.output_dir, f"ckpt_step_{step}"
                )
                save_fn = _save_intermediate_lora_adapter if args.mode == "lora" else _save_intermediate_ckpt
                save_fn(
                    student=student,
                    args=args,
                    step_count=step,
                    base_manifest=manifest,
                    log_history=log_history,
                    output_dir=ckpt_subdir,
                    elapsed_sec=time.time() - t0,
                    max_mem_mb=(
                        torch.cuda.max_memory_allocated() // (1024 * 1024)
                        if torch.cuda.is_available() else 0
                    ),
                )
                if bool(args.write_ready_markers):
                    _atomic_ready_marker(ckpt_subdir)
                logger.info(
                    "Intermediate ckpt saved at step %d -> %s",
                    step, ckpt_subdir,
                )
            if resume_path and (
                step == max_steps
                or step in save_at_steps_set
                or (
                    int(args.resume_save_every) > 0
                    and step % int(args.resume_save_every) == 0
                )
            ):
                _save_rolling_resume(
                    path=resume_path,
                    contract_sha256=resume_contract_sha256,
                    completed_step=step,
                    student=student,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    log_history=log_history,
                    validation_history=validation_history,
                    elapsed_sec=time.time() - t0,
                )
                logger.info(
                    "Rolling resume state saved at step %d -> %s",
                    step,
                    resume_path,
                )

    elapsed = time.time() - t0
    max_mem = torch.cuda.max_memory_allocated() // (1024 * 1024) if torch.cuda.is_available() else 0
    summary = {
        "max_steps": max_steps,
        "elapsed_sec": elapsed,
        "max_mem_mb": max_mem,
        "final_log": last_logged,
        "log_history": log_history,
        "gradient_history": gradient_history,
        "max_grad_norm": float(args.max_grad_norm),
        "gradient_clip_steps": sum(
            1 for record in gradient_history if bool(record.get("clipped"))
        ),
        "validation_history": validation_history,
        "manifest": manifest,
    }
    if distributed and not _distributed_is_primary(args):
        return summary
    logger.info("Training done in %.1fs, max_mem=%dMB", elapsed, max_mem)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.no_save:
        # Smoke/diagnostic runs should not write model weights, but the log
        # history is still the auditable evidence for router update checks.
        with open(os.path.join(args.output_dir, "cpt_log.json"), "w") as f:
            json.dump({k: v for k, v in summary.items() if k != "manifest"}, f, indent=2, default=str)

    # 6. Save (unless smoke / explicitly disabled).
    if not args.no_save:
        if needs_rrd_aux:
            unwrap_student_capture(student)
        skip_full = bool(getattr(args, "skip_full_state_dict", False))
        adapter_dir = None
        cmoe_extra_state_path = None
        if args.mode == "lora" and skip_full:
            adapter_dir = os.path.join(args.output_dir, "adapter")
            os.makedirs(adapter_dir, exist_ok=True)
            student.save_pretrained(
                adapter_dir,
                selected_adapters=["default"] if args.stack_init_adapter else None,
            )
            if moe_type == "cmoe":
                cmoe_extra_state_path = _save_cmoe_extra_state(
                    student, args.output_dir
                )
            logger.info(
                "Saved final LoRA adapter=%s extra_state=%s",
                adapter_dir,
                cmoe_extra_state_path,
            )
        elif args.mode == "lora":
            # Preserve the legacy merged-full-checkpoint path for callers that
            # did not opt into the storage-efficient experiment protocol.
            logger.info("LoRA mode: merging adapters into base via merge_and_unload()")
            student = student.merge_and_unload()

        trainable_delta_path = None
        trainable_delta_keys: List[str] = []
        if bool(getattr(args, "save_trainable_delta", False)):
            if args.mode == "lora":
                raise ValueError(
                    "--save_trainable_delta is CE-only; use adapter+extra-state "
                    "with --skip_full_state_dict for LoRA"
                )
            trainable_delta_path = os.path.join(
                args.output_dir, "trainable_delta.pt"
            )
            delta_sd = _trainable_delta_state(student)
            trainable_delta_keys = sorted(delta_sd)
            logger.info(
                "Saving trainable_delta -> %s (%d tensors)",
                trainable_delta_path,
                len(delta_sd),
            )
            torch.save(delta_sd, trainable_delta_path)

        sd_path = os.path.join(args.output_dir, "state_dict.pt")
        dense_hf_checkpoint = False
        if skip_full:
            logger.info(
                "Skipping full state_dict save; reconstruct from base+adapter/delta"
            )
        elif moe_type == "dense":
            logger.info(
                "Saving dense control as a Hugging Face pretrained checkpoint"
            )
            student_cpu = student.to("cpu")
            student_cpu.save_pretrained(
                args.output_dir,
                safe_serialization=True,
                max_shard_size="5GB",
            )
            tokenizer.save_pretrained(args.output_dir)
            sd_path = args.output_dir
            dense_hf_checkpoint = True
        else:
            logger.info("Saving state_dict -> %s (cpu)", sd_path)
            student_cpu = student.to("cpu")
            torch.save(student_cpu.state_dict(), sd_path)
        # Update manifest with CPT config.
        new_manifest = {
            **summary["manifest"],
            "cpt_mode": args.mode,
            "cpt_moe_type": args.moe_type,
            "cpt_shared_loss_form": args.shared_loss_form,
            "aux_loss_scope": args.aux_loss_scope,
            "cpt_aux_loss_scope": args.aux_loss_scope,
            "cpt_layer_local_aux_input": args.layer_local_aux_input,
            "cpt_shared_target_form": args.shared_target_form,
            "cpt_shared_target_input": args.shared_target_input,
            "cpt_router_target_input": args.router_target_input,
            "cpt_router_train_routing_mode": args.router_train_routing_mode,
            "router_train_routing_mode": args.router_train_routing_mode,
            "cpt_router_relax_epsilon": float(args.router_relax_epsilon),
            "router_relax_epsilon": float(args.router_relax_epsilon),
            "residual_prediction_source": (
                "routed_out_train_proxy" if args.router_train_routing_mode != "hard_topk" else "hard_routed_out"
            ),
            "hard_inference_preserved": True,
            "cpt_shared_grad_routed_detach": int(args.shared_grad_routed_detach),
            "cpt_shared_layer_mask": args.shared_layer_mask,
            "cpt_router_layer_mask": args.router_layer_mask,
        "cpt_moe_trainable_layer_mask": args.moe_trainable_layer_mask,
        "cpt_moe_trainable_layer_mask_info": getattr(args, "_moe_trainable_layer_mask_info", None),
            "cpt_shared_token_weight_mode": args.shared_token_weight_mode,
            "cpt_shared_token_weight_quantile": float(args.shared_token_weight_quantile),
            "cpt_shared_token_weight_floor": float(args.shared_token_weight_floor),
            "cpt_task_token_weight_mode": args.task_token_weight_mode,
            "cpt_task_token_weight_quantile": float(args.task_token_weight_quantile),
            "cpt_task_token_weight_floor": float(args.task_token_weight_floor),
            "cpt_extra_scale_trainable": int(args.cmoe_extra_scale_trainable),
            "cpt_extra_scale_init": args.cmoe_extra_scale_init,
            "cpt_cmoe_enable_load_balance": bool(args.cmoe_enable_load_balance),
            "cpt_add_eas": int(args.cmoe_add_eas),
            "cpt_eas_init_std": float(args.cmoe_eas_init_std),
            # Mirror to top-level manifest key so lmeval_cmoe.build_moe_skeleton_model
            # picks up add_eas during downstream evaluation.
            "add_eas": bool(args.cmoe_add_eas),
            "cpt_lr": args.learning_rate,
            "cpt_min_lr": args.min_learning_rate,
            "cpt_lr_schedule": args.lr_schedule,
            "cpt_optimizer": (
                "bitsandbytes.Adam8bit"
                if bool(args.use_8bit_adam)
                else "torch.optim.Adam"
            ),
            "cpt_optimizer_betas": [0.9, 0.95],
            "cpt_optimizer_eps": 1e-8,
            "cpt_weight_decay": 0.0,
            "cpt_shared_lr_multiplier": float(args.shared_lr_multiplier),
            "cpt_trainable_group_counts": getattr(args, "_trainable_group_counts", None),
            "cpt_optimizer_lr_group_counts": getattr(args, "_optimizer_lr_group_counts", None),
            "cpt_max_steps": max_steps,
            "cpt_seqlen": args.max_seqlen,
            "cpt_per_device_bsz": args.per_device_batch_size,
            "cpt_grad_accum_steps": args.gradient_accumulation_steps,
            "cpt_max_grad_norm": float(args.max_grad_norm),
            "cpt_global_bsz": int(args.per_device_batch_size)
            * int(args.gradient_accumulation_steps)
            * int(args._distributed_world_size),
            "cpt_one_epoch": bool(args.one_epoch),
            "cpt_effective_train_tokens": int(max_steps)
            * int(args.per_device_batch_size)
            * int(args.gradient_accumulation_steps)
            * int(getattr(args, "_distributed_world_size", 1))
            * int(args.max_seqlen),
            "cpt_distributed_sync": bool(args.distributed_sync),
            "cpt_distributed_world_size": int(args._distributed_world_size),
            "cpt_distributed_gradient_sync": (
                "in_place_all_reduce_mean"
                if bool(args.distributed_sync)
                else "none"
            ),
            "cpt_distributed_load_balance_counts": bool(args.distributed_sync),
            "cpt_fail_on_nonfinite": bool(args.fail_on_nonfinite),
            "cpt_alpha_task": args.alpha_task,
            "cpt_alpha_shared": args.alpha_shared,
            "cpt_alpha_router": args.alpha_router,
            "cpt_alpha_kd": float(args.alpha_kd),
            "cpt_single_loss_objective": args.single_loss_objective,
            "cpt_kd_temperature": float(args.kd_temperature),
            "cpt_kd_target_model": args.kd_target_model,
            "cpt_validation_calib_path": args.validation_calib_path,
            "cpt_validation_every": int(args.validation_every),
            "cpt_validation_batches": int(args.validation_batches),
            "cpt_validation_history_len": len(validation_history),
            "cpt_router_loss_form": args.router_loss_form,
            "cpt_activation_mass_mapping_source": (
                getattr(args, "_activation_mass_mapping_source", "recovered_exact_dense_neuron_indices")
                if args.router_loss_form in ROUTER_MAPPING_LOSS_FORMS
                else None
            ),
            "router_contribution_score": router_contribution_score_name(args.router_loss_form),
            "router_target_oracle": router_target_oracle_name(args.router_loss_form),
            "cpt_router_target_distribution": router_target_distribution_name(args.router_loss_form),
            "cpt_router_target_temperature": float(args.router_target_temperature),
            "cpt_router_margin_min": float(args.router_margin_min),
            "cpt_router_margin_power": float(args.router_margin_power),
            "cpt_router_margin_weight_clip": float(args.router_margin_weight_clip),
            "cpt_router_margin_threshold_quantile": float(args.router_margin_threshold_quantile),
            "cpt_router_min_active_frac": float(args.router_min_active_frac),
            "cpt_freeze_routed": bool(args.freeze_routed),
            "cpt_freeze_shared": bool(args.freeze_shared),
            "cpt_freeze_router": bool(args.freeze_router),
            "cpt_train_lm_head": bool(args.train_lm_head),
            "cpt_train_attention": bool(args.train_attention),
            "cpt_trainable_parameter_manifest": getattr(
                args, "_trainable_parameter_manifest", None
            ),
            "cpt_em_phase": args.em_phase,
            "cpt_data_source": args.data_source,
            "cpt_data_manifest_train_key": args.data_manifest_train_key,
            "cpt_data_manifest_validation_key": args.data_manifest_validation_key,
            "cpt_data_manifest_sha256": getattr(args, "_data_manifest_sha256", None),
            "cpt_data_manifest_seed_independent": bool(
                args.data_manifest_seed_independent
            ),
            "cpt_resume_schema": "cpt_ordered_resume_v1" if resume_path else None,
            "cpt_resume_contract_sha256": (
                resume_contract_sha256 if resume_path else None
            ),
            "cpt_resume_state_path": (
                os.path.abspath(resume_path) if resume_path else None
            ),
            "cpt_resume_save_every": int(args.resume_save_every),
            "cpt_data_packing": args.data_packing,
            "cpt_data_shuffle": bool(args.data_shuffle),
            "cpt_data_split": args.data_split,
            "cpt_continuation_field": args.continuation_field,
            "cpt_shared_down_init": args.shared_down_init,
            "cpt_shared_zero_jitter_std": float(args.shared_zero_jitter_std),
            "cpt_shared_loss_fp32": bool(args.shared_loss_fp32),
            "cpt_seed": args.seed,
            "cpt_dtype": args.dtype,
            "cpt_use_8bit_adam": bool(args.use_8bit_adam),
            # LoRA-mode-specific (None / unused outside --mode lora):
            "cpt_lora_r": (args.lora_r if args.mode == "lora" else None),
            "cpt_lora_alpha": (args.lora_alpha if args.mode == "lora" else None),
            "cpt_lora_dropout": (args.lora_dropout if args.mode == "lora" else None),
            "cpt_lora_target_modules": (
                [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
                if args.mode == "lora" else None
            ),
            "cpt_lora_extra_lr": (args.lora_extra_lr if args.mode == "lora" else None),
            "cpt_calib_path": args.calib_path,
            "cpt_source_moe_dir": os.path.abspath(args.moe_dir),
            "cpt_source_state_dict_path": (
                os.path.abspath(args.cmoe_state_dict_path)
                if (args.moe_type == "cmoe" and args.cmoe_state_dict_path)
                else None
            ),
            "cpt_source_checkpoint_manifest_path": getattr(
                args, "_source_checkpoint_manifest_path", None
            ),
            "cpt_source_checkpoint_manifest_sha256": getattr(
                args, "_source_checkpoint_manifest_sha256", None
            ),
            "cpt_preloaded_mlp_router": bool(
                getattr(args, "_preloaded_mlp_router", False)
            ),
            "cpt_initialization_artifacts": getattr(
                args, "_initialization_artifacts", {}
            ),
            "cpt_elapsed_sec": elapsed,
            "cpt_max_mem_mb": max_mem,
            "cpt_save_trainable_delta": bool(getattr(args, "save_trainable_delta", False)),
            "cpt_skip_full_state_dict": bool(getattr(args, "skip_full_state_dict", False)),
            "cpt_full_state_dict_path": (
                None if bool(getattr(args, "skip_full_state_dict", False)) else os.path.abspath(sd_path)
            ),
            "cpt_trainable_delta_path": os.path.abspath(trainable_delta_path) if trainable_delta_path else None,
            "cpt_trainable_delta_tensors": len(trainable_delta_keys),
            "cpt_adapter_dir": (
                os.path.abspath(adapter_dir) if adapter_dir else None
            ),
            "cpt_cmoe_extra_state_path": (
                os.path.abspath(cmoe_extra_state_path)
                if cmoe_extra_state_path
                else None
            ),
            "cpt_checkpoint_format": (
                "hf_pretrained"
                if dense_hf_checkpoint
                else (
                    (
                        "peft_adapter_plus_cmoe_extra_state"
                        if cmoe_extra_state_path
                        else "peft_adapter"
                    )
                    if adapter_dir
                    else (
                        "base_plus_trainable_delta"
                        if trainable_delta_path
                        else "full_state_dict"
                    )
                )
            ),
        }
        if getattr(args, "_mlp_router_manifest_fields", None):
            new_manifest.update(args._mlp_router_manifest_fields)
        with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
            json.dump(new_manifest, f, indent=2)
        # Copy tokenizer + config (mirror T2 layout) so lmeval_cmoe + future
        # CPT can reload directly from the output dir.
        if moe_type != "dense":
            for fname in os.listdir(args.moe_dir):
                full_src = os.path.join(args.moe_dir, fname)
                if not os.path.isfile(full_src):
                    continue
                if (
                    fname.startswith("tokenizer")
                    or fname == "config.json"
                    or fname == "special_tokens_map.json"
                    or fname == "generation_config.json"
                ):
                    shutil.copy(full_src, os.path.join(args.output_dir, fname))
        # Save log history as a separate file for downstream plotting.
        with open(os.path.join(args.output_dir, "cpt_log.json"), "w") as f:
            json.dump({k: v for k, v in summary.items() if k != "manifest"}, f, indent=2, default=str)
        if bool(args.write_ready_markers):
            _atomic_ready_marker(args.output_dir)
        logger.info("Saved CPT outputs to %s", args.output_dir)

    return summary


def main() -> None:
    args = parse_args()
    try:
        train(args)
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
