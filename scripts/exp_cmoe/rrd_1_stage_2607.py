#!/usr/bin/env python3
"""Reusable RRD-1-stage-2607 strategy for registry-backed CMoE models.

The strategy is the finalized Phase149
``04_full_e2e_original_lr_adam8bit`` configuration:

  L = CE + True-STF activation-mass router CE + 2 * joint RRD RMSE

All losses are active for one exact-token WikiText-2 epoch.  CE and joint RRD
update shared/routed experts; True-STF updates only the randomly initialized
per-layer MLP router.  Training and inference use hard top-A uniform expert
aggregation.

The ``--router-architecture cmoe`` ablation keeps the same objective, data,
optimizer, and per-layer LR policy but trains and evaluates the canonical
carved CMoE dual-gate router instead of attaching the random MLP router.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import math
import os
import random
import re
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
os.environ.setdefault("DISABLE_VERSION_CHECK", "1")

from llamafactory.train.distill.chain_losses import loss_router_topk_ce  # noqa: E402
from scripts.exp_cmoe import rrd_contract as rrd2  # noqa: E402
from scripts.exp_cmoe.cpt_train import (  # noqa: E402
    TokenIdsChunk,
    _collate,
    build_activation_mass_neuron_index_cache_from_state_dict,
    install_teacher_hooks,
    unwrap_student_capture,
    wrap_student_for_capture,
)
from scripts.exp_cmoe.mlp_router_patch import MLPProbeBank, attach_mlp_probe_router  # noqa: E402
from scripts.exp_cmoe.logit_distillation import logit_kd_loss  # noqa: E402
from scripts.exp_cmoe.rrd_router_targets import (  # noqa: E402
    load_cmoe_model,
    load_teacher,
    routed_modules,
    target_labels_for_batch,
)

STRATEGY = "rrd_1_stage_2607"
SOURCE_RUN = "04_full_e2e_original_lr_adam8bit"
PYTHON = ROOT / ".venv/bin/python"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)

SEQLEN = 2048
TOTAL_WINDOWS = 2048
BATCH_SIZE = 2
PROBE_HIDDEN = 1024
LR_MIN = 3e-5
LR_MAX = 1e-3
DEFAULT_ROUTER_BASE_LR = 5e-4
ROUTER_BASE_LR = DEFAULT_ROUTER_BASE_LR
ROUTER_ARCHITECTURE = "mlp"
RRD_TARGET_FORM = "joint"
RRD_GRADIENT_SCOPE = "target_graph"
TEACHER_TARGET_MODE = "teacher_trajectory"
LR_POLICY = "derived_middle_boost"
LR_RULE = "expert=clip(0.06*base); outer=base; middle=clip(2*base)"
EXPERT_LR = 3e-5
ROUTER_OUTER_LR = DEFAULT_ROUTER_BASE_LR
ROUTER_EARLY_LR = DEFAULT_ROUTER_BASE_LR
ROUTER_MIDDLE_LR = 1e-3
ROUTER_LATE_LR = DEFAULT_ROUTER_BASE_LR
CE_WEIGHT = 1.0
ROUTER_WEIGHT = 1.0
JOINT_RRD_WEIGHT = 2.0
KD_WEIGHT = 0.0
KD_TEMPERATURE = 1.0
KD_CHUNK_TOKENS = 128
EVAL_TASKS = rrd2.EVAL_TASKS
METRIC_DIRECTIONS = rrd2.METRIC_DIRECTIONS


def configure_loss_weights(
    ce_weight: float,
    router_weight: float,
    joint_rrd_weight: float,
) -> dict[str, float]:
    """Configure nonnegative objective weights while preserving canonical defaults."""
    global CE_WEIGHT, ROUTER_WEIGHT, JOINT_RRD_WEIGHT
    values = {
        "ce_weight": float(ce_weight),
        "router_weight": float(router_weight),
        "joint_rrd_weight": float(joint_rrd_weight),
    }
    for name, value in values.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative, got {value!r}")
    CE_WEIGHT = values["ce_weight"]
    ROUTER_WEIGHT = values["router_weight"]
    JOINT_RRD_WEIGHT = values["joint_rrd_weight"]
    return values


def configure_router_architecture(value: str) -> str:
    """Select the router implementation while preserving the MLP default."""
    global ROUTER_ARCHITECTURE
    architecture = str(value).strip().lower()
    if architecture not in {"mlp", "cmoe"}:
        raise ValueError(f"unknown router architecture: {value!r}")
    ROUTER_ARCHITECTURE = architecture
    return architecture


def configure_logit_kd(weight: float, temperature: float, chunk_tokens: int) -> None:
    global KD_WEIGHT, KD_TEMPERATURE, KD_CHUNK_TOKENS
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("KD weight must be finite and nonnegative")
    if not math.isfinite(temperature) or temperature <= 0 or chunk_tokens < 1:
        raise ValueError("KD temperature and chunk size must be positive")
    if weight > 0 and TEACHER_TARGET_MODE != "teacher_trajectory":
        raise ValueError("logit KD requires the full dense teacher trajectory")
    KD_WEIGHT, KD_TEMPERATURE, KD_CHUNK_TOKENS = float(weight), float(temperature), int(chunk_tokens)


def configure_rrd_target_form(value: str) -> str:
    """Select the RRD target while preserving the joint default."""
    global RRD_TARGET_FORM
    target_form = str(value).strip().lower()
    if target_form not in {"joint", "shared_residual_stopgrad"}:
        raise ValueError(f"unknown RRD target form: {value!r}")
    RRD_TARGET_FORM = target_form
    return target_form


def configure_rrd_gradient_scope(value: str) -> str:
    """Select which expert parameter groups receive the RRD gradient."""
    global RRD_GRADIENT_SCOPE
    gradient_scope = str(value).strip().lower()
    if gradient_scope not in {"target_graph", "shared_only"}:
        raise ValueError(f"unknown RRD gradient scope: {value!r}")
    RRD_GRADIENT_SCOPE = gradient_scope
    return gradient_scope


def configure_teacher_target_mode(value: str) -> str:
    """Select the canonical teacher trajectory or the FFN-only on-policy target."""
    global TEACHER_TARGET_MODE
    target_mode = str(value).strip().lower()
    if target_mode not in {"teacher_trajectory", "on_policy"}:
        raise ValueError(f"unknown teacher target mode: {value!r}")
    TEACHER_TARGET_MODE = target_mode
    return target_mode


def strategy_name() -> str:
    if KD_WEIGHT > 0:
        return f"{STRATEGY}_logit_kd"
    if TEACHER_TARGET_MODE == "on_policy":
        return f"{STRATEGY}_on_policy"
    return STRATEGY


def router_arch_manifest_value() -> str:
    return f"mlp_h{PROBE_HIDDEN}" if ROUTER_ARCHITECTURE == "mlp" else "cmoe_carved"


def router_description(contract: rrd2.Contract) -> str:
    if ROUTER_ARCHITECTURE == "mlp":
        return f"{contract.hidden_size}->GELU({PROBE_HIDDEN})->{contract.topology.routed_total}"
    return "canonical carved CMoE dual-gate (classifier * SiLU(gate)).abs()"


def aggregation_description(contract: rrd2.Contract) -> str:
    if ROUTER_ARCHITECTURE == "mlp":
        return f"hard top-{contract.topology.active} uniform sum"
    return f"canonical carved CMoE hard top-{contract.topology.active} weighted sum"


def configure_learning_rates(
    router_base_lr: float,
    *,
    expert_lr_override: float | None = None,
    uniform_router_lr: float | None = None,
    router_early_lr: float | None = None,
    router_middle_lr: float | None = None,
    router_late_lr: float | None = None,
) -> dict[str, float]:
    """Configure default, uniform, or explicit early/middle/late learning rates."""
    global LR_POLICY, LR_RULE, ROUTER_BASE_LR, EXPERT_LR
    global ROUTER_OUTER_LR, ROUTER_EARLY_LR, ROUTER_MIDDLE_LR, ROUTER_LATE_LR
    band_values = (router_early_lr, router_middle_lr, router_late_lr)
    has_any_band = any(value is not None for value in band_values)
    has_all_bands = all(value is not None for value in band_values)
    if has_any_band and not has_all_bands:
        raise ValueError(
            "--router-early-lr, --router-middle-lr, and --router-late-lr "
            "must be set together"
        )
    if has_any_band and uniform_router_lr is not None:
        raise ValueError("explicit three-band router LRs cannot be combined with --uniform-router-lr")
    if has_any_band and expert_lr_override is None:
        raise ValueError("explicit three-band router LRs require --expert-lr-override")
    if not has_any_band and (expert_lr_override is None) != (uniform_router_lr is None):
        raise ValueError("--expert-lr-override and --uniform-router-lr must be set together")
    if has_all_bands:
        expert = float(expert_lr_override)
        early = float(router_early_lr)
        middle = float(router_middle_lr)
        late = float(router_late_lr)
        for name, value in (
            ("expert", expert),
            ("router early", early),
            ("router middle", middle),
            ("router late", late),
        ):
            if not LR_MIN <= value <= LR_MAX:
                raise ValueError(f"{name} LR must stay in [{LR_MIN:g}, {LR_MAX:g}], got {value:g}")
        LR_POLICY = "explicit_three_band_router"
        LR_RULE = "expert=explicit; early=explicit; middle=explicit; late=explicit"
        ROUTER_BASE_LR = middle
        EXPERT_LR = expert
        ROUTER_EARLY_LR = early
        ROUTER_MIDDLE_LR = middle
        ROUTER_LATE_LR = late
        ROUTER_OUTER_LR = early
    elif uniform_router_lr is not None:
        expert = float(expert_lr_override)
        router = float(uniform_router_lr)
        for name, value in (("expert", expert), ("router", router)):
            if not LR_MIN <= value <= LR_MAX:
                raise ValueError(f"{name} LR must stay in [{LR_MIN:g}, {LR_MAX:g}], got {value:g}")
        LR_POLICY = "explicit_uniform_router"
        LR_RULE = "expert=explicit; outer=middle=uniform_router_lr"
        ROUTER_BASE_LR = router
        EXPERT_LR = expert
        ROUTER_OUTER_LR = router
        ROUTER_EARLY_LR = router
        ROUTER_MIDDLE_LR = router
        ROUTER_LATE_LR = router
    else:
        value = float(router_base_lr)
        if not LR_MIN <= value <= LR_MAX:
            raise ValueError(
                f"router base LR must stay in [{LR_MIN:g}, {LR_MAX:g}], got {value:g}"
            )
        LR_POLICY = "derived_middle_boost"
        LR_RULE = "expert=clip(0.06*base); outer=base; middle=clip(2*base)"
        ROUTER_BASE_LR = value
        ROUTER_OUTER_LR = value
        ROUTER_EARLY_LR = value
        EXPERT_LR = min(LR_MAX, max(LR_MIN, 0.06 * value))
        ROUTER_MIDDLE_LR = min(LR_MAX, max(LR_MIN, 2.0 * value))
        ROUTER_LATE_LR = value
    return {
        "lr_policy": LR_POLICY,
        "router_base_lr": ROUTER_BASE_LR,
        "expert_lr": EXPERT_LR,
        "router_outer_lr": ROUTER_OUTER_LR,
        "router_early_lr": ROUTER_EARLY_LR,
        "router_middle_lr": ROUTER_MIDDLE_LR,
        "router_late_lr": ROUTER_LATE_LR,
    }


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def cuda_peak_memory(devices: dict[str, torch.device]) -> dict[str, dict[str, float]]:
    """Return allocator peaks without changing the training graph."""
    peaks: dict[str, dict[str, float]] = {}
    for role, device in devices.items():
        if device.type != "cuda":
            continue
        with torch.cuda.device(device):
            peaks[role] = {
                "max_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
                "max_reserved_mib": torch.cuda.max_memory_reserved() / (1024**2),
            }
    return peaks


def output_root(contract: rrd2.Contract, smoke: bool = False) -> Path:
    root = contract.output_root
    return root / "smoke" if smoke else root


def run_dir(contract: rrd2.Contract, smoke: bool = False) -> Path:
    return output_root(contract, smoke) / "one_stage_e2e"


def ordered_indices(contract: rrd2.Contract) -> tuple[list[int], dict[str, Any]]:
    split = rrd2.split_payload(contract)
    indices = list(split["stage1_indices"]) + list(split["stage2_indices"])
    if len(indices) != TOTAL_WINDOWS or len(set(indices)) != TOTAL_WINDOWS:
        raise RuntimeError("one-stage deterministic data order is not a full permutation")
    payload = {
        "protocol": "phase146_seeded_permutation_stage1_half_then_stage2_half",
        "seed": contract.seed,
        "source": str(contract.train),
        "source_sha256": split["source_sha256"],
        "total_indices": len(indices),
        "unique_indices": len(set(indices)),
        "first_half_sha256": split["stage1_indices_sha256"],
        "second_half_sha256": split["stage2_indices_sha256"],
        "indices": indices,
    }
    return indices, payload


def strategy_audit(contract: rrd2.Contract) -> dict[str, Any]:
    audit = rrd2.audit_contract(contract)
    audit.pop("stage1", None)
    audit.pop("stage2", None)
    rrd_name = (
        "representation_RMSE"
        if RRD_TARGET_FORM == "joint"
        else "shared_residual_RMSE_stopgrad_routed"
    )
    teacher_input = (
        "student_e2e_trajectory_mlp_input"
        if TEACHER_TARGET_MODE == "on_policy"
        else "dense_teacher_trajectory_mlp_input"
    )
    teacher_output = f"dense_teacher_mlp({teacher_input})"
    rrd_description = (
        f"RMSE(shared_out + routed_out, stopgrad({teacher_output}))"
        if RRD_TARGET_FORM == "joint"
        else f"RMSE(shared_out, stopgrad({teacher_output} - routed_out))"
    )
    if RRD_GRADIENT_SCOPE == "shared_only":
        rrd_trainable = ["shared experts"]
    elif RRD_TARGET_FORM == "joint":
        rrd_trainable = ["shared experts", "routed experts"]
    else:
        rrd_trainable = [
            "shared experts",
            "routed experts through indirect end-to-end trajectory paths",
        ]
    audit.update({
        "strategy": strategy_name(),
        "strategy_source_run": SOURCE_RUN,
        "objective": (
            f"{CE_WEIGHT:g}*CE + {ROUTER_WEIGHT:g}*True-STF_router_CE"
            f" + {JOINT_RRD_WEIGHT:g}*{rrd_name}"
            + (f" + {KD_WEIGHT:g}*logit_KD" if KD_WEIGHT > 0 else "")
        ),
        "loss_terminology": {
            "RRD": "Routing and Residual Distillation = routing loss + representation loss (weighted)",
            "legacy_joint_RRD_RMSE": "representation loss: shared+routed vs dense MLP",
        },
        "logit_kd": {
            "weight": KD_WEIGHT, "temperature": KD_TEMPERATURE,
            "chunk_tokens": KD_CHUNK_TOKENS, "direction": "KL(teacher || student)",
            "reduction": "full_vocabulary_sum_valid_next_token_mean_times_temperature_squared",
            "student_trajectory": "ordinary_student_end_to_end",
            "gradient_policy": "shared_and_routed_only; teacher detached",
        },
        "joint_rrd": rrd_description,
        "rrd_target_form": RRD_TARGET_FORM,
        "rrd_gradient_scope": RRD_GRADIENT_SCOPE,
        "teacher_target_mode": TEACHER_TARGET_MODE,
        "teacher_parameter_scope": (
            "dense_ffn_only"
            if TEACHER_TARGET_MODE == "on_policy"
            else "full_dense_causal_lm"
        ),
        "teacher_target_input": teacher_input,
        "rrd": rrd_description,
        "router_target": (
            "activation_mass_sum_top_A_on_student_true_stf_input"
            if TEACHER_TARGET_MODE == "on_policy"
            else "activation_mass_sum_top_A_on_dense_teacher_input"
        ),
        "router_input": "student_true_stf_trajectory_mlp_input",
        "router_architecture": ROUTER_ARCHITECTURE,
        "router": router_description(contract),
        "aggregation": aggregation_description(contract),
        "data": {
            "total_windows": TOTAL_WINDOWS,
            "sequence_length": SEQLEN,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation": 1,
            "optimizer_steps": TOTAL_WINDOWS // BATCH_SIZE,
            "token_exposure": TOTAL_WINDOWS * SEQLEN,
            "epochs": 1,
            "baseline_multiplier": 1.0,
        },
        "training": {
            "windows": TOTAL_WINDOWS,
            "sequence_length": SEQLEN,
            "tokens": TOTAL_WINDOWS * SEQLEN,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation": 1,
            "optimizer_steps": TOTAL_WINDOWS // BATCH_SIZE,
            "epochs": 1,
            "optimizer": "bitsandbytes.Adam8bit",
            "betas": [0.9, 0.95],
            "weight_decay": 0.0,
            "schedule": "constant",
            "router_base_lr": ROUTER_BASE_LR,
            "expert_lr": EXPERT_LR,
            "router_outer_lr": ROUTER_OUTER_LR,
            "router_early_lr": ROUTER_EARLY_LR,
            "router_middle_lr": ROUTER_MIDDLE_LR,
            "router_late_lr": ROUTER_LATE_LR,
            "lr_policy": LR_POLICY,
            "lr_rule": LR_RULE,
            "lr_bounds": [LR_MIN, LR_MAX],
            "middle_layer_mask": contract.middle_mask,
        },
        "gradient_policy": {
            "CE": ["shared experts", "routed experts"],
            "RRD": rrd_trainable,
            "True-STF_router": [router_arch_manifest_value()],
            "frozen": ["attention", "embedding", "norm", "lm_head"]
            + (["legacy CMoE router"] if ROUTER_ARCHITECTURE == "mlp" else ["MLP router"]),
        },
        "router_quality_reporting": f"top-{contract.topology.active} hit-count only; exact-set disabled",
    })
    return audit


def make_batches(
    contract: rrd2.Contract,
    indices: list[int],
    smoke_steps: int = 0,
) -> list[torch.Tensor]:
    dataset = TokenIdsChunk(str(contract.train), SEQLEN, data_split="all")
    batches = [
        torch.stack([
            dataset[indices[offset]]["input_ids"],
            dataset[indices[offset + 1]]["input_ids"],
        ])
        for offset in range(0, len(indices), BATCH_SIZE)
    ]
    return batches[:smoke_steps] if smoke_steps else batches


def validation_loader(contract: rrd2.Contract, smoke_steps: int) -> Iterable[dict[str, torch.Tensor]]:
    dataset = TokenIdsChunk(str(contract.valtest), SEQLEN, data_split="all")
    loader: Any = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=_collate,
    )
    if smoke_steps:
        return [batch for idx, batch in enumerate(loader) if idx < 2]
    return loader


def report_hit_count(
    logits: torch.Tensor,
    target_idx: torch.Tensor,
    report_k: int,
) -> torch.Tensor:
    k = min(report_k, logits.shape[-1], target_idx.shape[-1])
    pred = logits.float().topk(k, dim=-1).indices
    target = target_idx[..., :k]
    return (
        pred.unsqueeze(-1) == target.unsqueeze(-2)
    ).any(dim=-1).sum(dim=-1).float()


def _dense_mlp_class(config: Any) -> type[nn.Module]:
    if str(config.model_type) == "qwen2":
        from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

        return Qwen2MLP
    if str(config.model_type) == "llama":
        from transformers.models.llama.modeling_llama import LlamaMLP

        return LlamaMLP
    raise ValueError(
        f"FFN-only on-policy teacher does not support model_type={config.model_type!r}"
    )


def _safetensor_weight_map(model_dir: Path) -> dict[str, str]:
    from safetensors import safe_open

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = load_json(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(f"missing weight_map object: {index_path}")
        return {str(key): str(value) for key, value in weight_map.items()}
    single_path = model_dir / "model.safetensors"
    if single_path.exists():
        with safe_open(single_path, framework="pt", device="cpu") as handle:
            return {str(key): single_path.name for key in handle.keys()}
    raise FileNotFoundError(
        "FFN-only on-policy teacher requires model.safetensors or "
        f"model.safetensors.index.json under {model_dir}"
    )


def load_dense_mlp_bank(
    model_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.ModuleList:
    """Load only per-layer dense FFNs from a local safetensors checkpoint."""
    from safetensors import safe_open
    from transformers import AutoConfig

    model_dir = model_dir.expanduser().resolve()
    config = AutoConfig.from_pretrained(str(model_dir), local_files_only=True)
    mlp_class = _dense_mlp_class(config)
    weight_map = _safetensor_weight_map(model_dir)
    bank = nn.ModuleList()
    for layer_idx in range(int(config.num_hidden_layers)):
        with torch.device("meta"):
            mlp = mlp_class(config)
        state: dict[str, torch.Tensor] = {}
        keys_by_shard: dict[str, list[tuple[str, str]]] = {}
        for local_key in mlp.state_dict():
            checkpoint_key = f"model.layers.{layer_idx}.mlp.{local_key}"
            shard = weight_map.get(checkpoint_key)
            if shard is None:
                raise KeyError(
                    f"dense FFN checkpoint key is missing: {checkpoint_key}"
                )
            keys_by_shard.setdefault(shard, []).append((local_key, checkpoint_key))
        for shard, keys in keys_by_shard.items():
            shard_path = model_dir / shard
            with safe_open(shard_path, framework="pt", device="cpu") as handle:
                for local_key, checkpoint_key in keys:
                    state[local_key] = handle.get_tensor(checkpoint_key)
        mlp.load_state_dict(state, strict=True, assign=True)
        mlp.to(device=device, dtype=dtype)
        mlp.requires_grad_(False)
        mlp.eval()
        bank.append(mlp)
    return bank


@torch.no_grad()
def dense_mlp_output_and_activation_mass_labels(
    dense_mlp: nn.Module,
    mlp_input: torch.Tensor,
    expert_neuron_indices: list[torch.Tensor],
    k: int,
    *,
    return_output: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Compute a dense FFN output and top-k activation-mass labels in one pass."""
    original_shape = mlp_input.shape
    hidden = mlp_input.reshape(-1, original_shape[-1])
    gate = dense_mlp.gate_proj(hidden)
    up = dense_mlp.up_proj(hidden)
    act_fn = getattr(dense_mlp, "act_fn", None)
    if act_fn is None:
        raise ValueError("dense MLP has no act_fn")
    intermediate = act_fn(gate) * up
    mass_parts = []
    for expert_idx, indices in enumerate(expert_neuron_indices):
        indices = indices.to(device=intermediate.device, dtype=torch.long)
        if indices.numel() == 0:
            raise ValueError(f"activation-mass mapping for expert {expert_idx} is empty")
        if int(indices.min().item()) < 0 or int(indices.max().item()) >= intermediate.shape[-1]:
            raise ValueError(
                f"activation-mass mapping for expert {expert_idx} is out of range"
            )
        mass_parts.append(
            intermediate.index_select(-1, indices).float().abs().sum(dim=-1)
        )
    if len(mass_parts) < int(k):
        raise ValueError(
            f"activation-mass mapping has {len(mass_parts)} groups, smaller than k={k}"
        )
    routed_mass = torch.stack(mass_parts, dim=-1)
    target_idx = routed_mass.topk(int(k), dim=-1).indices.contiguous()
    output = (
        dense_mlp.down_proj(intermediate).view(original_shape).detach()
        if return_output
        else None
    )
    return output, target_idx.detach()


class OnPolicyTeacherTargets:
    """Frozen dense-FFN targets evaluated on the current student trajectory."""

    def __init__(
        self,
        mlps: nn.ModuleList,
        activation_map: dict[int, list[torch.Tensor]],
        student_device: torch.device,
        teacher_device: torch.device,
        routed_total: int,
    ) -> None:
        self.mlps = mlps
        self.activation_map = activation_map
        self.student_device = student_device
        self.teacher_device = teacher_device
        self.routed_total = int(routed_total)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.mlps.parameters())

    def _teacher_input(self, value: torch.Tensor) -> torch.Tensor:
        return move_tensor_via_cpu(value.detach(), self.teacher_device)

    def _student_output(self, value: torch.Tensor) -> torch.Tensor:
        return move_tensor_via_cpu(value.detach(), self.student_device)

    @torch.no_grad()
    def output_for_input(self, layer_idx: int, mlp_input: torch.Tensor) -> torch.Tensor:
        teacher_input = self._teacher_input(mlp_input)
        output = self.mlps[int(layer_idx)](teacher_input)
        return self._student_output(output)

    @torch.no_grad()
    def output_and_labels_for_input(
        self,
        layer_idx: int,
        mlp_input: torch.Tensor,
        k: int,
        *,
        return_output: bool = True,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        teacher_input = self._teacher_input(mlp_input)
        output, target_idx = dense_mlp_output_and_activation_mass_labels(
            self.mlps[int(layer_idx)],
            teacher_input,
            self.activation_map[int(layer_idx)],
            int(k),
            return_output=return_output,
        )
        if output is not None:
            output = self._student_output(output)
        target_idx = self._student_output(target_idx)
        lower = int(target_idx.min().item())
        upper = int(target_idx.max().item())
        if lower < 0 or upper >= self.routed_total:
            raise RuntimeError(
                f"layer {layer_idx} router target range [{lower}, {upper}] is outside "
                f"[0, {self.routed_total})"
            )
        return output, target_idx

    def outputs_for_components(
        self,
        components: dict[int, dict[str, torch.Tensor]],
    ) -> dict[int, torch.Tensor]:
        outputs: dict[int, torch.Tensor] = {}
        for layer_idx in sorted(components):
            mlp_input = components[layer_idx].get("mlp_input")
            if mlp_input is None:
                raise RuntimeError(
                    f"on-policy target requires captured mlp_input for layer {layer_idx}"
                )
            outputs[layer_idx] = self.output_for_input(layer_idx, mlp_input)
        if len(outputs) != len(self.mlps):
            raise RuntimeError(
                f"on-policy target captured {len(outputs)} layers, expected {len(self.mlps)}"
            )
        return outputs


class TrueSTFRouterPatch:
    """Topology-aware True-STF branch; router gradients stay local to each head."""

    def __init__(
        self,
        student: Any,
        targets: dict[int, torch.Tensor],
        teacher_mlp_outs: dict[int, torch.Tensor],
        target_topk: int,
        report_k: int = 2,
        on_policy_teacher: OnPolicyTeacherTargets | None = None,
        record_metrics: bool = True,
    ) -> None:
        self.student = student
        self.targets = targets
        self.teacher_mlp_outs = teacher_mlp_outs
        self.target_topk = target_topk
        self.report_k = report_k
        self.on_policy_teacher = on_policy_teacher
        self.record_metrics = bool(record_metrics)
        self.originals: list[tuple[nn.Module, Any]] = []
        self.losses: dict[int, torch.Tensor] = {}
        self.layer_metrics: dict[int, dict[str, float]] = {}

    def __enter__(self) -> "TrueSTFRouterPatch":
        for layer_idx, layer in enumerate(self.student.model.layers):
            wrapper = layer.mlp
            self.originals.append((wrapper, wrapper.forward))
            wrapper.forward = types.MethodType(self._forward_for(layer_idx), wrapper)
        return self

    def __exit__(self, *_exc: object) -> None:
        for wrapper, original in self.originals:
            wrapper.forward = original

    def _forward_for(self, layer_idx: int):
        def forward(wrapper: nn.Module, x: torch.Tensor) -> torch.Tensor:
            inner = getattr(wrapper, "inner", wrapper)
            x_flat = x.reshape(-1, x.shape[-1])
            if self.on_policy_teacher is not None:
                if layer_idx not in self.targets:
                    dense, target_idx = self.on_policy_teacher.output_and_labels_for_input(
                        layer_idx,
                        x,
                        self.target_topk,
                        return_output=self.record_metrics,
                    )
                    self.targets[layer_idx] = target_idx
                    record_dense_metric = self.record_metrics
                else:
                    dense = None
                    record_dense_metric = False
                target_idx = self.targets[layer_idx].to(x_flat.device)
            else:
                target_idx = self.targets[layer_idx].to(x_flat.device)
                dense = self.teacher_mlp_outs[layer_idx].reshape(-1, x.shape[-1])
                record_dense_metric = True
            if ROUTER_ARCHITECTURE == "mlp":
                logits = self.student.phase127_mlp_router(layer_idx, x_flat)
                scores = logits.softmax(dim=-1, dtype=torch.float32)
            else:
                _weights, _indices, logits, scores = inner.gate(
                    x_flat, return_diagnostics=True
                )
            self.losses[layer_idx] = loss_router_topk_ce(
                scores,
                target_idx,
                k=self.target_topk,
                is_softmax=True,
            )
            with torch.no_grad():
                routed = torch.zeros_like(x_flat)
                for expert_idx, expert in enumerate(routed_modules(inner)):
                    token_idx = torch.where(
                        (target_idx == expert_idx).any(dim=-1)
                    )[0]
                    if token_idx.numel():
                        routed[token_idx] += expert(x_flat[token_idx])
                forced = inner.shared_experts(x_flat) + routed
                if record_dense_metric:
                    if dense is None:
                        raise RuntimeError("dense teacher output is missing")
                    self.layer_metrics[layer_idx] = {
                        "hit_count": float(
                            report_hit_count(
                                logits,
                                target_idx,
                                self.report_k,
                            ).mean().item()
                        ),
                        "cos_forced_dense": float(
                            F.cosine_similarity(
                                forced.float(),
                                dense.reshape_as(forced).float(),
                                dim=-1,
                            ).mean().item()
                        ),
                    }
            return forced.view_as(x)

        return forward


def move_tensor_via_cpu(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Use blocking CPU staging when tensors cross CUDA devices."""
    if value.device == device:
        return value
    return value.detach().cpu().to(device)


def teacher_targets(
    contract: rrd2.Contract,
    teacher: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    teacher_mlp_outs: dict[int, torch.Tensor],
    teacher_post_ln: dict[int, torch.Tensor],
    activation_map: Any,
    carve_manifest: dict[str, Any],
    teacher_logits: dict[str, torch.Tensor] | None = None,
) -> dict[int, torch.Tensor]:
    teacher_mlp_outs.clear()
    teacher_post_ln.clear()
    student_device = input_ids.device
    teacher_device = next(teacher.parameters()).device
    teacher_input_ids = move_tensor_via_cpu(input_ids, teacher_device)
    teacher_attention_mask = (
        move_tensor_via_cpu(attention_mask, teacher_device)
        if attention_mask is not None
        else None
    )
    with torch.no_grad():
        teacher_output = teacher(
            input_ids=teacher_input_ids,
            attention_mask=teacher_attention_mask,
            use_cache=False,
        )
        if teacher_logits is not None:
            teacher_logits.clear()
            teacher_logits["logits"] = move_tensor_via_cpu(teacher_output.logits.detach(), student_device)
        del teacher_output
        targets = target_labels_for_batch(
            teacher,
            teacher_post_ln,
            activation_map,
            carve_manifest,
            "mass",
            contract.topology.active,
        )
        for layer, value in tuple(teacher_mlp_outs.items()):
            teacher_mlp_outs[layer] = move_tensor_via_cpu(value, student_device)
        targets = {
            layer: move_tensor_via_cpu(value, student_device)
            for layer, value in targets.items()
        }
        for layer, value in targets.items():
            lower = int(value.min().item())
            upper = int(value.max().item())
            if lower < 0 or upper >= contract.topology.routed_total:
                raise RuntimeError(
                    f"layer {layer} router target range [{lower}, {upper}] is outside "
                    f"[0, {contract.topology.routed_total})"
                )
    teacher_post_ln.clear()
    return targets


def joint_rmse(
    components: dict[int, dict[str, torch.Tensor]],
    teacher_mlp_outs: dict[int, torch.Tensor],
) -> torch.Tensor:
    losses = {
        layer: (
            comp["shared_out"].float()
            + (
                comp["routed_out"].float()
                if RRD_TARGET_FORM == "joint"
                else comp["routed_out"].detach().float()
            )
            - teacher_mlp_outs[layer].detach().float()
        ).pow(2).mean().clamp_min(1e-12).sqrt()
        for layer, comp in components.items()
    }
    if not losses:
        raise RuntimeError("joint RRD captured no MoE components")
    return torch.stack([losses[layer] for layer in sorted(losses)]).mean()


def classify_param(name: str) -> str:
    if "phase127_mlp_router." in name:
        return "router"
    if any(marker in name for marker in (
        ".mlp.gate.gate",
        ".mlp.gate.classifier",
        ".mlp.inner.gate.gate",
        ".mlp.inner.gate.classifier",
    )):
        return "router"
    if ".mlp.experts." in name or ".mlp.inner.experts." in name:
        return "routed"
    if ".mlp.shared_experts." in name or ".mlp.inner.shared_experts." in name:
        return "shared"
    return "other"


def selected_router_param(name: str) -> bool:
    if ROUTER_ARCHITECTURE == "mlp":
        return "phase127_mlp_router." in name
    return classify_param(name) == "router" and "phase127_mlp_router." not in name


def set_trainable(student: nn.Module) -> dict[str, int]:
    counts = {
        "router": 0,
        "routed": 0,
        "shared": 0,
        "other": 0,
        "total": 0,
        "trainable": 0,
    }
    for name, parameter in student.named_parameters():
        group = classify_param(name)
        enabled = (
            group in {"shared", "routed"}
            or (group == "router" and selected_router_param(name))
        ) and parameter.numel() > 0
        parameter.requires_grad_(enabled)
        counts["total"] += int(parameter.numel())
        if enabled:
            counts[group] += int(parameter.numel())
            counts["trainable"] += int(parameter.numel())
    return counts


def named_group(
    student: nn.Module,
    group: str,
) -> list[tuple[str, nn.Parameter]]:
    return [
        (name, parameter)
        for name, parameter in student.named_parameters()
        if (
            parameter.requires_grad
            and parameter.numel() > 0
            and classify_param(name) == group
        )
    ]


def accumulate_loss_grads(
    loss: torch.Tensor,
    named_params: list[tuple[str, nn.Parameter]],
    *,
    scale: float,
    retain_graph: bool,
) -> dict[str, float]:
    parameters = [parameter for _name, parameter in named_params]
    grads = torch.autograd.grad(
        loss.float() * float(scale),
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    squared = {"router": 0.0, "routed": 0.0, "shared": 0.0}
    for (name, parameter), grad in zip(named_params, grads):
        if grad is None:
            continue
        grad = grad.detach()
        parameter.grad = grad if parameter.grad is None else parameter.grad + grad
        squared[classify_param(name)] += float(grad.float().norm().item()) ** 2
    return {key: math.sqrt(value) for key, value in squared.items()}


def mean_layer_metric(rows: dict[int, dict[str, float]], key: str) -> float:
    values = [row[key] for row in rows.values() if key in row]
    return float(sum(values) / max(len(values), 1))


def split_router_params(
    params: list[tuple[str, nn.Parameter]],
    middle_layers: tuple[int, ...],
    n_layers: int,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    middle_set = set(middle_layers)
    outer: list[nn.Parameter] = []
    middle: list[nn.Parameter] = []
    seen: set[int] = set()
    for name, param in params:
        pattern = (
            r"phase127_mlp_router\.heads\.(\d+)\."
            if ROUTER_ARCHITECTURE == "mlp"
            else r"model\.layers\.(\d+)\."
        )
        match = re.search(pattern, name)
        if match is None:
            raise RuntimeError(f"cannot recover router layer from {name}")
        layer = int(match.group(1))
        seen.add(layer)
        (middle if layer in middle_set else outer).append(param)
    if seen != set(range(n_layers)):
        raise RuntimeError(f"router layer coverage mismatch: {sorted(seen)}")
    return outer, middle


def split_router_params_three_band(
    params: list[tuple[str, nn.Parameter]],
    middle_layers: tuple[int, ...],
    n_layers: int,
) -> tuple[list[nn.Parameter], list[nn.Parameter], list[nn.Parameter]]:
    if not middle_layers or tuple(middle_layers) != tuple(range(middle_layers[0], middle_layers[-1] + 1)):
        raise RuntimeError(f"three-band routing requires a contiguous middle mask: {middle_layers}")
    middle_set = set(middle_layers)
    first_middle = middle_layers[0]
    last_middle = middle_layers[-1]
    early: list[nn.Parameter] = []
    middle: list[nn.Parameter] = []
    late: list[nn.Parameter] = []
    seen: set[int] = set()
    for name, param in params:
        pattern = (
            r"phase127_mlp_router\.heads\.(\d+)\."
            if ROUTER_ARCHITECTURE == "mlp"
            else r"model\.layers\.(\d+)\."
        )
        match = re.search(pattern, name)
        if match is None:
            raise RuntimeError(f"cannot recover router layer from {name}")
        layer = int(match.group(1))
        seen.add(layer)
        if layer < first_middle:
            early.append(param)
        elif layer in middle_set:
            middle.append(param)
        elif layer > last_middle:
            late.append(param)
        else:
            raise RuntimeError(f"router layer {layer} is not covered by a three-band partition")
    if seen != set(range(n_layers)):
        raise RuntimeError(f"router layer coverage mismatch: {sorted(seen)}")
    if not early or not middle or not late:
        raise RuntimeError("three-band router partition contains an empty parameter group")
    return early, middle, late


def build_optimizer(
    contract: rrd2.Contract,
    router_params: list[tuple[str, nn.Parameter]],
    expert_params: list[tuple[str, nn.Parameter]],
    *,
    paged_optimizer: bool = False,
) -> tuple[torch.optim.Optimizer, dict[str, Any]]:
    try:
        import bitsandbytes as bnb
    except ImportError as exc:
        raise RuntimeError("rrd_1_stage_2607 requires bitsandbytes Adam8bit") from exc
    groups = [
        {
            "params": [param for _name, param in expert_params],
            "lr": EXPERT_LR,
            "name": "shared_and_routed_experts",
        },
    ]
    if LR_POLICY == "explicit_three_band_router":
        early, middle, late = split_router_params_three_band(
            router_params,
            contract.middle_layers,
            contract.n_layers,
        )
        groups.extend([
            {"params": early, "lr": ROUTER_EARLY_LR, "name": "router_early"},
            {"params": middle, "lr": ROUTER_MIDDLE_LR, "name": "router_middle"},
            {"params": late, "lr": ROUTER_LATE_LR, "name": "router_late"},
        ])
    else:
        outer, middle = split_router_params(
            router_params,
            contract.middle_layers,
            contract.n_layers,
        )
        groups.extend([
            {"params": outer, "lr": ROUTER_OUTER_LR, "name": "router_outer"},
            {"params": middle, "lr": ROUTER_MIDDLE_LR, "name": "router_middle"},
        ])
    optimizer_cls = bnb.optim.PagedAdam8bit if paged_optimizer else bnb.optim.Adam8bit
    optimizer = optimizer_cls(
        groups,
        lr=EXPERT_LR,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )
    audit = {
        "optimizer": (
            "bitsandbytes.PagedAdam8bit"
            if paged_optimizer
            else "bitsandbytes.Adam8bit"
        ),
        "state_paging_only": bool(paged_optimizer),
        "betas": [0.9, 0.95],
        "weight_decay": 0.0,
        "groups": [
            {
                "name": str(group["name"]),
                "lr": float(group["lr"]),
                "parameter_count": sum(param.numel() for param in group["params"]),
            }
            for group in groups
        ],
    }
    return optimizer, audit


@torch.no_grad()
def evaluate_validation(
    contract: rrd2.Contract,
    student: Any,
    teacher: Any | None,
    on_policy_teacher: OnPolicyTeacherTargets | None,
    loader: Iterable[dict[str, torch.Tensor]],
    activation_map: Any,
    carve_manifest: dict[str, Any],
    teacher_mlp_outs: dict[int, torch.Tensor],
    teacher_post_ln: dict[int, torch.Tensor],
    components: dict[int, dict[str, torch.Tensor]],
    device: torch.device,
    step: int,
) -> dict[str, Any]:
    was_training = student.training
    student.eval()
    totals = {"ce": 0.0, "router": 0.0, "joint": 0.0, "kd": 0.0, "hit": 0.0, "cos": 0.0}
    batches = 0
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= 32:
            break
        ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        mask = batch.get("attention_mask")
        mask = mask.to(device) if mask is not None else None
        targets: dict[int, torch.Tensor] = {}
        teacher_logits: dict[str, torch.Tensor] = {}
        teacher_mlp_outs.clear()
        teacher_post_ln.clear()
        if on_policy_teacher is None:
            if teacher is None:
                raise RuntimeError("teacher-trajectory validation requires a teacher")
            targets = teacher_targets(
                contract,
                teacher,
                ids,
                mask,
                teacher_mlp_outs,
                teacher_post_ln,
                activation_map,
                carve_manifest,
                teacher_logits=teacher_logits if KD_WEIGHT > 0 else None,
            )
        components.clear()
        output = student(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False)
        kd = (
            logit_kd_loss(output.logits, teacher_logits["logits"], labels, mask,
                          temperature=KD_TEMPERATURE, chunk_tokens=KD_CHUNK_TOKENS)
            if KD_WEIGHT > 0 else output.loss.new_zeros(())
        )
        teacher_logits.clear()
        if on_policy_teacher is not None:
            teacher_mlp_outs.update(
                on_policy_teacher.outputs_for_components(components)
            )
        joint = joint_rmse(components, teacher_mlp_outs)
        with TrueSTFRouterPatch(
            student,
            targets,
            teacher_mlp_outs,
            contract.topology.active,
            on_policy_teacher=on_policy_teacher,
        ) as patch:
            student(input_ids=ids, attention_mask=mask, use_cache=False)
            router = torch.stack([patch.losses[layer] for layer in sorted(patch.losses)]).mean()
        totals["ce"] += float(output.loss)
        totals["router"] += float(router)
        totals["joint"] += float(joint)
        totals["kd"] += float(kd)
        totals["hit"] += mean_layer_metric(patch.layer_metrics, "hit_count")
        totals["cos"] += mean_layer_metric(patch.layer_metrics, "cos_forced_dense")
        batches += 1
    if was_training:
        student.train()
    count = max(batches, 1)
    ce = totals["ce"] / count
    router = totals["router"] / count
    joint = totals["joint"] / count
    return {
        "step": step,
        "loss_ce": ce,
        "loss_router": router,
        "loss_joint_rrd": joint,
        "loss_representation": joint,
        "loss_logit_kd": totals["kd"] / count,
        "loss_weights": {
            "ce": CE_WEIGHT,
            "router": ROUTER_WEIGHT,
            "joint_rrd": JOINT_RRD_WEIGHT,
            "logit_kd": KD_WEIGHT,
        },
        "weighted_total": (
            CE_WEIGHT * ce
            + ROUTER_WEIGHT * router
            + JOINT_RRD_WEIGHT * joint
            + KD_WEIGHT * totals["kd"] / count
        ),
        "ppl_proxy": math.exp(min(ce, 20.0)),
        "stf_hit_count": totals["hit"] / count,
        "stf_forced_cos": totals["cos"] / count,
        "validation_batches": batches,
    }


def train(
    contract: rrd2.Contract,
    *,
    device_name: str,
    teacher_device_name: str | None = None,
    smoke_steps: int = 0,
    delta_only: bool = False,
    activation_offload_cpu: bool = False,
    paged_optimizer: bool = False,
) -> None:
    out = run_dir(contract, bool(smoke_steps))
    complete = all((out / name).exists() for name in ("manifest.json", "cpt_log.json", "trainable_delta.pt", ".ready"))
    if complete and not smoke_steps:
        print(f"training already complete: {out}")
        return
    if out.exists() and not smoke_steps:
        allowed = {"researchctl_metadata.json"}
        unexpected = sorted(path.name for path in out.iterdir() if path.name not in allowed)
        if unexpected:
            raise RuntimeError(f"partial output exists; inspect before retrying: {out} unexpected={unexpected}")
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    random.seed(contract.seed)
    torch.manual_seed(contract.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(contract.seed)
    device = torch.device(device_name)
    teacher_device = torch.device(teacher_device_name or device_name)
    memory_devices = {"student": device, "teacher": teacher_device}
    for memory_device in {value for value in memory_devices.values() if value.type == "cuda"}:
        with torch.cuda.device(memory_device):
            torch.cuda.reset_peak_memory_stats()

    # phase120 resolves the teacher through its module-level path.
    import scripts.exp_cmoe.rrd_router_targets as p120

    p120.TEACHER = contract.teacher
    student, _tokenizer, carve_manifest = load_cmoe_model(
        contract.moe_dir,
        device,
        "bfloat16",
        train=True,
    )
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    probe: MLPProbeBank | None = None
    if ROUTER_ARCHITECTURE == "mlp":
        probe = MLPProbeBank(
            contract.n_layers,
            contract.hidden_size,
            PROBE_HIDDEN,
            contract.topology.routed_total,
        ).to(device=device, dtype=torch.bfloat16)
        attach_mlp_probe_router(student, probe, aggregation="uniform")
    counts = set_trainable(student)
    teacher: Any | None = None
    dense_mlp_bank: nn.ModuleList | None = None
    on_policy_teacher: OnPolicyTeacherTargets | None = None
    if TEACHER_TARGET_MODE == "on_policy":
        dense_mlp_bank = load_dense_mlp_bank(
            contract.teacher,
            teacher_device,
            torch.bfloat16,
        )
        teacher_view = types.SimpleNamespace(
            model=types.SimpleNamespace(
                layers=[
                    types.SimpleNamespace(mlp=mlp)
                    for mlp in dense_mlp_bank
                ]
            )
        )
        activation_map = build_activation_mass_neuron_index_cache_from_state_dict(
            teacher_view,
            str(contract.moe_dir / "state_dict.pt"),
            carve_manifest,
            teacher_device,
        )
        on_policy_teacher = OnPolicyTeacherTargets(
            dense_mlp_bank,
            activation_map,
            device,
            teacher_device,
            contract.topology.routed_total,
        )
        del teacher_view
    else:
        teacher = load_teacher(teacher_device, "bfloat16")
        activation_map = build_activation_mass_neuron_index_cache_from_state_dict(
            teacher,
            str(contract.moe_dir / "state_dict.pt"),
            carve_manifest,
            teacher_device,
        )
    student.config.use_cache = False
    student.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    if hasattr(student, "enable_input_require_grads"):
        student.enable_input_require_grads()

    components: dict[int, dict[str, torch.Tensor]] = {}
    wrap_student_for_capture(student, components)
    teacher_mlp_outs: dict[int, torch.Tensor] = {}
    teacher_post_ln: dict[int, torch.Tensor] = {}
    handles = (
        install_teacher_hooks(teacher, teacher_mlp_outs, teacher_post_ln)
        if teacher is not None
        else []
    )
    router_params = named_group(student, "router")
    shared_params = named_group(student, "shared")
    routed_params = named_group(student, "routed")
    expert_params = shared_params + routed_params
    optimizer, optimizer_audit = build_optimizer(
        contract,
        router_params,
        expert_params,
        paged_optimizer=paged_optimizer,
    )

    indices, order = ordered_indices(contract)
    batches = make_batches(contract, indices, smoke_steps)
    total_steps = len(batches)
    if not smoke_steps and total_steps != TOTAL_WINDOWS // BATCH_SIZE:
        raise RuntimeError(f"expected 1024 optimizer steps, got {total_steps}")
    val_loader = validation_loader(contract, smoke_steps)
    train_history: list[dict[str, Any]] = []
    validation_history = [evaluate_validation(
        contract,
        student,
        teacher,
        on_policy_teacher,
        val_loader,
        activation_map,
        carve_manifest,
        teacher_mlp_outs,
        teacher_post_ln,
        components,
        device,
        0,
    )]
    first_audit: dict[str, Any] = {}

    for step, ids_cpu in enumerate(batches, start=1):
        ids = ids_cpu.to(device)
        labels = ids.clone()
        targets: dict[int, torch.Tensor] = {}
        teacher_logits: dict[str, torch.Tensor] = {}
        teacher_mlp_outs.clear()
        teacher_post_ln.clear()
        if on_policy_teacher is None:
            if teacher is None:
                raise RuntimeError("teacher-trajectory training requires a teacher")
            targets = teacher_targets(
                contract,
                teacher,
                ids,
                None,
                teacher_mlp_outs,
                teacher_post_ln,
                activation_map,
                carve_manifest,
                teacher_logits=teacher_logits if KD_WEIGHT > 0 else None,
            )
        optimizer.zero_grad(set_to_none=True)
        components.clear()
        saved_tensor_context = (
            torch.autograd.graph.save_on_cpu(pin_memory=True)
            if activation_offload_cpu
            else contextlib.nullcontext()
        )
        with saved_tensor_context:
            output = student(input_ids=ids, labels=labels, use_cache=False)
        if CE_WEIGHT == 0:
            # Keep CE as a diagnostic while releasing its unused backward graph.
            output.loss = output.loss.detach()
        kd = (
            logit_kd_loss(output.logits, teacher_logits["logits"], labels,
                          temperature=KD_TEMPERATURE, chunk_tokens=KD_CHUNK_TOKENS)
            if KD_WEIGHT > 0 else output.loss.new_zeros(())
        )
        teacher_logits.clear()
        if on_policy_teacher is not None:
            teacher_mlp_outs.update(
                on_policy_teacher.outputs_for_components(components)
            )
        joint = joint_rmse(components, teacher_mlp_outs)
        ce_grads = accumulate_loss_grads(
            output.loss,
            expert_params,
            scale=CE_WEIGHT,
            retain_graph=True,
        ) if CE_WEIGHT > 0 else {"router": 0.0, "routed": 0.0, "shared": 0.0}
        rrd_params = (
            shared_params
            if RRD_GRADIENT_SCOPE == "shared_only"
            else expert_params
        )
        joint_grads = accumulate_loss_grads(
            joint,
            rrd_params,
            scale=JOINT_RRD_WEIGHT,
            retain_graph=KD_WEIGHT > 0,
        )
        kd_grads = accumulate_loss_grads(
            kd, expert_params, scale=KD_WEIGHT, retain_graph=False,
        ) if KD_WEIGHT > 0 else {"router": 0.0, "routed": 0.0, "shared": 0.0}
        weight_errors = [
            float(
                (
                    comp["topk_weights"].float().sum(dim=-1)
                    - contract.topology.active
                ).abs().max().item()
            )
            for comp in components.values()
            if comp.get("topk_weights") is not None
        ]
        log_step = step == 1 or step % 10 == 0 or step == total_steps
        with TrueSTFRouterPatch(
            student,
            targets,
            teacher_mlp_outs,
            contract.topology.active,
            on_policy_teacher=on_policy_teacher,
            record_metrics=on_policy_teacher is None or log_step,
        ) as patch:
            saved_tensor_context = (
                torch.autograd.graph.save_on_cpu(pin_memory=True)
                if activation_offload_cpu
                else contextlib.nullcontext()
            )
            with saved_tensor_context:
                student(input_ids=ids, use_cache=False)
            router = torch.stack([patch.losses[layer] for layer in sorted(patch.losses)]).mean()
            router_grads = accumulate_loss_grads(
                router,
                router_params,
                scale=ROUTER_WEIGHT,
                retain_graph=False,
            )
        loss_ce_value = float(output.loss.detach())
        loss_router_value = float(router.detach())
        loss_joint_value = float(joint.detach())
        loss_kd_value = float(kd.detach())
        weighted_total_value = (
            CE_WEIGHT * loss_ce_value
            + ROUTER_WEIGHT * loss_router_value
            + JOINT_RRD_WEIGHT * loss_joint_value
            + KD_WEIGHT * loss_kd_value
        )
        patch_layer_metrics = dict(patch.layer_metrics)
        del output, joint, kd, router, targets, patch
        components.clear()
        teacher_mlp_outs.clear()
        teacher_post_ln.clear()
        torch.cuda.empty_cache()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        if step == 1:
            first_audit = {
                "ce_grad_norms": ce_grads,
                "joint_rrd_grad_norms": joint_grads,
                "rrd_gradient_scope": RRD_GRADIENT_SCOPE,
                "router_grad_norms": router_grads,
                "logit_kd_grad_norms": kd_grads,
                "ce_weight": CE_WEIGHT,
                "kd_weight": KD_WEIGHT,
                "selected_weight_sum_reference": (
                    contract.topology.active
                    if ROUTER_ARCHITECTURE == "mlp"
                    else "canonical_cmoe"
                ),
                "selected_weight_sum_max_deviation_from_active": max(
                    weight_errors, default=0.0
                ),
            }
        if log_step:
            row = {
                "step": step,
                "loss_ce": loss_ce_value,
                "loss_router": loss_router_value,
                "loss_joint_rrd": loss_joint_value,
                "loss_representation": loss_joint_value,
                "loss_logit_kd": loss_kd_value,
                "weighted_total": weighted_total_value,
                "stf_hit_count": mean_layer_metric(patch_layer_metrics, "hit_count"),
                "stf_forced_cos": mean_layer_metric(patch_layer_metrics, "cos_forced_dense"),
                "optimizer_group_lrs": {
                    str(group.get("name", idx)): float(group["lr"])
                    for idx, group in enumerate(optimizer.param_groups)
                },
            }
            train_history.append(row)
            atomic_json(out / "progress.json", {
                "state": "training",
                "step": step,
                "total_steps": total_steps,
                "row": row,
            })
            print(json.dumps(row), flush=True)
        if step % 100 == 0 or step == total_steps:
            validation_history.append(evaluate_validation(
                contract,
                student,
                teacher,
                on_policy_teacher,
                val_loader,
                activation_map,
                carve_manifest,
                teacher_mlp_outs,
                teacher_post_ln,
                components,
                device,
                step,
            ))
            atomic_json(out / "cpt_log.json", {
                "strategy": strategy_name(),
                "elapsed_sec": time.time() - started,
                "log_history": train_history,
                "validation_history": validation_history,
            })

    for handle in handles:
        handle.remove()
    unwrap_student_capture(student)
    elapsed = time.time() - started
    peak_memory = cuda_peak_memory(memory_devices)
    trainable_names = {
        name for name, parameter in student.named_parameters() if parameter.requires_grad
    }
    state_names = {
        name.replace(".mlp.inner.", ".mlp.") for name in trainable_names
    }
    state = student.state_dict()
    if not smoke_steps:
        if not delta_only:
            torch.save(
                {key: value.detach().cpu() for key, value in state.items()},
                out / "state_dict.pt",
            )
        torch.save(
            {key: state[key].detach().cpu() for key in sorted(state_names)},
            out / "trainable_delta.pt",
        )
    else:
        torch.save({}, out / "trainable_delta.pt")

    active_strategy = strategy_name()
    manifest = {
        **carve_manifest,
        "strategy": active_strategy,
        "strategy_source_run": SOURCE_RUN,
        "phase": active_strategy,
        "run_name": (
            f"{contract.model_key}_{contract.topology.name.lower()}_seed{contract.seed}"
            + ("" if ROUTER_ARCHITECTURE == "mlp" else "_cmoe_router")
            + ("" if RRD_TARGET_FORM == "joint" else "_shared_residual_stopgrad")
            + ("" if RRD_GRADIENT_SCOPE == "target_graph" else "_rrd_shared_only")
            + ("" if TEACHER_TARGET_MODE == "teacher_trajectory" else "_on_policy")
        ),
        "cpt_mode": (
            (
                "one_stage_true_stf_joint_rrd"
                if RRD_TARGET_FORM == "joint"
                else (
                    "one_stage_true_stf_shared_residual_stopgrad_rrd_shared_only"
                    if RRD_GRADIENT_SCOPE == "shared_only"
                    else "one_stage_true_stf_shared_residual_stopgrad"
                )
            )
            + ("" if TEACHER_TARGET_MODE == "teacher_trajectory" else "_on_policy")
        ),
        "cpt_source_moe_dir": str(contract.moe_dir),
        "cpt_teacher_model_path": str(contract.teacher),
        "cpt_calib_path": str(contract.train),
        "cpt_validation_calib_path": str(contract.valtest),
        "cpt_data_order": order["protocol"],
        "cpt_data_order_manifest": str((output_root(contract, bool(smoke_steps)) / "data_order.json").resolve()),
        "cpt_seed": contract.seed,
        "cpt_one_epoch": True,
        "cpt_per_device_bsz": BATCH_SIZE,
        "cpt_grad_accum_steps": 1,
        "cpt_seqlen": SEQLEN,
        "cpt_max_steps": total_steps,
        "cpt_total_sequences": total_steps * BATCH_SIZE,
        "cpt_total_tokens": total_steps * BATCH_SIZE * SEQLEN,
        "cpt_optimizer": optimizer_audit["optimizer"],
        "cpt_optimizer_betas": [0.9, 0.95],
        "cpt_weight_decay": 0.0,
        "cpt_lr_schedule": "constant",
        "cpt_router_base_lr": ROUTER_BASE_LR,
        "cpt_expert_lr": EXPERT_LR,
        "cpt_router_outer_lr": ROUTER_OUTER_LR,
        "cpt_router_early_lr": ROUTER_EARLY_LR,
        "cpt_router_middle_lr": ROUTER_MIDDLE_LR,
        "cpt_router_late_lr": ROUTER_LATE_LR,
        "cpt_lr_policy": LR_POLICY,
        "cpt_lr_rule": LR_RULE,
        "cpt_lr_bounds": [LR_MIN, LR_MAX],
        "cpt_router_early_layers": list(range(0, contract.middle_layers[0])),
        "cpt_router_middle_layers": list(contract.middle_layers),
        "cpt_router_late_layers": list(range(contract.middle_layers[-1] + 1, contract.n_layers)),
        "cpt_optimizer_group_audit": optimizer_audit,
        "cpt_alpha_task": CE_WEIGHT,
        "cpt_alpha_router": ROUTER_WEIGHT,
        "cpt_alpha_residual": JOINT_RRD_WEIGHT,
        "cpt_alpha_kd": KD_WEIGHT,
        "cpt_kd_temperature": KD_TEMPERATURE,
        "cpt_kd_chunk_tokens": KD_CHUNK_TOKENS,
        "cpt_kd_loss_form": "temperature_squared_forward_KL_full_vocab_valid_next_token_mean",
        "cpt_kd_gradient_policy": "shared_and_routed_only",
        "cpt_kd_student_input_source": "ordinary_student_end_to_end",
        "cpt_loss_terminology": "RRD=routing+representation; legacy joint_rrd fields mean representation loss",
        "cpt_ce_gradient_policy": "shared_and_routed_only",
        "cpt_joint_rrd_gradient_policy": (
            "shared_parameters_only"
            if RRD_GRADIENT_SCOPE == "shared_only"
            else (
                "shared_and_routed_parameters"
                if RRD_TARGET_FORM == "joint"
                else "local_routed_output_detached_with_indirect_trajectory_gradients"
            )
        ),
        "cpt_router_gradient_policy": "true_stf_router_only",
        "cpt_rrd_target_form": RRD_TARGET_FORM,
        "cpt_rrd_gradient_scope": RRD_GRADIENT_SCOPE,
        "cpt_teacher_target_mode": TEACHER_TARGET_MODE,
        "cpt_teacher_parameter_scope": (
            "dense_ffn_only"
            if TEACHER_TARGET_MODE == "on_policy"
            else "full_dense_causal_lm"
        ),
        "cpt_teacher_full_model_loaded": TEACHER_TARGET_MODE != "on_policy",
        "cpt_teacher_ffn_parameter_count": (
            int(on_policy_teacher.parameter_count)
            if on_policy_teacher is not None
            else None
        ),
        "cpt_teacher_target_input_source": (
            "student_e2e_trajectory_mlp_input_detached"
            if TEACHER_TARGET_MODE == "on_policy"
            else "dense_teacher_trajectory_post_attention_layernorm"
        ),
        "cpt_shared_target_form": (
            "joint" if RRD_TARGET_FORM == "joint" else "residual_stopgrad_routed"
        ),
        "cpt_residual_loss_form": "fp32_global_rmse",
        "cpt_residual_target": (
            (
                "shared+routed->teacher_dense_mlp_on_student_input"
                if TEACHER_TARGET_MODE == "on_policy"
                else "shared+routed->teacher_dense_mlp_on_teacher_input"
            )
            if RRD_TARGET_FORM == "joint"
            else (
                "shared->stopgrad(teacher_dense_mlp_on_student_input-routed)"
                if TEACHER_TARGET_MODE == "on_policy"
                else "shared->stopgrad(teacher_dense_mlp_on_teacher_input-routed)"
            )
        ),
        "cpt_router_loss_form": "activation_mass_topA_ce",
        "router_contribution_score": "activation_mass_sum",
        "router_target_oracle": (
            "dense_teacher_ffn_on_student_true_stf_input_recovered_cmoe_neuron_mapping"
            if TEACHER_TARGET_MODE == "on_policy"
            else "dense_teacher_trajectory_recovered_cmoe_neuron_mapping"
        ),
        "cpt_router_input_source": "student_true_stf_trajectory_mlp_input",
        "cpt_stf_oracle_aggregation": "uniform_sum",
        "cpt_router_aggregation": aggregation_description(contract),
        "cpt_mlp_router_aggregation": "uniform" if ROUTER_ARCHITECTURE == "mlp" else "",
        "cpt_mlp_router_hidden_size": PROBE_HIDDEN if ROUTER_ARCHITECTURE == "mlp" else 0,
        "cpt_mlp_router_random_init": ROUTER_ARCHITECTURE == "mlp",
        "cpt_router_initialization": (
            "random" if ROUTER_ARCHITECTURE == "mlp" else "canonical_carve"
        ),
        "cpt_router_arch": router_arch_manifest_value(),
        "router_arch": router_arch_manifest_value(),
        "hard_inference_preserved": True,
        "cpt_freeze_attention": True,
        "cpt_freeze_lm_head": True,
        "cpt_cmoe_enable_load_balance": False,
        "cpt_router_quality_metric": f"top{contract.topology.active}_hit_count_average",
        "cpt_exact_set_match_reporting": "disabled",
        "cpt_trainable_parameter_counts": counts,
        "gradient_audit": first_audit,
        "cpt_train_log_every": 10,
        "cpt_validation_every": 100,
        "cpt_validation_batches": 32,
        "cpt_activation_offload_cpu": bool(activation_offload_cpu),
        "cpt_student_device": str(device),
        "cpt_teacher_device": str(teacher_device),
        "cpt_elapsed_sec": elapsed,
        "cpt_gpu_hours": elapsed / 3600.0,
        "cpt_cuda_peak_memory": peak_memory,
        "cpt_trainable_delta_path": str((out / "trainable_delta.pt").resolve()),
        "cpt_skip_full_state_dict": bool(delta_only),
        "checkpoint_status": "smoke" if smoke_steps else "saved",
    }
    atomic_json(out / "manifest.json", manifest)
    atomic_json(out / "cpt_log.json", {
        "strategy": active_strategy,
        "elapsed_sec": elapsed,
        "log_history": train_history,
        "validation_history": validation_history,
    })
    atomic_json(out / "training_timing.json", {
        "elapsed_sec": elapsed,
        "gpu_hours": elapsed / 3600.0,
        "cuda_peak_memory": peak_memory,
    })
    atomic_write_text(out / ".ready", "ready\n")
    atomic_json(out / "progress.json", {
        "state": "trained",
        "step": total_steps,
        "total_steps": total_steps,
    })
    del state, student, teacher, dense_mlp_bank, on_policy_teacher, probe, optimizer
    torch.cuda.empty_cache()


def run_command(command: list[str]) -> None:
    print("$", " ".join(str(value) for value in command), flush=True)
    subprocess.run(
        command,
        cwd=ROOT,
        env={**os.environ, "DISABLE_VERSION_CHECK": "1"},
        check=True,
    )


def configured_lmeval_command(
    contract: rrd2.Contract,
    out: Path,
    device_name: str,
    tasks: str,
    output_name: str,
) -> list[str]:
    if Path(output_name).name != output_name or not output_name.endswith(".json"):
        raise ValueError("--lmeval-output-name must be a plain .json filename")
    task_list = ",".join(part.strip() for part in tasks.split(",") if part.strip())
    if not task_list:
        raise ValueError("--lmeval-tasks must include at least one task")
    command = rrd2.lmeval_command(contract, out, device_name)
    command[command.index("--tasks") + 1] = task_list
    command[command.index("--output_json") + 1] = str(out / output_name)
    return command


def evaluate(
    contract: rrd2.Contract,
    *,
    device_name: str,
    force: bool = False,
    skip_alignment: bool = False,
    lmeval_tasks: str = EVAL_TASKS,
    lmeval_output_name: str = "lmeval.json",
) -> None:
    out = run_dir(contract)
    if not (out / "manifest.json").exists():
        raise FileNotFoundError(f"training is incomplete: {out}")
    if not (out / "state_dict.pt").exists() and not (out / "trainable_delta.pt").exists():
        raise FileNotFoundError(f"missing checkpoint artifacts: {out}")
    started = time.time()
    evaluation_success = False
    try:
        for kind in ("wt2_train", "wt2_valtest", "c4"):
            output = out / f"ppl_{kind}.json"
            if force or not output.exists():
                run_command(rrd2.ppl_command(contract, out, kind, device_name))
            write_leaderboard()
        if not skip_alignment:
            for kind in ("wt2_train", "wt2_valtest", "c4"):
                output = out / f"alignment_{kind}.json"
                if force or not output.exists():
                    run_command(rrd2.alignment_command(contract, out, kind, device_name))
                write_leaderboard()
        lmeval_output = out / lmeval_output_name
        if force or not lmeval_output.exists():
            run_command(configured_lmeval_command(
                contract, out, device_name, lmeval_tasks, lmeval_output_name
            ))
        evaluation_success = True
    finally:
        elapsed = time.time() - started
        atomic_json(out / "core_eval_timing.json", {
            "elapsed_sec": elapsed,
            "gpu_hours": elapsed / 3600.0,
        })
        atomic_json(out / "progress.json", {
            "state": "complete" if evaluation_success else "evaluation_failed",
        })
        write_leaderboard()


def strategy_contract_path(contract: rrd2.Contract, smoke: bool = False) -> Path:
    return output_root(contract, smoke) / "rrd_1_stage_2607_contract.json"


def prepare(contract: rrd2.Contract, smoke: bool = False) -> dict[str, Any]:
    audit = strategy_audit(contract)
    _indices, order = ordered_indices(contract)
    root = output_root(contract, smoke)
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "data_order.json", order)
    payload = {**audit, "smoke_steps": 4 if smoke else 0}
    atomic_json(strategy_contract_path(contract, smoke), payload)
    return payload


def metrics_for(out: Path) -> dict[str, float]:
    return rrd2.metrics_for(out)


def format_metric(value: Any, digits: int = 4) -> str:
    return "-" if not isinstance(value, (int, float)) else f"{float(value):.{digits}f}"


def leaderboard_path() -> Path:
    report = (
        "rrd_1_stage_2607"
        if ROUTER_ARCHITECTURE == "mlp"
        else "rrd_1_stage_2607_cmoe_router"
    )
    if RRD_TARGET_FORM != "joint":
        report += f"_{RRD_TARGET_FORM}"
    if RRD_GRADIENT_SCOPE != "target_graph":
        report += f"_{RRD_GRADIENT_SCOPE}"
    if TEACHER_TARGET_MODE != "teacher_trajectory":
        report += f"_{TEACHER_TARGET_MODE}"
    if KD_WEIGHT > 0:
        report += "_logit_kd"
    return ROOT / "outputs/reports" / report / "LEADERBOARD.md"


def write_leaderboard() -> Path:
    path = leaderboard_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.parent / ".leaderboard.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        rows = []
        for contract_path in sorted((ROOT / "saves").glob("**/rrd_1_stage_2607_contract.json")):
            payload = load_json(contract_path)
            if payload.get("strategy") != strategy_name() or int(payload.get("smoke_steps", 0)):
                continue
            if str(payload.get("router_architecture", "mlp")) != ROUTER_ARCHITECTURE:
                continue
            if str(payload.get("rrd_target_form", "joint")) != RRD_TARGET_FORM:
                continue
            if str(payload.get("rrd_gradient_scope", "target_graph")) != RRD_GRADIENT_SCOPE:
                continue
            if str(payload.get("teacher_target_mode", "teacher_trajectory")) != TEACHER_TARGET_MODE:
                continue
            out = contract_path.parent / "one_stage_e2e"
            metrics = metrics_for(out)
            if set(METRIC_DIRECTIONS).issubset(metrics):
                status = "completed"
            elif (out / "manifest.json").exists():
                status = "evaluating" if metrics else "trained"
            else:
                status = "planned"
            rows.append((payload, out, metrics, status))
        lines = [
            f"# {strategy_name()} Leaderboard",
            "",
            f"Updated: `{time.strftime('%Y-%m-%d %H:%M:%S %Z')}`",
            "",
            "Router quality uses top-A hit-count. Exact-set reporting is disabled.",
            "",
            "| Model | Topology | Seed | Status | WT2 train PPL | WT2 val PPL | C4 PPL | WT2 train hit | WT2 val hit | C4 hit | WT2 train cos | WT2 val cos | C4 cos | PIQA | WinoGrande | ARC-E | ARC-C | HellaSwag | Avg5 | Artifact |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for payload, out, metrics, status in rows:
            values = [
                payload.get("model_key", "?"),
                payload.get("topology", "?"),
                payload.get("seed", "?"),
                status,
                format_metric(metrics.get("ppl_wt2_train")),
                format_metric(metrics.get("ppl_wt2_valtest")),
                format_metric(metrics.get("ppl_c4")),
                format_metric(metrics.get("hit_wt2_train")),
                format_metric(metrics.get("hit_wt2_valtest")),
                format_metric(metrics.get("hit_c4")),
                format_metric(metrics.get("cos_wt2_train")),
                format_metric(metrics.get("cos_wt2_valtest")),
                format_metric(metrics.get("cos_c4")),
                format_metric(metrics.get("lmeval_piqa"), 2),
                format_metric(metrics.get("lmeval_winogrande"), 2),
                format_metric(metrics.get("lmeval_arc_easy"), 2),
                format_metric(metrics.get("lmeval_arc_challenge"), 2),
                format_metric(metrics.get("lmeval_hellaswag"), 2),
                format_metric(metrics.get("lmeval_avg"), 2),
                f"`{out / 'manifest.json'}`",
            ]
            lines.append("| " + " | ".join(str(value) for value in values) + " |")
        atomic_write_text(path, "\n".join(lines) + "\n")
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return path


def common_cli(args: argparse.Namespace) -> list[str]:
    command = [
        "--model", args.model,
        "--topology", args.topology,
        "--server", args.server,
        "--seed", str(args.seed),
        "--device", args.device,
        "--kd-weight", str(args.kd_weight),
        "--kd-temperature", str(args.kd_temperature),
        "--kd-chunk-tokens", str(args.kd_chunk_tokens),
    ]
    for flag, attr in (
        ("--teacher", "teacher"),
        ("--moe-dir", "moe_dir"),
        ("--train", "train"),
        ("--valtest", "valtest"),
        ("--data-manifest", "data_manifest"),
        ("--output-root", "output_root"),
        ("--teacher-device", "teacher_device"),
    ):
        value = getattr(args, attr)
        if value:
            command.extend([flag, str(value)])
    if args.delta_only:
        command.append("--delta-only")
    if args.skip_alignment:
        command.append("--skip-alignment")
    if args.activation_offload_cpu:
        command.append("--activation-offload-cpu")
    if args.paged_optimizer:
        command.append("--paged-optimizer")
    command.extend([
        "--router-base-lr", str(args.router_base_lr),
        "--router-architecture", str(args.router_architecture),
        "--rrd-target-form", str(args.rrd_target_form),
        "--rrd-gradient-scope", str(args.rrd_gradient_scope),
        "--teacher-target-mode", str(args.teacher_target_mode),
        "--ce-weight", str(args.ce_weight),
        "--router-weight", str(args.router_weight),
        "--joint-rrd-weight", str(args.joint_rrd_weight),
    ])
    if args.expert_lr_override is not None:
        command.extend(["--expert-lr-override", str(args.expert_lr_override)])
    if args.uniform_router_lr is not None:
        command.extend(["--uniform-router-lr", str(args.uniform_router_lr)])
    for flag, attr in (
        ("--router-early-lr", "router_early_lr"),
        ("--router-middle-lr", "router_middle_lr"),
        ("--router-late-lr", "router_late_lr"),
    ):
        value = getattr(args, attr)
        if value is not None:
            command.extend([flag, str(value)])
    command.extend([
        "--lmeval-tasks", str(args.lmeval_tasks),
        "--lmeval-output-name", str(args.lmeval_output_name),
    ])
    return command


def orchestrate(args: argparse.Namespace) -> None:
    common = common_cli(args)
    for command in ("train", "evaluate"):
        child = [str(PYTHON), str(Path(__file__).resolve()), command, *common]
        run_command(child)


def add_common(parser: argparse.ArgumentParser) -> None:
    rrd2.add_common(parser)
    parser.add_argument(
        "--router-base-lr",
        type=float,
        default=DEFAULT_ROUTER_BASE_LR,
        help="Outer-router LR; expert=clip(0.06x), middle=clip(2x).",
    )
    parser.add_argument(
        "--expert-lr-override",
        type=float,
        default=None,
        help="Explicit shared/routed expert LR; requires --uniform-router-lr.",
    )
    parser.add_argument(
        "--uniform-router-lr",
        type=float,
        default=None,
        help="Use one LR for every MLP-router layer; requires --expert-lr-override.",
    )
    parser.add_argument(
        "--router-early-lr",
        type=float,
        default=None,
        help="Explicit early-layer router LR; requires all three band LRs and expert override.",
    )
    parser.add_argument(
        "--router-middle-lr",
        type=float,
        default=None,
        help="Explicit middle-layer router LR; requires all three band LRs and expert override.",
    )
    parser.add_argument(
        "--router-late-lr",
        type=float,
        default=None,
        help="Explicit late-layer router LR; requires all three band LRs and expert override.",
    )
    parser.add_argument(
        "--router-architecture",
        choices=("mlp", "cmoe"),
        default="mlp",
        help="Use the best-method MLP router or the canonical carved CMoE router.",
    )
    parser.add_argument(
        "--rrd-target-form",
        choices=("joint", "shared_residual_stopgrad"),
        default="joint",
        help=(
            "Use joint RRD or the Option-A shared residual with routed_out detached; "
            "both retain the FP32 global-RMSE reduction."
        ),
    )
    parser.add_argument(
        "--rrd-gradient-scope",
        choices=("target_graph", "shared_only"),
        default="target_graph",
        help=(
            "Apply RRD gradients to every expert parameter reached by the target graph, "
            "or strictly to shared-expert parameters only. shared_only requires "
            "--rrd-target-form shared_residual_stopgrad."
        ),
    )
    parser.add_argument(
        "--teacher-target-mode",
        choices=("teacher_trajectory", "on_policy"),
        default="teacher_trajectory",
        help=(
            "Use the canonical full-teacher trajectory, or evaluate an FFN-only "
            "dense teacher on detached student MLP inputs. on_policy is a method "
            "ablation, not a byte-equivalent canonical optimization."
        ),
    )
    parser.add_argument(
        "--ce-weight",
        type=float,
        default=1.0,
        help="Causal language-modeling loss weight.",
    )
    parser.add_argument(
        "--router-weight",
        type=float,
        default=1.0,
        help="True-STF activation-mass router CE weight.",
    )
    parser.add_argument(
        "--joint-rrd-weight",
        type=float,
        default=2.0,
        help="Representation RMSE weight (legacy option name; RRD includes routing too).",
    )
    parser.add_argument("--kd-weight", type=float, default=0.0, help="Final-logit forward-KL weight; zero preserves canonical training.")
    parser.add_argument("--kd-temperature", type=float, default=1.0)
    parser.add_argument("--kd-chunk-tokens", type=int, default=128)
    parser.add_argument(
        "--lmeval-tasks",
        default=EVAL_TASKS,
        help="Comma-separated lm-eval task list; defaults to the canonical five tasks.",
    )
    parser.add_argument(
        "--lmeval-output-name",
        default="lmeval.json",
        help="Run-directory JSON filename for lm-eval output.",
    )
    parser.add_argument(
        "--skip-alignment",
        action="store_true",
        help="Evaluate only PPL and lm-eval; omit router alignment metrics.",
    )
    parser.add_argument(
        "--teacher-device",
        default=None,
        help="Optional separate device for the frozen dense teacher.",
    )
    parser.add_argument(
        "--activation-offload-cpu",
        action="store_true",
        help="Offload autograd-saved student activations to CPU during training.",
    )
    parser.add_argument(
        "--paged-optimizer",
        action="store_true",
        help="Use Adam8bit with UVM-paged optimizer-state storage.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in (
        "describe",
        "audit",
        "train",
        "evaluate",
        "run",
        "run-and-evaluate",
        "smoke",
        "leaderboard",
    ):
        child = sub.add_parser(name)
        add_common(child)
        child.add_argument("--force-eval", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_router_architecture(args.router_architecture)
    configure_rrd_target_form(args.rrd_target_form)
    configure_rrd_gradient_scope(args.rrd_gradient_scope)
    configure_teacher_target_mode(args.teacher_target_mode)
    configure_logit_kd(args.kd_weight, args.kd_temperature, args.kd_chunk_tokens)
    if KD_WEIGHT > 0 and not args.output_root:
        raise ValueError("logit KD requires an explicit new --output-root")
    if RRD_GRADIENT_SCOPE == "shared_only" and RRD_TARGET_FORM != "shared_residual_stopgrad":
        raise ValueError(
            "--rrd-gradient-scope shared_only requires "
            "--rrd-target-form shared_residual_stopgrad"
        )
    configure_loss_weights(
        args.ce_weight,
        args.router_weight,
        args.joint_rrd_weight,
    )
    configure_learning_rates(
        args.router_base_lr,
        expert_lr_override=args.expert_lr_override,
        uniform_router_lr=args.uniform_router_lr,
        router_early_lr=args.router_early_lr,
        router_middle_lr=args.router_middle_lr,
        router_late_lr=args.router_late_lr,
    )
    if args.command == "leaderboard":
        print(write_leaderboard())
        return
    contract = rrd2.resolve_contract(args)
    if not args.output_root:
        contract = rrd2.Contract(
            **{
                **contract.__dict__,
                "output_root": (
                    ROOT
                    / (
                        (
                            "saves/rrd_1_stage_2607_on_policy"
                            if TEACHER_TARGET_MODE == "on_policy"
                            else "saves/rrd_1_stage_2607"
                        )
                        if ROUTER_ARCHITECTURE == "mlp"
                        else (
                            "saves/rrd_1_stage_2607_cmoe_router_on_policy"
                            if TEACHER_TARGET_MODE == "on_policy"
                            else "saves/rrd_1_stage_2607_cmoe_router"
                        )
                    )
                    / f"{contract.model_key}_{contract.topology.name.lower()}_seed{contract.seed}"
                ).resolve(),
            }
        )
    if args.command in {"describe", "audit"}:
        payload = strategy_audit(contract)
        if args.command == "audit":
            _indices, order = ordered_indices(contract)
            payload["data_order"] = {
                key: value for key, value in order.items() if key != "indices"
            }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    if args.command == "smoke":
        prepare(contract, smoke=True)
        train(
            contract,
            device_name=args.device,
            teacher_device_name=args.teacher_device,
            smoke_steps=4,
            delta_only=True,
            activation_offload_cpu=bool(args.activation_offload_cpu),
            paged_optimizer=bool(args.paged_optimizer),
        )
        manifest = load_json(run_dir(contract, True) / "manifest.json")
        audit = manifest.get("gradient_audit", {})
        rrd_grads = audit.get("joint_rrd_grad_norms", {})
        routed_rrd_ok = (
            float(rrd_grads.get("routed", 0.0)) == 0.0
            if RRD_GRADIENT_SCOPE == "shared_only"
            else (
                float(rrd_grads.get("routed", 0.0)) > 0.0
                if RRD_TARGET_FORM == "joint"
                else True
            )
        )
        if not (
            float(audit.get("ce_grad_norms", {}).get("router", 0.0)) == 0.0
            and float(rrd_grads.get("router", 0.0)) == 0.0
            and float(rrd_grads.get("shared", 0.0)) > 0.0
            and routed_rrd_ok
            and float(audit.get("router_grad_norms", {}).get("router", 0.0)) > 0.0
        ):
            raise RuntimeError(f"gradient smoke failed: {audit}")
        if KD_WEIGHT > 0:
            kd_grads = audit["logit_kd_grad_norms"]
            if not (kd_grads["router"] == 0 and kd_grads["shared"] > 0 and kd_grads["routed"] > 0):
                raise RuntimeError(f"KD gradient smoke failed: {audit}")
        if CE_WEIGHT == 0 and any(audit["ce_grad_norms"].values()):
            raise RuntimeError(f"CE replacement still received CE gradients: {audit}")
        print(json.dumps({"status": "passed", "audit": audit}, indent=2))
        return
    prepare(contract)
    if args.command in {"train", "run"}:
        train(
            contract,
            device_name=args.device,
            teacher_device_name=args.teacher_device,
            delta_only=bool(args.delta_only),
            activation_offload_cpu=bool(args.activation_offload_cpu),
            paged_optimizer=bool(args.paged_optimizer),
        )
        write_leaderboard()
    elif args.command == "evaluate":
        evaluate(
            contract,
            device_name=args.device,
            force=bool(args.force_eval),
            skip_alignment=bool(args.skip_alignment),
            lmeval_tasks=args.lmeval_tasks,
            lmeval_output_name=args.lmeval_output_name,
        )
    elif args.command == "run-and-evaluate":
        orchestrate(args)


if __name__ == "__main__":
    main()
