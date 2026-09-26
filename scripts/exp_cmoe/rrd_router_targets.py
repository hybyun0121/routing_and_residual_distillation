from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from scripts.exp_cmoe.cpt_train import (
    activation_mass_targets_from_dense_mlp,
    activation_rank_targets_from_dense_mlp,
    best_subset_topk_from_routed_stack,
    dense_vector_targets_from_dense_mlp,
    marginal_gain_targets_from_routed_stack,
)
from scripts.exp_cmoe.lmeval_cmoe import (
    build_dtype,
    build_moe_skeleton_model,
    load_manifest,
    load_state_dict_into_model,
    resolve_moe_type,
)
from llamafactory.model.cmoe_moe_llama import freeze_extra_bias_and_scale

TEACHER = Path("models/Qwen2.5-7B")

def load_jsonl_token_stream(jsonl_path: Path, tokenizer: Any) -> torch.Tensor:
    eos = tokenizer.eos_token_id
    ids: List[int] = []
    row_kind: Optional[str] = None
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            token_ids = row.get("input_ids")
            text = row.get("text")
            current_kind = "token_ids" if isinstance(token_ids, list) else "text"
            if row_kind is None:
                row_kind = current_kind
            elif row_kind != current_kind:
                raise ValueError(f"{jsonl_path}:{line_no} mixes text and input_ids rows")
            if current_kind == "token_ids":
                if not token_ids:
                    raise ValueError(f"{jsonl_path}:{line_no} has empty input_ids")
                ids.extend(int(x) for x in token_ids)
            else:
                if not isinstance(text, str):
                    raise ValueError(
                        f"{jsonl_path}:{line_no} requires string 'text' or list 'input_ids'"
                    )
                ids.extend(tokenizer.encode(text, add_special_tokens=False))
                if eos is not None:
                    ids.append(int(eos))
    if not ids:
        raise ValueError(f"{jsonl_path} produced no tokens")
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


def iter_windows(input_ids: torch.Tensor, seqlen: int, max_windows: Optional[int]) -> Iterable[torch.Tensor]:
    n = input_ids.numel() // seqlen
    if max_windows is not None:
        n = min(n, int(max_windows))
    for i in range(n):
        yield input_ids[:, i * seqlen: (i + 1) * seqlen]


def iter_window_batches(
    input_ids: torch.Tensor,
    seqlen: int,
    batch_size: int,
    max_steps: Optional[int],
) -> Iterable[torch.Tensor]:
    n = input_ids.numel() // seqlen
    if max_steps is not None:
        n = min(n, int(max_steps) * int(batch_size))
    n = (n // int(batch_size)) * int(batch_size)
    for offset in range(0, n, int(batch_size)):
        rows = []
        for j in range(int(batch_size)):
            i = offset + j
            rows.append(input_ids[:, i * seqlen: (i + 1) * seqlen].squeeze(0))
        yield torch.stack(rows, dim=0)


def normalize_manifest(raw: Dict[str, Any]) -> Dict[str, Any]:
    n_total = int(raw.get("nexperts", raw.get("n_experts", 8)))
    n_shared = int(raw.get("nshared", 0))
    n_active = int(raw.get("nactivated", raw.get("n_active", 2)))
    return {
        **raw,
        "moe_type": "cmoe",
        "n_experts": n_total,
        "n_active": n_active,
        "n_routed_experts": n_total - n_shared,
        "has_shared": n_shared > 0,
    }


def load_cmoe_model(model_dir: Path, device: torch.device, dtype_name: str, *, train: bool) -> Tuple[Any, Any, Dict[str, Any]]:
    manifest_path = model_dir / "manifest.json"
    state_path = model_dir / "state_dict.pt"
    manifest_raw = load_manifest(str(manifest_path), moe_type="cmoe")
    manifest = normalize_manifest(manifest_raw)
    dtype = build_dtype(dtype_name)
    model, tok = build_moe_skeleton_model(
        base_model_path=str(TEACHER),
        manifest=manifest_raw,
        dtype=dtype,
        moe_type=resolve_moe_type(manifest_raw, "cmoe"),
    )
    load_state_dict_into_model(model, str(state_path), strict=True)
    for layer in model.model.layers:
        freeze_extra_bias_and_scale(layer.mlp, extra_scale_trainable=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.to(device)
    model.train(bool(train))
    model.config.use_cache = False
    return model, tok, manifest


def load_teacher(device: torch.device, dtype_name: str) -> Any:
    from transformers import AutoModelForCausalLM

    teacher = AutoModelForCausalLM.from_pretrained(str(TEACHER), torch_dtype=build_dtype(dtype_name), low_cpu_mem_usage=True)
    teacher.config.use_cache = False
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.to(device).eval()
    return teacher


def routed_modules(mlp: nn.Module) -> List[nn.Module]:
    return [
        mlp.experts[i]
        for i in range(int(mlp.experts_start_idx), int(mlp.experts_end_idx))
        if mlp.experts[i] is not None
    ]


def target_labels_for_batch(
    teacher: Any,
    teacher_post_ln: Dict[int, torch.Tensor],
    activation_map: Dict[int, List[torch.Tensor]],
    manifest: Dict[str, Any],
    target_kind: str,
    k: int,
    oracle_student: Optional[Any] = None,
) -> Dict[int, torch.Tensor]:
    labels: Dict[int, torch.Tensor] = {}
    for layer_idx, t_inp in teacher_post_ln.items():
        x_flat = t_inp.reshape(-1, t_inp.shape[-1])
        groups = activation_map[int(layer_idx)]
        if target_kind == "mass":
            idx, _vals, _margin, _full = activation_mass_targets_from_dense_mlp(
                teacher.model.layers[layer_idx].mlp,
                x_flat,
                int(manifest["n_experts"]),
                int(manifest["n_routed_experts"]),
                k,
                expert_neuron_indices=groups,
            )
        elif target_kind == "rank":
            idx, _vals, _margin, _full = activation_rank_targets_from_dense_mlp(
                teacher.model.layers[layer_idx].mlp,
                x_flat,
                int(manifest["n_experts"]),
                int(manifest["n_routed_experts"]),
                k,
                expert_neuron_indices=groups,
            )
        elif target_kind == "dense_vector":
            idx, _vals, _margin, _full = dense_vector_targets_from_dense_mlp(
                teacher.model.layers[layer_idx].mlp,
                x_flat,
                int(manifest["n_routed_experts"]),
                k,
                expert_neuron_indices=groups,
            )
        elif target_kind in {"best_subset", "marginal_gain"}:
            if oracle_student is None:
                raise ValueError(f"target_kind={target_kind!r} requires oracle_student")
            oracle_mlp = oracle_student.model.layers[layer_idx].mlp
            routed = torch.stack([m(x_flat) for m in routed_modules(oracle_mlp)], dim=0).float()
            shared = oracle_mlp.shared_experts(x_flat).float()
            dense = teacher.model.layers[layer_idx].mlp(x_flat).float()
            residual = dense - shared
            if target_kind == "best_subset":
                idx, _vals, _margin = best_subset_topk_from_routed_stack(routed, residual, k)
            else:
                idx, _vals, _margin, _full = marginal_gain_targets_from_routed_stack(routed, residual, k)
        else:
            raise ValueError(f"unknown target_kind={target_kind!r}")
        labels[int(layer_idx)] = idx.detach()
    return labels
