#!/usr/bin/env python3
"""Utilities for attaching Phase126-style MLP probe routers to CMoE models."""

from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn


class MLPProbeBank(nn.Module):
    """Per-layer MLP router heads: hidden -> probe_hidden -> routed experts."""

    def __init__(self, n_layers: int, hidden_size: int, probe_hidden: int, n_experts: int) -> None:
        super().__init__()
        self.n_layers = int(n_layers)
        self.hidden_size = int(hidden_size)
        self.probe_hidden = int(probe_hidden)
        self.n_experts = int(n_experts)
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, probe_hidden, bias=True),
                    nn.GELU(),
                    nn.Linear(probe_hidden, n_experts, bias=True),
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        head = self.heads[int(layer_idx)]
        dtype = next(head.parameters()).dtype
        return head(x.to(dtype=dtype))


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_probe_from_manifest(
    manifest: Dict[str, Any],
    model: Any,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> MLPProbeBank:
    layers = model.model.layers
    n_layers = len(layers)
    hidden_size = int(layers[0].mlp.dim)
    probe_hidden = int(manifest.get("cpt_mlp_router_hidden_size", 1024))
    n_routed = int(manifest.get("nexperts", manifest.get("n_experts", 8))) - int(manifest.get("nshared", 2))
    if "n_routed_experts" in manifest:
        n_routed = int(manifest["n_routed_experts"])
    probe = MLPProbeBank(n_layers, hidden_size, probe_hidden, n_routed)
    if dtype is not None:
        probe.to(dtype=dtype)
    if device is not None:
        probe.to(device)
    return probe


def build_probe_from_dir(
    probe_dir: str | Path,
    model: Any,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[MLPProbeBank, Dict[str, Any]]:
    """Build a Phase126 MLP probe bank from ``probe_state_dict.pt``."""

    root = Path(probe_dir)
    manifest = _load_json(root / "manifest.json")
    head_kind = str(manifest.get("probe_head_kind"))
    if head_kind != "mlp_probe":
        raise ValueError(f"{root}: expected probe_head_kind='mlp_probe', got {head_kind!r}")
    probe_hidden = int(manifest.get("probe_hidden_size", 0))
    if probe_hidden <= 0:
        raise ValueError(f"{root}: invalid probe_hidden_size={probe_hidden}")

    layers = model.model.layers
    n_layers = len(layers)
    hidden_size = int(layers[0].mlp.dim)
    n_routed = int(manifest.get("routed_expert_domain", manifest.get("n_routed_experts", 6)))
    probe = MLPProbeBank(n_layers, hidden_size, probe_hidden, n_routed)
    state_path = root / "probe_state_dict.pt"
    if not state_path.exists():
        raise FileNotFoundError(state_path)
    state = torch.load(state_path, map_location="cpu")
    probe.load_state_dict(state, strict=True)
    if dtype is not None:
        probe.to(dtype=dtype)
    if device is not None:
        probe.to(device)
    return probe, manifest


def is_mlp_router_arch(value: Any) -> bool:
    """Return True for Phase126+ MLP probe router manifest values."""

    text = str(value or "")
    return text == "mlp_probe" or text.startswith("mlp_h") or text.startswith("hybrid_mlp_h")


def parse_layer_mask(value: Any) -> Optional[Set[int]]:
    """Parse a comma/range layer mask such as ``9-18`` or ``0,3,9-18``."""

    text = str(value or "").strip()
    if not text:
        return None
    out: Set[int] = set()
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


def _compute_selected_proxy(
    mlp: nn.Module,
    x_flat: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    y = torch.zeros_like(x_flat)
    counts = torch.bincount(indices.flatten(), minlength=int(mlp.n_routed_experts)).tolist()
    for expert_idx in range(int(mlp.experts_start_idx), int(mlp.experts_end_idx)):
        if counts[expert_idx] == 0:
            continue
        token_idx, top_pos = torch.where(indices == expert_idx)
        expert_out = mlp.experts[expert_idx](x_flat[token_idx])
        y[token_idx] += expert_out * weights[token_idx, top_pos, None].to(expert_out.dtype)
    return y


def attach_mlp_probe_router(
    model: Any,
    probe: MLPProbeBank,
    *,
    aggregation: str = "uniform",
    layer_mask: Any = None,
    module_name: str = "phase127_mlp_router",
) -> None:
    """Patch CMoE MLP forwards so top-k selection comes from ``probe``.

    The main inference path remains hard top-k.  ``aggregation='uniform'`` uses
    weight 1 for every selected expert, matching the Phase126 PPL winner.
    ``selected_topk_renorm`` is still exposed as a train-time auxiliary proxy
    when the caller requests ``router_train_routing_mode='selected_topk_renorm'``.
    """

    if aggregation not in {"uniform", "probe_renorm"}:
        raise ValueError(f"unknown MLP router aggregation={aggregation!r}")
    if hasattr(model, module_name):
        raise ValueError(f"model already has module {module_name!r}")
    parsed_layer_mask = parse_layer_mask(layer_mask)
    model.add_module(module_name, probe)
    probe_bank = getattr(model, module_name)

    for layer_idx, layer in enumerate(model.model.layers):
        if parsed_layer_mask is not None and layer_idx not in parsed_layer_mask:
            continue
        mlp = layer.mlp
        original_forward = mlp.forward

        def make_forward(idx: int, orig: Any):
            def forward(patched_mlp: nn.Module, x: torch.Tensor, return_components: bool = False, **kwargs: Any) -> Any:
                shape = x.size()
                x_flat = x.reshape(-1, patched_mlp.dim)
                logits = probe_bank(idx, x_flat)
                scores = logits.softmax(dim=-1, dtype=torch.float32)
                topk = int(patched_mlp.n_activated_experts)
                selected_scores, indices = scores.topk(topk, dim=-1)
                if aggregation == "uniform":
                    weights = torch.ones(indices.shape, dtype=x_flat.dtype, device=x_flat.device)
                else:
                    weights = selected_scores / selected_scores.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                    weights = weights.to(dtype=x_flat.dtype)

                requested_mode = str(kwargs.pop("router_train_routing_mode", "hard_topk"))
                kwargs.pop("override_topk_indices", None)
                kwargs.pop("override_topk_weights", None)
                result = orig(
                    x,
                    return_components=return_components,
                    override_topk_indices=indices,
                    override_topk_weights=weights,
                    router_train_routing_mode="hard_topk",
                    **kwargs,
                )
                if not return_components:
                    return result

                result["router_logits"] = logits
                result["router_scores"] = scores
                result["topk_indices"] = indices
                result["topk_weights"] = weights
                result["mlp_router_aggregation"] = aggregation
                if requested_mode == "selected_topk_renorm":
                    renorm = selected_scores / selected_scores.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                    renorm = renorm.to(dtype=x_flat.dtype)
                    result["selected_topk_renorm_weights"] = renorm
                    result["routed_out_train_proxy"] = _compute_selected_proxy(
                        patched_mlp,
                        x_flat,
                        indices,
                        renorm,
                    ).view(shape)
                elif requested_mode != "hard_topk":
                    raise ValueError(
                        "MLP probe router currently supports hard_topk and "
                        "selected_topk_renorm train proxy only, got "
                        f"{requested_mode!r}"
                    )
                return result

            return forward

        mlp.forward = types.MethodType(make_forward(layer_idx, original_forward), mlp)


def mlp_router_manifest_fields(
    *,
    probe_dir: str | Path,
    probe_manifest: Dict[str, Any],
    aggregation: str,
    layer_mask: Any = None,
    router_arch: str | None = None,
) -> Dict[str, Any]:
    hidden = int(probe_manifest.get("probe_hidden_size", 1024))
    arch = str(router_arch or f"mlp_h{hidden}")
    parsed_layer_mask = parse_layer_mask(layer_mask)
    return {
        "cpt_router_arch": arch,
        "router_arch": arch,
        "cpt_mlp_router_probe_dir": str(Path(probe_dir).resolve()),
        "cpt_mlp_router_source_run": Path(probe_dir).name,
        "cpt_mlp_router_hidden_size": hidden,
        "cpt_mlp_router_source_input": probe_manifest.get("input_source"),
        "cpt_mlp_router_source_target": probe_manifest.get("router_target", "activation_mass_sum"),
        "cpt_mlp_router_source_train_hit": probe_manifest.get("final_train_hit_count"),
        "cpt_mlp_router_source_val_hit": probe_manifest.get("final_val_hit_count"),
        "cpt_mlp_router_aggregation": aggregation,
        "cpt_mlp_router_layer_mask": str(layer_mask or ""),
        "cpt_mlp_router_parsed_layer_mask": sorted(parsed_layer_mask) if parsed_layer_mask is not None else [],
        "cpt_mlp_router_hybrid_default_router_outside_mask": bool(parsed_layer_mask is not None),
        "cpt_mlp_router_parameter_count": None,
    }
