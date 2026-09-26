"""CMoE carved MoE module mirrored for RRD RRD trainer.

Source: third_party/cmoe/CMoE_model.py (lines 1-110).

Mirror policy: forward semantics are preserved exactly. The Router/MoE classes
are extended with optional diagnostic outputs so that the RRD loss stack
(magnitude-weighted KL on router, RMSE on shared+routed reconstruction) can
attach to the same forward pass without an extra pass over experts.

No external project imports — this module is self-contained so that the
RRD codebase does not depend on the CMoE repository at runtime.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Mirrored modules
# ---------------------------------------------------------------------------


class LlamaMLP(nn.Module):
    """Standard Llama MLP block: down(silu(gate(x)) * up(x)).

    Mirrored from CMoE_model.py:10-29 unchanged.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = F.silu if hidden_act == "silu" else getattr(F, hidden_act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        intermediate = gate * up
        output = self.down_proj(intermediate)
        return output


class Router(nn.Module):
    """CMoE dual-gate router with extra_scale (Parameter) and extra_bias (buffer).

    Forward semantics (default):
        scores       = (classifier(x) * silu(gate(x))).abs()        # raw logits
        softmax_scs  = softmax(scores, dim=-1, dtype=fp32)          # pre-bias dist
        biased       = softmax_scs + extra_bias[None, :]            # routing only
        indices      = topk(biased, k).indices                      # [N, k]
        weighted     = 1 + softmax_scs * extra_scale                # value scaling
        weights      = weighted.gather(1, indices)                  # [N, k]
        return (weights, indices)

    Note: extra_bias is added *only* to the topk decision; the magKL distribution
    must use ``softmax_scs`` (post-softmax, pre-bias).

    Args:
        hidden_size: input feature dim.
        n_experts: number of routed experts (NOT total — caller passes
            ``n_experts - n_shared`` from the parent MoE).
        n_activated: top-K size.
        bias_speed: in-place adjustment magnitude for ``update_bias`` (only
            applied when ``cus_training=True`` on the parent MoE).
    """

    def __init__(
        self,
        hidden_size: int,
        n_experts: int,
        n_activated: int,
        bias_speed: float = 0.001,
    ) -> None:
        super().__init__()
        self.dim = hidden_size
        self.topk = n_activated

        self.act_fn = F.silu
        self.gate = nn.Linear(hidden_size, n_experts, bias=False)
        self.classifier = nn.Linear(hidden_size, n_experts, bias=False)

        # Caller controls dtype/device via .to(); originally torch.bfloat16/cuda.
        self.extra_scale = nn.Parameter(torch.zeros(n_experts))
        # extra_bias persists in state_dict for save/load (load-balancing offset).
        self.register_buffer("extra_bias", torch.zeros(n_experts, dtype=torch.float32))
        self.bias_update_speed = bias_speed

    def update_bias(self, counts: torch.Tensor) -> None:
        """In-place load-balancing nudge. Only invoked when parent
        MoE.cus_training is True."""
        mean_load = counts.mean()
        overloaded = counts > mean_load
        underloaded = counts < mean_load
        self.extra_bias.data[overloaded] -= self.bias_update_speed
        self.extra_bias.data[underloaded] += self.bias_update_speed

    def forward(
        self,
        x: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """Compute routing decision.

        Args:
            x: ``[N, hidden_size]`` flat tokens.
            return_diagnostics: if True, also return the pre-softmax raw_logits
                and the post-softmax (pre-bias) distribution used by the magKL
                router loss.

        Returns:
            Default: ``(weights, indices)``.
            With ``return_diagnostics=True``:
                ``(weights, indices, raw_logits, softmax_scores)`` where
                ``raw_logits`` is the pre-softmax abs-product
                ``(classifier(x) * silu(gate(x))).abs()`` (shape ``[N, n_experts]``)
                and ``softmax_scores`` is the post-softmax fp32 distribution
                pre-bias (shape ``[N, n_experts]``).
        """
        raw_logits = (self.classifier(x) * self.act_fn(self.gate(x))).abs()
        softmax_scores = raw_logits.softmax(dim=-1, dtype=torch.float32)

        # Routing decision uses bias-adjusted scores.
        biased = softmax_scores + self.extra_bias[None, :]
        indices = torch.topk(biased, self.topk, dim=-1)[1]

        # Weight scaling uses bias-free distribution times learnable per-expert
        # scale (anchored at 1.0).
        scaled = 1 + softmax_scores * self.extra_scale
        weights = scaled.gather(1, indices)
        weights = weights.type_as(x)

        if return_diagnostics:
            return weights, indices, raw_logits, softmax_scores
        return weights, indices


class MoE(nn.Module):
    """CMoE S+A MoE block: routed experts (top-K) + a single fat shared expert.

    For S3A3E8: ``n_experts=8``, ``n_shared=3``, ``n_activated=3`` →
    ``n_routed_experts = 5``. The shared expert is a single LlamaMLP with width
    ``n_shared * moe_inter_dim`` (3× the per-expert width).

    Mirrored from CMoE_model.py:71-110 with one optional output mode added.
    """

    def __init__(
        self,
        hidden_size: int,
        moe_inter_dim: int,
        n_experts: int,
        n_shared: int,
        n_activated: int,
        bias_speed: float = 0.001,
        add_eas: bool = False,
    ) -> None:
        super().__init__()
        self.dim = hidden_size
        self.moe_inter_dim = moe_inter_dim
        n_routed_experts = n_experts - n_shared
        self.n_routed_experts = n_routed_experts
        self.n_activated_experts = n_activated
        self.experts_start_idx = 0
        self.experts_end_idx = n_routed_experts
        self.gate = Router(
            hidden_size=hidden_size,
            n_experts=n_routed_experts,
            n_activated=n_activated,
            bias_speed=bias_speed,
        )
        self.n_shared_experts = n_shared
        # Mirror of the original list-comp: in S3A3E8 every i is in
        # [start, end) so no None slots arise. Kept as ModuleList for parity.
        self.experts = nn.ModuleList(
            [
                LlamaMLP(self.dim, moe_inter_dim)
                if self.experts_start_idx <= i < self.experts_end_idx
                else None
                for i in range(self.n_routed_experts)
            ]
        )
        self.shared_experts = LlamaMLP(self.dim, self.n_shared_experts * moe_inter_dim)

        # Extra Additional Shared expert (EAS): always-active extra LlamaMLP of
        # routed-expert width (moe_inter_dim). Trainable, designed to learn the
        # residual `teacher_rep - (shared + routed_top_k)` via L_shared.
        # In the residual stream (main forward path) the EAS contribution is
        # detached so L_task / L_router gradients do NOT flow into EAS — only
        # L_shared computed from the captured `eas_out` updates EAS.
        self.add_eas = bool(add_eas)
        if self.add_eas:
            self.eas = LlamaMLP(self.dim, moe_inter_dim)
        else:
            self.eas = None

        self.cus_training = False
        # Opt-in distributed load-balancing for data-parallel CPT.  When
        # enabled, every rank updates extra_bias from the global token counts
        # so the mutable routing state remains identical across replicas.
        self.sync_load_balance_counts = False
        self.enable_scale = True

    def forward(
        self,
        x: torch.Tensor,
        return_components: bool = False,
        override_topk_indices: Optional[torch.Tensor] = None,
        override_topk_weights: Optional[torch.Tensor] = None,
        router_train_routing_mode: str = "hard_topk",
        router_relax_epsilon: float = 0.05,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass.

        Args:
            x: ``[..., hidden_size]`` (typically ``[B, S, H]``).
            return_components: if True, return a dict with all intermediate
                tensors needed by the RRD loss stack.

        Returns:
            Default: ``[..., hidden_size]`` combined output (routed + shared).
            With ``return_components=True``:
                {
                    "output":         [..., H] combined,
                    "routed_out":     [..., H] routed-expert sum (post weights),
                    "shared_out":     [..., H] shared expert output,
                    "router_logits":  [N, n_routed_experts] pre-softmax,
                    "router_scores":  [N, n_routed_experts] post-softmax (pre-bias),
                    "topk_weights":   [N, topk],
                    "topk_indices":   [N, topk] long,
                }
                where ``N = prod(x.shape[:-1])``.
            router_train_routing_mode: Training-only proxy used only in
                ``return_components`` mode. ``hard_topk`` preserves the normal
                output; ``selected_topk_renorm`` keeps the hard selected experts
                but renormalizes their gate probabilities for the auxiliary
                routed-output proxy; ``soft_full``/``st_topk``/``relaxed_topk``
                expose ``routed_out_train_proxy`` so auxiliary losses can send
                gradient to the router distribution without changing hard
                inference.
        """
        valid_routing_modes = {
            "hard_topk",
            "selected_topk_renorm",
            "soft_full",
            "st_topk",
            "relaxed_topk",
        }
        if router_train_routing_mode not in valid_routing_modes:
            raise ValueError(f"unknown router_train_routing_mode={router_train_routing_mode!r}")
        shape = x.size()
        x_flat = x.view(-1, self.dim)

        weights, indices, raw_logits, softmax_scores = self.gate(
            x_flat, return_diagnostics=True
        )
        if override_topk_indices is not None:
            indices = override_topk_indices.to(indices.device).long()
            if override_topk_weights is not None:
                weights = override_topk_weights.to(weights.device, dtype=weights.dtype)
            else:
                weights = torch.ones(
                    indices.shape,
                    dtype=x_flat.dtype,
                    device=x_flat.device,
                )

        y = torch.zeros_like(x_flat)
        selected_renorm_y = None
        selected_renorm_weights = None
        if return_components and router_train_routing_mode == "selected_topk_renorm":
            selected_scores = softmax_scores.gather(1, indices).float()
            selected_renorm_weights = (
                selected_scores
                / selected_scores.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            ).type_as(x_flat)
            selected_renorm_y = torch.zeros_like(x_flat)
        counts_t = torch.bincount(indices.flatten(), minlength=self.n_routed_experts)
        if self.cus_training:
            counts_for_bias = counts_t
            if (
                self.sync_load_balance_counts
                and torch.distributed.is_available()
                and torch.distributed.is_initialized()
            ):
                counts_for_bias = counts_t.clone()
                torch.distributed.all_reduce(
                    counts_for_bias,
                    op=torch.distributed.ReduceOp.SUM,
                )
            self.gate.update_bias(counts_for_bias.to(dtype=torch.bfloat16))

        counts = counts_t.tolist()
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            expert_out = expert(x_flat[idx])
            if self.enable_scale:
                y[idx] += expert_out * weights[idx, top, None]
            else:
                y[idx] += expert_out
            if selected_renorm_y is not None and selected_renorm_weights is not None:
                selected_renorm_y[idx] += expert_out * selected_renorm_weights[idx, top, None]

        z = self.shared_experts(x_flat)
        if self.add_eas and self.eas is not None:
            eas_out_flat = self.eas(x_flat)
            # Residual stream: EAS contribution is .detach()ed so the LM-head /
            # next-layer graph does not propagate L_task / L_router gradients
            # into EAS. The non-detached `eas_out` is exposed via
            # return_components for the L_shared loss only.
            combined_flat = y + z + eas_out_flat.detach()
        else:
            eas_out_flat = None
            combined_flat = y + z

        routed_out_train_proxy = None
        if return_components and router_train_routing_mode == "selected_topk_renorm":
            routed_out_train_proxy = selected_renorm_y
        elif return_components and router_train_routing_mode != "hard_topk":
            all_expert_outs = []
            for i in range(self.experts_start_idx, self.experts_end_idx):
                expert = self.experts[i]
                all_expert_outs.append(expert(x_flat))
            expert_stack = torch.stack(all_expert_outs, dim=1)  # [N, E, H]
            hard_gate = torch.zeros(
                x_flat.shape[0],
                self.n_routed_experts,
                dtype=softmax_scores.dtype,
                device=x_flat.device,
            )
            hard_gate.scatter_add_(1, indices, weights.float())
            if router_train_routing_mode == "soft_full":
                proxy_gate = softmax_scores.float()
            elif router_train_routing_mode == "st_topk":
                proxy_gate = hard_gate + (softmax_scores.float() - softmax_scores.float().detach())
            else:
                eps = float(router_relax_epsilon)
                hard_mask = torch.zeros_like(hard_gate)
                hard_mask.scatter_(1, indices, 1.0)
                proxy_gate = hard_gate + eps * (1.0 - hard_mask) * softmax_scores.float()
            routed_out_train_proxy = (
                expert_stack * proxy_gate.to(expert_stack.dtype).unsqueeze(-1)
            ).sum(dim=1)

        if return_components:
            comp: Dict[str, torch.Tensor] = {
                "output": combined_flat.view(shape),
                "routed_out": y.view(shape),
                "hard_routed_out": y.view(shape),
                "shared_out": z.view(shape),
                "router_logits": raw_logits,
                "router_scores": softmax_scores,
                "topk_weights": weights,
                "topk_indices": indices,
            }
            if routed_out_train_proxy is not None:
                comp["routed_out_train_proxy"] = routed_out_train_proxy.view(shape)
            if selected_renorm_weights is not None:
                comp["selected_topk_renorm_weights"] = selected_renorm_weights
            if eas_out_flat is not None:
                comp["eas_out"] = eas_out_flat.view(shape)
            return comp
        return combined_flat.view(shape)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def freeze_extra_bias_and_scale(
    moe: MoE,
    extra_scale_trainable: bool = False,
) -> None:
    """Reset ``extra_bias`` to 0 and freeze the auto-update path.

    The extra_bias buffer is normally adjusted in-place during CMoE's LoRA
    fine-tuning (``cus_training=True``). For RRD CPT we want it disabled and
    zeroed so it does not re-introduce a routing offset that magKL cannot see.

    Args:
        moe: a CMoE :class:`MoE` instance.
        extra_scale_trainable: if False (default), freeze the per-expert
            ``extra_scale`` parameter so routing weight scaling is fixed at the
            carving-time value. If True, keep it trainable.
    """
    moe.gate.extra_bias.zero_()
    moe.cus_training = False
    moe.gate.extra_scale.requires_grad = bool(extra_scale_trainable)


def swap_llama_mlp_to_cmoe_moe(
    model: nn.Module,
    n_experts: int = 8,
    n_activated: int = 3,
    n_shared: int = 3,
    bias_speed: float = 0.001,
    add_eas: bool = False,
    eas_init_std: float = 0.0,
) -> None:
    """Replace each ``layer.mlp`` on a Llama-style model with a CMoE :class:`MoE`.

    Args:
        model: HuggingFace LlamaForCausalLM (or compatible).
        n_experts: total expert count (routed + shared, e.g. 8).
        n_activated: top-K routed (e.g. 3).
        n_shared: shared expert count (e.g. 2 for E8A3S2 + EAS).
        bias_speed: Router bias_speed.
        add_eas: if True, add an Extra Additional Shared expert (EAS) per layer
            with ``intermediate_size = moe_inter_dim``. EAS is always-active and
            its contribution to the residual stream is .detach()ed inside
            forward — only L_shared (which uses the captured ``eas_out``)
            propagates gradients into EAS.
        eas_init_std: if >0, initialise all EAS LlamaMLP weights (gate/up/down)
            to N(0, std) instead of LlamaMLP's default (Kaiming) init. Use a
            small std (e.g. 1e-3) to keep EAS contribution near-zero at step 0
            while preserving a gradient path through every weight (mirrors the
            ``shared_zero_jitter_std`` pattern for RRD-MoE shared experts).
    """
    layers = model.model.layers if hasattr(model, "model") else model.layers
    hidden_size = layers[0].mlp.gate_proj.weight.shape[1]
    base_intermediate = layers[0].mlp.gate_proj.weight.shape[0]
    if base_intermediate % n_experts != 0:
        raise ValueError(
            f"intermediate_size {base_intermediate} not divisible by n_experts {n_experts}"
        )
    moe_inter_dim = base_intermediate // n_experts

    for layer in layers:
        old_mlp = layer.mlp
        ref_param = next(old_mlp.parameters())
        dtype = ref_param.dtype
        device = ref_param.device

        new_moe = MoE(
            hidden_size=hidden_size,
            moe_inter_dim=moe_inter_dim,
            n_experts=n_experts,
            n_shared=n_shared,
            n_activated=n_activated,
            bias_speed=bias_speed,
            add_eas=add_eas,
        ).to(device=device, dtype=dtype)

        # extra_bias is fp32 by convention even when the rest is bf16.
        new_moe.gate.extra_bias = new_moe.gate.extra_bias.to(
            device=device, dtype=torch.float32
        )

        # EAS zero-jitter init (LlamaMLP default Kaiming would inject too much
        # signal at step 0; small N(0, std) keeps |eas_out| ~0 while gradient
        # paths to gate/up/down all stay live).
        if add_eas and eas_init_std > 0.0 and new_moe.eas is not None:
            with torch.no_grad():
                new_moe.eas.gate_proj.weight.normal_(mean=0.0, std=eas_init_std)
                new_moe.eas.up_proj.weight.normal_(mean=0.0, std=eas_init_std)
                new_moe.eas.down_proj.weight.normal_(mean=0.0, std=eas_init_std)

        layer.mlp = new_moe


def load_cmoe_state_dict(
    model: nn.Module,
    sd_path: str,
    strict: bool = True,
) -> None:
    """Load a CMoE-format state_dict (e.g. ``llama2_S3A3E8/state_dict.pt``).

    Caller must invoke :func:`swap_llama_mlp_to_cmoe_moe` first so that the
    parameter names (e.g. ``model.layers.{i}.mlp.experts.0.gate_proj.weight``,
    ``mlp.gate.classifier.weight``, ``mlp.gate.extra_scale``,
    ``mlp.gate.extra_bias``) line up with the carved checkpoint.

    Args:
        model: post-swap model.
        sd_path: path to a torch ``state_dict.pt``.
        strict: passed through to :func:`nn.Module.load_state_dict`.
    """
    sd = torch.load(sd_path, map_location="cpu")
    model.load_state_dict(sd, strict=strict)


# ---------------------------------------------------------------------------
# Self-test (no large-model load)
# ---------------------------------------------------------------------------


def _self_test() -> None:
    torch.manual_seed(0)

    failures = []

    def _check(name: str, cond: bool, detail: str = "") -> None:
        status = "PASS" if cond else "FAIL"
        print(f"  [{status}] {name}{('  — ' + detail) if detail else ''}")
        if not cond:
            failures.append(name)

    print("Test 1: Router.return_diagnostics interface")
    B, S, H = 2, 8, 128
    n_experts_routed, k = 5, 3
    router = Router(hidden_size=H, n_experts=n_experts_routed, n_activated=k)
    x = torch.randn(B * S, H)
    out_default = router(x)
    out_diag = router(x, return_diagnostics=True)
    _check("router default 2-tuple", isinstance(out_default, tuple) and len(out_default) == 2)
    _check("router diag 4-tuple", isinstance(out_diag, tuple) and len(out_diag) == 4)
    w, idx, raw_logits, softmax_scores = out_diag
    _check("weights shape", tuple(w.shape) == (B * S, k), f"got {tuple(w.shape)}")
    _check("indices shape", tuple(idx.shape) == (B * S, k), f"got {tuple(idx.shape)}")
    _check(
        "raw_logits shape",
        tuple(raw_logits.shape) == (B * S, n_experts_routed),
        f"got {tuple(raw_logits.shape)}",
    )
    _check(
        "softmax_scores shape",
        tuple(softmax_scores.shape) == (B * S, n_experts_routed),
        f"got {tuple(softmax_scores.shape)}",
    )
    _check(
        "softmax_scores sums to 1",
        torch.allclose(softmax_scores.sum(dim=-1), torch.ones(B * S), atol=1e-5),
    )

    print("\nTest 2: MoE.return_components interface")
    hidden, intermediate = 128, 64
    n_experts, n_shared, n_act = 8, 3, 3
    moe_inter = intermediate // n_experts  # 8
    assert moe_inter == 8
    moe = MoE(
        hidden_size=hidden,
        moe_inter_dim=moe_inter,
        n_experts=n_experts,
        n_shared=n_shared,
        n_activated=n_act,
    )
    x = torch.randn(B, S, hidden)
    out = moe(x)
    comps = moe(x, return_components=True)
    _check("default returns tensor", isinstance(out, torch.Tensor))
    _check("default output shape", tuple(out.shape) == (B, S, hidden))
    expected_keys = {
        "output",
        "routed_out",
        "shared_out",
        "router_logits",
        "router_scores",
        "topk_weights",
        "topk_indices",
    }
    _check("comps keys", set(comps.keys()) == expected_keys, f"got {set(comps.keys())}")
    _check("comps.output shape", tuple(comps["output"].shape) == (B, S, hidden))
    _check("comps.routed_out shape", tuple(comps["routed_out"].shape) == (B, S, hidden))
    _check("comps.shared_out shape", tuple(comps["shared_out"].shape) == (B, S, hidden))
    n_routed_experts = n_experts - n_shared  # 5
    _check(
        "comps.router_logits shape",
        tuple(comps["router_logits"].shape) == (B * S, n_routed_experts),
    )
    _check(
        "comps.router_scores shape",
        tuple(comps["router_scores"].shape) == (B * S, n_routed_experts),
    )
    _check(
        "comps.topk_weights shape",
        tuple(comps["topk_weights"].shape) == (B * S, n_act),
    )
    _check(
        "comps.topk_indices shape",
        tuple(comps["topk_indices"].shape) == (B * S, n_act),
    )
    _check("topk_indices is long", comps["topk_indices"].dtype == torch.long)
    _check(
        "router_scores rows sum to 1",
        torch.allclose(
            comps["router_scores"].sum(dim=-1),
            torch.ones(B * S),
            atol=1e-5,
        ),
    )

    print("\nTest 3: freeze_extra_bias_and_scale")
    # Pre-perturb both extras to confirm reset/freeze.
    with torch.no_grad():
        moe.gate.extra_bias.add_(0.5)
        moe.gate.extra_scale.add_(0.3)
    moe.cus_training = True
    freeze_extra_bias_and_scale(moe)  # default: extra_scale_trainable=False
    _check(
        "extra_bias zeroed",
        torch.all(moe.gate.extra_bias == 0).item(),
        f"sum={moe.gate.extra_bias.sum().item()}",
    )
    _check("cus_training False", moe.cus_training is False)
    _check(
        "extra_scale frozen by default",
        moe.gate.extra_scale.requires_grad is False,
    )
    # Now opt-in to trainable.
    freeze_extra_bias_and_scale(moe, extra_scale_trainable=True)
    _check(
        "extra_scale trainable when requested",
        moe.gate.extra_scale.requires_grad is True,
    )

    print("\nTest 4: forward parity (default vs return_components)")
    # Reset freshly and use no-grad eval on a deterministic input to compare.
    moe2 = MoE(
        hidden_size=hidden,
        moe_inter_dim=moe_inter,
        n_experts=n_experts,
        n_shared=n_shared,
        n_activated=n_act,
    )
    moe2.eval()
    x2 = torch.randn(B, S, hidden)
    with torch.no_grad():
        y_default = moe2(x2)
        y_comps = moe2(x2, return_components=True)["output"]
    _check(
        "default == comps['output']",
        torch.allclose(y_default, y_comps, atol=1e-6),
        f"max_abs_diff={(y_default - y_comps).abs().max().item():.2e}",
    )

    print("\nTest 5: swap_llama_mlp_to_cmoe_moe smoke (mock model)")

    class MockLlamaMLP(nn.Module):
        def __init__(self, hidden: int, intermediate: int) -> None:
            super().__init__()
            self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
            self.up_proj = nn.Linear(hidden, intermediate, bias=False)
            self.down_proj = nn.Linear(intermediate, hidden, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    class MockLlamaLayer(nn.Module):
        def __init__(self, hidden: int, intermediate: int) -> None:
            super().__init__()
            self.mlp = MockLlamaMLP(hidden, intermediate)

    class MockLlamaInner(nn.Module):
        def __init__(self, hidden: int, intermediate: int, n_layers: int = 3) -> None:
            super().__init__()
            self.layers = nn.ModuleList(
                [MockLlamaLayer(hidden, intermediate) for _ in range(n_layers)]
            )

    class MockLlamaForCausalLM(nn.Module):
        def __init__(self, hidden: int, intermediate: int, n_layers: int = 3) -> None:
            super().__init__()
            self.model = MockLlamaInner(hidden, intermediate, n_layers)

    mock_hidden = 64
    mock_intermediate = 64  # divisible by n_experts=8 → moe_inter=8
    mock_model = MockLlamaForCausalLM(mock_hidden, mock_intermediate, n_layers=3)
    swap_llama_mlp_to_cmoe_moe(
        mock_model,
        n_experts=8,
        n_activated=3,
        n_shared=3,
    )
    all_swapped = all(isinstance(layer.mlp, MoE) for layer in mock_model.model.layers)
    _check("all 3 layers swapped to MoE", all_swapped)
    # Forward through swapped model to confirm no shape mismatch.
    x_mock = torch.randn(2, 4, mock_hidden)
    h = x_mock
    for layer in mock_model.model.layers:
        h = layer.mlp(h)
    _check("swapped model forward shape", tuple(h.shape) == (2, 4, mock_hidden))
    # moe_inter_dim inference correctness.
    first_moe = mock_model.model.layers[0].mlp
    _check(
        "moe_inter_dim inferred",
        first_moe.experts[0].gate_proj.weight.shape[0] == mock_intermediate // 8,
    )
    _check(
        "shared expert width = n_shared * moe_inter",
        first_moe.shared_experts.gate_proj.weight.shape[0]
        == 3 * (mock_intermediate // 8),
    )

    print("")
    if failures:
        print(f"FAILED ({len(failures)}): {failures}")
        raise SystemExit(1)
    print("All tests passed")


if __name__ == "__main__":
    _self_test()
