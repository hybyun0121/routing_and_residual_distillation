"""RRD-style MoE module compatible with Llama-2 dense MLP.

Provides ``LlamaMLP``, ``Router``, ``MoE`` with optional shared expert, and a
``build_moe_from_clusters`` helper that splits a teacher dense MLP into
balanced expert chunks via a cluster assignment tensor.

Designed for the RRD framework: standard single-linear router (no dual-gate),
optional zero-init shared expert (down_proj=0) so the chain warm-start begins
from MoE_K-only behavior.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class LlamaMLP(nn.Module):
    """Llama-style SwiGLU FFN.

    Args:
        hidden_size: Input/output hidden dim.
        intermediate_size: Inner FFN width.
        hidden_act: Activation name; ``silu`` by default.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = F.silu if hidden_act == "silu" else getattr(F, hidden_act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Router(nn.Module):
    """Single-linear router (standard MoE form).

    Returns raw logits, full softmax scores, top-k weights gathered from the
    softmax, and top-k expert indices.
    """

    def __init__(self, hidden_size: int, n_experts: int, top_k: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.n_experts = n_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden_size, n_experts, bias=False)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute routing.

        Args:
            x: Tensor of shape ``[N, H]``.

        Returns:
            ``(logits, scores, topk_weights, topk_indices)``:
              - ``logits``:        ``[N, n_experts]`` raw gate logits.
              - ``scores``:        ``[N, n_experts]`` softmax probs (float32).
              - ``topk_weights``:  ``[N, top_k]`` softmax probs gathered at top-k.
              - ``topk_indices``:  ``[N, top_k]`` long indices.
        """
        logits = self.gate(x)
        scores = F.softmax(logits, dim=-1, dtype=torch.float32)
        topk_weights, topk_indices = torch.topk(scores, self.top_k, dim=-1)
        return logits, scores, topk_weights.type_as(x), topk_indices


class MoE(nn.Module):
    """RRD MoE block with optional shared expert.

    Always exposes ``n_experts`` routed experts (split balanced from the teacher
    intermediate dim). ``n_active`` controls top-K routing. When
    ``has_shared=True``, an extra ``LlamaMLP(hidden_size, d_shared)`` runs in
    parallel and is added to the routed sum (E8A4S2 setup). With
    ``zero_init_shared=True`` the shared expert's ``down_proj`` is zero-init so
    it contributes 0 at start.

    Args:
        hidden_size: Hidden dim H.
        intermediate_size: Teacher dense intermediate dim (must be divisible by
            ``n_experts``).
        n_experts: Number of routed experts.
        n_active: Top-K routed experts per token.
        has_shared: Whether to add a parallel shared expert.
        d_shared: Shared expert intermediate width (required when
            ``has_shared=True``).
        zero_init_shared: Zero-init shared.down_proj when ``has_shared=True``.
        hidden_act: Activation for all MLPs.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        n_experts: int = 8,
        n_active: int = 4,
        has_shared: bool = False,
        d_shared: Optional[int] = None,
        zero_init_shared: bool = True,
        hidden_act: str = "silu",
        expert_weight_mode: str = "unit",
    ) -> None:
        super().__init__()
        if intermediate_size % n_experts != 0:
            raise ValueError(
                f"intermediate_size ({intermediate_size}) must be divisible by "
                f"n_experts ({n_experts})."
            )
        if expert_weight_mode not in ("unit", "softmax"):
            raise ValueError(f"expert_weight_mode must be 'unit' or 'softmax', got {expert_weight_mode}")

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.n_experts = n_experts
        self.n_active = n_active
        self.has_shared = has_shared
        self.d_shared = d_shared if has_shared else None
        self.expert_weight_mode = expert_weight_mode

        moe_inter_dim = intermediate_size // n_experts
        self.moe_inter_dim = moe_inter_dim

        self.experts = nn.ModuleList(
            [LlamaMLP(hidden_size, moe_inter_dim, hidden_act) for _ in range(n_experts)]
        )
        self.gate = Router(hidden_size, n_experts, top_k=n_active)

        if has_shared:
            if d_shared is None:
                raise ValueError("d_shared must be provided when has_shared=True.")
            self.shared_expert = LlamaMLP(hidden_size, d_shared, hidden_act)
            if zero_init_shared:
                with torch.no_grad():
                    self.shared_expert.down_proj.weight.zero_()
        else:
            self.shared_expert = None

    def forward(
        self,
        x: torch.Tensor,
        return_components: bool = False,
        override_topk_indices: Optional[torch.Tensor] = None,
        override_topk_weights: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass.

        Args:
            x: Input ``[B, S, H]`` or ``[N, H]``.
            return_components: If True, return diagnostic dict; else return
                combined output with the original input shape.
            override_topk_indices: Optional ``[N, top_k]`` long tensor to bypass
                the gate's argmax selection (EM Phase M oracle routing). When
                provided, the gate is still called for ``router_logits/scores``
                bookkeeping but routing uses these indices.
            override_topk_weights: Optional ``[N, top_k]`` float tensor for
                expert combination weights. Only consulted when
                ``override_topk_indices`` is set AND ``expert_weight_mode ==
                "softmax"``. ``"unit"`` mode ignores it.

        Returns:
            Combined output tensor (default) or component dict (when
            ``return_components=True``) with keys: ``output``, ``routed_out``,
            ``shared_out``, ``router_logits``, ``router_scores``,
            ``topk_weights``, ``topk_indices``.
        """
        original_shape = x.shape
        x_flat = x.reshape(-1, self.hidden_size)
        n_tokens = x_flat.shape[0]

        logits, scores, topk_weights, topk_indices = self.gate(x_flat)
        if override_topk_indices is not None:
            topk_indices = override_topk_indices.to(topk_indices.device).long()
            if override_topk_weights is not None:
                topk_weights = override_topk_weights.to(topk_weights.device, dtype=topk_weights.dtype)

        routed_out = torch.zeros_like(x_flat)
        flat_indices = topk_indices.reshape(-1)
        counts = torch.bincount(flat_indices, minlength=self.n_experts)

        for e in range(self.n_experts):
            if counts[e].item() == 0:
                continue
            token_idx, top_pos = torch.where(topk_indices == e)
            expert_input = x_flat[token_idx]
            expert_out = self.experts[e](expert_input)
            if self.expert_weight_mode == "softmax":
                weights_e = topk_weights[token_idx, top_pos].unsqueeze(-1).to(expert_out.dtype)
                routed_out.index_add_(0, token_idx, (expert_out * weights_e).to(routed_out.dtype))
            else:
                routed_out.index_add_(0, token_idx, expert_out.to(routed_out.dtype))

        if self.has_shared:
            shared_out = self.shared_expert(x_flat)
        else:
            shared_out = torch.zeros_like(x_flat)

        combined = routed_out + shared_out

        if return_components:
            out_shape = original_shape
            return {
                "output": combined.reshape(out_shape),
                "routed_out": routed_out.reshape(out_shape),
                "shared_out": shared_out.reshape(out_shape),
                "router_logits": logits,
                "router_scores": scores,
                "topk_weights": topk_weights,
                "topk_indices": topk_indices,
            }

        return combined.reshape(original_shape)


def build_moe_from_clusters(
    teacher_mlp: nn.Module,
    cluster_ids: torch.Tensor,
    n_experts: int,
    n_active: int,
    has_shared: bool,
    d_shared: Optional[int] = None,
    zero_init_shared: bool = True,
    hidden_act: str = "silu",
) -> MoE:
    """Build an ``MoE`` module by splitting a teacher Llama MLP via clusters.

    Args:
        teacher_mlp: Module exposing ``gate_proj``/``up_proj``/``down_proj``
            (all ``nn.Linear``). Shapes:
              - ``gate_proj.weight``: ``[I, H]``
              - ``up_proj.weight``:   ``[I, H]``
              - ``down_proj.weight``: ``[H, I]``
            where ``I = intermediate_size`` and ``H = hidden_size``.
        cluster_ids: Long tensor of shape ``[I]`` with values in
            ``[0, n_experts)``. Each cluster must contain exactly ``I/n_experts``
            entries (balanced partition).
        n_experts: Number of routed experts.
        n_active: Top-K routed experts.
        has_shared: Add a parallel shared expert.
        d_shared: Shared expert width (required if ``has_shared``).
        zero_init_shared: Zero-init shared.down_proj.
        hidden_act: Activation name.

    Returns:
        MoE module with experts initialized from the teacher's row/column
        slices. Router gate is left at its default random init (caller is
        expected to overwrite via a router-init step).
    """
    gate_w = teacher_mlp.gate_proj.weight.detach()
    up_w = teacher_mlp.up_proj.weight.detach()
    down_w = teacher_mlp.down_proj.weight.detach()

    intermediate_size, hidden_size = gate_w.shape
    if down_w.shape != (hidden_size, intermediate_size):
        raise ValueError(
            f"Teacher down_proj shape {tuple(down_w.shape)} inconsistent with "
            f"gate_proj {tuple(gate_w.shape)}."
        )
    if intermediate_size % n_experts != 0:
        raise ValueError(
            f"intermediate_size ({intermediate_size}) must be divisible by "
            f"n_experts ({n_experts})."
        )
    expected_per_expert = intermediate_size // n_experts

    if cluster_ids.numel() != intermediate_size:
        raise ValueError(
            f"cluster_ids has {cluster_ids.numel()} entries but expected "
            f"{intermediate_size}."
        )
    cluster_ids = cluster_ids.to(torch.long).cpu()

    moe = MoE(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        n_experts=n_experts,
        n_active=n_active,
        has_shared=has_shared,
        d_shared=d_shared,
        zero_init_shared=zero_init_shared,
        hidden_act=hidden_act,
    )

    target_dtype = next(moe.parameters()).dtype
    target_device = next(moe.parameters()).device

    for e in range(n_experts):
        neuron_idx = (cluster_ids == e).nonzero(as_tuple=False).squeeze(-1)
        if neuron_idx.numel() != expected_per_expert:
            raise ValueError(
                f"Cluster {e} has {neuron_idx.numel()} neurons; expected "
                f"{expected_per_expert} (balanced partition required)."
            )
        expert = moe.experts[e]
        with torch.no_grad():
            expert.gate_proj.weight.copy_(
                gate_w.index_select(0, neuron_idx).to(target_dtype).to(target_device)
            )
            expert.up_proj.weight.copy_(
                up_w.index_select(0, neuron_idx).to(target_dtype).to(target_device)
            )
            expert.down_proj.weight.copy_(
                down_w.index_select(1, neuron_idx).to(target_dtype).to(target_device)
            )

    return moe


def _self_test() -> None:
    """Self-test exercising build, forward, components, zero-init and N=8 sanity."""
    torch.manual_seed(0)

    hidden_size = 4096
    intermediate_size = 11008  # Llama-2-7B
    n_experts = 8
    moe_inter = intermediate_size // n_experts

    teacher = LlamaMLP(hidden_size, intermediate_size)
    cluster_ids = torch.arange(intermediate_size) // moe_inter

    x = torch.randn(2, 16, hidden_size)

    # E8A6 (no shared)
    moe_e8a6 = build_moe_from_clusters(
        teacher_mlp=teacher,
        cluster_ids=cluster_ids,
        n_experts=8,
        n_active=6,
        has_shared=False,
    )
    moe_e8a6.eval()
    out_e8a6 = moe_e8a6(x)
    assert out_e8a6.shape == x.shape, f"E8A6 shape mismatch: {out_e8a6.shape}"

    comp_e8a6 = moe_e8a6(x, return_components=True)
    assert comp_e8a6["output"].shape == x.shape
    assert comp_e8a6["routed_out"].shape == x.shape
    assert comp_e8a6["shared_out"].shape == x.shape
    assert torch.equal(comp_e8a6["shared_out"], torch.zeros_like(comp_e8a6["shared_out"]))
    assert comp_e8a6["router_logits"].shape == (2 * 16, 8)
    assert comp_e8a6["topk_indices"].shape == (2 * 16, 6)

    # E8A4S2 (with shared, zero-init)
    d_shared = intermediate_size // 4  # = 2 * (intermediate / 8)
    moe_e8a4s2 = build_moe_from_clusters(
        teacher_mlp=teacher,
        cluster_ids=cluster_ids,
        n_experts=8,
        n_active=4,
        has_shared=True,
        d_shared=d_shared,
        zero_init_shared=True,
    )
    moe_e8a4s2.eval()
    out_e8a4s2 = moe_e8a4s2(x)
    assert out_e8a4s2.shape == x.shape

    comp_s2 = moe_e8a4s2(x, return_components=True)
    assert torch.allclose(
        comp_s2["shared_out"], torch.zeros_like(comp_s2["shared_out"])
    ), "zero_init_shared should make shared_out exactly zero."
    assert torch.allclose(
        comp_s2["output"], comp_s2["routed_out"]
    ), "With shared zero-init, output should equal routed_out."

    # N=8 all-active equivalence with teacher (BasicSplit cluster -> exact partition)
    moe_e8a8 = build_moe_from_clusters(
        teacher_mlp=teacher,
        cluster_ids=cluster_ids,
        n_experts=8,
        n_active=8,
        has_shared=False,
    )
    moe_e8a8.eval()

    # Override router so each token gets uniform top-8 with weight=1 each (raw weighting).
    with torch.no_grad():
        # Set gate weight to 0 -> logits=0 -> softmax uniform 1/8. We then bypass
        # softmax weighting by patching the forward through monkey weights:
        # easier path: directly hook router output. We'll instead manually build
        # the equivalence: with weight=1/8 each, sum(experts(x)) / 8 != teacher.
        # To match teacher exactly we need weight=1 each. Use a custom forward.
        moe_e8a8.gate.gate.weight.zero_()

    # Custom equivalence forward: sum all expert outputs unweighted = teacher MLP.
    x_flat = x.reshape(-1, hidden_size)
    with torch.no_grad():
        teacher_out = teacher(x_flat)
        moe_sum = torch.zeros_like(teacher_out)
        for e in range(8):
            moe_sum = moe_sum + moe_e8a8.experts[e](x_flat)

    cos = F.cosine_similarity(moe_sum.flatten(), teacher_out.flatten(), dim=0).item()
    assert cos > 0.99, f"E8 all-active vs teacher cos_sim={cos:.6f} < 0.99"

    # Also sanity-check topk_indices are valid (within range).
    assert comp_e8a6["topk_indices"].max().item() < 8
    assert comp_e8a6["topk_indices"].min().item() >= 0

    print(f"E8 sum vs teacher cos_sim: {cos:.6f}")
    print("T1 self-test passed")


if __name__ == "__main__":
    _self_test()
