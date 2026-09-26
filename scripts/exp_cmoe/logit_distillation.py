"""Full-vocabulary, token-mean forward KL with bounded FP32 workspace.

The teacher is always detached. The custom backward recomputes probabilities
per token chunk, avoiding a full B*S*V FP32 probability graph for Qwen.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class _ForwardKL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, student, teacher, valid, temperature, chunk_tokens):
        indices = valid.reshape(-1).nonzero(as_tuple=False).flatten()
        if not indices.numel():
            raise ValueError("logit KD has no valid next-token positions")
        ctx.save_for_backward(student, teacher, indices)
        ctx.temperature = temperature
        ctx.chunk_tokens = chunk_tokens
        student_flat = student.reshape(-1, student.shape[-1])
        teacher_flat = teacher.reshape(-1, teacher.shape[-1])
        total = torch.zeros((), device=student.device, dtype=torch.float32)
        for idx in indices.split(chunk_tokens):
            log_s = F.log_softmax(student_flat.index_select(0, idx).float() / temperature, dim=-1)
            log_t = F.log_softmax(teacher_flat.index_select(0, idx).float() / temperature, dim=-1)
            total += (log_t.exp() * (log_t - log_s)).sum()
        return total * (temperature**2 / indices.numel())

    @staticmethod
    def backward(ctx, grad_output):
        student, teacher, indices = ctx.saved_tensors
        student_flat = student.reshape(-1, student.shape[-1])
        teacher_flat = teacher.reshape(-1, teacher.shape[-1])
        gradient = torch.zeros_like(student)
        gradient_flat = gradient.reshape(-1, gradient.shape[-1])
        scale = grad_output.float() * (ctx.temperature / indices.numel())
        for idx in indices.split(ctx.chunk_tokens):
            p_s = F.softmax(student_flat.index_select(0, idx).float() / ctx.temperature, dim=-1)
            p_t = F.softmax(teacher_flat.index_select(0, idx).float() / ctx.temperature, dim=-1)
            gradient_flat.index_copy_(0, idx, ((p_s - p_t) * scale).to(student.dtype))
        return gradient, None, None, None, None


def logit_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    *,
    temperature: float = 1.0,
    chunk_tokens: int = 128,
) -> torch.Tensor:
    """tau² * KL(teacher || student), averaged over causal-CE target positions.

Logits at t predict labels at t+1. Each sequence's last logits are excluded;
ignore-index labels and masked target positions are excluded as well.
"""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("KD temperature must be finite and positive")
    if chunk_tokens < 1:
        raise ValueError("KD chunk_tokens must be positive")
    if student_logits.ndim != 3 or student_logits.shape != teacher_logits.shape:
        raise ValueError("teacher/student KD logits must have identical [B,S,V] shapes")
    if student_logits.device != teacher_logits.device:
        raise ValueError("teacher/student KD logits must be on the same device")
    if labels.shape != student_logits.shape[:2]:
        raise ValueError("KD labels must have shape [B,S]")
    valid = torch.zeros_like(labels, dtype=torch.bool)
    valid[:, :-1] = labels[:, 1:] != -100
    if attention_mask is not None:
        if attention_mask.shape != labels.shape:
            raise ValueError("KD attention_mask must match labels")
        valid[:, :-1] &= attention_mask[:, 1:].bool()
    return _ForwardKL.apply(
        student_logits.contiguous(), teacher_logits.detach().contiguous(),
        valid, float(temperature), int(chunk_tokens),
    )
