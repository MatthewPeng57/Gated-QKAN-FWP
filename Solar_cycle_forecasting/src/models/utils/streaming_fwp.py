# Copyright 2026 Matthew Peng and contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Streaming fast-weight programmer — fused prefix-scan replacement.

Replaces the materialize-and-cumsum block in FWP.forward with a
left-to-right recurrence that never materializes the
(B, L, O, I, R+1, 2) delta_theta tensor.

The recurrence is theta_t = g_t * theta_{t-1} + (1 - g_t) * (A_t (x) B_t),
so only the running theta is kept in memory. Both paths compute the same
final theta; the cumsum form folds the same decay factors in one shot.

Dispatch:
  CUDA + Triton available + env QFWP_STREAMING_FWP != 0  -> Triton kernel
  otherwise                                              -> pure-torch recurrence
"""
from __future__ import annotations

import os
from typing import Callable

import torch

try:
    from .streaming_fwp_ops import (
        streaming_fwp_forward_triton,
        streaming_fwp_backward_triton,
    )

    _TRITON_AVAILABLE = True
except Exception:
    streaming_fwp_forward_triton = None  # type: ignore[assignment]
    streaming_fwp_backward_triton = None  # type: ignore[assignment]
    _TRITON_AVAILABLE = False


def _streaming_fwp_torch_forward(
    A: torch.Tensor,      # (B, L, O, R+1)
    B: torch.Tensor,      # (B, L, I, 2)
    gates: torch.Tensor,  # (B, L)
) -> torch.Tensor:
    """Pure-torch forward recurrence.

    theta_0 = 0
    for t = 1..L:
        theta_t = (1 - g_t) * A_t (x) B_t + g_t * theta_{t-1}
    return theta_L

    Used as:
      - CPU / non-Triton fallback
      - correctness oracle for the Triton kernel
    """
    Bsz, L, O, Rp1 = A.shape
    assert B.shape[:2] == (Bsz, L), (
        f"B shape {tuple(B.shape)} incompatible with A batch/length ({Bsz}, {L})"
    )
    _, _, I, two = B.shape
    assert two == 2, f"B last dim must be 2, got {two}"
    assert gates.shape == (Bsz, L), (
        f"gates shape {tuple(gates.shape)} != ({Bsz}, {L})"
    )
    assert A.device == B.device == gates.device, (
        f"device mismatch: A={A.device} B={B.device} gates={gates.device}"
    )
    assert A.dtype == B.dtype == gates.dtype, (
        f"dtype mismatch: A={A.dtype} B={B.dtype} gates={gates.dtype}"
    )
    theta = torch.zeros(Bsz, O, I, Rp1, two, dtype=A.dtype, device=A.device)
    for t in range(L):
        A_t = A[:, t]                              # (B, O, Rp1)
        B_t = B[:, t]                              # (B, I, 2)
        g_t = gates[:, t].view(Bsz, 1, 1, 1, 1)    # (B, 1, 1, 1, 1)
        # delta_t[b, o, i, l, k] = A_t[b, o, l] * B_t[b, i, k]
        delta = torch.einsum("bol,bik->boilk", A_t, B_t)
        theta = (1 - g_t) * delta + g_t * theta
    return theta


def streaming_fwp_final_theta(
    A: torch.Tensor,      # (B, L, O, R+1)
    B: torch.Tensor,      # (B, L, I, 2)
    gates: torch.Tensor,  # (B, L)
) -> torch.Tensor:
    """Compute theta_L under the FWP gated recurrence. Autograd-safe.

    Uses the fused Triton kernel on CUDA when available, otherwise a
    pure-torch left-to-right recurrence. Both implementations compute
    the same value in exact arithmetic; fp32 drift is O(eps*L) ~ 5e-7.
    """
    kernel_enabled = os.environ.get("QFWP_STREAMING_FWP", "1") != "0"
    use_triton = (
        kernel_enabled
        and _TRITON_AVAILABLE
        and A.is_cuda
        and A.dtype == torch.float32
    )
    if use_triton:
        return _StreamingFWPFunction.apply(A, B, gates)
    return _streaming_fwp_torch_forward(A, B, gates)


class _StreamingFWPFunction(torch.autograd.Function):
    """Triton-backed autograd function: fused forward (saves θ-checkpoints)
    + fused backward kernel. No Python chain-rule loop."""

    @staticmethod
    def forward(ctx, A, B, gates):
        out, theta_ckpts = streaming_fwp_forward_triton(A, B, gates)
        ctx.save_for_backward(A, B, gates, theta_ckpts)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        A, B, gates, theta_ckpts = ctx.saved_tensors
        grad_A, grad_B, grad_gates = streaming_fwp_backward_triton(
            A, B, gates, grad_out, theta_ckpts
        )
        return grad_A, grad_B, grad_gates


def streaming_fwp_per_step_outputs(
    A: torch.Tensor,        # (B, L, O, R+1)
    B: torch.Tensor,        # (B, L, I, 2)
    gates: torch.Tensor,    # (B, L)
    qkan_call: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x_resized_seq: torch.Tensor,  # (B, L, I)
) -> torch.Tensor:
    """Run the FWP recurrence and call qkan_call at every timestep.

    Returns a (B, L, O) tensor of per-step QKAN outputs. Differentiable
    end-to-end via torch autograd through the recurrence and the qkan_call.
    Pure-torch path; a Triton-fused per-step kernel is deferred to a
    follow-up task once profiling shows L>>16 wall-time matters.
    """
    Bsz, L, O, Rp1 = A.shape
    _, _, I, two = B.shape
    assert two == 2 and gates.shape == (Bsz, L)
    assert x_resized_seq.shape == (Bsz, L, I), (
        f"x_resized_seq shape {tuple(x_resized_seq.shape)} != ({Bsz}, {L}, {I})"
    )
    theta = torch.zeros(Bsz, O, I, Rp1, two, dtype=A.dtype, device=A.device)
    outs = []
    for t in range(L):
        delta = torch.einsum("bol,bik->boilk", A[:, t], B[:, t])
        g = gates[:, t].view(Bsz, 1, 1, 1, 1)
        theta = (1 - g) * delta + g * theta
        outs.append(qkan_call(x_resized_seq[:, t], theta))
    return torch.stack(outs, dim=1)
