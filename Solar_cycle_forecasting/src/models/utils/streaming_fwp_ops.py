# Copyright 2026 Kuo-Chung Peng and Samuel Yen-Chi Chen
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
Triton kernels for the streaming FWP prefix-scan.

The recurrence is documented in streaming_fwp.py. This file is the CUDA-only
fast path; the pure-torch fallback lives in streaming_fwp.py.
"""
from __future__ import annotations

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore


# Segment length for the backward-pass recompute. Tunable: smaller SEG uses
# more checkpoints (more memory, cheaper per-segment recompute) but shorter
# unrolled register-histories. SEG=16 is a good balance at L=528.
_SEG = 16


def _next_pow2(n: int) -> int:
    """Smallest power of two >= n. Triton tl.arange requires power-of-2 extents."""
    p = 1
    while p < n:
        p *= 2
    return p


def _num_ckpts(L: int, seg: int = _SEG) -> int:
    """Number of θ-checkpoints saved by the forward kernel.

    Checkpoint k stores θ_{k * seg}. We save k = 0, 1, ..., floor(L/seg),
    plus a trailing θ_L if L is not a multiple of seg. At L=528, seg=16 (clean
    multiple) we get 34 checkpoints: θ_0, θ_16, ..., θ_528.
    """
    last = (L + seg - 1) // seg   # ceil(L / seg)
    # k = 0 .. last, inclusive → last + 1 checkpoints.
    return last + 1


@triton.jit
def _streaming_fwp_forward_kernel(
    A_ptr,            # (B, L, O, R+1)  fp32
    B_ptr,            # (B, L, I, 2)    fp32
    gates_ptr,        # (B, L)          fp32
    out_ptr,          # (B, O, I, R+1, 2)  fp32
    theta_ckpts_ptr,  # (B, N_CKPTS, O, I, R+1, 2)  fp32
    # strides for A
    sA_b, sA_l, sA_o, sA_r,
    # strides for B
    sB_b, sB_l, sB_i, sB_k,
    # strides for gates
    sG_b, sG_l,
    # strides for out
    sO_b, sO_o, sO_i, sO_r, sO_k,
    # strides for theta_ckpts
    sCk_b, sCk_k, sCk_o, sCk_i, sCk_r, sCk_kk,
    # shapes (constexpr)
    L: tl.constexpr,
    RP1: tl.constexpr,    # reps + 1
    K_DIM: tl.constexpr,  # RP1 * 2 (the true state size)
    K_PAD: tl.constexpr,  # next_pow2(K_DIM) — the tl.arange extent
    SEG: tl.constexpr,    # segment length for backward checkpoint save
    N_CKPTS: tl.constexpr,
):
    """One program per (b, o, i) triple.

    Also saves θ at every SEG steps into `theta_ckpts_ptr` for the fused
    backward kernel. Since t comes from a Python-level `range(L)` with L
    constexpr, the `if` on (t+1)%SEG is evaluated at compile time and only
    emits a store at the exact boundary timesteps.
    """
    pid_b = tl.program_id(0)
    pid_o = tl.program_id(1)
    pid_i = tl.program_id(2)

    k_range = tl.arange(0, K_PAD)
    valid = k_range < K_DIM
    # Decompose valid k into (r, kk) with kk in {0, 1}: k = r * 2 + kk
    r_idx = k_range // 2
    kk_idx = k_range % 2

    # Save θ_0 = 0 at ckpt[0]
    ckpt0_off = (
        pid_b * sCk_b + 0 * sCk_k + pid_o * sCk_o + pid_i * sCk_i
        + r_idx * sCk_r + kk_idx * sCk_kk
    )
    theta = tl.zeros([K_PAD], dtype=tl.float32)
    tl.store(theta_ckpts_ptr + ckpt0_off, theta, mask=valid)

    for t in range(L):
        # Load A_t[b, o, r_idx[k]] (masked: invalid lanes read 0.0)
        a_offsets = pid_b * sA_b + t * sA_l + pid_o * sA_o + r_idx * sA_r
        A_t = tl.load(A_ptr + a_offsets, mask=valid, other=0.0)

        # Load B_t[b, i, kk_idx[k]]
        b_offsets = pid_b * sB_b + t * sB_l + pid_i * sB_i + kk_idx * sB_k
        B_t = tl.load(B_ptr + b_offsets, mask=valid, other=0.0)

        # delta[k] = A_t[k] * B_t[k]
        delta = A_t * B_t

        # Scalar gate
        g = tl.load(gates_ptr + pid_b * sG_b + t * sG_l)

        # theta_{t+1} = (1 - g) * delta + g * theta
        # Inert lanes: delta=0, theta=0, result stays 0 regardless of g.
        theta = (1.0 - g) * delta + g * theta

        # Segment checkpoint: save θ_{t+1} if (t+1) is a multiple of SEG.
        # t is a Python int (constexpr loop), so this `if` is compile-time.
        tp1 = t + 1
        if (tp1 % SEG) == 0:
            save_k = tp1 // SEG
            if save_k < N_CKPTS:
                ckpt_off = (
                    pid_b * sCk_b + save_k * sCk_k + pid_o * sCk_o + pid_i * sCk_i
                    + r_idx * sCk_r + kk_idx * sCk_kk
                )
                tl.store(theta_ckpts_ptr + ckpt_off, theta, mask=valid)

    # If L is not a multiple of SEG, save the trailing θ_L to the last ckpt slot.
    # Compile-time guard: emits a store only when L isn't a clean multiple.
    if (L % SEG) != 0:
        last_k = (L // SEG) + 1
        if last_k < N_CKPTS:
            ckpt_off = (
                pid_b * sCk_b + last_k * sCk_k + pid_o * sCk_o + pid_i * sCk_i
                + r_idx * sCk_r + kk_idx * sCk_kk
            )
            tl.store(theta_ckpts_ptr + ckpt_off, theta, mask=valid)

    # Store final theta_L into (B, O, I, R+1, 2); only valid lanes
    out_offsets = (
        pid_b * sO_b + pid_o * sO_o + pid_i * sO_i
        + r_idx * sO_r + kk_idx * sO_k
    )
    tl.store(out_ptr + out_offsets, theta, mask=valid)


def streaming_fwp_forward_triton(
    A: torch.Tensor,      # (B, L, O, R+1) fp32 CUDA
    B: torch.Tensor,      # (B, L, I, 2)   fp32 CUDA
    gates: torch.Tensor,  # (B, L)         fp32 CUDA
):
    """Launch the streaming-FWP forward kernel.

    Returns (final_theta, theta_ckpts):
      - final_theta: (B, O, I, R+1, 2)
      - theta_ckpts: (B, N_CKPTS, O, I, R+1, 2)  — segment θ checkpoints
        used by the fused backward kernel. Its size is
        B × N_CKPTS × O × I × (R+1) × 2 × 4 bytes; at this repository's
        configuration (L=528, SEG=16 → N_CKPTS=34, B=32, O=20, I=10, R=2)
        that is 32 × 34 × 20 × 10 × 3 × 2 × 4 B ≈ 5.2 MB.
    """
    assert A.is_cuda and B.is_cuda and gates.is_cuda, "kernel requires CUDA tensors"
    assert A.dtype == torch.float32 == B.dtype == gates.dtype, (
        f"kernel requires fp32; got A={A.dtype} B={B.dtype} gates={gates.dtype}"
    )
    Bsz, L, O, Rp1 = A.shape
    _, _, I, two = B.shape
    assert two == 2 and gates.shape == (Bsz, L)

    A = A.contiguous()
    B = B.contiguous()
    gates = gates.contiguous()

    out = torch.empty(Bsz, O, I, Rp1, 2, device=A.device, dtype=torch.float32)

    n_ckpts = _num_ckpts(L, _SEG)
    theta_ckpts = torch.empty(
        Bsz, n_ckpts, O, I, Rp1, 2, device=A.device, dtype=torch.float32
    )

    k_dim = Rp1 * 2
    k_pad = _next_pow2(k_dim)

    grid = (Bsz, O, I)
    _streaming_fwp_forward_kernel[grid](
        A, B, gates, out, theta_ckpts,
        A.stride(0), A.stride(1), A.stride(2), A.stride(3),
        B.stride(0), B.stride(1), B.stride(2), B.stride(3),
        gates.stride(0), gates.stride(1),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
        theta_ckpts.stride(0), theta_ckpts.stride(1), theta_ckpts.stride(2),
        theta_ckpts.stride(3), theta_ckpts.stride(4), theta_ckpts.stride(5),
        L=L, RP1=Rp1, K_DIM=k_dim, K_PAD=k_pad,
        SEG=_SEG, N_CKPTS=n_ckpts,
    )
    return out, theta_ckpts


@triton.jit
def _streaming_fwp_backward_kernel(
    A_ptr,            # (B, L, O, R+1)  fp32
    B_ptr,            # (B, L, I, 2)    fp32
    gates_ptr,        # (B, L)          fp32
    theta_ckpts_ptr,  # (B, N_CKPTS, O, I, R+1, 2)  fp32
    grad_out_ptr,     # (B, O, I, R+1, 2)           fp32
    grad_A_ptr,       # (B, L, O, R+1)              fp32 (atomic_add)
    grad_B_ptr,       # (B, L, I, 2)                fp32 (atomic_add)
    grad_gates_ptr,   # (B, L)                      fp32 (atomic_add)
    # strides A
    sA_b, sA_l, sA_o, sA_r,
    # strides B
    sB_b, sB_l, sB_i, sB_k,
    # strides gates
    sG_b, sG_l,
    # strides theta_ckpts
    sCk_b, sCk_k, sCk_o, sCk_i, sCk_r, sCk_kk,
    # strides grad_out
    sGO_b, sGO_o, sGO_i, sGO_r, sGO_k,
    # strides grad_A
    sgA_b, sgA_l, sgA_o, sgA_r,
    # strides grad_B
    sgB_b, sgB_l, sgB_i, sgB_k,
    # strides grad_gates
    sgG_b, sgG_l,
    # shapes (constexpr)
    L: tl.constexpr,
    RP1: tl.constexpr,
    K_DIM: tl.constexpr,
    K_PAD: tl.constexpr,
    SEG: tl.constexpr,
    N_CKPTS: tl.constexpr,
):
    """Fused backward for the streaming FWP gated prefix-scan.

    One program per (b, o, i). Walks segments in reverse. For each segment:
      1. Re-runs forward from the checkpoint at segment start, caching θ at
         every step in a Python-unrolled list of register vectors.
      2. Sweeps backward through the segment, accumulating grad_A, grad_B,
         grad_gates via `tl.atomic_add` (since grad_A sums over i, grad_B
         sums over o, grad_gates sums over (o, i) — all across programs),
         and propagating grad_θ through the gate.
    """
    pid_b = tl.program_id(0)
    pid_o = tl.program_id(1)
    pid_i = tl.program_id(2)

    k_range = tl.arange(0, K_PAD)
    valid = k_range < K_DIM
    r_idx = k_range // 2
    kk_idx = k_range % 2

    # Load initial grad_θ at t=L (boundary condition: grad_θ_L = grad_out).
    go_off = (
        pid_b * sGO_b + pid_o * sGO_o + pid_i * sGO_i
        + r_idx * sGO_r + kk_idx * sGO_k
    )
    grad_theta = tl.load(grad_out_ptr + go_off, mask=valid, other=0.0)

    # Walk segments in reverse. We have N_CKPTS - 1 "segments" total, covering
    # the timesteps [0, L). The last checkpoint (index N_CKPTS-1) holds θ_L;
    # earlier checkpoints are segment-start θ values.
    N_SEGS: tl.constexpr = N_CKPTS - 1

    # Row-selector axis for 2D register tiles (used to index θ_prev history
    # and to scatter per-step outputs into batched atomic writes).
    seg_idx_1d = tl.arange(0, SEG)                                    # (SEG,)

    for seg_rev in range(N_SEGS):
        seg = N_SEGS - 1 - seg_rev   # segment index (compile-time)
        t_start = seg * SEG          # first timestep of this segment

        # Load θ at segment start from checkpoint.
        ckpt_off = (
            pid_b * sCk_b + seg * sCk_k + pid_o * sCk_o + pid_i * sCk_i
            + r_idx * sCk_r + kk_idx * sCk_kk
        )
        theta_seg_start = tl.load(theta_ckpts_ptr + ckpt_off, mask=valid, other=0.0)

        # --- Pass 1: forward recompute within segment, caching θ_prev (the
        # state BEFORE each step's update) into a 2D register tile of shape
        # (SEG, K_PAD). Row k_in = θ_prev for step t_start + k_in.
        #
        # Using a 2D tl tensor (not a Python list) bypasses Triton's
        # restrictions on `.append` / `__setitem__` / `enumerate` /
        # `for x in list` / integer indexing of Python lists. We index with
        # tl.where(seg_idx_1d == k_in, ...) to broadcast a row value into
        # the 2D tile, and collapse with tl.sum(..., axis=0) to extract a
        # row back out.
        theta_hist = tl.zeros([SEG, K_PAD], dtype=tl.float32)
        cur_theta = theta_seg_start

        for k_in in range(SEG):
            t = t_start + k_in
            # Write θ_prev for step `t` into row k_in of the history tile.
            row_mask = (seg_idx_1d == k_in)[:, None]      # (SEG, 1) bool
            theta_hist = tl.where(row_mask, cur_theta[None, :], theta_hist)

            # Compile-time guard for ragged trailing segments (t >= L).
            if t < L:
                a_off = pid_b * sA_b + t * sA_l + pid_o * sA_o + r_idx * sA_r
                A_t_fwd = tl.load(A_ptr + a_off, mask=valid, other=0.0)
                b_off = pid_b * sB_b + t * sB_l + pid_i * sB_i + kk_idx * sB_k
                B_t_fwd = tl.load(B_ptr + b_off, mask=valid, other=0.0)
                g_fwd = tl.load(gates_ptr + pid_b * sG_b + t * sG_l)
                cur_theta = (1.0 - g_fwd) * (A_t_fwd * B_t_fwd) + g_fwd * cur_theta

        # --- Pass 2: backward sweep through segment. Instead of issuing a
        # scalar atomic_add per timestep (which bottlenecks on atomic
        # contention — grad_A has I-way i-contention, grad_B has O-way
        # o-contention, and grad_gates has O*I-way (o,i)-contention; at this
        # repository's O=20, I=10 that is 10-, 20- and 200-way), we
        # accumulate each per-step gradient into a 2D (SEG, K_PAD) tile or
        # 1D (SEG,) tile, then issue ONE vectorized atomic_add per tile at
        # the end of the segment. This reduces atomic_add calls by SEG×
        # (the hardware still does SEG-wide atomic coalescing, but a single
        # call avoids per-step launch overhead and lets the compiler batch
        # better).
        gA_seg = tl.zeros([SEG, K_PAD], dtype=tl.float32)
        gB_seg = tl.zeros([SEG, K_PAD], dtype=tl.float32)
        gG_seg = tl.zeros([SEG], dtype=tl.float32)

        for k_in in range(SEG):
            t = t_start + (SEG - 1 - k_in)
            if t < L:
                a_off = pid_b * sA_b + t * sA_l + pid_o * sA_o + r_idx * sA_r
                A_t = tl.load(A_ptr + a_off, mask=valid, other=0.0)
                b_off = pid_b * sB_b + t * sB_l + pid_i * sB_i + kk_idx * sB_k
                B_t = tl.load(B_ptr + b_off, mask=valid, other=0.0)
                g = tl.load(gates_ptr + pid_b * sG_b + t * sG_l)

                # Extract θ_prev for this step from theta_hist.
                bwd_row = SEG - 1 - k_in
                row_sel_b = (seg_idx_1d == bwd_row)[:, None]            # (SEG, 1)
                row_sel = tl.where(row_sel_b, 1.0, 0.0)                 # (SEG, 1)
                theta_prev = tl.sum(row_sel * theta_hist, axis=0)       # (K_PAD,)

                delta = A_t * B_t
                coef = 1.0 - g

                # Per-step contributions.
                gA_elem = coef * grad_theta * B_t                       # (K_PAD,)
                gB_elem = coef * grad_theta * A_t                       # (K_PAD,)
                gG_scalar = tl.sum(
                    tl.where(valid, (theta_prev - delta) * grad_theta, 0.0)
                )                                                       # scalar

                # Scatter each scalar/vector into the corresponding row of
                # the segment-local tile, indexed by `bwd_row`. The final
                # atomic_add masks invalid K lanes, so we don't need to
                # pre-zero them here.
                row_mask_bwd = (seg_idx_1d == bwd_row)[:, None]         # (SEG, 1)
                gA_seg = tl.where(row_mask_bwd, gA_elem[None, :], gA_seg)
                gB_seg = tl.where(row_mask_bwd, gB_elem[None, :], gB_seg)
                row_mask_g = (seg_idx_1d == bwd_row)
                gG_seg = tl.where(row_mask_g, gG_scalar, gG_seg)

                # Propagate grad_θ backward through the gate.
                grad_theta = g * grad_theta

        # --- Batched atomic_add: one per target tensor per segment.
        t_range = t_start + seg_idx_1d                                  # (SEG,)
        t_range_valid = t_range < L                                     # (SEG,)

        # grad_A: (SEG, K_PAD) tile → addresses (b, t, o, r_idx).
        gA_addr = (
            pid_b * sgA_b
            + t_range[:, None] * sgA_l
            + pid_o * sgA_o
            + r_idx[None, :] * sgA_r
        )
        gA_mask = t_range_valid[:, None] & valid[None, :]
        tl.atomic_add(grad_A_ptr + gA_addr, gA_seg, mask=gA_mask)

        # grad_B: (SEG, K_PAD) tile → addresses (b, t, i, kk_idx).
        gB_addr = (
            pid_b * sgB_b
            + t_range[:, None] * sgB_l
            + pid_i * sgB_i
            + kk_idx[None, :] * sgB_k
        )
        gB_mask = t_range_valid[:, None] & valid[None, :]
        tl.atomic_add(grad_B_ptr + gB_addr, gB_seg, mask=gB_mask)

        # grad_gates: (SEG,) vector → addresses (b, t).
        gG_addr = pid_b * sgG_b + t_range * sgG_l
        tl.atomic_add(grad_gates_ptr + gG_addr, gG_seg, mask=t_range_valid)


def streaming_fwp_backward_triton(
    A: torch.Tensor,            # (B, L, O, R+1)
    B: torch.Tensor,            # (B, L, I, 2)
    gates: torch.Tensor,        # (B, L)
    grad_out: torch.Tensor,     # (B, O, I, R+1, 2)
    theta_ckpts: torch.Tensor,  # (B, N_CKPTS, O, I, R+1, 2)
):
    """Fused backward: reads `theta_ckpts` saved by the forward kernel,
    re-runs forward + backward within each SEG-sized segment entirely in
    registers, and writes grad_A/grad_B/grad_gates via `atomic_add`."""
    assert (
        A.is_cuda and B.is_cuda and gates.is_cuda
        and grad_out.is_cuda and theta_ckpts.is_cuda
    )
    assert (
        A.dtype == torch.float32 == B.dtype == gates.dtype
        == grad_out.dtype == theta_ckpts.dtype
    ), "backward requires fp32 inputs + grad_out + theta_ckpts"

    Bsz, L, O, Rp1 = A.shape
    _, _, I, two = B.shape
    assert two == 2 and gates.shape == (Bsz, L)
    assert theta_ckpts.shape[0] == Bsz and theta_ckpts.shape[2:] == (O, I, Rp1, 2)
    n_ckpts = theta_ckpts.shape[1]
    assert n_ckpts == _num_ckpts(L, _SEG), (
        f"theta_ckpts.shape[1]={n_ckpts} != expected {_num_ckpts(L, _SEG)}"
    )

    A = A.contiguous()
    B = B.contiguous()
    gates = gates.contiguous()
    grad_out = grad_out.contiguous()
    theta_ckpts = theta_ckpts.contiguous()

    # grad_A, grad_B, grad_gates accumulated via atomic_add across programs.
    grad_A = torch.zeros_like(A)
    grad_B = torch.zeros_like(B)
    grad_gates = torch.zeros_like(gates)

    k_dim = Rp1 * 2
    k_pad = _next_pow2(k_dim)

    grid = (Bsz, O, I)
    _streaming_fwp_backward_kernel[grid](
        A, B, gates, theta_ckpts, grad_out,
        grad_A, grad_B, grad_gates,
        A.stride(0), A.stride(1), A.stride(2), A.stride(3),
        B.stride(0), B.stride(1), B.stride(2), B.stride(3),
        gates.stride(0), gates.stride(1),
        theta_ckpts.stride(0), theta_ckpts.stride(1), theta_ckpts.stride(2),
        theta_ckpts.stride(3), theta_ckpts.stride(4), theta_ckpts.stride(5),
        grad_out.stride(0), grad_out.stride(1), grad_out.stride(2),
        grad_out.stride(3), grad_out.stride(4),
        grad_A.stride(0), grad_A.stride(1), grad_A.stride(2), grad_A.stride(3),
        grad_B.stride(0), grad_B.stride(1), grad_B.stride(2), grad_B.stride(3),
        grad_gates.stride(0), grad_gates.stride(1),
        L=L, RP1=Rp1, K_DIM=k_dim, K_PAD=k_pad,
        SEG=_SEG, N_CKPTS=n_ckpts,
    )

    return grad_A, grad_B, grad_gates
