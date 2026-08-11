# mypy: ignore-errors
# Copyright (c) 2026, Jiun-Cheng Jiang. All rights reserved.
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

# ---------------------------------------------------------------------------
# NOTICE (Apache-2.0 §4(b)) — this file is a MODIFIED copy.
#
# Original: https://github.com/Jim137/qkan (Apache-2.0),
#           Copyright (c) Jiun-Cheng Jiang.
# Modifications: Copyright 2026 Matthew Peng and contributors,
#           licensed under the same Apache-2.0 terms.
#
# Summary of changes made to the upstream file:
#   * per-sample batched ``theta`` carrying a leading batch axis
#     ``(B, out_dim, in_dim, reps+1, 2)``, with ``forward`` / ``forward_no_sum``
#     taking ``theta`` / ``base_weight`` as call arguments rather than reading
#     them from module state — required because the GQKAN-QKANFWP fast
#     programmer's ``theta`` is an input produced by the slow programmer;
#   * ``qkan.solver`` imports replaced by the repo-local ``fast_solver``;
#   * added a ``cutile`` batched-solver path;
#   * optional hardware backends guarded so the main training paths import
#     cleanly when those SDKs are absent.
# ---------------------------------------------------------------------------

"""
cuTile-fused kernels for QKAN with per-sample batched ``theta``.

This is a snapshot of upstream ``qkan/cutile_ops.py`` (Jim137/qkan), kept
in-repo and extended to accept ``theta`` with a leading batch axis
``(B, out_dim, in_dim, reps+1, 2)`` — required by the GQKAN-QKANFWP fast
programmer, whose ``theta`` is a per-sample input produced by the slow
programmer rather than a shared trainable parameter.

Only the ``pz_encoding`` ansatz is supported in this module. ``rpz`` and
``real`` fall back to the pure-PyTorch scalar-recurrence path in
``fast_solver.py``.

Key deltas from upstream ``_ct_pz_encoding_kernel``:
  * Every ``ct.gather(theta, …)`` takes ``b_offs`` as its leading index so
    each of ``BLOCK_B`` lanes reads its own per-sample theta. For an
    unbatched theta (``theta.shape[0] == 1``), the wrapper ``expand()``s
    with a stride-0 axis so the same kernel handles both cases without a
    dedicated code path.
  * Backward: ``grad_theta`` is written per-(b,o,i,l,k) via plain
    ``ct.scatter`` — no atomic needed, since for batched theta each
    program is the unique writer for its (b,o,i) slab.
  * Circuit, observable, and ``fast_measure`` semantics are unchanged.
"""

import math

import cuda.tile as ct  # type: ignore
import torch

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]


def _select_block_b(n_oi: int, batch: int) -> int:
    """Adaptive BLOCK_B selection — mirrors qkan._kernel_utils."""
    if n_oi >= 256 and batch >= 128:
        return 128
    if n_oi >= 256 and batch >= 64:
        return 64
    return 32


# ── pz_encoding forward kernel (batched theta) ────────────────────────────


@ct.kernel
def _ct_pz_encoding_kernel_batched(
    x,  # [Batch, In_Dim]
    theta,  # [Batch, Out_Dim, In_Dim, Reps+1, 2]  (stride-0 on dim 0 when unbatched)
    pw,  # [Out_Dim, In_Dim, Reps]
    pb,  # [Out_Dim, In_Dim, Reps]
    out,  # [Batch, Out_Dim, In_Dim]
    batch_size: ConstInt,
    in_dim: ConstInt,
    out_dim: ConstInt,
    reps: ConstInt,
    PREACTS_TRAINABLE: ConstBool,
    FAST_MEASURE: ConstBool,
    BLOCK_B: ConstInt,
):
    """
    Fused QKAN pz_encoding forward kernel with per-sample theta.

    Grid: (out_dim * in_dim, cdiv(batch, BLOCK_B), 1).
    Circuit: H|0> -> [Rz(t0) Ry(t1) Rz(enc)]×reps -> Rz(t0) Ry(t1) -> measure Z
    """
    pid_oi = ct.bid(0)
    pid_b = ct.bid(1)

    idx_i = pid_oi % in_dim
    idx_o = pid_oi // in_dim

    b_offs = pid_b * BLOCK_B + ct.arange(BLOCK_B, dtype=ct.int32)
    b_mask = b_offs < batch_size

    x_vals = ct.astype(
        ct.gather(x, (b_offs, idx_i), mask=b_mask, padding_value=0.0), ct.float32
    )

    INV_SQRT2 = 0.7071067811865476
    r0 = ct.full((BLOCK_B,), INV_SQRT2, dtype=ct.float32)
    i0 = ct.zeros((BLOCK_B,), dtype=ct.float32)
    r1 = ct.full((BLOCK_B,), INV_SQRT2, dtype=ct.float32)
    i1 = ct.zeros((BLOCK_B,), dtype=ct.float32)

    for layer in range(reps):
        # Rz(t0) — per-sample theta load
        t0 = ct.astype(
            ct.gather(theta, (b_offs, idx_o, idx_i, layer, 0), mask=b_mask, padding_value=0.0),
            ct.float32,
        )
        a = t0 * 0.5
        c = ct.cos(a)
        s = ct.sin(a)
        nr0 = r0 * c + i0 * s
        ni0 = i0 * c - r0 * s
        nr1 = r1 * c - i1 * s
        ni1 = i1 * c + r1 * s
        r0, i0, r1, i1 = nr0, ni0, nr1, ni1

        # Ry(t1)
        t1 = ct.astype(
            ct.gather(theta, (b_offs, idx_o, idx_i, layer, 1), mask=b_mask, padding_value=0.0),
            ct.float32,
        )
        a = t1 * 0.5
        c = ct.cos(a)
        s = ct.sin(a)
        nr0 = c * r0 - s * r1
        ni0 = c * i0 - s * i1
        nr1 = s * r0 + c * r1
        ni1 = s * i0 + c * i1
        r0, i0, r1, i1 = nr0, ni0, nr1, ni1

        # Rz(enc)
        enc = x_vals
        if PREACTS_TRAINABLE:
            w = ct.astype(ct.gather(pw, (idx_o, idx_i, layer)), ct.float32)
            b = ct.astype(ct.gather(pb, (idx_o, idx_i, layer)), ct.float32)
            enc = w * x_vals + b

        a = enc * 0.5
        c = ct.cos(a)
        s = ct.sin(a)
        nr0 = r0 * c + i0 * s
        ni0 = i0 * c - r0 * s
        nr1 = r1 * c - i1 * s
        ni1 = i1 * c + r1 * s
        r0, i0, r1, i1 = nr0, ni0, nr1, ni1

    # Final Rz(t0), Ry(t1)
    t0 = ct.astype(
        ct.gather(theta, (b_offs, idx_o, idx_i, reps, 0), mask=b_mask, padding_value=0.0),
        ct.float32,
    )
    t1 = ct.astype(
        ct.gather(theta, (b_offs, idx_o, idx_i, reps, 1), mask=b_mask, padding_value=0.0),
        ct.float32,
    )

    a = t0 * 0.5
    c = ct.cos(a)
    s = ct.sin(a)
    nr0 = r0 * c + i0 * s
    ni0 = i0 * c - r0 * s
    nr1 = r1 * c - i1 * s
    ni1 = i1 * c + r1 * s
    r0, i0, r1, i1 = nr0, ni0, nr1, ni1

    a = t1 * 0.5
    c = ct.cos(a)
    s = ct.sin(a)
    nr0 = c * r0 - s * r1
    ni0 = c * i0 - s * i1
    nr1 = s * r0 + c * r1
    ni1 = s * i0 + c * i1
    r0, i0, r1, i1 = nr0, ni0, nr1, ni1

    if FAST_MEASURE:
        result = ct.sqrt(r0 * r0 + i0 * i0) - ct.sqrt(r1 * r1 + i1 * i1)
    else:
        result = (r0 * r0 + i0 * i0) - (r1 * r1 + i1 * i1)

    ct.scatter(out, (b_offs, idx_o, idx_i), result, mask=b_mask)


def cutile_pz_forward_batched(
    x: torch.Tensor,
    theta: torch.Tensor,
    preacts_w: torch.Tensor,
    preacts_b: torch.Tensor,
    preacts_trainable: bool,
    fast_measure: bool = True,
) -> torch.Tensor:
    """
    Launch the batched-theta pz_encoding forward kernel.

    Args
    ----
    x : (batch, in_dim) fp32 on CUDA
    theta : (batch, out_dim, in_dim, reps+1, 2) fp32 on CUDA
            (or (1, out_dim, in_dim, reps+1, 2) broadcasted via expand())
    preacts_w, preacts_b : (out_dim, in_dim, reps) fp32

    Returns
    -------
    (batch, out_dim, in_dim) fp32
    """
    assert x.is_cuda and theta.is_cuda, "cutile kernels require CUDA tensors"
    assert x.dim() == 2 and theta.dim() == 5, (
        f"Expected x.dim()==2, theta.dim()==5, got {x.dim()} and {theta.dim()}"
    )

    batch, in_dim = x.shape
    _, out_dim, _, reps_plus_1, _ = theta.shape
    reps = reps_plus_1 - 1

    x = x.to(torch.float32).contiguous()
    # If theta is (1, O, I, R+1, 2) we expand with stride-0 so the kernel's
    # (b_offs, ...) gather still works for the unbatched case. .expand() keeps
    # stride-0 on the batch axis without allocation.
    if theta.shape[0] == 1 and batch != 1:
        theta = theta.to(torch.float32).expand(batch, -1, -1, -1, -1).contiguous()
        # NOTE: .contiguous() here *does* allocate. This is intentional — the
        # gather kernel reads theta by (b_offs, o, i, l, k) and a stride-0 gather
        # is not correctly modelled by cuTile's scheduler. For the training
        # case theta is always already batched, so this branch is only hit in
        # tests / parity checks.
    else:
        theta = theta.to(torch.float32).contiguous()

    output = torch.empty(batch, out_dim, in_dim, device=x.device, dtype=torch.float32)

    n_oi = out_dim * in_dim
    BLOCK_B = _select_block_b(n_oi, batch)
    grid = (n_oi, math.ceil(batch / BLOCK_B), 1)

    if preacts_trainable:
        preacts_w = preacts_w.to(torch.float32).contiguous()
        preacts_b = preacts_b.to(torch.float32).contiguous()
    else:
        # Not used, but cuTile still passes the tensor pointers.
        preacts_w = preacts_w.to(torch.float32).contiguous() if preacts_w.numel() else \
            torch.empty(out_dim, in_dim, max(reps, 1), device=x.device, dtype=torch.float32)
        preacts_b = preacts_b.to(torch.float32).contiguous() if preacts_b.numel() else \
            torch.empty(out_dim, in_dim, max(reps, 1), device=x.device, dtype=torch.float32)

    ct.launch(
        torch.cuda.current_stream(),
        grid,
        _ct_pz_encoding_kernel_batched,
        (
            x,
            theta,
            preacts_w,
            preacts_b,
            output,
            batch,
            in_dim,
            out_dim,
            reps,
            preacts_trainable,
            fast_measure,
            BLOCK_B,
        ),
    )

    return output


# ── pz_encoding backward kernel (batched theta) ───────────────────────────


@ct.kernel(occupancy=4)
def _ct_pz_encoding_backward_kernel_batched(
    x,  # [Batch, In_Dim]
    theta,  # [Batch, Out_Dim, In_Dim, Reps+1, 2]
    pw,  # [Out_Dim, In_Dim, Reps]
    pb,  # [Out_Dim, In_Dim, Reps]
    grad_out,  # [Batch, Out_Dim, In_Dim]
    states,  # [N_Programs, N_States, 4, BLOCK_B] — fp32
    trig_cache,  # [N_Programs, 2*Reps+2, BLOCK_B, 2] — fp32 per-sample cos/sin
    grad_theta,  # [Batch, Out_Dim, In_Dim, Reps+1, 2]  (plain scatter, no atomic)
    grad_x,  # [Batch, In_Dim]
    grad_pw,  # [Out_Dim, In_Dim, Reps]
    grad_pb,  # [Out_Dim, In_Dim, Reps]
    batch_size: ConstInt,
    in_dim: ConstInt,
    out_dim: ConstInt,
    reps: ConstInt,
    n_b_blocks: ConstInt,
    PREACTS_TRAINABLE: ConstBool,
    FAST_MEASURE: ConstBool,
    BLOCK_B: ConstInt,
):
    """Backward kernel for pz_encoding with batched theta.

    Per-sample theta means ``grad_theta[b, o, i, l, k]`` is written by exactly
    one program — use plain scatter, not atomic_add. ``grad_x[b, i]`` is still
    summed across ``o``, so it still needs atomic_add.
    """
    pid_oi = ct.bid(0)
    pid_b = ct.bid(1)

    idx_i = pid_oi % in_dim
    idx_o = pid_oi // in_dim

    b_offs = pid_b * BLOCK_B + ct.arange(BLOCK_B, dtype=ct.int32)
    b_mask = b_offs < batch_size
    b_range = ct.arange(BLOCK_B, dtype=ct.int32)

    x_vals = ct.astype(
        ct.gather(x, (b_offs, idx_i), mask=b_mask, padding_value=0.0), ct.float32
    )

    pi = pid_oi * n_b_blocks + pid_b

    def _save4(sidx, v0, v1, v2, v3):
        ct.scatter(states, (pi, sidx, 0, b_range), v0, mask=b_mask)
        ct.scatter(states, (pi, sidx, 1, b_range), v1, mask=b_mask)
        ct.scatter(states, (pi, sidx, 2, b_range), v2, mask=b_mask)
        ct.scatter(states, (pi, sidx, 3, b_range), v3, mask=b_mask)

    def _load4(sidx):
        return (
            ct.gather(states, (pi, sidx, 0, b_range), mask=b_mask, padding_value=0.0),
            ct.gather(states, (pi, sidx, 1, b_range), mask=b_mask, padding_value=0.0),
            ct.gather(states, (pi, sidx, 2, b_range), mask=b_mask, padding_value=0.0),
            ct.gather(states, (pi, sidx, 3, b_range), mask=b_mask, padding_value=0.0),
        )

    def _save_trig(tidx, c_vec, s_vec):
        """Per-sample trig cache — scatter full BLOCK_B vector."""
        ct.scatter(trig_cache, (pi, tidx, b_range, 0), c_vec, mask=b_mask)
        ct.scatter(trig_cache, (pi, tidx, b_range, 1), s_vec, mask=b_mask)

    def _load_trig(tidx):
        return (
            ct.gather(trig_cache, (pi, tidx, b_range, 0), mask=b_mask, padding_value=0.0),
            ct.gather(trig_cache, (pi, tidx, b_range, 1), mask=b_mask, padding_value=0.0),
        )

    # ── Phase 1: Forward recompute, saving states + trig cache ──
    INV_SQRT2 = 0.7071067811865476
    r0 = ct.full((BLOCK_B,), INV_SQRT2, dtype=ct.float32)
    i0 = ct.zeros((BLOCK_B,), dtype=ct.float32)
    r1 = ct.full((BLOCK_B,), INV_SQRT2, dtype=ct.float32)
    i1 = ct.zeros((BLOCK_B,), dtype=ct.float32)

    _save4(0, r0, i0, r1, i1)

    state_idx = 1
    trig_idx = 0
    for layer in range(reps):
        # Rz(t0) — per-sample theta; cache per-sample trig
        t0 = ct.astype(
            ct.gather(theta, (b_offs, idx_o, idx_i, layer, 0), mask=b_mask, padding_value=0.0),
            ct.float32,
        )
        a = t0 * 0.5
        c = ct.cos(a)
        s = ct.sin(a)
        _save_trig(trig_idx, c, s)
        trig_idx = trig_idx + 1
        nr0 = r0 * c + i0 * s
        ni0 = i0 * c - r0 * s
        nr1 = r1 * c - i1 * s
        ni1 = i1 * c + r1 * s
        r0, i0, r1, i1 = nr0, ni0, nr1, ni1
        _save4(state_idx, r0, i0, r1, i1)
        state_idx = state_idx + 1

        # Ry(t1)
        t1 = ct.astype(
            ct.gather(theta, (b_offs, idx_o, idx_i, layer, 1), mask=b_mask, padding_value=0.0),
            ct.float32,
        )
        a = t1 * 0.5
        c = ct.cos(a)
        s = ct.sin(a)
        _save_trig(trig_idx, c, s)
        trig_idx = trig_idx + 1
        nr0 = c * r0 - s * r1
        ni0 = c * i0 - s * i1
        nr1 = s * r0 + c * r1
        ni1 = s * i0 + c * i1
        r0, i0, r1, i1 = nr0, ni0, nr1, ni1
        _save4(state_idx, r0, i0, r1, i1)
        state_idx = state_idx + 1

        # Rz(enc)
        enc = x_vals
        if PREACTS_TRAINABLE:
            w = ct.astype(ct.gather(pw, (idx_o, idx_i, layer)), ct.float32)
            b = ct.astype(ct.gather(pb, (idx_o, idx_i, layer)), ct.float32)
            enc = w * x_vals + b

        a = enc * 0.5
        c = ct.cos(a)
        s = ct.sin(a)
        nr0 = r0 * c + i0 * s
        ni0 = i0 * c - r0 * s
        nr1 = r1 * c - i1 * s
        ni1 = i1 * c + r1 * s
        r0, i0, r1, i1 = nr0, ni0, nr1, ni1
        _save4(state_idx, r0, i0, r1, i1)
        state_idx = state_idx + 1

    # Final Rz(t0)
    t0 = ct.astype(
        ct.gather(theta, (b_offs, idx_o, idx_i, reps, 0), mask=b_mask, padding_value=0.0),
        ct.float32,
    )
    a = t0 * 0.5
    c = ct.cos(a)
    s = ct.sin(a)
    _save_trig(trig_idx, c, s)
    trig_idx = trig_idx + 1
    nr0 = r0 * c + i0 * s
    ni0 = i0 * c - r0 * s
    nr1 = r1 * c - i1 * s
    ni1 = i1 * c + r1 * s
    r0, i0, r1, i1 = nr0, ni0, nr1, ni1
    _save4(state_idx, r0, i0, r1, i1)
    state_idx = state_idx + 1

    # Final Ry(t1)
    t1 = ct.astype(
        ct.gather(theta, (b_offs, idx_o, idx_i, reps, 1), mask=b_mask, padding_value=0.0),
        ct.float32,
    )
    a = t1 * 0.5
    c = ct.cos(a)
    s = ct.sin(a)
    _save_trig(trig_idx, c, s)
    trig_idx = trig_idx + 1
    nr0 = c * r0 - s * r1
    ni0 = c * i0 - s * i1
    nr1 = s * r0 + c * r1
    ni1 = s * i0 + c * i1
    r0, i0, r1, i1 = nr0, ni0, nr1, ni1

    # ── Phase 2: Measurement gradient ──
    go = ct.astype(
        ct.gather(grad_out, (b_offs, idx_o, idx_i), mask=b_mask, padding_value=0.0),
        ct.float32,
    )

    if FAST_MEASURE:
        alpha_norm = ct.sqrt(r0 * r0 + i0 * i0)
        beta_norm = ct.sqrt(r1 * r1 + i1 * i1)
        inv_alpha = ct.where(alpha_norm > 1e-30, 1.0 / alpha_norm, 0.0)
        inv_beta = ct.where(beta_norm > 1e-30, 1.0 / beta_norm, 0.0)
        ar0 = go * r0 * inv_alpha
        ai0 = go * i0 * inv_alpha
        ar1 = -go * r1 * inv_beta
        ai1 = -go * i1 * inv_beta
    else:
        ar0 = 2.0 * go * r0
        ai0 = 2.0 * go * i0
        ar1 = -2.0 * go * r1
        ai1 = -2.0 * go * i1

    # ── Phase 3: Backward sweep ──
    grad_x_local = ct.zeros((BLOCK_B,), dtype=ct.float32)

    # Backward through final Ry(t1_final)
    trig_idx = trig_idx - 1
    state_idx = state_idx - 1
    sr0, si0, sr1, si1 = _load4(state_idx)
    c, s = _load_trig(trig_idx)

    grad_t1_vec = 0.5 * (
        ar0 * (-s * sr0 - c * sr1)
        + ai0 * (-s * si0 - c * si1)
        + ar1 * (c * sr0 - s * sr1)
        + ai1 * (c * si0 - s * si1)
    )
    # Per-sample grad_theta write — plain scatter, NOT atomic_add.
    ct.scatter(
        grad_theta,
        (b_offs, idx_o, idx_i, reps, 1),
        grad_t1_vec,
        mask=b_mask,
    )

    nar0 = c * ar0 + s * ar1
    nai0 = c * ai0 + s * ai1
    nar1 = -s * ar0 + c * ar1
    nai1 = -s * ai0 + c * ai1
    ar0, ai0, ar1, ai1 = nar0, nai0, nar1, nai1

    # Backward through final Rz(t0_final)
    trig_idx = trig_idx - 1
    state_idx = state_idx - 1
    sr0, si0, sr1, si1 = _load4(state_idx)
    c, s = _load_trig(trig_idx)

    grad_t0_vec = 0.5 * (
        -s * (ar0 * sr0 + ai0 * si0 + ar1 * sr1 + ai1 * si1)
        + c * (ar0 * si0 - ai0 * sr0 - ar1 * si1 + ai1 * sr1)
    )
    ct.scatter(
        grad_theta,
        (b_offs, idx_o, idx_i, reps, 0),
        grad_t0_vec,
        mask=b_mask,
    )

    nar0 = c * ar0 - s * ai0
    nai0 = s * ar0 + c * ai0
    nar1 = c * ar1 + s * ai1
    nai1 = -s * ar1 + c * ai1
    ar0, ai0, ar1, ai1 = nar0, nai0, nar1, nai1

    for _ri in range(reps):
        layer = reps - 1 - _ri

        # Backward through Rz(enc)
        state_idx = state_idx - 1
        sr0, si0, sr1, si1 = _load4(state_idx)

        enc = x_vals
        if PREACTS_TRAINABLE:
            w = ct.astype(ct.gather(pw, (idx_o, idx_i, layer)), ct.float32)
            b = ct.astype(ct.gather(pb, (idx_o, idx_i, layer)), ct.float32)
            enc = w * x_vals + b

        ae = enc * 0.5
        ce = ct.cos(ae)
        se = ct.sin(ae)

        grad_enc = 0.5 * (
            -se * (ar0 * sr0 + ai0 * si0 + ar1 * sr1 + ai1 * si1)
            + ce * (ar0 * si0 - ai0 * sr0 - ar1 * si1 + ai1 * sr1)
        )

        if PREACTS_TRAINABLE:
            ct.atomic_add(
                grad_pw,
                (idx_o, idx_i, layer),
                ct.sum(ct.where(b_mask, grad_enc * x_vals, 0.0)),
                memory_order=ct.MemoryOrder.RELAXED,
            )
            ct.atomic_add(
                grad_pb,
                (idx_o, idx_i, layer),
                ct.sum(ct.where(b_mask, grad_enc, 0.0)),
                memory_order=ct.MemoryOrder.RELAXED,
            )
            grad_x_local = grad_x_local + grad_enc * w
        else:
            grad_x_local = grad_x_local + grad_enc

        nar0 = ce * ar0 - se * ai0
        nai0 = se * ar0 + ce * ai0
        nar1 = ce * ar1 + se * ai1
        nai1 = -se * ar1 + ce * ai1
        ar0, ai0, ar1, ai1 = nar0, nai0, nar1, nai1

        # Backward through Ry(t1)
        trig_idx = trig_idx - 1
        state_idx = state_idx - 1
        sr0, si0, sr1, si1 = _load4(state_idx)
        ct1, st1 = _load_trig(trig_idx)

        grad_t1_vec = 0.5 * (
            ar0 * (-st1 * sr0 - ct1 * sr1)
            + ai0 * (-st1 * si0 - ct1 * si1)
            + ar1 * (ct1 * sr0 - st1 * sr1)
            + ai1 * (ct1 * si0 - st1 * si1)
        )
        ct.scatter(
            grad_theta,
            (b_offs, idx_o, idx_i, layer, 1),
            grad_t1_vec,
            mask=b_mask,
        )

        nar0 = ct1 * ar0 + st1 * ar1
        nai0 = ct1 * ai0 + st1 * ai1
        nar1 = -st1 * ar0 + ct1 * ar1
        nai1 = -st1 * ai0 + ct1 * ai1
        ar0, ai0, ar1, ai1 = nar0, nai0, nar1, nai1

        # Backward through Rz(t0)
        trig_idx = trig_idx - 1
        state_idx = state_idx - 1
        sr0, si0, sr1, si1 = _load4(state_idx)
        ct0, st0 = _load_trig(trig_idx)

        grad_t0_vec = 0.5 * (
            -st0 * (ar0 * sr0 + ai0 * si0 + ar1 * sr1 + ai1 * si1)
            + ct0 * (ar0 * si0 - ai0 * sr0 - ar1 * si1 + ai1 * sr1)
        )
        ct.scatter(
            grad_theta,
            (b_offs, idx_o, idx_i, layer, 0),
            grad_t0_vec,
            mask=b_mask,
        )

        nar0 = ct0 * ar0 - st0 * ai0
        nai0 = st0 * ar0 + ct0 * ai0
        nar1 = ct0 * ar1 + st0 * ai1
        nai1 = -st0 * ar1 + ct0 * ai1
        ar0, ai0, ar1, ai1 = nar0, nai0, nar1, nai1

    # grad_x gets contributions from all (o, i) pairs — needs atomic
    ct.atomic_add(
        grad_x,
        (b_offs, idx_i),
        ct.where(b_mask, grad_x_local, 0.0),
        memory_order=ct.MemoryOrder.RELAXED,
    )


def cutile_pz_backward_batched(
    x: torch.Tensor,
    theta: torch.Tensor,
    pw: torch.Tensor,
    pb: torch.Tensor,
    grad_output: torch.Tensor,
    preacts_trainable: bool,
    fast_measure: bool,
) -> tuple:
    """Launch the batched-theta pz_encoding backward kernel.

    Returns (grad_x, grad_theta, grad_pw, grad_pb). grad_theta has the
    same leading batch dim as the input ``theta``.
    """
    assert theta.dim() == 5, f"Expected theta.dim()==5, got {theta.dim()}"
    batch, in_dim = x.shape
    _, out_dim, _, reps_plus_1, _ = theta.shape
    reps = reps_plus_1 - 1

    x = x.to(torch.float32).contiguous()

    # Expand unbatched theta to a contiguous batched tensor for the scatter path
    # (the backward writes per-sample gradients — a stride-0 grad_theta would
    # over-write the same memory from multiple programs).
    theta_was_broadcast = theta.shape[0] == 1 and batch != 1
    if theta_was_broadcast:
        theta = theta.to(torch.float32).expand(batch, -1, -1, -1, -1).contiguous()
    else:
        theta = theta.to(torch.float32).contiguous()

    grad_output = grad_output.to(torch.float32).contiguous()

    n_oi = out_dim * in_dim
    BLOCK_B = _select_block_b(n_oi, batch)
    n_states = 3 * reps + 3
    n_b_blocks = math.ceil(batch / BLOCK_B)
    n_programs = n_oi * n_b_blocks

    states = torch.empty(
        n_programs, n_states, 4, BLOCK_B, device=x.device, dtype=torch.float32
    )
    n_trig = 2 * reps + 2
    trig_cache = torch.empty(
        n_programs, n_trig, BLOCK_B, 2, device=x.device, dtype=torch.float32
    )

    grad_theta = torch.zeros(
        batch, out_dim, in_dim, reps + 1, 2, device=x.device, dtype=torch.float32
    )
    grad_x = torch.zeros(batch, in_dim, device=x.device, dtype=torch.float32)

    if preacts_trainable:
        pw = pw.to(torch.float32).contiguous()
        pb = pb.to(torch.float32).contiguous()
        grad_pw = torch.zeros(pw.shape, device=x.device, dtype=torch.float32)
        grad_pb = torch.zeros(pb.shape, device=x.device, dtype=torch.float32)
    else:
        if pw.numel() == 0 or pw.dim() != 3:
            pw = torch.empty(out_dim, in_dim, max(reps, 1), device=x.device, dtype=torch.float32)
            pb = torch.empty(out_dim, in_dim, max(reps, 1), device=x.device, dtype=torch.float32)
        else:
            pw = pw.to(torch.float32).contiguous()
            pb = pb.to(torch.float32).contiguous()
        grad_pw = torch.zeros(1, device=x.device, dtype=torch.float32)
        grad_pb = torch.zeros(1, device=x.device, dtype=torch.float32)

    grid = (n_oi, n_b_blocks, 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        _ct_pz_encoding_backward_kernel_batched,
        (
            x,
            theta,
            pw,
            pb,
            grad_output,
            states,
            trig_cache,
            grad_theta,
            grad_x,
            grad_pw,
            grad_pb,
            batch,
            in_dim,
            out_dim,
            reps,
            n_b_blocks,
            preacts_trainable,
            fast_measure,
            BLOCK_B,
        ),
    )

    # If the input was broadcast from a 1-batch theta, sum back to that shape
    # so upstream's accumulator pattern (theta shared across batch) still works.
    if theta_was_broadcast:
        grad_theta = grad_theta.sum(dim=0, keepdim=True)

    return (
        grad_x,
        grad_theta,
        grad_pw if preacts_trainable else None,
        grad_pb if preacts_trainable else None,
    )
