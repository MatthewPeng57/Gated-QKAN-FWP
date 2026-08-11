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

"""Batched-theta CuTe kernels for the QKAN fast-programmer readout (pz forward + backward).

Loads ``csrc/cute_batched_kernels.cu`` through ``torch.utils.cpp_extension.load``.
The first call JIT-compiles the kernel, which takes roughly 60 seconds; the build
is cached afterwards and subsequent loads are near-instant.

Prerequisites, in the shell that first triggers the build:

  * an NVIDIA GPU and a working CUDA toolchain — ``nvcc`` on ``PATH``, or
    ``CUDA_HOME`` pointing at the CUDA installation;
  * ``ninja`` installed and on ``PATH`` (``pip install ninja``);
  * ``CUTLASS_PATH`` pointing at a CUTLASS checkout — this module appends
    ``include/`` to it, so point it at the repository root::

        git clone --depth 1 https://github.com/NVIDIA/cutlass
        export CUTLASS_PATH="$PWD/cutlass"

  * optionally ``TORCH_EXTENSIONS_DIR`` to control where the build is cached.

This backend is opt-in (``--fast_solver cute``); the default ``flash`` path has
no such build requirement.
"""
import functools
import os

import torch
from torch.utils.cpp_extension import load


_CUTLASS_HINT = (
    "CUTLASS_PATH is not set — the CuTe backend needs the CUTLASS headers. "
    "Get them with:\n"
    "    git clone --depth 1 https://github.com/NVIDIA/cutlass\n"
    '    export CUTLASS_PATH="$PWD/cutlass"\n'
    "(this module appends 'include/' to CUTLASS_PATH). A CUDA toolchain "
    "(nvcc on PATH or CUDA_HOME set) and ninja are also required."
)


def cute_unavailable_reason():
    """Return ``None`` if the CuTe backend looks usable, else a reason string.

    Deliberately cheap — it inspects the environment only and never triggers the
    JIT build, so callers can pre-flight the backend and fail with a clear
    message instead of surfacing an error from inside ``torch.autograd``.
    A ``None`` result is necessary, not sufficient: the build itself can still
    fail, and that error is raised by :func:`_mod`.
    """
    if not torch.cuda.is_available():
        return "no CUDA device is available (the CuTe backend is CUDA-only)"
    cutlass = os.environ.get("CUTLASS_PATH")
    if not cutlass:
        return _CUTLASS_HINT
    if not os.path.isdir(os.path.join(cutlass, "include")):
        return (
            f"CUTLASS_PATH={cutlass!r} has no 'include/' subdirectory — point it "
            "at the root of a CUTLASS checkout."
        )
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(here, "csrc", "cute_batched_kernels.cu")
    if not os.path.isfile(src):
        return f"kernel source is missing: {src}"
    return None


@functools.lru_cache(maxsize=1)
def _mod():
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(here, "csrc", "cute_batched_kernels.cu")
    reason = cute_unavailable_reason()
    if reason is not None:
        raise RuntimeError(
            f"{reason}\nUse --fast_solver flash to avoid the CuTe backend entirely."
        )
    cutlass = os.environ.get("CUTLASS_PATH")
    inc = os.path.join(cutlass, "include")
    return load(
        name="cute_batched",
        sources=[src],
        extra_include_paths=[inc],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3", "-std=c++17", "--expt-relaxed-constexpr",
            "--expt-extended-lambda", "--use_fast_math",
        ],
        verbose=True,
    )


def cute_pz_forward_batched(x, theta, pw=None, pb=None, reps=None,
                            preacts_trainable=False, fast_measure=True):
    """x: (B, in_dim); theta: (B, out_dim, in_dim, reps+1, 2). Returns (B, out_dim, in_dim)."""
    x = x.to(torch.float32).contiguous()
    theta = theta.to(torch.float32).contiguous()
    dev = x.device
    if pw is None:
        pw = torch.zeros(1, device=dev, dtype=torch.float32)
    if pb is None:
        pb = torch.zeros(1, device=dev, dtype=torch.float32)
    pw = pw.to(torch.float32).contiguous()
    pb = pb.to(torch.float32).contiguous()
    return _mod().pz_forward_batched(
        x, theta, pw, pb, bool(preacts_trainable), bool(fast_measure)
    )


class _CuteBatchedPZ(torch.autograd.Function):
    """Autograd boundary over the batched pz CuTe fwd/bwd kernels.

    The backward RECONSTRUCTS the circuit states via the reversible rev_* sweep
    (no forward state is buffered), so it only needs the inputs — we save
    (x, theta, pw, pb) and the two mode flags. Inputs are assumed already
    float32 + contiguous (prepped by `cute_pz_batched`), so the float32 grads
    returned here match the tensors passed to `.apply()`.
    """

    @staticmethod
    def forward(ctx, x, theta, pw, pb, preacts_trainable, fast_measure):
        out = _mod().pz_forward_batched(
            x, theta, pw, pb, bool(preacts_trainable), bool(fast_measure)
        )
        ctx.save_for_backward(x, theta, pw, pb)
        ctx.preacts_trainable = bool(preacts_trainable)
        ctx.fast_measure = bool(fast_measure)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, theta, pw, pb = ctx.saved_tensors
        gx, gth, gpw, gpb = _mod().pz_backward_batched(
            x, theta, pw, pb, grad_out.to(torch.float32).contiguous(),
            ctx.preacts_trainable, ctx.fast_measure,
        )
        if not ctx.preacts_trainable:  # kernel returns undefined tensors → force None
            gpw = gpb = None
        # forward inputs: (x, theta, pw, pb, preacts_trainable, fast_measure)
        return gx, gth, gpw, gpb, None, None


def cute_pz_batched(x, theta, pw=None, pb=None, reps=None,
                    preacts_trainable=False, fast_measure=True):
    """Differentiable batched-theta pz readout (autograd-wrapped CuTe fwd/bwd).

    x: (B, in_dim); theta: (B, out_dim, in_dim, reps+1, 2); returns (B, out_dim, in_dim).
    `reps` is optional/validated only — the kernel infers it from theta's shape. On
    `preacts_trainable=True`, pw/pb are (out_dim, in_dim, reps) and receive grads.
    """
    x = x.to(torch.float32).contiguous()
    theta = theta.to(torch.float32).contiguous()
    dev = x.device
    if pw is None:
        pw = torch.zeros(1, device=dev, dtype=torch.float32)
    if pb is None:
        pb = torch.zeros(1, device=dev, dtype=torch.float32)
    pw = pw.to(torch.float32).contiguous()
    pb = pb.to(torch.float32).contiguous()
    if reps is not None and theta.shape[-2] - 1 != reps:
        raise ValueError(
            f"reps={reps} inconsistent with theta shape {tuple(theta.shape)} "
            f"(expected theta.shape[-2] == reps+1 == {reps + 1})"
        )
    return _CuteBatchedPZ.apply(
        x, theta, pw, pb, bool(preacts_trainable), bool(fast_measure)
    )
