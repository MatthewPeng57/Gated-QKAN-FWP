# Copyright (c) 2024, Jiun-Cheng Jiang. All rights reserved.
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
# Modifications: Copyright 2026 Kuo-Chung Peng and Samuel Yen-Chi Chen,
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
QKAN layer simulating solver

This module provides a solver for quantum neural networks using PyTorch, PennyLane,
Triton-accelerated kernels, or cuQuantum tensor network contraction.
"""

import math

import numpy as np
import torch

from qkan.torch_qc import StateVector, TorchGates

# cuQuantum / opt_einsum availability for cutn_solver
try:
    from cuquantum.tensornet import contract_path as _cutn_contract_path  # type: ignore

    _CUTN_AVAILABLE = True
except ImportError:
    _CUTN_AVAILABLE = False

try:
    from opt_einsum import contract_path as _oe_contract_path  # type: ignore

    _OE_AVAILABLE = True
except ImportError:
    _OE_AVAILABLE = False

# Triton fused kernels for flash solver
try:
    from .fast_fused_ops import (
        triton_pz_backward,
        triton_pz_forward,
        triton_real_backward,
        triton_real_forward,
        triton_rpz_backward,
        triton_rpz_forward,
    )

    _FLASH_AVAILABLE = True
except ImportError:
    _FLASH_AVAILABLE = False

_INV_SQRT2 = math.sqrt(2.0) / 2.0


def qml_solver(x: torch.Tensor, theta: torch.Tensor, reps: int, **kwargs):
    """
    Single-qubit data reuploading circuit using PennyLane.

    Args
    ----
        x : torch.Tensor
            shape: (batch_size, in_dim)
        theta : torch.Tensor
            shape: (reps, 2)
        reps : int
        qml_device : str
            default: "default.qubit"
    """
    import pennylane as qml  # type: ignore

    qml_device: str = kwargs.get("qml_device", "default.qubit")
    dev = qml.device(qml_device, wires=1)

    @qml.qnode(dev, interface="torch")
    def circuit(x: torch.Tensor, theta: torch.Tensor):
        """
        Args
        ----
            x : torch.Tensor
                shape: (batch_size, in_dim)
            theta : torch.Tensor
                shape: (reps, 2)
        """
        qml.RY(np.pi / 2, wires=0)
        for l in range(reps):
            qml.RZ(theta[l, 0], wires=0)
            qml.RY(theta[l, 1], wires=0)
            qml.RZ(x, wires=0)
        qml.RZ(theta[reps, 0], wires=0)
        qml.RY(theta[reps, 1], wires=0)
        return qml.expval(qml.PauliZ(0))

    return circuit(x, theta)


def torch_exact_solver(
    x: torch.Tensor,
    theta: torch.Tensor,
    preacts_weight: torch.Tensor,
    preacts_bias: torch.Tensor,
    reps: int,
    **kwargs,
) -> torch.Tensor:
    """
    Single-qubit data reuploading circuit.

    Args
    ----
        x : torch.Tensor
            shape: (batch_size, in_dim)
        theta : torch.Tensor
            shape: (*group, reps, 2)
        preacts_weight : torch.Tensor
            shape: (*group, reps)
        preacts_bias : torch.Tensor
            shape: (*group, reps)
        reps : int
        ansatz : str
            options: ["pz_encoding", "px_encoding"], default: "pz_encoding"
        n_group : int
            number of neurons in a group, default: in_dim of x

    Returns
    -------
        torch.Tensor
            shape: (batch_size, out_dim, in_dim)
    """
    batch, in_dim = x.shape
    device = x.device
    ansatz = kwargs.get("ansatz", "pz_encoding")
    # group = kwargs.get("group", in_dim)
    preacts_trainable = kwargs.get("preacts_trainable", False)
    fast_measure = kwargs.get("fast_measure", True)
    out_dim: int = kwargs.get("out_dim", in_dim)
    dtype = kwargs.get("dtype", torch.complex64)

    if len(theta.shape) != 4:
        theta = theta.unsqueeze(0)
    if theta.shape[1] != in_dim:
        repeat_out = out_dim
        repeat_in = in_dim // theta.shape[1] + 1
        theta = theta.repeat(repeat_out, repeat_in, 1, 1)[:, :in_dim, :, :]
    # rpz_encoding always needs encoded_x (with bias), even when preacts_trainable=False
    _needs_encoded_x = preacts_trainable or ansatz in ("rpz_encoding", "rpz")
    if _needs_encoded_x:
        if len(preacts_weight.shape) != 3:
            preacts_weight = preacts_weight.unsqueeze(0)
            preacts_bias = preacts_bias.unsqueeze(0)
        if preacts_weight.shape[1] != in_dim:
            repeat_out = out_dim
            repeat_in = in_dim // preacts_weight.shape[1] + 1
            preacts_weight = preacts_weight.repeat(repeat_out, repeat_in, 1)[
                :, :in_dim, :
            ]
            preacts_bias = preacts_bias.repeat(repeat_out, repeat_in, 1)[:, :in_dim, :]
        encoded_x = torch.einsum("oir,bi->boir", preacts_weight, x).add(preacts_bias)
        # encoded_x shape: (batch_size, out_dim, in_dim, reps)

    def _pz_encoding(theta: torch.Tensor):
        """
        Args
        ----
            theta : torch.Tensor
                shape: (*group, reps, 2)
        """
        psi = StateVector(
            x.shape[0],
            theta.shape[0],
            theta.shape[1],
            device=device,
            dtype=dtype,
        )  # psi.state: torch.Tensor, shape: (batch_size, out_dim, in_dim, 2)
        psi.h()
        if not preacts_trainable:
            rug = TorchGates.rz_gate(x, dtype=dtype)
        for l in range(reps):
            psi.rz(theta[:, :, l, 0])
            psi.ry(theta[:, :, l, 1])
            if not preacts_trainable:
                psi.state = torch.einsum("mnbi,boin->boim", rug, psi.state)
            else:
                psi.state = torch.einsum(
                    "mnboi,boin->boim",
                    TorchGates.rz_gate(encoded_x[:, :, :, l], dtype=dtype),
                    psi.state,
                )

        psi.rz(theta[:, :, reps, 0])
        psi.ry(theta[:, :, reps, 1])
        return psi.measure_z(fast_measure)  # shape: (batch_size, out_dim, in_dim)

    def _rpz_encoding(theta: torch.Tensor):
        """
        Args
        ----
            theta : torch.Tensor
                shape: (*group, reps, 2)
        """
        psi = StateVector(
            x.shape[0],
            theta.shape[0],
            theta.shape[1],
            device=device,
            dtype=dtype,
        )
        psi.h()
        for l in range(reps):
            psi.ry(theta[:, :, l, 0])
            psi.state = torch.einsum(
                "mnboi,boin->boim",
                TorchGates.rz_gate(encoded_x[:, :, :, l], dtype=dtype),
                psi.state,
            )
        psi.ry(theta[:, :, reps, 0])
        return psi.measure_z(fast_measure)  # shape: (batch_size, out_dim, in_dim)

    def _px_encoding(theta: torch.Tensor):
        """
        Args
        ----
            theta: torch.Tensor
                shape: (*group, reps, 1)
        """
        psi = StateVector(
            x.shape[0],
            theta.shape[0],
            theta.shape[1],
            device=device,
            dtype=dtype,
        )  # psi.state: torch.Tensor, shape: (batch_size * g, out_dim, n_group, 2)
        psi.h()
        for l in range(reps):
            psi.rz(theta[:, :, l, 0])
            psi.state = torch.einsum(
                "mnboi,boin->boim",
                TorchGates.rx_gate(
                    torch.acos(
                        # torch.sin(
                        encoded_x[:, :, :, l]
                        # )
                        # add sin to prevent input from exceeding pm 1
                    ),
                    dtype=dtype,
                ),
                psi.state,
            )
            """
            # complex extension implementation
            psi.state = torch.einsum(
                "mnboi,boin->boim",
                TorchGates.acrx_gate(
                    torch.einsum("oi,bi->boi", preacts_weight[:, :, l], x)
                ),
                psi.state,
            )
            """
        psi.rz(theta[:, :, reps, 0])
        return psi.measure_z(fast_measure)  # shape: (batch_size, out_dim, in_dim)

    def _real(theta: torch.Tensor):
        """
        Args
        ----
            theta: torch.Tensor
                shape: (*group, reps, 1)
        """
        psi = StateVector(
            x.shape[0],
            theta.shape[0],
            theta.shape[1],
            device=device,
            dtype=dtype,
        )  # psi.state: torch.Tensor, shape: (batch_size, out_dim, in_dim, 2)
        psi.h()
        if not preacts_trainable:
            rug = TorchGates.ry_gate(x, dtype=dtype)
        for l in range(reps):
            psi.x()
            # psi.z()
            psi.ry(theta[:, :, l, 0])
            psi.z()
            if not preacts_trainable:
                psi.state = torch.einsum("mnbi,boin->boim", rug, psi.state)
            else:
                psi.state = torch.einsum(
                    "mnboi,boin->boim",
                    TorchGates.ry_gate(encoded_x[:, :, :, l], dtype=dtype),
                    psi.state,
                )
        return psi.measure_z(fast_measure)  # shape: (batch_size, out_dim, in_dim)

    def _mix(theta: torch.Tensor):
        """
        Args
        ----
            theta: torch.Tensor
                shape: (*group, reps, 2)
        """
        psi = StateVector(
            x.shape[0],
            theta.shape[0],
            theta.shape[1],
            device=device,
            dtype=dtype,
        )  # psi.state: torch.Tensor, shape: (batch_size, out_dim, in_dim, 2)
        psi.h()
        if not preacts_trainable:
            rug_y = TorchGates.ry_gate(x, dtype=dtype)
        for l in range(reps):
            psi.rz(theta[:, :, l, 0])
            psi.rx(theta[:, :, l, 1])
            if not preacts_trainable:
                psi.state = torch.einsum("mnbi,boin->boim", rug_y, psi.state)
            else:
                psi.state = torch.einsum(
                    "mnboi,boin->boim",
                    TorchGates.ry_gate(encoded_x[:, :, :, l], dtype=dtype),
                    psi.state,
                )
        psi.rz(theta[:, :, reps, 0])
        psi.rx(theta[:, :, reps, 1])
        return psi.measure_z(fast_measure)  # shape: (batch_size, out_dim, in_dim)

    if ansatz == "pz_encoding" or ansatz == "pz":
        circuit = _pz_encoding
    elif ansatz == "rpz_encoding" or ansatz == "rpz":
        circuit = _rpz_encoding
    elif ansatz == "px_encoding" or ansatz == "px":
        circuit = _px_encoding
    elif ansatz == "real":
        circuit = _real
    elif ansatz == "mix":
        circuit = _mix
    elif callable(ansatz):
        circuit = ansatz
    else:
        raise NotImplementedError()
    x = circuit(theta)  # shape: (batch_size, out_dim, in_dim)
    return x


def _combined_xryz_gate(theta, dtype=torch.complex64):
    """
    Analytically compute X @ RY(theta) @ Z as a single 2x2 gate (real ansatz).

    X @ RY(θ) @ Z = [[sin(θ/2), -cos(θ/2)],
                      [cos(θ/2),  sin(θ/2)]]
    """
    cos = torch.cos(theta / 2)
    sin = torch.sin(theta / 2)
    return torch.stack(
        [
            torch.stack([sin, -cos]),
            torch.stack([cos, sin]),
        ]
    ).to(dtype)


def _combined_rz_ry_gate(alpha, beta, dtype=torch.complex64):
    """
    Fused gate for the pz ansatz sequence: first RZ(alpha), then RY(beta).

    Matrix product RY(β) @ RZ(α) (rightmost acts first on the state):

        [[cos(β/2)·e^{-iα/2}, -sin(β/2)·e^{+iα/2}],
         [sin(β/2)·e^{-iα/2},  cos(β/2)·e^{+iα/2}]]
    """
    cos = torch.cos(beta / 2)
    sin = torch.sin(beta / 2)
    exp_neg = torch.exp(-0.5j * alpha)
    exp_pos = torch.exp(0.5j * alpha)
    return torch.stack(
        [
            torch.stack([cos * exp_neg, -sin * exp_pos]),
            torch.stack([sin * exp_neg, cos * exp_pos]),
        ]
    ).to(dtype)


def _find_contraction_path(expression, operands):
    """Find optimal contraction path using cuQuantum or opt_einsum."""
    if _CUTN_AVAILABLE:
        path, _ = _cutn_contract_path(expression, *operands)
        return path
    if _OE_AVAILABLE:
        path, _ = _oe_contract_path(expression, *operands)
        return path
    return None


# def _build_real_expression(reps, preacts_trainable):
#     """
#     Build einsum expression for the real-ansatz circuit.
#     Circuit: |0> -> H -> [XRyZ(theta) -> RY(x)]^reps -> measure
#     Operands per rep: 2 (fused gate + data encoding). No final gate.
#     """
#     chain = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
#     n_needed = 2 + 2 * reps
#     if n_needed > 26:
#         return None
#     ci = 0
#     q = chain[ci]
#     ci += 1
#     subs = [f"boi{q}"]
#     q_new = chain[ci]
#     ci += 1
#     subs.append(f"{q_new}{q}")
#     q = q_new
#     for _ in range(reps):
#         q_new = chain[ci]
#         ci += 1
#         subs.append(f"{q_new}{q}oi")
#         q = q_new
#         q_new = chain[ci]
#         ci += 1
#         subs.append(f"{q_new}{q}boi" if preacts_trainable else f"{q_new}{q}bi")
#         q = q_new
#     return ",".join(subs) + "->" + f"boi{q}"


# def _build_pz_expression(reps, preacts_trainable):
#     """
#     Build einsum expression for the pz_encoding circuit.
#     Circuit: |0> -> H -> [RzRy_fused(theta) -> RZ(x)]^reps -> RzRy_fused(theta_final) -> measure
#     Operands per rep: 2 (fused gate + data encoding). +1 final gate.
#     """
#     chain = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
#     n_needed = 3 + 2 * reps  # H(2) + reps*2 + final(1)
#     if n_needed > 26:
#         return None
#     ci = 0
#     q = chain[ci]
#     ci += 1
#     subs = [f"boi{q}"]
#     q_new = chain[ci]
#     ci += 1
#     subs.append(f"{q_new}{q}")
#     q = q_new
#     for _ in range(reps):
#         q_new = chain[ci]
#         ci += 1
#         subs.append(f"{q_new}{q}oi")
#         q = q_new
#         q_new = chain[ci]
#         ci += 1
#         subs.append(f"{q_new}{q}boi" if preacts_trainable else f"{q_new}{q}bi")
#         q = q_new
#     # Final RzRy gate
#     q_new = chain[ci]
#     ci += 1
#     subs.append(f"{q_new}{q}oi")
#     q = q_new
#     return ",".join(subs) + "->" + f"boi{q}"


# def _build_rpz_expression(reps):
#     """
#     Build einsum expression for the rpz_encoding circuit.
#     Circuit: |0> -> H -> [RY(theta) -> RZ(encoded_x)]^reps -> RY(theta_final) -> measure
#     rpz always uses encoded_x so data gates are (batch, out, in).
#     Operands per rep: 2 (RY + RZ_data). +1 final RY.
#     """
#     chain = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
#     n_needed = 3 + 2 * reps
#     if n_needed > 26:
#         return None
#     ci = 0
#     q = chain[ci]
#     ci += 1
#     subs = [f"boi{q}"]
#     q_new = chain[ci]
#     ci += 1
#     subs.append(f"{q_new}{q}")
#     q = q_new
#     for _ in range(reps):
#         q_new = chain[ci]
#         ci += 1
#         subs.append(f"{q_new}{q}oi")
#         q = q_new
#         q_new = chain[ci]
#         ci += 1
#         subs.append(f"{q_new}{q}boi")
#         q = q_new
#     # Final RY gate
#     q_new = chain[ci]
#     ci += 1
#     subs.append(f"{q_new}{q}oi")
#     q = q_new
#     return ",".join(subs) + "->" + f"boi{q}"



def _build_real_expression(reps, preacts_trainable, is_batched_theta):
    chain = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    n_needed = 2 + 2 * reps
    if n_needed > 26:
        return None
    ci = 0
    q = chain[ci]; ci += 1
    subs = [f"boi{q}"]
    q_new = chain[ci]; ci += 1
    subs.append(f"{q_new}{q}")
    q = q_new
    
    for _ in range(reps):
        q_new = chain[ci]; ci += 1
        # Evaluate dynamically inside the loop
        subs.append(f"{q_new}{q}boi" if is_batched_theta else f"{q_new}{q}oi")
        q = q_new
        q_new = chain[ci]; ci += 1
        subs.append(f"{q_new}{q}boi" if preacts_trainable else f"{q_new}{q}bi")
        q = q_new
        
    return ",".join(subs) + "->" + f"boi{q}"


def _build_pz_expression(reps, preacts_trainable, is_batched_theta):
    chain = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    n_needed = 3 + 2 * reps  
    if n_needed > 26:
        return None
    ci = 0
    q = chain[ci]; ci += 1
    subs = [f"boi{q}"]
    q_new = chain[ci]; ci += 1
    subs.append(f"{q_new}{q}")
    q = q_new
    
    for _ in range(reps):
        q_new = chain[ci]; ci += 1
        # Evaluate dynamically inside the loop
        subs.append(f"{q_new}{q}boi" if is_batched_theta else f"{q_new}{q}oi")
        q = q_new
        q_new = chain[ci]; ci += 1
        subs.append(f"{q_new}{q}boi" if preacts_trainable else f"{q_new}{q}bi")
        q = q_new
        
    # Final RzRy gate
    q_new = chain[ci]; ci += 1
    # Evaluate dynamically for the final gate
    subs.append(f"{q_new}{q}boi" if is_batched_theta else f"{q_new}{q}oi")
    q = q_new
    
    return ",".join(subs) + "->" + f"boi{q}"


def _build_rpz_expression(reps, is_batched_theta):
    chain = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    n_needed = 3 + 2 * reps
    if n_needed > 26:
        return None
    ci = 0
    q = chain[ci]; ci += 1
    subs = [f"boi{q}"]
    q_new = chain[ci]; ci += 1
    subs.append(f"{q_new}{q}")
    q = q_new
    
    for _ in range(reps):
        q_new = chain[ci]; ci += 1
        # Evaluate dynamically inside the loop
        subs.append(f"{q_new}{q}boi" if is_batched_theta else f"{q_new}{q}oi")
        q = q_new
        q_new = chain[ci]; ci += 1
        subs.append(f"{q_new}{q}boi")
        q = q_new
        
    # Final RY gate
    q_new = chain[ci]; ci += 1
    # Evaluate dynamically for the final gate
    subs.append(f"{q_new}{q}boi" if is_batched_theta else f"{q_new}{q}oi")
    q = q_new
    
    return ",".join(subs) + "->" + f"boi{q}"

# Cache for precompiled contraction plans: {(expression, shapes_tuple): plan}
_CUTN_PLAN_CACHE: dict = {}


def _precompile_plan(equation, operand_shapes):
    """
    Precompile a contraction plan: find the optimal path once and convert it
    into a list of pairwise einsum strings that can be executed without any
    string parsing in the hot path.

    Returns (steps, permute_str) or None if no path optimizer is available.
        steps: list of (idx1, idx2, einsum_str)
        permute_str: final transposition einsum or None
    """
    dummy_ops = [torch.empty(*s) for s in operand_shapes]
    path = _find_contraction_path(equation, dummy_ops)
    if path is None:
        return None

    input_str, output_str = equation.split("->")
    subscripts = input_str.split(",")
    final_indices = set(output_str)

    steps = []
    for i, j in path:
        idx1, idx2 = sorted((i, j))
        sub1, sub2 = subscripts[idx1], subscripts[idx2]

        remaining = [s for k, s in enumerate(subscripts) if k != idx1 and k != idx2]
        needed = set("".join(remaining)) | final_indices

        out_chars = [
            c for c in (sub1 + sub2) if c in (set(sub1) | set(sub2)) and c in needed
        ]
        out_sub = "".join(dict.fromkeys(out_chars))

        steps.append((idx1, idx2, f"{sub1},{sub2}->{out_sub}"))

        subscripts.pop(idx2)
        subscripts.pop(idx1)
        subscripts.append(out_sub)

    permute = f"{subscripts[0]}->{output_str}" if subscripts[0] != output_str else None
    return steps, permute


def _execute_plan(plan, operands):
    """Execute a precompiled contraction plan (hot path, no string parsing)."""
    steps, permute = plan
    ops = list(operands)
    for idx1, idx2, einsum_str in steps:
        new_op = torch.einsum(einsum_str, ops[idx1], ops[idx2])
        ops.pop(idx2)
        ops.pop(idx1)
        ops.append(new_op)
    if permute:
        return torch.einsum(permute, ops[0])
    return ops[0]


def _get_plan(expression, operands):
    """Get (or compute and cache) a contraction plan for the given expression."""
    key = (expression, tuple(op.shape for op in operands))
    if key not in _CUTN_PLAN_CACHE:
        _CUTN_PLAN_CACHE[key] = _precompile_plan(
            expression, [op.shape for op in operands]
        )
    return _CUTN_PLAN_CACHE[key]


def cutn_solver(
    x: torch.Tensor,
    theta: torch.Tensor,
    preacts_weight: torch.Tensor,
    preacts_bias: torch.Tensor,
    reps: int,
    **kwargs,
) -> torch.Tensor:
    
    batch, in_dim = x.shape
    device = x.device
    ansatz = kwargs.get("ansatz", "pz_encoding")
    preacts_trainable = kwargs.get("preacts_trainable", False)
    fast_measure = kwargs.get("fast_measure", True)
    out_dim: int = kwargs.get("out_dim", in_dim)
    dtype = kwargs.get("dtype", torch.complex64)

    _SUPPORTED = {"pz_encoding", "pz", "rpz_encoding", "rpz", "real"}
    if ansatz not in _SUPPORTED:
        return torch_exact_solver(
            x, theta, preacts_weight, preacts_bias, reps, **kwargs
        )

    # NEW: Detect if theta is batched
    is_batched_theta = len(theta.shape) == 5

    # Build whole-circuit expression based on ansatz (pass the batching flag)
    if ansatz in ("pz_encoding", "pz"):
        expression = _build_pz_expression(reps, preacts_trainable, is_batched_theta)
    elif ansatz in ("rpz_encoding", "rpz"):
        expression = _build_rpz_expression(reps, is_batched_theta)
    else:  # real
        expression = _build_real_expression(reps, preacts_trainable, is_batched_theta)

    if expression is None:  # reps too large for single-char indices
        return torch_exact_solver(
            x, theta, preacts_weight, preacts_bias, reps, **kwargs
        )

    # Broadcasting logic
    if is_batched_theta:
        if theta.shape[0] != batch:
            raise ValueError(f"Theta batch size {theta.shape[0]} must match input batch size {batch}")
        if theta.shape[2] != in_dim:
            repeat_out = out_dim
            repeat_in = in_dim // theta.shape[2] + 1
            theta = theta.repeat(1, repeat_out, repeat_in, 1, 1)[:, :, :in_dim, :, :]
    else:
        if len(theta.shape) != 4:
            theta = theta.unsqueeze(0)
        if theta.shape[1] != in_dim:
            repeat_out = out_dim
            repeat_in = in_dim // theta.shape[1] + 1
            theta = theta.repeat(repeat_out, repeat_in, 1, 1)[:, :in_dim, :, :]

    _needs_encoded_x = preacts_trainable or ansatz in ("rpz_encoding", "rpz")
    if _needs_encoded_x:
        if len(preacts_weight.shape) != 3:
            preacts_weight = preacts_weight.unsqueeze(0)
            preacts_bias = preacts_bias.unsqueeze(0)
        if preacts_weight.shape[1] != in_dim:
            repeat_out = out_dim
            repeat_in = in_dim // preacts_weight.shape[1] + 1
            preacts_weight = preacts_weight.repeat(repeat_out, repeat_in, 1)[
                :, :in_dim, :
            ]
            preacts_bias = preacts_bias.repeat(repeat_out, repeat_in, 1)[:, :in_dim, :]
        encoded_x = torch.einsum("oir,bi->boir", preacts_weight, x).add(preacts_bias)

    # Build 2x2 H gate
    inv_sqrt2 = torch.tensor(_INV_SQRT2, device=device, dtype=dtype)
    h_gate = torch.stack(
        [
            torch.stack([inv_sqrt2, inv_sqrt2]),
            torch.stack([inv_sqrt2, -inv_sqrt2]),
        ]
    )

    # -- Build initial state --
    psi = torch.zeros(batch, out_dim, in_dim, 2, dtype=dtype, device=device)
    psi[:, :, :, 0] = 1.0

    # -- Build operands based on ansatz --
    operands = [psi, h_gate]

    if ansatz in ("pz_encoding", "pz"):
        if not preacts_trainable:
            rz_data = TorchGates.rz_gate(x, dtype=dtype) 
            
        for l in range(reps):
            # NEW: Use `...` so it dynamically routes 4D vs 5D appropriately
            fused_l = _combined_rz_ry_gate(
                theta[..., l, 0], theta[..., l, 1], dtype=dtype
            )
            operands.append(fused_l)
            if not preacts_trainable:
                operands.append(rz_data)
            else:
                operands.append(TorchGates.rz_gate(encoded_x[:, :, :, l], dtype=dtype))
                
        # Final fused gate
        operands.append(
            _combined_rz_ry_gate(
                theta[..., reps, 0], theta[..., reps, 1], dtype=dtype
            )
        )

    elif ansatz in ("rpz_encoding", "rpz"):
        for l in range(reps):
            operands.append(TorchGates.ry_gate(theta[..., l, 0], dtype=dtype))
            operands.append(TorchGates.rz_gate(encoded_x[:, :, :, l], dtype=dtype))
            
        # Final RY gate
        operands.append(TorchGates.ry_gate(theta[..., reps, 0], dtype=dtype))

    else:  # real
        if not preacts_trainable:
            ry_data = TorchGates.ry_gate(x, dtype=dtype)
            
        for l in range(reps):
            operands.append(_combined_xryz_gate(theta[..., l, 0], dtype=dtype))
            if not preacts_trainable:
                operands.append(ry_data)
            else:
                operands.append(TorchGates.ry_gate(encoded_x[:, :, :, l], dtype=dtype))

    # Get cached contraction plan (path computed only once per shape config)
    plan = _get_plan(expression, operands)

    if plan is not None:
        psi = _execute_plan(plan, operands)
    else:
        psi = torch.einsum(expression, *operands)

    # Measurement (Z basis)
    return (
        psi[:, :, :, 0].abs() - psi[:, :, :, 1].abs()
        if fast_measure
        else torch.square(psi[:, :, :, 0].abs()) - torch.square(psi[:, :, :, 1].abs())
    )


# ---------------------------------------------------------------------------
# Flash (Triton-accelerated) solver
# ---------------------------------------------------------------------------

_SUPPORTED_FLASH_ANSATZES = {"pz_encoding", "pz", "rpz_encoding", "rpz", "real"}


class _FlashFunction(torch.autograd.Function):
    """
    Custom autograd function: Triton forward and backward.

    Forward dispatches to the appropriate Triton kernel based on ansatz.
    Backward uses direct Triton kernels with forward recomputation.
    """

    @staticmethod
    def forward(
        ctx,
        x,
        theta,
        preacts_w,
        preacts_b,
        reps,
        fast_measure,
        preacts_trainable,
        out_dim,
        c_dtype,
        ansatz,
    ):
        ctx.save_for_backward(x, theta, preacts_w, preacts_b)
        ctx.reps = reps
        ctx.fast_measure = fast_measure
        ctx.preacts_trainable = preacts_trainable
        ctx.out_dim = out_dim
        ctx.c_dtype = c_dtype
        ctx.ansatz = ansatz

        if ansatz in ("pz_encoding", "pz"):
            return triton_pz_forward(
                x, theta, preacts_w, preacts_b, preacts_trainable, fast_measure
            )
        elif ansatz in ("rpz_encoding", "rpz"):
            return triton_rpz_forward(x, theta, preacts_w, preacts_b, fast_measure)
        elif ansatz == "real":
            return triton_real_forward(
                x,
                theta,
                preacts_w,
                preacts_b,
                preacts_trainable,
                fast_measure,
                c_dtype=c_dtype,
            )
        else:
            raise ValueError(f"Unsupported ansatz for flash: {ansatz}")

    @staticmethod
    def backward(ctx, grad_output):
        x, theta, preacts_w, preacts_b = ctx.saved_tensors
        ansatz = ctx.ansatz

        if ansatz in ("pz_encoding", "pz"):
            grad_x, grad_theta, grad_pw, grad_pb = triton_pz_backward(
                x,
                theta,
                preacts_w,
                preacts_b,
                grad_output,
                ctx.preacts_trainable,
                ctx.fast_measure,
            )
        elif ansatz in ("rpz_encoding", "rpz"):
            grad_x, grad_theta, grad_pw, grad_pb = triton_rpz_backward(
                x,
                theta,
                preacts_w,
                preacts_b,
                grad_output,
                ctx.fast_measure,
            )
        elif ansatz == "real":
            grad_x, grad_theta, grad_pw, grad_pb = triton_real_backward(
                x,
                theta,
                preacts_w,
                preacts_b,
                grad_output,
                ctx.preacts_trainable,
                ctx.fast_measure,
                c_dtype=ctx.c_dtype,
            )
            # Cast gradients back to parameter dtype
            p_dtype = x.dtype
            grad_x = grad_x.to(p_dtype)
            grad_theta = grad_theta.to(p_dtype)
            if grad_pw is not None:
                grad_pw = grad_pw.to(p_dtype)
            if grad_pb is not None:
                grad_pb = grad_pb.to(p_dtype)
        else:
            raise ValueError(f"Unsupported ansatz for flash backward: {ansatz}")

        return (
            grad_x,
            grad_theta,
            grad_pw,
            grad_pb,
            None,  # reps
            None,  # fast_measure
            None,  # preacts_trainable
            None,  # out_dim
            None,  # c_dtype
            None,  # ansatz
        )


def flash_exact_solver(
    x: torch.Tensor,
    theta: torch.Tensor,
    preacts_weight: torch.Tensor,
    preacts_bias: torch.Tensor,
    reps: int,
    **kwargs,
) -> torch.Tensor:
    """
    Triton-accelerated exact solver. Drop-in replacement for torch_exact_solver.

    Uses fused Triton kernels for pz_encoding, rpz_encoding, and real ansatzes.
    Falls back to torch_exact_solver for unsupported ansatzes.

    Args:
        Same as torch_exact_solver.

    Returns:
        torch.Tensor, shape: (batch_size, out_dim, in_dim)
    """
    if not _FLASH_AVAILABLE:
        raise ImportError(
            "Triton fused kernels not available. Install triton to use flash solver."
        )

    ansatz = kwargs.get("ansatz", "pz_encoding")
    preacts_trainable = kwargs.get("preacts_trainable", False)
    fast_measure = kwargs.get("fast_measure", True)
    out_dim: int = kwargs.get("out_dim", x.shape[1])
    c_dtype = kwargs.get("dtype", torch.complex64)
    batch, in_dim = x.shape

    # Fallback for unsupported ansatzes
    if ansatz not in _SUPPORTED_FLASH_ANSATZES:
        return torch_exact_solver(
            x, theta, preacts_weight, preacts_bias, reps, **kwargs
        )

    # # Broadcasting logic (mirrors torch_exact_solver)
    # if len(theta.shape) != 4:
    #     theta = theta.unsqueeze(0)
    # if theta.shape[1] != in_dim:
    #     repeat_out = out_dim
    #     repeat_in = in_dim // theta.shape[1] + 1
    #     theta = theta.repeat(repeat_out, repeat_in, 1, 1)[:, :in_dim, :, :]
    # Check if theta has a batch dimension (5D: B, out_dim, in_dim, reps, params)
    is_batched_theta = len(theta.shape) == 5

    if is_batched_theta:
        # Broadcasting logic for batched theta
        if theta.shape[0] != batch:
            raise ValueError(f"Theta batch size {theta.shape[0]} must match input batch size {batch}")
        
        if theta.shape[2] != in_dim:
            repeat_out = out_dim
            repeat_in = in_dim // theta.shape[2] + 1
            # Repeat across out_dim and in_dim, but leave batch, reps, and params alone
            theta = theta.repeat(1, repeat_out, repeat_in, 1, 1)[:, :, :in_dim, :, :]
    else:
        # Original broadcasting logic for static theta
        if len(theta.shape) != 4:
            theta = theta.unsqueeze(0)
        if theta.shape[1] != in_dim:
            repeat_out = out_dim
            repeat_in = in_dim // theta.shape[1] + 1
            theta = theta.repeat(repeat_out, repeat_in, 1, 1)[:, :in_dim, :, :]

    # rpz always needs encoded_x; others only when preacts_trainable
    _needs_encoded_x = preacts_trainable or ansatz in ("rpz_encoding", "rpz")
    if _needs_encoded_x:
        if len(preacts_weight.shape) != 3:
            preacts_weight = preacts_weight.unsqueeze(0)
            preacts_bias = preacts_bias.unsqueeze(0)
        if preacts_weight.shape[1] != in_dim:
            repeat_out = out_dim
            repeat_in = in_dim // preacts_weight.shape[1] + 1
            preacts_weight = preacts_weight.repeat(repeat_out, repeat_in, 1)[
                :, :in_dim, :
            ]
            preacts_bias = preacts_bias.repeat(repeat_out, repeat_in, 1)[:, :in_dim, :]

    # Check if gradients are needed (training)
    needs_grad = theta.requires_grad or x.requires_grad
    if _needs_encoded_x:
        needs_grad = (
            needs_grad or preacts_weight.requires_grad or preacts_bias.requires_grad
        )
    elif preacts_trainable:
        needs_grad = (
            needs_grad or preacts_weight.requires_grad or preacts_bias.requires_grad
        )

    if needs_grad:
        return _FlashFunction.apply(
            x,
            theta,
            preacts_weight,
            preacts_bias,
            reps,
            fast_measure,
            preacts_trainable,
            out_dim,
            c_dtype,
            ansatz,
        )
    else:
        if ansatz in ("pz_encoding", "pz"):
            return triton_pz_forward(
                x,
                theta,
                preacts_weight,
                preacts_bias,
                preacts_trainable,
                fast_measure,
            )
        elif ansatz in ("rpz_encoding", "rpz"):
            return triton_rpz_forward(
                x,
                theta,
                preacts_weight,
                preacts_bias,
                fast_measure,
            )
        elif ansatz == "real":
            return triton_real_forward(
                x,
                theta,
                preacts_weight,
                preacts_bias,
                preacts_trainable,
                fast_measure,
                c_dtype=c_dtype,
            )
        else:
            raise NotImplementedError


# ---------------------------------------------------------------------------
# cuTile batched-theta kernel fast path (optional)
# ---------------------------------------------------------------------------
#
# Uses the repo-local cuTile kernels in ``cutile_batched_ops.py`` when running
# on CUDA with ansatz ``pz_encoding``. The kernel mirrors upstream
# ``_ct_pz_encoding_kernel`` but accepts per-sample theta via a leading batch
# index on every ``ct.gather(theta, …)`` and writes ``grad_theta`` with a
# plain ``ct.scatter`` (no atomic, since each (b, o, i) is a unique writer).
#
# When cuda-tile is unavailable or ansatz is not pz, we fall back to the pure
# PyTorch scalar recurrence defined below.

try:
    from .cutile_batched_ops import (
        cutile_pz_backward_batched,
        cutile_pz_forward_batched,
    )

    _CUTILE_BATCHED_AVAILABLE = True
except Exception:  # cuda.tile not installed, or any other import error
    _CUTILE_BATCHED_AVAILABLE = False

# CuTe batched-theta kernel (optional, opt-in via `--fast_solver cute`).
# Importing it JIT-compiles nothing by itself, but it can still fail for many
# reasons on a machine without a CUDA toolchain: RuntimeError when CUTLASS_PATH
# is unset, OSError/CalledProcessError from a failed ninja build, ImportError
# when torch's cpp_extension machinery is unusable. Catch Exception so that a
# user without any of this can still import the module and train with `flash`.
try:
    from .cute_batched_ops import cute_pz_batched, cute_unavailable_reason

    _CUTE_BATCHED_AVAILABLE = True
except Exception:
    cute_pz_batched = None
    cute_unavailable_reason = None
    _CUTE_BATCHED_AVAILABLE = False


class _CuTileBatchedPZFunction(torch.autograd.Function):
    """Custom autograd function: cuTile batched-theta pz forward + backward."""

    @staticmethod
    def forward(
        ctx,
        x,
        theta,
        preacts_w,
        preacts_b,
        reps,
        fast_measure,
        preacts_trainable,
    ):
        ctx.save_for_backward(x, theta, preacts_w, preacts_b)
        ctx.reps = reps
        ctx.fast_measure = fast_measure
        ctx.preacts_trainable = preacts_trainable

        return cutile_pz_forward_batched(
            x,
            theta,
            preacts_w,
            preacts_b,
            preacts_trainable,
            fast_measure,
        )

    @staticmethod
    def backward(ctx, grad_output):
        x, theta, preacts_w, preacts_b = ctx.saved_tensors
        grad_x, grad_theta, grad_pw, grad_pb = cutile_pz_backward_batched(
            x,
            theta,
            preacts_w,
            preacts_b,
            grad_output,
            ctx.preacts_trainable,
            ctx.fast_measure,
        )

        # If theta was unbatched (shape[0] == 1) the backward returned a
        # (1, O, I, R+1, 2) grad; the solver wrapper will handle squeezing
        # to match the original shape. For the batched case, shapes match
        # out-of-the-box.
        return (
            grad_x,
            grad_theta,
            grad_pw,
            grad_pb,
            None,  # reps
            None,  # fast_measure
            None,  # preacts_trainable
        )


# ---------------------------------------------------------------------------
# cuTile-inspired batched-theta solver (pure PyTorch scalar recurrence)
# ---------------------------------------------------------------------------
#
# Mirrors the single-qubit scalar-recurrence algorithm used by upstream qkan's
# cuTile and Triton fused kernels (qkan/cutile_ops.py, qkan/fused_ops.py), but
# reimplemented in pure PyTorch so that `theta` can carry a leading batch axis
# `(B, out_dim, in_dim, reps+1, 2)` — required by the fast programmer in
# GQKAN-QKANFWP where `theta` is a per-sample input, not an nn.Parameter.
#
# Correctness target: numerically matches flash_exact_solver within float
# nondeterminism (rel_mse < 1e-5 GPU, < 1e-6 CPU).
#
# The gate convention is taken verbatim from upstream `_ct_pz_encoding_kernel`
# (qkan/cutile_ops.py) and `_pz_encoding_kernel` (qkan/fused_ops.py), which are
# consistent with one another:
#   Rz(θ) : diag(e^{-iθ/2}, e^{+iθ/2})
#   Ry(θ) : [[cos(θ/2), -sin(θ/2)], [sin(θ/2),  cos(θ/2)]]
#   Circuit (pz):  H|0>  → [Rz(θ₀) Ry(θ₁) Rz(enc)]^reps  → Rz(θ_f,0) Ry(θ_f,1)  → <Z>
#   Circuit (rpz): H|0>  → [Ry(θ₀) Rz(w·x+b)]^reps       → Ry(θ_f,0)            → <Z>
#   Circuit (real): H|0> → [X Ry(θ₀) Z Ry(enc)]^reps                             → <Z>
#
# Measurement matches `fast_measure` flag of the existing solvers:
#     fast_measure=True :  |α| − |β|         (quantum-inspired shortcut)
#     fast_measure=False:  |α|² − |β|²       (Born rule)
# ---------------------------------------------------------------------------


_SUPPORTED_CUTILE_ANSATZES = {"pz_encoding", "pz", "rpz_encoding", "rpz", "real"}


def _cutile_pz_scalar(
    x_b1i: torch.Tensor,
    theta_oi: torch.Tensor,
    encoded_x_boir: "torch.Tensor | None",
    reps: int,
    fast_measure: bool,
    preacts_trainable: bool,
    batch: int,
    out_dim: int,
    in_dim: int,
) -> torch.Tensor:
    """Scalar-state pz_encoding recurrence. State r0,i0,r1,i1 broadcast to (B,O,I)."""
    state_dtype = torch.float32
    r0 = torch.full(
        (batch, out_dim, in_dim), _INV_SQRT2, dtype=state_dtype, device=x_b1i.device
    )
    i0 = torch.zeros_like(r0)
    r1 = r0.clone()
    i1 = torch.zeros_like(r0)

    # theta_oi: (B, O, I, R+1, 2) or (O, I, R+1, 2) — both broadcast against (B, O, I)
    x_boi = x_b1i  # (B, 1, I), broadcasts over O

    for layer in range(reps):
        # Rz(θ₀)  — diag(e^{-iα/2}, e^{+iα/2}) per upstream sign convention
        a = theta_oi[..., layer, 0] * 0.5
        c = torch.cos(a)
        s = torch.sin(a)
        nr0 = r0 * c + i0 * s
        ni0 = i0 * c - r0 * s
        nr1 = r1 * c - i1 * s
        ni1 = i1 * c + r1 * s
        r0, i0, r1, i1 = nr0, ni0, nr1, ni1

        # Ry(θ₁)
        a = theta_oi[..., layer, 1] * 0.5
        c = torch.cos(a)
        s = torch.sin(a)
        nr0 = c * r0 - s * r1
        ni0 = c * i0 - s * i1
        nr1 = s * r0 + c * r1
        ni1 = s * i0 + c * i1
        r0, i0, r1, i1 = nr0, ni0, nr1, ni1

        # Rz(enc)  — data gate; enc is broadcast over O when preacts not trainable
        if preacts_trainable:
            enc = encoded_x_boir[..., layer]  # (B, O, I)
        else:
            enc = x_boi  # (B, 1, I) broadcasts to (B, O, I)
        a = enc * 0.5
        c = torch.cos(a)
        s = torch.sin(a)
        nr0 = r0 * c + i0 * s
        ni0 = i0 * c - r0 * s
        nr1 = r1 * c - i1 * s
        ni1 = i1 * c + r1 * s
        r0, i0, r1, i1 = nr0, ni0, nr1, ni1

    # Final Rz(θ_f,0)
    a = theta_oi[..., reps, 0] * 0.5
    c = torch.cos(a)
    s = torch.sin(a)
    nr0 = r0 * c + i0 * s
    ni0 = i0 * c - r0 * s
    nr1 = r1 * c - i1 * s
    ni1 = i1 * c + r1 * s
    r0, i0, r1, i1 = nr0, ni0, nr1, ni1

    # Final Ry(θ_f,1)
    a = theta_oi[..., reps, 1] * 0.5
    c = torch.cos(a)
    s = torch.sin(a)
    nr0 = c * r0 - s * r1
    ni0 = c * i0 - s * i1
    nr1 = s * r0 + c * r1
    ni1 = s * i0 + c * i1
    r0, i0, r1, i1 = nr0, ni0, nr1, ni1

    if fast_measure:
        return torch.sqrt(r0 * r0 + i0 * i0) - torch.sqrt(r1 * r1 + i1 * i1)
    return (r0 * r0 + i0 * i0) - (r1 * r1 + i1 * i1)


def _cutile_rpz_scalar(
    theta_oi: torch.Tensor,
    encoded_x_boir: torch.Tensor,
    reps: int,
    fast_measure: bool,
    batch: int,
    out_dim: int,
    in_dim: int,
    device,
) -> torch.Tensor:
    """Scalar-state rpz_encoding recurrence. rpz always uses encoded_x."""
    state_dtype = torch.float32
    r0 = torch.full(
        (batch, out_dim, in_dim), _INV_SQRT2, dtype=state_dtype, device=device
    )
    i0 = torch.zeros_like(r0)
    r1 = r0.clone()
    i1 = torch.zeros_like(r0)

    for layer in range(reps):
        # Ry(θ)
        a = theta_oi[..., layer, 0] * 0.5
        c = torch.cos(a)
        s = torch.sin(a)
        nr0 = c * r0 - s * r1
        ni0 = c * i0 - s * i1
        nr1 = s * r0 + c * r1
        ni1 = s * i0 + c * i1
        r0, i0, r1, i1 = nr0, ni0, nr1, ni1

        # Rz(encoded_x)
        a = encoded_x_boir[..., layer] * 0.5
        c = torch.cos(a)
        s = torch.sin(a)
        nr0 = r0 * c + i0 * s
        ni0 = i0 * c - r0 * s
        nr1 = r1 * c - i1 * s
        ni1 = i1 * c + r1 * s
        r0, i0, r1, i1 = nr0, ni0, nr1, ni1

    # Final Ry
    a = theta_oi[..., reps, 0] * 0.5
    c = torch.cos(a)
    s = torch.sin(a)
    nr0 = c * r0 - s * r1
    ni0 = c * i0 - s * i1
    nr1 = s * r0 + c * r1
    ni1 = s * i0 + c * i1
    r0, i0, r1, i1 = nr0, ni0, nr1, ni1

    if fast_measure:
        return torch.sqrt(r0 * r0 + i0 * i0) - torch.sqrt(r1 * r1 + i1 * i1)
    return (r0 * r0 + i0 * i0) - (r1 * r1 + i1 * i1)


def _cutile_real_scalar(
    x_b1i: torch.Tensor,
    theta_oi: torch.Tensor,
    encoded_x_boir: "torch.Tensor | None",
    reps: int,
    fast_measure: bool,
    preacts_trainable: bool,
    batch: int,
    out_dim: int,
    in_dim: int,
) -> torch.Tensor:
    """Scalar-state real ansatz recurrence. No imaginary components needed."""
    state_dtype = torch.float32
    r0 = torch.full(
        (batch, out_dim, in_dim), _INV_SQRT2, dtype=state_dtype, device=x_b1i.device
    )
    r1 = r0.clone()
    x_boi = x_b1i

    for layer in range(reps):
        # X gate
        r0, r1 = r1, r0

        # Ry(θ)
        a = theta_oi[..., layer, 0] * 0.5
        c = torch.cos(a)
        s = torch.sin(a)
        nr0 = c * r0 - s * r1
        nr1 = s * r0 + c * r1
        r0, r1 = nr0, nr1

        # Z gate
        r1 = -r1

        # Ry(enc)
        if preacts_trainable:
            enc = encoded_x_boir[..., layer]
        else:
            enc = x_boi
        a = enc * 0.5
        c = torch.cos(a)
        s = torch.sin(a)
        nr0 = c * r0 - s * r1
        nr1 = s * r0 + c * r1
        r0, r1 = nr0, nr1

    if fast_measure:
        return torch.abs(r0) - torch.abs(r1)
    return r0 * r0 - r1 * r1


def cutile_batched_solver(
    x: torch.Tensor,
    theta: torch.Tensor,
    preacts_weight: torch.Tensor,
    preacts_bias: torch.Tensor,
    reps: int,
    **kwargs,
) -> torch.Tensor:
    """
    Batched-theta cuTile-inspired solver (pure PyTorch scalar recurrence).

    Drop-in replacement for ``flash_exact_solver``. Accepts ``theta`` with
    shape ``(B, out_dim, in_dim, reps+1, K)`` (batched, the GQKAN fast-
    programmer case) **or** ``(out_dim, in_dim, reps+1, K)`` (unbatched,
    upstream-compatible). ``K`` is 2 for pz, 1 for rpz, 1 for real.

    Mirrors upstream qkan's cuTile kernel logic at the PyTorch level
    (see ``qkan/cutile_ops.py::_ct_pz_encoding_kernel``) so that the same
    gate sequence and sign convention apply. This keeps numerical parity
    with ``flash_exact_solver`` and preserves the ``fast_measure=True`` →
    ``|α|−|β|`` transform used by the trained decoder.

    Args
    ----
    x : (B, in_dim) float tensor
    theta : (B, out_dim, in_dim, reps+1, K) or (out_dim, in_dim, reps+1, K)
    preacts_weight, preacts_bias : (out_dim, in_dim, reps) or broadcastable
    reps : int

    kwargs
    ------
    ansatz : one of {"pz_encoding"/"pz", "rpz_encoding"/"rpz", "real"};
        other ansatzes fall back to ``torch_exact_solver``.
    preacts_trainable : bool (default False)
    fast_measure : bool (default True) — match training-time decoder distribution.
    out_dim : int (default x.shape[1])
    dtype : cast output to this dtype (state is always fp32 for stability).

    Returns
    -------
    postacts : (B, out_dim, in_dim) tensor, same dtype as x.
    """
    ansatz = kwargs.get("ansatz", "pz_encoding")
    preacts_trainable = kwargs.get("preacts_trainable", False)
    fast_measure = kwargs.get("fast_measure", True)
    out_dim: int = kwargs.get("out_dim", x.shape[1])

    if ansatz not in _SUPPORTED_CUTILE_ANSATZES:
        return torch_exact_solver(
            x, theta, preacts_weight, preacts_bias, reps, **kwargs
        )

    batch, in_dim = x.shape
    device = x.device
    p_dtype = x.dtype

    # Normalize theta to always include a leading (possibly broadcast) batch dim.
    is_batched_theta = theta.dim() == 5
    if not is_batched_theta:
        if theta.dim() != 4:
            theta = theta.reshape(-1, in_dim, reps + 1, theta.shape[-1])
        theta = theta.unsqueeze(0)  # (1, O, I, R+1, K) — broadcasts against state
    else:
        if theta.shape[0] not in (1, batch):
            raise ValueError(
                f"Theta batch size {theta.shape[0]} must be 1 or match input batch {batch}"
            )

    # Broadcast out_dim / in_dim when theta was allocated on a smaller shape.
    if theta.shape[2] != in_dim:
        repeat_in = in_dim // theta.shape[2] + 1
        repeat_out = out_dim if theta.shape[1] != out_dim else 1
        theta = theta.repeat(1, repeat_out, repeat_in, 1, 1)[
            :, :out_dim, :in_dim, :, :
        ]
    elif theta.shape[1] != out_dim:
        repeat_out = out_dim // theta.shape[1] + 1
        theta = theta.repeat(1, repeat_out, 1, 1, 1)[:, :out_dim, :, :, :]

    # Cast theta to fp32 state dtype (matches cuTile/Triton kernels' compute dtype).
    theta_f32 = theta.to(torch.float32)

    # encoded_x is only materialized when preacts are trainable or ansatz requires it.
    _needs_encoded_x = preacts_trainable or ansatz in ("rpz_encoding", "rpz")

    # ── Fast path: cuTile batched-theta kernel for pz_encoding on CUDA ──
    # Opt-in via env var QFWP_CUTILE_KERNEL=0 to force the pure-torch path.
    import os as _os

    _kernel_enabled = _os.environ.get("QFWP_CUTILE_KERNEL", "1") != "0"
    if (
        _kernel_enabled
        and _CUTILE_BATCHED_AVAILABLE
        and device.type == "cuda"
        and ansatz in ("pz_encoding", "pz")
    ):
        # Normalize preacts shape for the kernel wrapper.
        if _needs_encoded_x:
            if preacts_weight.dim() == 2:
                preacts_weight = preacts_weight.unsqueeze(0)
                preacts_bias = preacts_bias.unsqueeze(0)
            if preacts_weight.shape[1] != in_dim:
                repeat_out = out_dim
                repeat_in = in_dim // preacts_weight.shape[1] + 1
                preacts_weight = preacts_weight.repeat(
                    repeat_out, repeat_in, 1
                )[:, :in_dim, :]
                preacts_bias = preacts_bias.repeat(
                    repeat_out, repeat_in, 1
                )[:, :in_dim, :]
        pw_arg = (
            preacts_weight.to(torch.float32)
            if _needs_encoded_x
            else torch.empty(0, device=device)
        )
        pb_arg = (
            preacts_bias.to(torch.float32)
            if _needs_encoded_x
            else torch.empty(0, device=device)
        )
        postacts = _CuTileBatchedPZFunction.apply(
            x.to(torch.float32).contiguous(),
            theta_f32.contiguous(),
            pw_arg,
            pb_arg,
            reps,
            fast_measure,
            preacts_trainable,
        )
        return postacts.to(p_dtype)

    # ── Fallback: pure-PyTorch scalar recurrence ──
    encoded_x_boir = None
    if _needs_encoded_x:
        if preacts_weight.dim() == 2:
            preacts_weight = preacts_weight.unsqueeze(0)
            preacts_bias = preacts_bias.unsqueeze(0)
        if preacts_weight.shape[1] != in_dim:
            repeat_out = out_dim
            repeat_in = in_dim // preacts_weight.shape[1] + 1
            preacts_weight = preacts_weight.repeat(repeat_out, repeat_in, 1)[
                :, :in_dim, :
            ]
            preacts_bias = preacts_bias.repeat(repeat_out, repeat_in, 1)[
                :, :in_dim, :
            ]
        # encoded_x: (B, O, I, R) in fp32
        encoded_x_boir = torch.einsum(
            "oir,bi->boir", preacts_weight.to(torch.float32), x.to(torch.float32)
        ).add(preacts_bias.to(torch.float32))

    x_b1i = x.to(torch.float32).unsqueeze(1)  # (B, 1, I)

    if ansatz in ("pz_encoding", "pz"):
        postacts = _cutile_pz_scalar(
            x_b1i=x_b1i,
            theta_oi=theta_f32,
            encoded_x_boir=encoded_x_boir,
            reps=reps,
            fast_measure=fast_measure,
            preacts_trainable=preacts_trainable,
            batch=batch,
            out_dim=out_dim,
            in_dim=in_dim,
        )
    elif ansatz in ("rpz_encoding", "rpz"):
        postacts = _cutile_rpz_scalar(
            theta_oi=theta_f32,
            encoded_x_boir=encoded_x_boir,
            reps=reps,
            fast_measure=fast_measure,
            batch=batch,
            out_dim=out_dim,
            in_dim=in_dim,
            device=device,
        )
    else:  # real
        postacts = _cutile_real_scalar(
            x_b1i=x_b1i,
            theta_oi=theta_f32,
            encoded_x_boir=encoded_x_boir,
            reps=reps,
            fast_measure=fast_measure,
            preacts_trainable=preacts_trainable,
            batch=batch,
            out_dim=out_dim,
            in_dim=in_dim,
        )

    return postacts.to(p_dtype)


# ---------------------------------------------------------------------------
# CuTe batched-theta kernel fast path (optional, opt-in)
# ---------------------------------------------------------------------------

_SUPPORTED_CUTE_ANSATZES = {"pz_encoding", "pz"}


def cute_batched_solver(
    x: torch.Tensor,
    theta: torch.Tensor,
    preacts_weight: torch.Tensor,
    preacts_bias: torch.Tensor,
    reps: int,
    **kwargs,
) -> torch.Tensor:
    """
    Batched-theta CuTe solver (JIT-compiled CUDA kernel).

    Drop-in replacement for ``flash_exact_solver`` / ``cutile_batched_solver``:
    same signature, same ``(batch, out_dim, in_dim)`` return shape.

    Unlike those two, this backend implements **only the pz ansatz** and has no
    fallback — anything else raises, so a silent switch to a different numerical
    path can never happen behind your back. Use ``--fast_solver flash`` for the
    other ansatzes.

    Args
    ----
    x : (B, in_dim) float tensor
    theta : (B, out_dim, in_dim, reps+1, 2), or (out_dim, in_dim, reps+1, 2)
        which is broadcast to a batch of 1.
    preacts_weight, preacts_bias : (out_dim, in_dim, reps) or broadcastable
    reps : int

    kwargs
    ------
    ansatz : must be "pz_encoding" or "pz".
    preacts_trainable : bool (default False)
    fast_measure : bool (default True)
    out_dim : int (default x.shape[1])
    dtype : accepted for signature compatibility; the kernel is fp32 internally
        and the result is returned in ``x``'s dtype.

    Returns
    -------
    postacts : (B, out_dim, in_dim) tensor, same dtype as x.
    """
    ansatz = kwargs.get("ansatz", "pz_encoding")
    preacts_trainable = kwargs.get("preacts_trainable", False)
    fast_measure = kwargs.get("fast_measure", True)
    out_dim: int = kwargs.get("out_dim", x.shape[1])

    if not _CUTE_BATCHED_AVAILABLE:
        raise RuntimeError(
            "The CuTe backend is not available: `from .cute_batched_ops import "
            "cute_pz_batched` failed at import time. It needs an NVIDIA GPU, a CUDA "
            "toolchain (nvcc on PATH or CUDA_HOME set), ninja, and CUTLASS_PATH "
            "pointing at a CUTLASS checkout. Use --fast_solver flash instead."
        )

    if ansatz not in _SUPPORTED_CUTE_ANSATZES:
        raise RuntimeError(
            f"The CuTe backend implements only the pz ansatz, got ansatz={ansatz!r}. "
            f"Supported: {sorted(_SUPPORTED_CUTE_ANSATZES)}. "
            "Use --fast_solver flash for this ansatz."
        )

    if x.device.type != "cuda":
        raise RuntimeError(
            f"The CuTe backend is CUDA-only, got a tensor on device {x.device!r}. "
            "Use --fast_solver flash (GPU) or --fast_solver cutile (CPU-capable)."
        )

    batch, in_dim = x.shape
    p_dtype = x.dtype

    # Normalize theta to a leading batch axis, mirroring cutile_batched_solver.
    if theta.dim() == 4:
        theta = theta.unsqueeze(0)
    elif theta.dim() != 5:
        raise ValueError(
            f"theta must be 4D (out_dim, in_dim, reps+1, 2) or 5D with a leading "
            f"batch axis, got shape {tuple(theta.shape)}"
        )
    if theta.shape[0] not in (1, batch):
        raise ValueError(
            f"Theta batch size {theta.shape[0]} must be 1 or match input batch {batch}"
        )

    # Broadcast out_dim / in_dim when theta was allocated on a smaller shape.
    if theta.shape[2] != in_dim:
        repeat_in = in_dim // theta.shape[2] + 1
        repeat_out = out_dim if theta.shape[1] != out_dim else 1
        theta = theta.repeat(1, repeat_out, repeat_in, 1, 1)[
            :, :out_dim, :in_dim, :, :
        ]
    elif theta.shape[1] != out_dim:
        repeat_out = out_dim // theta.shape[1] + 1
        theta = theta.repeat(1, repeat_out, 1, 1, 1)[:, :out_dim, :, :, :]

    # The kernel reads its own per-sample theta, so a batch-1 theta must be
    # materialized across the batch rather than relied on to broadcast.
    if theta.shape[0] == 1 and batch != 1:
        theta = theta.expand(batch, -1, -1, -1, -1)

    # Pre-flight the backend so a missing toolchain fails here, with a clear
    # message, rather than from inside torch.autograd on the first apply().
    reason = cute_unavailable_reason()
    if reason is not None:
        raise RuntimeError(
            f"The CuTe backend is unavailable: {reason}\n"
            "Use --fast_solver flash instead."
        )

    pw_arg = preacts_weight if preacts_trainable else None
    pb_arg = preacts_bias if preacts_trainable else None

    postacts = cute_pz_batched(
        x,
        theta,
        pw_arg,
        pb_arg,
        reps=reps,
        preacts_trainable=preacts_trainable,
        fast_measure=fast_measure,
    )
    return postacts.to(p_dtype)
