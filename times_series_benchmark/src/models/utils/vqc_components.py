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

"""Reusable PennyLane circuit layers for the GQKAN-QFWP variational ansatz.

Each rotation layer indexes its parameter tensor as ``w[..., idx]``, so a 1-D
tensor of angles and a batched 2-D tensor route through the same code path.

The layer decomposition follows the standard PennyLane variational-classifier
tutorial: https://pennylane.ai/qml/demos/tutorial_variational_classifier
"""

import pennylane as qml


def H_layer(nqubits):
	"""Layer of single-qubit Hadamard gates."""
	for idx in range(nqubits):
		qml.Hadamard(wires=idx)


def RX_layer(w):
	"""Layer of parametrized qubit rotations around the x axis."""
	for idx in range(w.shape[-1]):
		qml.RX(w[..., idx], wires=idx)


def RY_layer(w):
	"""Layer of parametrized qubit rotations around the y axis."""
	for idx in range(w.shape[-1]):
		qml.RY(w[..., idx], wires=idx)


def RZ_layer(w):
	"""Layer of parametrized qubit rotations around the z axis."""
	for idx in range(w.shape[-1]):
		qml.RZ(w[..., idx], wires=idx)


def entangling_layer(nqubits):
	"""Layer of CNOTs followed by another shifted layer of CNOT.

	Applies CNOTs on even-indexed pairs first, then on odd-indexed pairs::

	    CNOT  CNOT  CNOT  CNOT ...  CNOT
	      CNOT  CNOT  CNOT ...  CNOT
	"""
	for i in range(0, nqubits - 1, 2):  # even indices: i = 0, 2, ... N-2
		qml.CNOT(wires=[i, i + 1])
	for i in range(1, nqubits - 1, 2):  # odd indices:  i = 1, 3, ... N-3
		qml.CNOT(wires=[i, i + 1])
