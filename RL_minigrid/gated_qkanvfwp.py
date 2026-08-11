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

"""Gated QKAN-VQC fast-weight programmer.

A QKAN-based slow programmer reads the observation and emits the rotation
angles of a variational quantum circuit, so the circuit is reprogrammed at every
timestep rather than holding a single trained parameter set. A sigmoid gate
blends the new angles with those carried over from the previous step. A
classical linear layer preprocesses the observation before it is encoded into
the circuit.
"""

from functools import partial

import torch
import torch.nn as nn

import pennylane as qml
from pennylane import numpy as np

from qkan import QKAN
from utils import set_init


### VQC

def H_layer(nqubits):
	"""Layer of single-qubit Hadamard gates.
	"""
	for idx in range(nqubits):
		qml.Hadamard(wires=idx)

def RX_layer(w):
	"""Layer of parametrized qubit rotations around the y axis.
	"""
	for idx, element in enumerate(w):
		qml.RX(element, wires=idx)

def RY_layer(w):
	"""Layer of parametrized qubit rotations around the y axis.
	"""
	for idx, element in enumerate(w):
		qml.RY(element, wires=idx)

def RZ_layer(w):
	"""Layer of parametrized qubit rotations around the y axis.
	"""
	for idx, element in enumerate(w):
		qml.RZ(element, wires=idx)


def entangling_layer(nqubits):
	"""Layer of CNOTs followed by another shifted layer of CNOT.
	"""
	# In other words it should apply something like :
	# CNOT  CNOT  CNOT  CNOT...  CNOT
	#   CNOT  CNOT  CNOT...  CNOT
	for i in range(0, nqubits - 1, 2):  # Loop over even indices: i=0,2,...N-2
		qml.CNOT(wires=[i, i + 1])
	for i in range(1, nqubits - 1, 2):  # Loop over odd indices:  i=1,3,...N-3
		qml.CNOT(wires=[i, i + 1])


def cycle_entangling_layer(nqubits):
	for i in range(0, nqubits):
		qml.CNOT(wires=[i, (i + 1) % nqubits])


##############


### VQC function

def quantum_net(inputs, q_weights, n_outputs):
	n_dep = q_weights.shape[0]
	n_qub = q_weights.shape[1]

	H_layer(n_qub)

	RY_layer(inputs)

	for k in range(n_dep):
		entangling_layer(n_qub)
		RY_layer(q_weights[k])

	# Expectation values in the Z basis
	return [qml.expval(qml.PauliZ(position)) for position in range(n_outputs)]


##############

class HideSignature:
	def __init__(self, partial_func):
		self.partial_func = partial_func

	def __call__(self, inputs, q_weights):
		return self.partial_func(inputs, q_weights)

##############

### VQC batch wrapper

class BatchVQC:
	def __init__(self, q_func):
		self.q_func = q_func

	def __call__(self, inputs, q_weights):

		res_all = []
		for input_item, q_weight_item in zip(inputs, q_weights):
			res = self.q_func(input_item, q_weight_item) # not yet consider the memory of previous time-step
			res_all.append(torch.stack(res))  # stack so a tensor, not a list, is returned

		return torch.stack(res_all)


##############


### FWP Cell Module

class FWPCell(nn.Module):
	def __init__(self, s_dim, latent_dim):
		super().__init__()

		self.n_qubits = latent_dim
		self.q_depth = 2

		dev = qml.device("default.qubit", wires = self.n_qubits)
		self.q_func = BatchVQC(qml.QNode(HideSignature(partial(quantum_net, n_outputs = self.n_qubits)), dev, interface = "torch"))

		qkan_s_dim = int(np.ceil(np.log2(latent_dim)))
		self.slow_program_encoder = nn.Sequential(nn.Linear(s_dim,qkan_s_dim),  QKAN(
                        width=[qkan_s_dim, qkan_s_dim],
                        reps=1,
                        ba_trainable=True,
                    ), nn.Linear(qkan_s_dim,qkan_s_dim))
		self.slow_program_layer_idx = torch.nn.Linear(qkan_s_dim, self.q_depth)
		self.slow_program_qubit_idx = torch.nn.Linear(qkan_s_dim, self.n_qubits)

		self.classical_preprocessing = torch.nn.Linear(s_dim, self.n_qubits)
		self.fast_gate = torch.nn.Linear(qkan_s_dim, 1)

		set_init([self.slow_program_layer_idx,self.slow_program_qubit_idx,self.classical_preprocessing, self.fast_gate])

	def forward(self, batch_item, previous_circuit_param):
		res = self.slow_program_encoder(batch_item)
		m_1 = nn.Tanh()
		m_2 = nn.Tanh()

		res_layer_idx = self.slow_program_layer_idx(m_1(res))
		res_qubit_idx = self.slow_program_qubit_idx(m_2(res))

		# Calculate the VQC params
		out_circuit_params = []
		for layer_idx, qubit_idx in zip(res_layer_idx, res_qubit_idx):
			outer_product = torch.outer(layer_idx, qubit_idx)
			out_circuit_params.append(outer_product)
		out_circuit_params = torch.stack(out_circuit_params)

		# Blend the new circuit params with the ones carried over
		gate = torch.sigmoid(self.fast_gate(res))
		out_circuit_params = gate * previous_circuit_param + (1-gate) * out_circuit_params

		# Go through the VQC
		res = self.classical_preprocessing(batch_item)
		res = self.q_func(res, out_circuit_params)

		return res, out_circuit_params

	def initial_fast_params(self, batch_size):

		return torch.zeros(batch_size, self.q_depth, self.n_qubits)


### FWP Module


class FWP(nn.Module):
	# FWP module: processes the whole sequence
	# (N, T, F)
	# N: number of batch
	# T: number of time-step
	# F: number of features
	def __init__(self, s_dim, latent_dim):
		super().__init__()
		self.fwp_cell = FWPCell(s_dim = s_dim, latent_dim = latent_dim)

	def forward(self, batch_item):
		batch_size = batch_item.shape[0]
		initial_fast_params = self.fwp_cell.initial_fast_params(batch_size)
		time_length = batch_item.shape[1]

		output_collection_list = []

		for t in range(time_length):
			out_batch, initial_fast_params = self.fwp_cell(batch_item[:, t, :], initial_fast_params)
			output_collection_list.append(out_batch)

		res = torch.stack(output_collection_list)

		return res
