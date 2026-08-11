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

from qkan import QKAN

import torch
import torch.nn as nn

import pennylane as qml
from pennylane import numpy as np

from .utils.vqc_components import H_layer
from .utils.vqc_components import RY_layer
from .utils.vqc_components import entangling_layer

torch.set_default_dtype(torch.float32)

# Define actual circuit architecture
def q_function(x, q_weights, n_class):
# def q_function(x, q_weights):
	""" The variational quantum circuit. """

	# Reshape weights
	# θ = θ.reshape(vqc_depth, n_qubits)

	# Start from state |+> , unbiased w.r.t. |0> and |1>

	n_dep = q_weights.shape[1]
	n_qub = q_weights.shape[2]

	H_layer(n_qub)

	# Embed features in the quantum node
	RY_layer(x)

	# Sequence of trainable variational layers
	for k in range(n_dep):
		# print("k: {}".format(k))
		entangling_layer(n_qub)
		RY_layer(q_weights[:,k])

	# Expectation values in the Z basis
	exp_vals = [qml.expval(qml.PauliZ(position)) for position in range(n_class)]  # only measure first "n_class" of qubits and discard the rest
	return exp_vals




class VQC(nn.Module):
	def __init__(self, vqc_depth, n_qubits, n_class):
		super().__init__()
		# self.weights = nn.Parameter(0.01 * torch.randn(vqc_depth, n_qubits))  # g rotation params
		self.dev = qml.device("default.qubit", wires=n_qubits)  # Can use different simulation backend or quantum computers.

		# q_layer_func = partial(q_function, n_class = n_class)

		self.VQC = qml.QNode(q_function, self.dev, interface = "torch")
		# self.VQC = qml.QNode(HideSignature(q_layer_func), self.dev, interface = "torch")
		

		self.n_class = n_class


	def forward(self, X, fast_param):
		# y_preds = torch.stack([torch.stack(self.VQC(x, self.weights, self.n_class)) for x in X]) # PennyLane 0.35.1
		# print("X (QLSTM-v0-Batch): {}".format(X))
		y_preds = torch.stack(self.VQC(X, fast_param, self.n_class)).T
		# y_preds = torch.stack(self.VQC(X, self.weights)).T
		# print("y_preds (QLSTM-v0-Batch): {}".format(y_preds))
		# print("y_preds.shape: {}".format(y_preds.shape))
		return y_preds

##############



### FWP Cell Module


class FWPCell(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, vqc_depth):
		super().__init__()

		a_dim = 4
		# latent_dim = 8
		latent_dim = hidden_size
		self.n_qubits = latent_dim
		self.q_depth = vqc_depth
		# self.q_depth = 2

		dev = qml.device("default.qubit", wires = self.n_qubits)
		# self.q_func = BatchVQC(VQCWrapper(qml.QNode(quantum_net, dev, interface = "torch"), a_dim))
		# self.q_func = BatchVQC(qml.QNode(quantum_net, dev, interface = "torch"))
		# self.q_func = BatchVQC(qml.QNode(HideSignature(partial(quantum_net, n_outputs = a_dim)), dev, interface = "torch"))
		self.q_func =  VQC(vqc_depth = vqc_depth, n_qubits = self.n_qubits , n_class = a_dim)

		# self.slow_program_encoder = torch.nn.Linear(input_size, latent_dim)
		qkan_s_dim = int(np.ceil(np.log2(latent_dim)))
		# self.slow_program_encoder = torch.nn.Linear(input_size, latent_dim)
		self.slow_program_encoder = nn.Sequential(nn.Linear(input_size,qkan_s_dim),  QKAN(
                        width=[qkan_s_dim, qkan_s_dim],
                        reps=1,
                        ba_trainable=True,
                    ))
		self.slow_program_layer_idx = torch.nn.Linear(qkan_s_dim, self.q_depth)
		self.slow_program_qubit_idx = torch.nn.Linear(qkan_s_dim, self.n_qubits)

		self.post_processing = torch.nn.Linear(a_dim, output_size) 
		self.fast_gate = torch.nn.Linear(qkan_s_dim, output_size)

		self.fast_gate.bias.data.fill_(2.0)  #removing this will prob cause some platau

	def forward(self, batch_item, previous_circuit_param):
		res = self.slow_program_encoder(batch_item)

		res_layer_idx = self.slow_program_layer_idx(res)
		res_qubit_idx = self.slow_program_qubit_idx(res)

		# Calculate the VQC params
		out_circuit_params = []
		for layer_idx, qubit_idx in zip(res_layer_idx, res_qubit_idx):
			outer_product = torch.outer(layer_idx, qubit_idx)
			# print("outer_product: {}".format(outer_product))
			out_circuit_params.append(outer_product)
		
		out_circuit_params = torch.stack(out_circuit_params)

		# out_circuit_params = torch.tanh(out_circuit_params)  # performance will be a bit better
  
		gate = torch.sigmoid(self.fast_gate(res))
		gate_expanded = gate.view(-1,1,1)
		
  

		# Add of previous_circuit_param and out_circuit_params
		# out_circuit_params = torch.add(out_circuit_params, previous_circuit_param)
		# one = torch.ones_like(gate_expanded)
		out_circuit_params =  (1 - gate_expanded) * out_circuit_params + gate_expanded * previous_circuit_param 

		# Go through the VQC
		# print(f'batch_item :{batch_item.shape}, out_circuit_params : {out_circuit_params.shape}')
		res = self.q_func(batch_item, out_circuit_params).float()

		# Post-processing 

		res = self.post_processing(res)


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
	def __init__(self, qfwp_cell):
		super().__init__()
		self.fwp_cell = qfwp_cell

		

	def forward(self, x):
		batch_size, seq_len, _ = x.size()
		initial_fast_params = self.fwp_cell.initial_fast_params(batch_size)

		output_collection_list = []

		for t in range(seq_len):
			x_t = x[:, t, :]  # Extract the t-th time step
			out_batch, initial_fast_params = self.fwp_cell(x_t, initial_fast_params)
			output_collection_list.append(out_batch)

		res = torch.stack(output_collection_list)
		
		# print(f'res shape {res.shape}')

		return res
