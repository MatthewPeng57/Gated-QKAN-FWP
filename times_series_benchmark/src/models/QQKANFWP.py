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

import torch
import torch.nn as nn

from pennylane import numpy as np

from qkan import QKAN
from .utils.qkan_fast import QKAN as fast_QKAN

torch.set_default_dtype(torch.float32)




### FWP Cell Module


class FWPCell(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, vqc_depth):
		super().__init__()

		a_dim = 4
		# latent_dim = 8
		
		# self.n_qubits = latent_dim
		# self.q_depth = vqc_depth
		# self.q_depth = 2

		
		qkan_s_dim = int(np.ceil(np.log2(hidden_size)))  
		# qkan_s_dim  = 4
		latent_dim = qkan_s_dim + 2
		# latent_dim = 5

  
		self.qkan_layer = fast_QKAN(
			width=[qkan_s_dim, qkan_s_dim],
			reps=1,
			solver="cutn",
		)
  
		layer = self.qkan_layer.layers[0]

		self.theta_shape = layer.theta.shape      # (3,3,2,2)
		self.base_shape  = layer.base_weight.shape # (3,3)

		self.theta_num = layer.theta.numel()
		self.base_num  = layer.base_weight.numel()
		
		# self.slow_program_encoder = torch.nn.Linear(input_size, latent_dim)
		self.slow_program_encoder = nn.Sequential(nn.Linear(input_size,qkan_s_dim),  QKAN(
                        width=[qkan_s_dim, qkan_s_dim],
                        reps=1,
                        ba_trainable=True,
                        solver="cutn",
                    ), nn.Linear(qkan_s_dim,latent_dim))
  
		# low-rank theta generator
		self.theta_head_A = nn.Linear(latent_dim, self.theta_shape[0]*self.theta_shape[2])
		self.theta_head_B = nn.Linear(latent_dim, self.theta_shape[1]*self.theta_shape[3])

		# base weight generator
		# self.base_head = nn.Linear(latent_dim, self.base_num)
  
		# self.slow_program_layer_idx = torch.nn.Linear(qkan_s_dim, self.q_depth)
		# self.slow_program_qubit_idx = torch.nn.Linear(qkan_s_dim, self.n_qubits)
		

		self.prelinear =  torch.nn.Linear(input_size, qkan_s_dim) 
		self.post_processing = torch.nn.Linear(qkan_s_dim, output_size) 
		
		self.fast_gate = torch.nn.Linear(latent_dim, output_size)

		self.fast_gate.bias.data.fill_(2.0)

	def forward(self, batch_item, fast_theta, fast_base):
		
		batch = batch_item.shape[0]
		res = self.slow_program_encoder(batch_item)

		# -------- theta generation --------

		A = self.theta_head_A(res)
		B = self.theta_head_B(res)

		A = A.view(batch,
				self.theta_shape[0],
				self.theta_shape[2])

		B = B.view(batch,
				self.theta_shape[1],
				self.theta_shape[3])

		# outer product
		theta_new = torch.einsum(
			"bld,bqe->blqde",
			A,
			B
		)

		gate = torch.sigmoid(self.fast_gate(res))
		gate_expanded = gate.view(-1,1,1,1,1)
		
		theta = (1 - gate_expanded) * theta_new + gate_expanded * fast_theta
		
		batch_item = self.prelinear(batch_item)
		
		out = self.qkan_layer(batch_item, theta, None)
		res = self.post_processing(out)


		return res, theta, fast_base

	def initial_fast_params(self, batch_size, device):

		theta = torch.zeros(batch_size, *self.theta_shape, device = device)
		base  = torch.zeros(batch_size, *self.base_shape, device = device)

		return theta, base

### FWP Module


class FWP(nn.Module):
	# FWP module: processes the whole sequence
	# (N, T, F)
	# N: number of batch
	# T: number of time-step
	# F: number of features
	def __init__(self, qfwp_cell, device):
		super().__init__()
		self.fwp_cell = qfwp_cell

		self.device  = device

	def forward(self, x):
		batch_size, seq_len, _ = x.size()
		theta_fast, base_fast = self.fwp_cell.initial_fast_params(batch_size, self.device)

		output_collection_list = []

		for t in range(seq_len):
			x_t = x[:, t, :]  # Extract the t-th time step
			out_batch, theta_fast, base_fast = self.fwp_cell(x_t, theta_fast, base_fast)
			output_collection_list.append(out_batch)

		res = torch.stack(output_collection_list)
		
		# print(f'res shape {res.shape}')

		return res
