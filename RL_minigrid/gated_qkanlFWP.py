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

"""Gated QKAN-linear fast-weight programmer.

A QKAN-based slow programmer reads the observation and emits the rows and
columns of a fast weight matrix, whose outer product forms a full ``(B, O, F)``
fast linear layer plus bias. A sigmoid gate smooths that fast layer against the
one carried over from the previous step. A classical linear layer preprocesses
the observation before the fast layer is applied.
"""

import torch
import torch.nn as nn

from utils import set_init
from qkan import QKAN


### FWP Cell Module

class FWPCell(nn.Module):
	def __init__(self, s_dim, latent_dim):
		super().__init__()
		hidden_size = latent_dim
		self.input_size = s_dim
		self.in_size = latent_dim
		self.out_size = latent_dim
		# ---- Input preprocessing ----
		self.input_pre = nn.Linear(self.input_size , self.in_size)
		qkan_s_dim = 5

		# ---- Slow network ----
		self.slow_program_encoder = nn.Sequential(nn.Linear(self.input_size ,qkan_s_dim),  QKAN(
                        width=[qkan_s_dim, qkan_s_dim],
                        reps=1,
                        ba_trainable=True,
                    ), nn.Linear(qkan_s_dim,hidden_size))

		# ---- Fast weight generators (NO projection) ----
		self.to_l = nn.Linear(hidden_size, self.out_size)   # rows (O)
		self.to_q = nn.Linear(hidden_size, self.in_size)    # cols (F)

		# ---- Bias head ----
		self.to_bias = nn.Linear(hidden_size, self.out_size)

		# ---- Gating ----
		# the positive bias starts the gate near 1 so the carried-over fast
		# weights dominate early in training
		self.fast_gate = nn.Linear(hidden_size, 1)
		self.fast_gate.bias.data.fill_(2.0)
		set_init([self.to_l,self.to_q,self.input_pre, self.fast_gate,self.slow_program_encoder,self.to_bias])

	def forward(self, x, prev_weight, prev_bias):
		# x: (B, F), prev_weight: (B, O, F), prev_bias: (B, O)
		B = x.size(0)
		h = self.slow_program_encoder(x)
		m_1 = nn.Tanh()
		m_2 = nn.Tanh()

		# ---- Generate fast weights ----
		l = self.to_l(m_1(h))
		q = self.to_q(m_2(h))

		# ---- Input preprocessing ----
		x = self.input_pre(x)  # (B, F)

		# Outer product using einsum -> (B, O, F)
		W = torch.einsum('bi,bj->bij', l, q)

		# ---- Gating (temporal smoothing) ----
		gate = torch.sigmoid(self.fast_gate(h))  # (B, 1)
		gate_w = gate.view(B, 1, 1)

		W = (1 - gate_w) * W + gate_w * prev_weight

		# ---- Bias ----
		gate_b = gate.view(B, 1)
		b = self.to_bias(h)  # (B, O)
		b = (1 - gate_b) * b + gate_b * prev_bias

		# ---- Apply fast linear layer (using bmm) ----
		x_unsq = x.unsqueeze(-1)              # (B, F, 1)
		out = torch.bmm(W, x_unsq).squeeze(-1) + b  # (B, O)

		return out, W, b

	def initial_fast_params(self, batch_size):

		return torch.zeros(batch_size, self.out_size, self.in_size), torch.zeros(batch_size, self.out_size)


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
		initial_fast_weight,initial_fast_bias = self.fwp_cell.initial_fast_params(batch_size)
		time_length = batch_item.shape[1]

		output_collection_list = []

		for t in range(time_length):
			out_batch, initial_fast_weight,initial_fast_bias = self.fwp_cell(batch_item[:, t, :], initial_fast_weight,initial_fast_bias)
			output_collection_list.append(out_batch)

		res = torch.stack(output_collection_list)

		return res
