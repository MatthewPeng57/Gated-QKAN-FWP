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

import torch
import torch.nn as nn

from .utils.qkan_fast import QKAN as fast_QKAN

torch.set_default_dtype(torch.float32)

class FWPCell(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super().__init__()
		hidden_size = hidden_size - 1
		self.input_size = input_size
		self.output_size = output_size
		self.in_size = 2
		self.out_size = 3
  
		self.qkan_layer = fast_QKAN(
			width=[self.in_size, self.out_size],
			solver = "cutn",
			reps=1,
			ba_trainable=True,
		)
  
		layer = self.qkan_layer.layers[0]

		self.theta_shape = layer.theta.shape      # (out_dim, in_dim, reps+1, 2)
		self.base_shape  = layer.base_weight.shape # (out_dim, in_dim)

		self.theta_num = layer.theta.numel()
		self.base_num  = layer.base_weight.numel()
  
		# ---- Input preprocessing ----
		self.input_pre = nn.Linear(input_size, self.in_size)
  
		# ---- post preprocessing ----
		self.output_post = nn.Linear(self.out_size, output_size)
		# ---- Slow network ----
		self.encoder = nn.Linear(input_size, hidden_size)

		# ---- Fast weight generators (NO projection) ----
		self.theta_head_A = nn.Linear(hidden_size, self.theta_shape[0]*self.theta_shape[2])
		self.theta_head_B = nn.Linear(hidden_size, self.theta_shape[1]*self.theta_shape[3])


		# ---- Gating ----
		self.fast_gate = nn.Linear(hidden_size, 1)
		self.fast_gate.bias.data.fill_(2.0)

	def forward(self, x, prev_theta):
		batch = x.shape[0]
		res = self.encoder(x)

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
		
		theta = (1 - gate_expanded) * theta_new + gate_expanded * prev_theta

		x = self.input_pre(x)

		out = self.qkan_layer(x, theta, None)
		out = self.output_post(out)


		return out, theta

	def initial_fast_params(self, batch_size, device):
		theta = torch.zeros(batch_size, *self.theta_shape, device=device)

		return theta


# ---- Sequence wrapper (unchanged logic, slightly cleaned) ----
class FWP(nn.Module):
    def __init__(self, fwp_cell, device):
        super().__init__()
        self.fwp_cell = fwp_cell
        self.device = device

    def forward(self, x):
        """
        x: (B, T, F)
        """
        B, T, _ = x.size()

        fast_theta = self.fwp_cell.initial_fast_params(B, self.device)

        outputs = []

        for t in range(T):
            x_t = x[:, t, :]
            out, fast_theta = self.fwp_cell(x_t, fast_theta)
            outputs.append(out)

        # (T, B, O)
        return torch.stack(outputs)
