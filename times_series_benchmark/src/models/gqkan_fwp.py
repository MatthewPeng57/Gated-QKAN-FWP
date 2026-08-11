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

from qkan import QKAN

torch.set_default_dtype(torch.float32)

class FWPCell(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super().__init__()
		hidden_size = 4
		self.input_size = input_size
		self.output_size = output_size
		self.in_size = 4
		self.out_size = 5
		# ---- Input preprocessing ----
		self.input_pre = nn.Linear(input_size, self.in_size)
  
		# ---- post preprocessing ----
		self.output_post = nn.Linear(self.out_size, output_size)
		input_resize = 2
		# ---- Slow network ----
		self.encoder = nn.Sequential(nn.Linear(input_size,input_resize),  QKAN(
                        width=[input_resize, hidden_size],
                        reps=1,
                        # postact_bias_trainable=True,
                		# postact_weight_trainable=True,
                        ba_trainable=True,
                        solver="cutn",
                    ))

		# ---- Fast weight generators (NO projection) ----
		self.to_l = nn.Linear(hidden_size, self.out_size)   # rows (O)
		self.to_q = nn.Linear(hidden_size, self.in_size)    # cols (F)

		# ---- Bias head ----
		self.to_bias = nn.Linear(hidden_size, output_size)

		# ---- Gating ----
		self.fast_gate = nn.Linear(hidden_size, 1)
		self.fast_gate.bias.data.fill_(2.0)

	def forward(self, x, prev_weight, prev_bias):
		"""
		x: (B, F)
		prev_W: (B, O, F)
		"""

		B = x.size(0)

		# ---- Slow features ----
		h = self.encoder(x)  # (B, H)

		# ---- Input preprocessing ----
		x = self.input_pre(x)  # (B, F)

		

		# ---- Generate fast weights ----
		l = self.to_l(h)  # (B, O)
		q = self.to_q(h)  # (B, F)

        # Outer product using einsum → (B, O, F)
		W = torch.einsum('bi,bj->bij', l, q)

        # Optional stabilization (recommended)
        # W = torch.tanh(W)

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

		out = self.output_post(out)
		return out, W, b

	def initial_fast_params(self, batch_size, device):
		# Fast weight matches W from forward(): (B, out_size, in_size).
		# Fast bias matches to_bias output: (B, output_size), a shared bias that
		# broadcasts across the out_size rows.
		return (torch.zeros(batch_size, self.out_size, self.in_size, device=device),
				torch.zeros(batch_size, self.output_size, device=device))


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

        fast_weight, fast_bias = self.fwp_cell.initial_fast_params(B, self.device)

        outputs = []

        for t in range(T):
            x_t = x[:, t, :]
            out, fast_weight, fast_bias = self.fwp_cell(x_t, fast_weight, fast_bias)
            outputs.append(out)

        # (T, B, O)
        return torch.stack(outputs)
