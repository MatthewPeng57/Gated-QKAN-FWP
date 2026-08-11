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

"""Actor-critic policy wrapper around the GQKAN-FWP fast-weight programmer.

Exposes ``QuantumFWPNet``, the network used by ``run_gqkan_fwp.py``. The fast
weights are carried between steps by the caller as ``(fast_weight, fast_bias)``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from gqkan_fwp import FWP


class QuantumFWPNet(nn.Module):
	def __init__(self, s_dim, a_dim):
		super(QuantumFWPNet, self).__init__()

		self.state_space = s_dim
		self.action_space = a_dim

		self.latent_dim = 8

		self.fwp  = FWP(s_dim = self.state_space, latent_dim = self.latent_dim)

		# Actor and critic heads
		self.Linear2 = nn.Linear(self.latent_dim, self.action_space)
		self.Linear3 = nn.Linear(self.latent_dim, 1)

		self.distribution = torch.distributions.Categorical

	def forward_action(self, x, prev_weight, prev_bias):
		"""Single-step forward used while acting; carries the fast weights forward."""
		out_batch, fast_weight, fast_bias = self.fwp.fwp_cell(x, prev_weight, prev_bias)

		logits = self.Linear2(out_batch).squeeze(0)
		values = self.Linear3(out_batch).squeeze(0)

		return logits, values, fast_weight, fast_bias

	def forward_loss(self, x):
		"""Whole-sequence forward used to build the update loss."""
		x = x.unsqueeze(0)  # (T, F) -> (1, T, F): the update buffer is one sequence

		out_batch = self.fwp(x)

		# squeeze(0) keeps logits and values shaped like the non-recurrent case
		logits = self.Linear2(out_batch).squeeze(0)
		values = self.Linear3(out_batch).squeeze(0)

		return logits, values

	def choose_action(self, s, prev_weight, prev_bias):
		self.eval()

		logits, _, fast_weight, fast_bias = self.forward_action(s, prev_weight, prev_bias)

		prob = F.softmax(logits, dim = -1).data
		m = self.distribution(prob)

		return m.sample().numpy(), fast_weight, fast_bias

	def loss_func(self, s, a, v_t):
		self.train()

		logits, values = self.forward_loss(s)
		td = v_t - values.squeeze(1)
		c_loss = td.pow(2)

		probs = F.softmax(logits.squeeze(1), dim=1)
		m = self.distribution(probs)
		exp_v = m.log_prob(a) * td.detach().squeeze()
		a_loss = -exp_v
		total_loss = (c_loss + a_loss).mean()
		return total_loss
