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

"""Actor-critic policy wrapper around the GQKAN-QKANFWP fast-weight programmer.

Exposes ``QuantumFWPNet``, the network used by ``run_gqkan_qkanfwp.py``. The fast
weights are carried between steps by the caller as ``(theta, base_weight)``.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gqkan_qkanfwp import FWP


class QuantumFWPNet(nn.Module):
	def __init__(self, s_dim, a_dim, rnn_as_reservoir=False):
		super(QuantumFWPNet, self).__init__()

		self.state_space = s_dim
		self.action_space = a_dim

		self.latent_dim = 8

		qkan_s_dim = int(np.ceil(np.log2(self.latent_dim)))
		self.fwp    = FWP(s_dim = self.state_space, latent_dim = self.latent_dim, qkan_s_dim = qkan_s_dim)

		# Actor and critic heads
		self.Linear2 = nn.Linear(qkan_s_dim, self.action_space)
		self.Linear3 = nn.Linear(qkan_s_dim, 1)

		self.distribution = torch.distributions.Categorical

	def forward_action(self, x, previous_theta_fast, previous_base_fast):
		"""Single-step forward used while acting; carries the fast weights forward."""
		out_batch, updated_theta_fast, updated_base_fast = self.fwp.fwp_cell(x, previous_theta_fast, previous_base_fast)

		logits = self.Linear2(out_batch).squeeze(0)
		values = self.Linear3(out_batch).squeeze(0)

		return logits, values, updated_theta_fast, updated_base_fast

	def forward_loss(self, x):
		"""Whole-sequence forward used to build the update loss."""
		x = x.unsqueeze(0)  # (T, F) -> (1, T, F): the update buffer is one sequence

		out_batch = self.fwp(x)

		# squeeze(0) keeps logits and values shaped like the non-recurrent case
		logits = self.Linear2(out_batch).squeeze(0)
		values = self.Linear3(out_batch).squeeze(0)

		return logits, values

	def choose_action(self, s, previous_theta_fast, previous_base_fast):
		self.eval()

		logits, _, updated_theta_fast, updated_base_fast = self.forward_action(s, previous_theta_fast, previous_base_fast)

		prob = F.softmax(logits, dim = -1).data
		m = self.distribution(prob)

		return m.sample().numpy(), updated_theta_fast, updated_base_fast

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
