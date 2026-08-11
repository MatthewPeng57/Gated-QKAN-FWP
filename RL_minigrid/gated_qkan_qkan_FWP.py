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

"""Gated QKAN-QKAN fast-weight programmer.

Both halves of the programmer are QKAN layers: the slow programmer that reads
the observation is an upstream ``qkan.QKAN``, and the layer it reprograms is the
vendored ``qkan_fast.QKAN``, which accepts per-call ``theta``. A sigmoid gate
blends the newly generated theta with the fast theta carried over from the
previous step.

This variant generates ``theta`` only; the base weights are left untouched and
passed through, and ``None`` is handed to the QKAN layer in their place.
"""

import torch
import torch.nn as nn

from qkan_fast import QKAN as fast_QKAN
from qkan import QKAN

from utils import set_init


class FWPCell(nn.Module):

	def __init__(self, s_dim, latent_dim, qkan_s_dim):
		super().__init__()

		self.qkan_layer = fast_QKAN(
			width=[qkan_s_dim, qkan_s_dim],
			reps=1,
		)

		layer = self.qkan_layer.layers[0]

		self.theta_shape = layer.theta.shape      # (3,3,2,2)
		self.base_shape  = layer.base_weight.shape # (3,3)

		self.theta_num = layer.theta.numel()
		self.base_num  = layer.base_weight.numel()

		# slow programmer
		self.encoder = nn.Sequential(nn.Linear(s_dim,qkan_s_dim),  QKAN(
                        width=[qkan_s_dim, qkan_s_dim],
                        reps=1,
                        ba_trainable=True,
                    ), nn.Linear(qkan_s_dim,latent_dim))

		# low-rank theta generator
		self.theta_head_A = nn.Linear(latent_dim, self.theta_shape[0]*self.theta_shape[2])
		self.theta_head_B = nn.Linear(latent_dim, self.theta_shape[1]*self.theta_shape[3])

		# input preprocessing
		self.classical_preprocessing = nn.Linear(s_dim, qkan_s_dim)

		# fast weight decay gate; the positive bias starts the gate near 1 so the
		# carried-over fast theta dominates early in training
		self.fast_gate = nn.Linear(latent_dim,1)
		self.fast_gate.bias.data.fill_(2.0)
		set_init([self.theta_head_A,self.theta_head_B,self.classical_preprocessing, self.fast_gate])

	def forward(self, x, fast_theta, fast_base):

		batch = x.shape[0]

		latent = torch.tanh(self.encoder(x))

		# -------- theta generation --------

		A = self.theta_head_A(latent)
		B = self.theta_head_B(latent)

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

		# -------- fast weight update --------

		gate = torch.sigmoid(self.fast_gate(latent))

		fast_theta = gate*fast_theta + (1-gate) * theta_new

		# -------- QKAN forward --------

		res = self.classical_preprocessing(x)

		# QKAN expects single weight set
		theta = fast_theta.squeeze(0)

		res = self.qkan_layer(res, theta, None)

		# fast_base is returned unchanged: this variant programs theta only
		return res, fast_theta, fast_base

	def initial_fast_params(self, batch_size):

		theta = torch.zeros(batch_size, *self.theta_shape)
		base  = torch.zeros(batch_size, *self.base_shape)

		return theta, base


class FWP(nn.Module):

    def __init__(self, s_dim, latent_dim, qkan_s_dim):
        super().__init__()

        self.fwp_cell = FWPCell(s_dim, latent_dim, qkan_s_dim)

    def forward(self, seq):

        batch_size, T, _ = seq.shape

        theta_fast, base_fast = self.fwp_cell.initial_fast_params(batch_size)

        outputs = []

        for t in range(T):

            out, theta_fast, base_fast = self.fwp_cell(
                seq[:,t,:],
                theta_fast,
                base_fast
            )

            outputs.append(out)

        return torch.stack(outputs)
