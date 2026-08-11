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

# Damped simple-harmonic-motion (damped pendulum) dataset.
#
# The 2nd-order ODE formulation and its odeint integration are adapted from:
#   "Solving 2nd order ODE for a simple pendulum using python", skill-lync.
#   https://skill-lync.com/projects/Solving-2nd-order-ODE-for-a-simple-pendulum-using-python-40080

import numpy as np
from scipy.integrate import odeint
import math

import torch
from torch.utils.data import Dataset

from sklearn.preprocessing import MinMaxScaler

def system(theta,t,b,g,l,m):
	theta1 = theta[0]
	theta2 = theta[1]
	dtheta1_dt = theta2
	dtheta2_dt = -(b/m)*theta2-g*math.sin(theta1)
	dtheta_dt=[dtheta1_dt,dtheta2_dt]

	return dtheta_dt


b=0.15
g=9.81
l=1
m=1


theta_0 = [0,3]


t = np.linspace(0,20,240)


theta = odeint(system,theta_0,t,args = (b,g,l,m))


# normalize the dataset
scaler = MinMaxScaler(feature_range=(-1, 1))
dataset = scaler.fit_transform(theta[:,1].reshape(-1, 1))


# PyTorch Dataset

class DampedSHMSequenceDataset(Dataset):
	"""
	Turn a time-series (e.g. theta[:,1] or an already-scaled dataset) into a set of
	(x, y) samples that predict the next point from seq_len consecutive points.

	By default data_src is assumed to be already scaled (pre_scaled=True);
	pass pre_scaled=False to let this Dataset do the scaling internally.
	"""
	def __init__(self,
				 data_src=dataset,         # defaults to the precomputed module-level `dataset`
				 seq_len=4,
				 pre_scaled=True,
				 feature_range=(-1, 1),
				 dtype=torch.float32):
		self.seq_len = int(seq_len)
		self.dtype = dtype

		arr = np.asarray(data_src).reshape(-1)

		# Whether to scale inside the Dataset
		self.scaler = None
		if pre_scaled:
			scaled = arr
		else:
			self.scaler = MinMaxScaler(feature_range=feature_range)
			scaled = self.scaler.fit_transform(arr.reshape(-1, 1)).reshape(-1)

		# Pre-slice all (x, y) pairs
		xs, ys = [], []
		for i in range(len(scaled) - self.seq_len - 1):
			xs.append(scaled[i : i + self.seq_len])
			ys.append(scaled[i + self.seq_len])

		self.x = torch.tensor(np.array(xs), dtype=self.dtype)   # [N, seq_len]
		self.y = torch.tensor(np.array(ys), dtype=self.dtype)   # [N]

	def __len__(self):
		return self.x.shape[0]

	def __getitem__(self, idx):
		# return self.x[idx], self.y[idx]
		return self.x[idx].unsqueeze(-1), self.y[idx] # unsqueeze(-1) gives a trailing feature axis: (seq_len, 1)

	def inverse_transform_y(self, y_tensor: torch.Tensor):
		"""
		If this Dataset scaled internally (pre_scaled=False), use this to map outputs
		back to the original units. If pre_scaled=True (the default), inputs are
		returned unchanged.
		"""
		if self.scaler is None:
			return y_tensor
		y_np = y_tensor.detach().cpu().numpy().reshape(-1, 1)
		inv = self.scaler.inverse_transform(y_np).reshape(-1)
		return torch.tensor(inv, dtype=y_tensor.dtype)
