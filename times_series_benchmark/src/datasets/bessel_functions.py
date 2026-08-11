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

import numpy as np
import scipy.special

import torch

from torch.utils.data import Dataset

from sklearn.preprocessing import MinMaxScaler

# Bessel function of the first kind, order 2, sampled on a fixed grid.
x = np.linspace(2, 100, 256)
j2 = scipy.special.jv(2, x)


# PyTorch Dataset
class BesselSequenceDataset(Dataset):
	"""
	Turn a single series (by default the J2 samples above) into a
	(seq_len -> next step) PyTorch Dataset.
	- Input: data_src (1D numpy array); MinMaxScaler to [-1, 1] is applied automatically
	- Sampling: seq_len consecutive points as x, the next point as y
	- Returns: x shape = (seq_len,); y shape = ()
	"""
	def __init__(self, data_src = j2, seq_len=4, feature_range=(-1, 1), dtype=torch.float32):
		# Keep the scaler so the scaling can be inverted later
		self.scaler = MinMaxScaler(feature_range=feature_range)
		# Make sure the input is a 1D np.ndarray
		data_src = np.asarray(data_src).reshape(-1)
		# Scale as 2D, then flatten back to 1D
		scaled = self.scaler.fit_transform(data_src.reshape(-1, 1)).reshape(-1)

		self.seq_len = int(seq_len)
		self.dtype = dtype

		# Pre-slice all (x, y) pairs
		xs, ys = [], []
		for i in range(len(scaled) - self.seq_len - 1):
			xs.append(scaled[i : i + self.seq_len])
			ys.append(scaled[i + self.seq_len])
		# Convert to torch.Tensor
		self.x = torch.tensor(np.array(xs), dtype=self.dtype)        # [N, seq_len]
		self.y = torch.tensor(np.array(ys), dtype=self.dtype)        # [N]

	def __len__(self):
		return self.x.shape[0]

	def __getitem__(self, idx):
		return self.x[idx].unsqueeze(-1), self.y[idx] # unsqueeze(-1) gives a trailing feature axis: (seq_len, 1)

	# Optional helper for inverting the scaling
	def inverse_transform_y(self, y_tensor: torch.Tensor):
		"""
		Convert a scalar or vector y (scaled) back to the original scale.
		"""
		y_np = y_tensor.detach().cpu().numpy().reshape(-1, 1)
		return torch.tensor(self.scaler.inverse_transform(y_np).reshape(-1), dtype=y_tensor.dtype)
