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

import torch

from sklearn.preprocessing import MinMaxScaler


from torch.utils.data import Dataset


x = np.arange(-2, 20, 0.01)
data = 0.
for n in range(11):
	data += np.exp(-10*(x-2*n)**2)*np.exp(-x/16)


###
class DelayedQCDataset(Dataset):
	"""
	Slice a single time-series into a (seq_len -> next point) PyTorch Dataset.
	- Default pre_scaled=False: a MinMaxScaler rescales the data to [-1, 1] internally
	- If the data is already scaled, pass pre_scaled=True
	- Outputs:
		x: [N, seq_len]
		y: [N]
	"""
	def __init__(self,
				 data_src: np.ndarray = data,
				 seq_len: int = 4,
				 pre_scaled: bool = False,
				 feature_range = (-1, 1),
				 dtype = torch.float32):
		self.seq_len = int(seq_len)
		self.dtype = dtype

		arr = np.asarray(data_src).reshape(-1)

		# Scaling control
		self.scaler = None
		if pre_scaled:
			scaled = arr
		else:
			self.scaler = MinMaxScaler(feature_range=feature_range)
			scaled = self.scaler.fit_transform(arr.reshape(-1, 1)).reshape(-1)

		# Slice into (x, y) pairs
		xs, ys = [], []
		for i in range(len(scaled) - self.seq_len - 1):
			xs.append(scaled[i : i + self.seq_len])
			ys.append(scaled[i + self.seq_len])

		self.x = torch.tensor(np.array(xs), dtype=self.dtype)   # [N, seq_len]
		self.y = torch.tensor(np.array(ys), dtype=self.dtype)   # [N]

	def __len__(self):
		return self.x.shape[0]

	def __getitem__(self, idx):
		return self.x[idx].unsqueeze(-1), self.y[idx] # unsqueeze(-1) gives a trailing feature axis: (seq_len, 1)

	def inverse_transform_y(self, y_tensor: torch.Tensor):
		"""
		If this Dataset applied scaling (pre_scaled=False), map y from the scaled
		domain back to the original scale; if pre_scaled=True, return y unchanged.
		"""
		if self.scaler is None:
			return y_tensor
		y_np = y_tensor.detach().cpu().numpy().reshape(-1, 1)
		inv = self.scaler.inverse_transform(y_np).reshape(-1)
		return torch.tensor(inv, dtype=y_tensor.dtype)


	

###
