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
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from .cudaq_data import load_cudaq_series

# t_max used to produce the shipped transmon.csv (see make_cudaq_data.py):
# a smooth 25 ns window of oscillations.
TRANSMON_T_MAX = 25.0


def _simulate(num_steps):
    # Imported lazily: requires cudaq + cupy + an NVIDIA GPU.
    from .transmon_gen import generate_transmon_dynamics
    return generate_transmon_dynamics(num_steps=num_steps, t_max=TRANSMON_T_MAX)


class TransmonDataset(Dataset):
    def __init__(self, seq_len=10, num_steps=3000):
        self.seq_len = int(seq_len)

        # Load the shipped 25ns window (or re-simulate when opted in)
        raw_data = load_cudaq_series(
            "transmon",
            lambda: _simulate(num_steps),
            expected_len=num_steps,
        )

        if len(raw_data) <= self.seq_len + 1:
            raise ValueError(f"Data length ({len(raw_data)}) is too short for seq_len ({self.seq_len}).")

        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_data = self.scaler.fit_transform(raw_data.reshape(-1, 1)).reshape(-1)
        
        self.x, self.y = [], []
        for i in range(len(scaled_data) - self.seq_len - 1):
            self.x.append(scaled_data[i : i + self.seq_len])
            self.y.append(scaled_data[i + self.seq_len])
            
        self.x = torch.tensor(np.array(self.x), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx].unsqueeze(-1), self.y[idx]

    def inverse_transform(self, data_tensor: torch.Tensor):
        if getattr(self, 'scaler', None) is None:
            return data_tensor
        data_np = data_tensor.detach().cpu().numpy().reshape(-1, 1)
        inv_data = self.scaler.inverse_transform(data_np).reshape(-1)
        return torch.tensor(inv_data, dtype=data_tensor.dtype, device=data_tensor.device)
