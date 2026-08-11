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


def _simulate(num_steps, decay):
    # Imported lazily: requires cudaq + cupy + an NVIDIA GPU.
    from .jaynes_cummings_gen import generate_jc_dynamics
    return generate_jc_dynamics(num_steps=num_steps, decay=decay)


class JaynesCummingsDataset(Dataset):
    """Qubit excitation dynamics of the driven Jaynes-Cummings model."""

    def __init__(self, seq_len=10, num_steps=3000, decay=True):
        self.seq_len = seq_len
        # 1. Load the shipped physical data (or re-simulate when opted in)
        raw_data = load_cudaq_series(
            "jaynes_cummings",
            lambda: _simulate(num_steps, decay),
            expected_len=num_steps,
        )

        # 2. Normalize
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_data = self.scaler.fit_transform(raw_data.reshape(-1, 1)).reshape(-1)
        
        # 3. Create sequences (Windowing)
        self.x, self.y = [], []
        for i in range(len(scaled_data) - seq_len - 1):
            self.x.append(scaled_data[i : i + seq_len])
            self.y.append(scaled_data[i + seq_len])
            
        self.x = torch.tensor(np.array(self.x), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        # Return [seq_len, 1] for the FWP/LSTM input
        return self.x[idx].unsqueeze(-1), self.y[idx]
    
    def inverse_transform(self, data_tensor: torch.Tensor):
        if getattr(self, 'scaler', None) is None:
            return data_tensor
        data_np = data_tensor.detach().cpu().numpy().reshape(-1, 1)
        inv_data = self.scaler.inverse_transform(data_np).reshape(-1)
        return torch.tensor(inv_data, dtype=data_tensor.dtype, device=data_tensor.device)
