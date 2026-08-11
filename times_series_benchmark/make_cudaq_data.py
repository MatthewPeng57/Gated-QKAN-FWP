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

"""Regenerate the shipped CUDA-Q benchmark CSVs in ``cuda_q_data/``.

This is the authoritative tool that produced the two shipped CSVs. The
dataset classes in ``src/datasets/`` load those CSVs by default so the
benchmark runs without CUDA-Q; run this script only to reproduce them.

Requires ``cudaq`` + ``cupy`` + an NVIDIA GPU. The generators are
deterministic, so re-running reproduces the shipped CSVs bit-exactly.

The parameters below are the single source of truth and must stay in sync
with the dataset classes (``jaynes_cummings``, ``transmon_resonator``),
which re-simulate with the same values when ``QFWP_REGEN_CUDAQ=1`` is set.

Usage:
    python make_cudaq_data.py
"""

import numpy as np
import os

# Import the CUDA-Q generators
from src.datasets.jaynes_cummings_gen import generate_jc_dynamics
from src.datasets.transmon_gen import generate_transmon_dynamics

# Ensure the data directory exists
os.makedirs("cuda_q_data", exist_ok=True)

print("1. Generating Jaynes-Cummings...")
jc_data = generate_jc_dynamics(num_steps=3000, decay=True)
np.savetxt("cuda_q_data/jaynes_cummings.csv", jc_data, delimiter=",")

print("2. Generating Transmon Resonator...")
transmon_data = generate_transmon_dynamics(num_steps=3000, t_max=25.0)
np.savetxt("cuda_q_data/transmon.csv", transmon_data, delimiter=",")

print("\nSuccess! All datasets saved to the 'cuda_q_data/' folder.")
