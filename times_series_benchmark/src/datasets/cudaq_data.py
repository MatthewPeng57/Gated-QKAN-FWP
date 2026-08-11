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

"""Access to the shipped, pre-simulated CUDA-Q benchmark series.

The two quantum datasets (``jaynes_cummings``, ``transmon``) ship as
pre-simulated CSV files under ``cuda_q_data/``.  Those CSVs are loaded
**by default** so that the benchmark runs on any machine, including one
without NVIDIA CUDA-Q, ``cupy`` or a GPU.

Live re-simulation is an explicit opt-in::

    QFWP_REGEN_CUDAQ=1 python train.py --dataset jaynes_cummings ...

Live regeneration requires ``cudaq`` + ``cupy`` + an NVIDIA GPU.  The
generators are deterministic, so regeneration reproduces the shipped CSVs
bit-exactly (verified: max |diff| = 0.0 for both datasets).

``make_cudaq_data.py`` at the project root is the authoritative tool that
produced the shipped CSVs; the dataset classes here call the same generators
with the same parameters.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Callable, Optional

import numpy as np

#: Environment variable that opts in to live CUDA-Q re-simulation.
REGEN_ENV_VAR = "QFWP_REGEN_CUDAQ"

_TRUTHY = {"1", "true", "yes", "on"}

# src/datasets/cudaq_data.py -> project root (times_series_benchmark/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CUDAQ_DATA_DIR = PROJECT_ROOT / "cuda_q_data"


def regeneration_requested() -> bool:
    """Return True when live CUDA-Q re-simulation was explicitly requested."""
    return os.environ.get(REGEN_ENV_VAR, "").strip().lower() in _TRUTHY


def load_cudaq_series(
    name: str,
    generator: Callable[[], np.ndarray],
    expected_len: Optional[int] = None,
) -> np.ndarray:
    """Return the 1-D series for a CUDA-Q dataset.

    Args:
        name: Dataset stem, matching ``cuda_q_data/{name}.csv``.
        generator: Zero-argument callable that re-simulates the series live.
            It must import ``cudaq`` lazily so that this module stays usable
            without CUDA-Q installed.
        expected_len: Number of steps the caller asked for. Used only to warn
            when the shipped CSV has a different length.

    Returns:
        The series as a 1-D float64 array.

    Raises:
        FileNotFoundError: If the shipped CSV is missing and regeneration was
            not requested.
    """
    if regeneration_requested():
        return np.asarray(generator(), dtype=np.float64).reshape(-1)

    csv_path = CUDAQ_DATA_DIR / f"{name}.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Shipped CUDA-Q data '{csv_path}' not found. Either restore the file "
            f"or set {REGEN_ENV_VAR}=1 to re-simulate it live "
            f"(requires cudaq + cupy + an NVIDIA GPU; see make_cudaq_data.py)."
        )

    data = np.loadtxt(csv_path, delimiter=",").reshape(-1)

    if expected_len is not None and len(data) != expected_len:
        warnings.warn(
            f"Shipped '{name}.csv' has {len(data)} steps but num_steps={expected_len} "
            f"was requested; using the shipped {len(data)} steps. Set "
            f"{REGEN_ENV_VAR}=1 to simulate {expected_len} steps instead.",
            RuntimeWarning,
            stacklevel=2,
        )

    return data
