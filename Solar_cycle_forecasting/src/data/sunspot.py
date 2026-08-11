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

"""Sunspot data pipeline: sliding-window dataset + chronological split.

This is a verbatim numerical reproduction of the data pipeline the published
GQKAN-QKANFWP checkpoints were trained with. Two properties are deliberately
frozen so that new models remain directly comparable against those
checkpoints:

  1. **The shipped CSV is pinned.** ``data/Sunspots.csv`` is the exact CSV
     used for training (sha256
     ``15c3d116ad6c5a5427837ae4cec39aa9b4b2e4a0d8e374d501a4ac760fc50b35``,
     3265 monthly rows, ending 2021-01-31), read via a repo-relative path.
  2. **The MinMaxScaler is fit on the full series** (train+val+test), exactly
     as executed. This is a normalization leakage; it is kept rather than
     fixed so the comparison stays apples-to-apples. It is NOT a train-only
     fit.

For L=528, H=132 this yields N=2606 windows and a window-level chronological
80/10/10 split of (2084 / 260 / 262).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, Subset

# Repo root = the Solar_cycle_forecasting/ directory (this file is
# src/data/sunspot.py, so two levels up).
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = _REPO_ROOT / "data" / "Sunspots.csv"

_TARGET_COLUMN = "Monthly Mean Total Sunspot Number"


class SunspotSequenceDataset(Dataset):
    """Univariate direct-multi-step sunspot dataset.

    ``x`` shape ``(seq_len, 1)``; ``y`` shape ``(horizon,)``.

    Args:
        csv_path: Path to the SILSO monthly-mean CSV.
        seq_len: Input window length L (published setting: 528).
        horizon: Forecast horizon H (published setting: 132).
        dtype: Tensor dtype (default ``torch.float32``).
        smooth_window: Centered rolling-mean window applied *before* scaling.
            ``0`` (the default) keeps the series bit-identical to the trained
            checkpoints. A positive value smooths the series with a centered
            rolling mean of that many months before scaling.
    """

    def __init__(
        self,
        csv_path: str | Path = DEFAULT_CSV,
        seq_len: int = 528,
        horizon: int = 132,
        dtype: torch.dtype = torch.float32,
        smooth_window: int = 0,
    ) -> None:
        import pandas as pd  # local import keeps module import cheap

        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.seq_len = int(seq_len)
        self.horizon = int(horizon)
        self.dtype = dtype
        self.smooth_window = int(smooth_window)

        df = pd.read_csv(csv_path)
        data_src = df[_TARGET_COLUMN].ffill().bfill().values

        if self.smooth_window > 0:
            data_src = (
                pd.Series(data_src)
                .rolling(window=self.smooth_window, center=True, min_periods=1)
                .mean()
                .values
            )

        # Fit on the FULL series (frozen leakage — see module docstring).
        scaled = self.scaler.fit_transform(data_src.reshape(-1, 1)).reshape(-1)

        self.full_unscaled_df = df
        self.full_scaled_data = scaled

        xs, ys = [], []
        for i in range(len(scaled) - self.seq_len - self.horizon + 1):
            xs.append(scaled[i : i + self.seq_len])
            ys.append(scaled[i + self.seq_len : i + self.seq_len + self.horizon])

        self.x = torch.tensor(np.array(xs), dtype=self.dtype).unsqueeze(-1)  # (N, L, 1)
        self.y = torch.tensor(np.array(ys), dtype=self.dtype)                # (N, H)

    def inverse_transform(self, scaled_values) -> np.ndarray:
        """Invert the min-max scaling, preserving input shape."""
        original_shape = np.array(scaled_values).shape
        flat = np.array(scaled_values).reshape(-1, 1)
        return self.scaler.inverse_transform(flat).reshape(original_shape)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


@dataclass
class SunspotSplit:
    """Chronological 80/10/10 window-level split of a ``SunspotSequenceDataset``."""

    dataset: SunspotSequenceDataset
    train: Subset
    val: Subset
    test: Subset
    n_total: int
    n_train: int
    n_val: int
    n_test: int


def make_sunspot_split(
    csv_path: str | Path = DEFAULT_CSV,
    seq_len: int = 528,
    horizon: int = 132,
    smooth_window: int = 0,
) -> SunspotSplit:
    """Build the chronological sunspot window split.

    ``n_train = int(0.8 N)``, ``n_val = int(0.1 N)``, ``test = remainder`` — all
    chronological, with no shuffling of the split boundaries (the training
    DataLoader shuffles within the train subset).
    """
    ds = SunspotSequenceDataset(csv_path, seq_len, horizon, smooth_window=smooth_window)
    n_total = len(ds)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    train = Subset(ds, range(0, n_train))
    val = Subset(ds, range(n_train, n_train + n_val))
    test = Subset(ds, range(n_train + n_val, n_total))
    return SunspotSplit(
        dataset=ds,
        train=train,
        val=val,
        test=test,
        n_total=n_total,
        n_train=n_train,
        n_val=n_val,
        n_test=n_total - n_train - n_val,
    )


@dataclass
class DatasetBundle:
    """Datasets consumed by ``train_loading.make_loaders`` / ``run_training``.

    ``simulation_ds`` is the full windowed dataset (used for the rolling
    reconstruction figure); ``train_len`` is the number of training windows.
    """

    train_ds: Dataset
    test_ds: Dataset
    simulation_ds: Dataset
    train_len: int
    val_ds: Optional[Dataset] = None


def make_datasets(args) -> DatasetBundle:
    """Build the train/val/test/simulation dataset bundle from ``args``.

    Only ``dataset == "sunspots"`` is supported. Produces the 80/10/10
    window-level chronological split (``simulation_ds`` = the full windowed
    dataset, ``train_len`` = ``n_train``).
    """
    dataset = getattr(args, "dataset", "sunspots")
    if dataset != "sunspots":
        raise ValueError(
            f"Only the 'sunspots' dataset is available; got {dataset!r}."
        )
    split = make_sunspot_split(
        csv_path=DEFAULT_CSV,
        seq_len=args.window_len,
        horizon=args.horizon,
    )
    return DatasetBundle(
        train_ds=split.train,
        test_ds=split.test,
        simulation_ds=split.dataset,
        train_len=split.n_train,
        val_ds=split.val,
    )
