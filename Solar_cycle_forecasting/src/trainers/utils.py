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

"""Logging and prediction dumping for training runs.

This repository deliberately ships no plotting code: runs emit CSVs
(``train_log.csv``, ``prediction_log.csv``, the final metrics summary) and
leave visualisation to the reader.
"""

from __future__ import annotations

import csv
import logging
import os
from typing import Optional, Union

import torch


def log_epoch(epoch, train_loss, test_loss, path):
	file_exists = path.is_file()
	with open(path, "a", newline="") as f:
		writer = csv.writer(f)
		# Write the header row only when creating the file.
		if not file_exists:
			writer.writerow(["epoch", "train_loss", "test_loss"])
		writer.writerow([epoch, train_loss, test_loss])


def setup_logger(path: str) -> logging.Logger:
	logger = logging.getLogger("train_logger")
	logger.setLevel(logging.INFO)
	logger.handlers.clear()

	fh = logging.FileHandler(path, mode="w")
	fh.setLevel(logging.INFO)
	fmt = logging.Formatter("%(asctime)s - %(message)s")
	fh.setFormatter(fmt)

	logger.addHandler(fh)
	logger.propagate = False
	return logger


def predict_and_log(
    args,
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    csv_path: Union[str, os.PathLike],
    split: str = "simulation",
    epoch: Optional[int] = None,
) -> None:
    model.eval()
    all_ytrue, all_ypred = [], []

    # Safely unwrap the dataset in case it's a torch.utils.data.Subset
    base_dataset = loader.dataset
    if hasattr(base_dataset, 'dataset'):
        base_dataset = base_dataset.dataset

    with torch.no_grad(), open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["epoch", "model", "split", "horizon", "y_true", "y_pred"])

        for xb, yb in loader:
            xb = xb.to(args.device, non_blocking=True)
            yb = yb.to(args.device, non_blocking=True)

            yhat = model(xb)

            # Model-specific output handling.
            if args.model == "gqkan_qkanfwp":
                yhat = yhat[-1]
            else:
                raise ValueError(f"Model '{args.model}' is not a valid choice.")

            # --- CRITICAL FIX: Extract to numpy arrays cleanly ---
            # If multi-step (horizon > 1), yhat shape is [Batch, Horizon]. 
            # We slice [:, 0] to keep only the 1st future step, giving one
            # continuous series over the simulation window.
            if yhat.ndim > 1 and yhat.shape[1] > 1:
                yhat_np = yhat[:, 0].detach().cpu().numpy()
                ytrue_np = yb[:, 0].detach().cpu().numpy()
            else:
                yhat_np = yhat.detach().cpu().view(-1).numpy()
                ytrue_np = yb.detach().cpu().view(-1).numpy()
            # -----------------------------------------------------

            # --- DENORMALIZATION BLOCK ---
            # Check if the underlying dataset has our custom inverse_transform method
            if hasattr(base_dataset, 'inverse_transform'):
                yhat_np = base_dataset.inverse_transform(yhat_np)
                ytrue_np = base_dataset.inverse_transform(ytrue_np)
            # -----------------------------

            # Convert back to lists for CSV writing
            yhat_list = yhat_np.tolist()
            ytrue_list = ytrue_np.tolist()

            for yt, yp in zip(ytrue_list, yhat_list):
                writer.writerow([epoch, args.model, split, args.horizon, yt, yp])
                all_ytrue.append(yt)
                all_ypred.append(yp)
