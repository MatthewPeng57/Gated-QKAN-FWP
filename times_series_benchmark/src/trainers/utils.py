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

from __future__ import annotations

import csv
import logging
import os
from typing import Optional
from typing import Union

import torch

def log_epoch(epoch, train_loss, test_loss, path):
	file_exists = path.is_file()
	with open(path, "a", newline="") as f:
		writer = csv.writer(f)
		# Write the header row only for a new file
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
	"""Run the model over `loader` and append every prediction to `csv_path`."""
	model.eval()

	with torch.no_grad(), open(csv_path, "a", newline="") as f:
		writer = csv.writer(f)
		if f.tell() == 0:
			writer.writerow(["epoch", "model", "split", "horizon", "y_true", "y_pred"])

		for xb, yb in loader:
			xb = xb.to(args.device, non_blocking=True)
			yb = yb.to(args.device, non_blocking=True)

			yhat = model(xb)

			# NOTE: model-specific output handling
			if args.model in ("gqkan_qkanfwp", "gqkanfwp", "gqkan_fwp", "gqkan_qfwp"):
				yhat = yhat[-1]
			else: raise ValueError(f"Model '{args.model}' is not a valid choice.")

			yhat_list = yhat.detach().cpu().view(-1).tolist()
			ytrue_list = yb.detach().cpu().view(-1).tolist()

			for yt, yp in zip(ytrue_list, yhat_list):
				writer.writerow([epoch, args.model, split, args.horizon, yt, yp])
