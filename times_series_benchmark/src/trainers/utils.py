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

from __future__ import annotations

import csv
import logging
import os
from typing import Optional
from typing import Union

import matplotlib.pyplot as plt
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


def plot_losses(train_losses, test_losses, epoch, save_path):
	"""
	Plot and save the train/test loss curves.
	"""
	plt.figure()
	plt.plot(range(1, len(train_losses)+1), train_losses, label="Train Loss")
	plt.plot(range(1, len(test_losses)+1), test_losses, label="Test Loss")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.title("Training vs Test Loss")
	plt.legend()
	plt.savefig(save_path)
	plt.close()


def predict_and_log(
	args,
	model: torch.nn.Module,
	loader: torch.utils.data.DataLoader,
	train_len: int,
	csv_path: Union[str, os.PathLike],
	split: str = "simulation",
	epoch: Optional[int] = None,
	debug_plot: bool = True,
	debug_path: Union[str, os.PathLike] = "debug_plot.png",
) -> None:
	model.eval()
	all_ytrue, all_ypred = [], []

	with torch.no_grad(), open(csv_path, "a", newline="") as f:
		writer = csv.writer(f)
		if f.tell() == 0:
			writer.writerow(["epoch", "model", "split", "horizon", "y_true", "y_pred"])

		for xb, yb in loader:
			xb = xb.to(args.device, non_blocking=True)
			yb = yb.to(args.device, non_blocking=True)

			yhat = model(xb)

			# NOTE: model-specific output handling
			if args.model in ("qqkanfwp", "lqkanfwp", "qkanlfwp", "qkanvfwp"):
				yhat = yhat[-1]
			else: raise ValueError(f"Model '{args.model}' is not a valid choice.")

			yhat_list = yhat.detach().cpu().view(-1).tolist()
			ytrue_list = yb.detach().cpu().view(-1).tolist()

			for yt, yp in zip(ytrue_list, yhat_list):
				writer.writerow([epoch, args.model, split, args.horizon, yt, yp])
				all_ytrue.append(yt)
				all_ypred.append(yp)

	if debug_plot and all_ytrue:
		plt.figure(figsize=(8, 4))
		plt.plot(all_ytrue, label="Ground Truth", linewidth=1.2)
		plt.plot(all_ypred, label="Prediction", linewidth=1.2)
		plt.axvline(x=train_len, c="r", linestyle="--")
		plt.title(f"[DEBUG] {args.model} {split} (epoch={epoch})")
		plt.xlabel("timestep (index)")
		plt.ylabel("value")
		plt.legend()
		plt.tight_layout()
		plt.savefig(debug_path, dpi=200)
		plt.close()
