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

# src/trainers/train_loading.py

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .utils import log_epoch, plot_losses, predict_and_log


def count_effective_trainable(model):
    total = 0
    grad_params = 0

    for p in model.parameters():
        n = p.numel()
        total += n

        if p.grad is not None:
            grad_params += n

    return total, grad_params

@dataclass
class LoaderBundle:
	train_loader: DataLoader
	test_loader: DataLoader
	simulation_loader: DataLoader
	train_len: int


def make_loaders(args, dataset_bundle) -> LoaderBundle:
	train_loader = DataLoader(dataset_bundle.train_ds, batch_size=args.batch_size, shuffle=True)
	test_loader = DataLoader(dataset_bundle.test_ds, batch_size=args.batch_size, shuffle=False)
	simulation_loader = DataLoader(dataset_bundle.simulation_ds, batch_size=args.batch_size, shuffle=False)

	return LoaderBundle(
		train_loader=train_loader,
		test_loader=test_loader,
		simulation_loader=simulation_loader,
		train_len=dataset_bundle.train_len,
	)


def _extract_model_output(args, out):
	"""
	Normalize model forward outputs into a tensor aligned with y.
	This keeps your old per-model handling in one place.
	"""
	if args.model in ("gqkan_qkanfwp", "gqkanfwp", "gqkan_fwp", "gqkan_qfwp"):
		return out[-1]
	else:
		raise ValueError(f"Model '{args.model}' is not a valid choice.")


def run_training(
	args,
	model: torch.nn.Module,
	loaders: LoaderBundle,
	result_path: Union[str, os.PathLike],
	logger,
) -> None:
	result_path.mkdir(parents=True, exist_ok=True)

	# save args snapshot
	logger.info("Args: %s", json.dumps(vars(args), indent=4))

	model = model.to(args.device)
	criterion = nn.MSELoss()
	optimizer = optim.Adam(model.parameters(), lr=args.lr)

	train_losses = []
	test_losses = []

	csv_path = result_path / "train_log.csv"
	prediction_csv_path = result_path / "prediction_log.csv"

	for epoch in range(1, args.epochs + 1):
		# ---- train ----
		model.train()
		train_loss = 0.0

		for X, y in loaders.train_loader:
			X = X.to(args.device, non_blocking=True)
			y = y.to(args.device, non_blocking=True)

			optimizer.zero_grad()
			out = model(X)
			out = _extract_model_output(args, out)

			loss = criterion(out.squeeze(), y.float())
			loss.backward()
			
     
			optimizer.step()

			train_loss += loss.item() * X.size(0)

		if epoch == 1:
			with torch.no_grad():
				total_param, trainable_param = count_effective_trainable(model)
				logger.info("===== Parameter Summary =====")
				logger.info(f"Total parameters: {total_param:,}")
				logger.info(f"Trainable parameters: {trainable_param:,}")
				for name, param in model.named_parameters():
					if param.grad is None:
						print(f"No grad: {name}")
					else:
						print(f"Gradients exist: {name}, norm={param.grad.norm()}")
		train_loss /= len(loaders.train_loader.dataset)

		# ---- eval ----
		model.eval()
		test_loss = 0.0
		with torch.no_grad():
			for X, y in loaders.test_loader:
				X = X.to(args.device, non_blocking=True)
				y = y.to(args.device, non_blocking=True)

				out = model(X)
				out = _extract_model_output(args, out)

				loss = criterion(out.squeeze(), y.float())
				test_loss += loss.item() * X.size(0)

		test_loss /= len(loaders.test_loader.dataset)
		
		print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

		logger.info("Epoch %d: train loss=%.8f, test loss=%.8f", epoch, train_loss, test_loss)
		log_epoch(epoch, train_loss, test_loss, csv_path)

		train_losses.append(train_loss)
		test_losses.append(test_loss)

		# prediction plot/log
		if epoch in {1,15,30,50,100}:
			prediction_plot_path = result_path / f"prediction_plot_epoch_{epoch}.png"
			predict_and_log(
				args=args,
				model=model,
				loader=loaders.simulation_loader,
				train_len=loaders.train_len,
				csv_path=prediction_csv_path,
				split="simulation",
				epoch=epoch,
				debug_path=prediction_plot_path,
			)

			# loss plot
			loss_plot_path = result_path / f"loss_compare_plot_epoch_{epoch}.png"
			plot_losses(train_losses, test_losses, epoch, save_path=loss_plot_path)

			# checkpoint
			model_path = result_path / f"saved_checkpoint_epoch_{epoch}.pth"
			torch.save(
				{
					"epoch": epoch,
					"model_state_dict": model.state_dict(),
					"optimizer_state_dict": optimizer.state_dict(),
					"loss": float(loss.detach().cpu().item()),
				},
				model_path,
			)
