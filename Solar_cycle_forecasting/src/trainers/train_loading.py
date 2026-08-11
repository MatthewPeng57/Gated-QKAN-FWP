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

"""Training and evaluation loop for the sunspot forecasting benchmark.

``run_training`` owns the full run: the epoch loop (peak-aware or plain MSE,
Adam with a per-batch or per-epoch LR schedule), best-validation
checkpointing, the periodic figure/checkpoint dumps, and the final
denormalized test evaluation that writes ``<model>_final_metrics_summary.csv``.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .utils import (
    log_epoch,
    plot_losses,
    predict_and_log,
    plot_sampled_multistep_forecasts,
    plot_sunspot_paper_style,
    generate_sunspot_reconstruction_figure_13_style,
)


def count_effective_trainable(model):
    """Return (total_params, effective_trainable_params).

    "Effective trainable" = only parameters whose ``.grad`` is populated after
    at least one ``loss.backward()`` call. This correctly excludes
    ``nn.Parameter``s that are declared with ``requires_grad=True`` but
    never participate in the forward pass (dead weights).

    The canonical example in this codebase is the GQKAN-QKANFWP fast
    programmer's ``self.qkan_layer.layers[k].theta``: it is declared as an
    ``nn.Parameter`` by the upstream QKAN class, but ``FWP.forward`` always
    overrides it by passing a slow-programmer-generated tensor as the
    ``theta`` argument, so it never receives a gradient and must not be
    counted against the model's capacity budget.

    PRECONDITION: must be called AFTER at least one ``loss.backward()`` —
    before any backward pass every ``p.grad`` is ``None`` and this returns 0.
    """
    total = 0
    grad_params = 0

    for p in model.parameters():
        n = p.numel()
        total += n

        if p.grad is not None:
            grad_params += n

    return total, grad_params


def peak_aware_loss(y_pred, y_true, alpha=2.0):
    """
    Standard MSE, but errors on high target values are multiplied by (1 + alpha * y_true).
    Because y_true is scaled [0, 1], peaks get a penalty multiplier up to (1 + alpha).
    """
    mse_base = (y_pred - y_true) ** 2
    # Weight map: 1.0 for 0s, up to 3.0 for peaks (if alpha=2.0)
    weight_map = 1.0 + (alpha * y_true)
    return torch.mean(mse_base * weight_map)


def plain_mse_loss(y_pred, y_true):
    """Unweighted MSE — EFC 2023 §B.4 optimization target."""
    return torch.mean((y_pred - y_true) ** 2)

@dataclass
class LoaderBundle:
    train_loader: DataLoader
    test_loader: DataLoader
    simulation_loader: DataLoader
    train_len: int
    val_loader: Optional[DataLoader] = None # --- NEW ---

def make_loaders(args, dataset_bundle) -> LoaderBundle:
    train_loader = DataLoader(dataset_bundle.train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset_bundle.test_ds, batch_size=args.batch_size, shuffle=False)
    simulation_loader = DataLoader(dataset_bundle.simulation_ds, batch_size=args.batch_size, shuffle=False)

    # --- NEW: Fallback logic for safety ---
    # If a dataset uses 80/10/10, use val_ds. If a legacy dataset uses 80/20, default to test_ds.
    if dataset_bundle.val_ds is not None:
        val_loader = DataLoader(dataset_bundle.val_ds, batch_size=args.batch_size, shuffle=False)
    else:
        val_loader = test_loader

    return LoaderBundle(
        train_loader=train_loader,
        test_loader=test_loader,
        simulation_loader=simulation_loader,
        train_len=dataset_bundle.train_len,
        val_loader=val_loader
    )


def _extract_model_output(args, out):
	"""
	Normalize model forward outputs into a tensor aligned with y.
	"""
	if args.model == "qqkanfwp":
		# FWP returns a per-timestep stack; the last entry is the (B, H) forecast.
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
	# os.makedirs(result_path, exist_ok=True)
	result_path.mkdir(parents=True, exist_ok=True)

	# save args snapshot
	logger.info("Args: %s", json.dumps(vars(args), indent=4))

	model = model.to(args.device)

	# Loss selection per --loss (default peak_aware_mse).
	_loss_name = getattr(args, "loss", "peak_aware_mse")
	if _loss_name == "plain_mse":
		criterion = lambda y_pred, y_true: plain_mse_loss(y_pred, y_true)
	else:  # peak_aware_mse
		criterion = lambda y_pred, y_true: peak_aware_loss(y_pred, y_true, alpha=args.alpha)
	optimizer = optim.Adam(model.parameters(), lr=args.lr)

	# LR schedule per --lr_schedule. Default `keras_decay` = per-step
	# 1/(1 + 1e-6 * step), the schedule the published checkpoints were
	# trained with. `efc_stepwise` = MultiStepLR with gamma=0.9 at epochs/3
	# and 2*epochs/3 (EFC 2023 §B.2). The two paths differ in per-batch vs
	# per-epoch `.step()` cadence; `scheduler_per_step` controls which.
	_schedule = getattr(args, "lr_schedule", "keras_decay")
	if _schedule == "efc_stepwise":
		milestones = [max(1, args.epochs // 3), max(2, 2 * args.epochs // 3)]
		scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.9)
		scheduler_per_step = False  # step per epoch
	else:  # keras_decay
		decay_rate = 1e-6
		keras_decay = lambda step: 1.0 / (1.0 + decay_rate * step)
		scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=keras_decay)
		scheduler_per_step = True  # step per batch

	train_losses = []
	test_losses = []
	train_wall_start = time.time()

	# Mean training-batch wall time. Wraps only the forward/backward/optimizer
	# step — excludes validation.
	sum_batch_time_s = 0.0
	total_batches = 0

	# Best-validation checkpoint tracking. Training always runs the full
	# --epochs; there is no early-stopping break.
	best_test_loss = float('inf')

	csv_path = result_path / "train_log.csv"
	prediction_csv_path = result_path / "prediction_log.csv"

	for epoch in range(1, args.epochs + 1):
		# ---- train ----
		model.train()
		train_loss = 0.0

		for X, y in loaders.train_loader:
			_batch_t0 = time.perf_counter()
			X = X.to(args.device, non_blocking=True)
			y = y.to(args.device, non_blocking=True)

			optimizer.zero_grad()
			out = model(X)
			out = _extract_model_output(args, out)

			y_for_loss = y.float()

			loss = criterion(out.squeeze(), y_for_loss)
			loss.backward()

			optimizer.step()

			# Keras-decay scheduler steps per-batch; EFC-stepwise steps per-epoch.
			if scheduler_per_step:
				scheduler.step()

			sum_batch_time_s += time.perf_counter() - _batch_t0
			total_batches += 1
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
		# ---- eval ----
		model.eval()
		val_loss = 0.0 # Renamed from test_loss
		with torch.no_grad():
			for X, y in loaders.val_loader: # Switched to val_loader
				X = X.to(args.device, non_blocking=True)
				y = y.to(args.device, non_blocking=True)

				out = model(X)
				out = _extract_model_output(args, out)

				loss = criterion(out.squeeze(), y.float())
				val_loss += loss.item() * X.size(0)

		val_loss /= len(loaders.val_loader.dataset)

		current_lr = optimizer.param_groups[0]['lr']
		print(f"Epoch {epoch:03d} | LR: {current_lr:.6f} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

		logger.info("Epoch %d: train loss=%.8f, val loss=%.8f", epoch, train_loss, val_loss)
		log_epoch(epoch, train_loss, val_loss, csv_path)

		train_losses.append(train_loss)
		test_losses.append(val_loss) # Leaving array name as test_losses so your plot_losses function doesn't break

		# EFC-stepwise scheduler advances at the end of each epoch. Keras-decay
		# already stepped per batch inside the training loop above.
		if not scheduler_per_step:
			scheduler.step()

		# scheduler.step(test_loss)
		# current_lr = optimizer.param_groups[0]['lr']

		# print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
		# print(f"Epoch {epoch:03d} | LR: {current_lr:.6f} | Train Loss: {train_loss:.4f} | Test Loss: {val_loss:.4f}")

		
		
		
		if epoch in {1,15,30,50,100} and not getattr(args, "skip_plots", False):
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

			if args.dataset == 'sunspots':
				paper_plot_path = result_path / f"sunspot_paper_style_epoch_{epoch}.png"
				plot_sunspot_paper_style(
					args=args,
					model=model,
					loader=loaders.test_loader, 
					debug_path=paper_plot_path,
					num_samples=4 
				)

		# Best-validation checkpoint.
		if val_loss < best_test_loss:
			best_test_loss = val_loss
			best_model_path = result_path / "best_model.pth"
			torch.save(model.state_dict(), best_model_path)
			print(f"  -> New best validation loss! Saved model to {best_model_path.name}")

	train_wall_time_seconds = float(time.time() - train_wall_start)
	mean_batch_time_ms = 1000.0 * sum_batch_time_s / max(total_batches, 1)
	logger.info(f"Training wall time: {train_wall_time_seconds:.2f}s "
		f"({train_wall_time_seconds / max(args.epochs, 1):.2f}s per epoch); "
		f"mean batch time: {mean_batch_time_ms:.3f} ms/batch over {total_batches} batches")

	# ==========================================
	# FINAL EVALUATION (DENORMALIZED METRICS)
	# ==========================================

	# FIX 2: Point to the actual best_model.pth file
	best_model_path = result_path / "best_model.pth"

	# Load the best weights back into the model
	if best_model_path.exists():
		checkpoint = torch.load(best_model_path, map_location=args.device)
		if "model_state_dict" in checkpoint:
			model.load_state_dict(checkpoint["model_state_dict"])
		else:
			model.load_state_dict(checkpoint)
		logger.info(f"Loaded best model weights from {best_model_path.name} for final evaluation.")

	logger.info("Starting final evaluation on test set (Denormalized)...")
	model.eval()

	# Unwrap the test dataset once so `base_dataset` is available to both the
	# plotting block and the generic evaluation loop below.
	base_dataset = loaders.test_loader.dataset
	if hasattr(base_dataset, 'dataset'):
		base_dataset = base_dataset.dataset

	# Generate a final plot for the BEST model
	final_snapshot_plot_path = result_path / "multistep_snapshot_FINAL.png"
	if args.dataset == 'sunspots' and not getattr(args, "skip_plots", False):
				paper_plot_path = result_path / f"sunspot_paper_style_epoch_final_{epoch}.png"
				plot_sunspot_paper_style(
					args=args,
					model=model,
					loader=loaders.test_loader, 
					debug_path=paper_plot_path,
					num_samples=4 
				)
	logger.info("Starting final sunspot cycle reconstruction visualization...")

	if args.dataset == 'sunspots' and not getattr(args, "skip_plots", False):
		final_snapshot_plot_path = result_path / "multistep_snapshot_FINAL.png"
		plot_sampled_multistep_forecasts(
					args=args,
					model=model,
					loader=loaders.test_loader, 
					csv_path=result_path / "multistep_snapshot_log.csv", # Append to the existing log
					epoch=epoch, # Will log as the final epoch number
					debug_path=final_snapshot_plot_path,
					num_samples=5 # You can easily change this to 3, 5, or 10!
				)

		# Full-history reconstruction figure: needs the raw (unscaled) monthly
		# sunspot series and its dates, read from the dataset's unscaled
		# DataFrame. Column names follow the shipped SILSO monthly CSV.
		full_df = loaders.simulation_loader.dataset.full_unscaled_df
		history_dates_real = pd.to_datetime(base_dataset.full_unscaled_df['Date']).values
		history_actual_real = full_df['Monthly Mean Total Sunspot Number'].values

		final_reconstruction_path = result_path / "final_sunspot_cycle_reconstruction_Figure_13.png"
		
		
		generate_sunspot_reconstruction_figure_13_style(
			args=args,
			model=model,
			unscaled_target_series=history_actual_real,
			dates_decimal=history_dates_real,
			sunspot_dataset_instances=base_dataset,
			debug_path=final_reconstruction_path,
			epoch=epoch,
			train_len=loaders.train_len
		)
	
	# We will track both scaled and denormalized (real) predictions
	all_true_real = []
	all_pred_real = []
	all_true_scaled = []
	all_pred_scaled = []
	peak_amplitude_errors = []
	peak_timing_errors = []

	with torch.no_grad():
		for X, y in loaders.test_loader:
			X = X.to(args.device, non_blocking=True)
			y = y.to(args.device, non_blocking=True)

			out = model(X)
			out = _extract_model_output(args, out)

			# Convert to numpy
			y_pred_np = out.detach().cpu().numpy()
			y_true_np = y.detach().cpu().numpy()

			# --- 1. STORE SCALED PREDICTIONS ---
			all_true_scaled.extend(y_true_np.flatten().tolist())
			all_pred_scaled.extend(y_pred_np.flatten().tolist())

			# --- 2. DENORMALIZE ---
			if hasattr(base_dataset, 'inverse_transform'):
				y_pred_np = base_dataset.inverse_transform(y_pred_np)
				y_true_np = base_dataset.inverse_transform(y_true_np)

			# --- NEW: CALCULATE PEAK ERRORS PER 132-STEP CYCLE ---
            # y_pred_np and y_true_np are shape [Batch_Size, 132]
			for b in range(y_true_np.shape[0]):
				true_seq = y_true_np[b]
				pred_seq = y_pred_np[b]
				
				# 1. Amplitude Error (How many sunspots off?)
				true_peak_val = np.max(true_seq)
				pred_peak_val = np.max(pred_seq)
				peak_amplitude_errors.append(abs(true_peak_val - pred_peak_val))
				
				# 2. Timing Error (How many months early/late?)
				true_peak_month = np.argmax(true_seq)
				pred_peak_month = np.argmax(pred_seq)
				peak_timing_errors.append(abs(true_peak_month - pred_peak_month))
			# --- 3. STORE DENORMALIZED PREDICTIONS ---
			all_true_real.extend(y_true_np.flatten().tolist())
			all_pred_real.extend(y_pred_np.flatten().tolist())
			
	# Calculate Denormalized Metrics
	true_real_arr = np.array(all_true_real)
	pred_real_arr = np.array(all_pred_real)
	mae_real = float(np.mean(np.abs(pred_real_arr - true_real_arr)))
	mse_real = float(np.mean((pred_real_arr - true_real_arr)**2))
	rmse_real = float(np.sqrt(mse_real))

	# Calculate Scaled Metrics
	true_scaled_arr = np.array(all_true_scaled)
	pred_scaled_arr = np.array(all_pred_scaled)
	mae_scaled = float(np.mean(np.abs(pred_scaled_arr - true_scaled_arr)))
	mse_scaled = float(np.mean((pred_scaled_arr - true_scaled_arr)**2))
	rmse_scaled = float(np.sqrt(mse_scaled))
	# R^2 on scaled data — required for the Task A baseline comparison.
	ss_res_scaled = float(np.sum((true_scaled_arr - pred_scaled_arr) ** 2))
	ss_tot_scaled = float(np.sum((true_scaled_arr - np.mean(true_scaled_arr)) ** 2))
	r2_scaled = 1.0 - ss_res_scaled / ss_tot_scaled if ss_tot_scaled > 0.0 else float('nan')
	mean_peak_amplitude_error = float(np.mean(peak_amplitude_errors))
	mean_peak_timing_error = float(np.mean(peak_timing_errors))

	# Get Trainable Parameters
	total_param, trainable_param = count_effective_trainable(model)

	# Log to console
	logger.info(f"Final Scaled Test MAE:      {mae_scaled:.4f}")
	logger.info(f"Final Scaled Test MSE:      {mse_scaled:.4f}")
	logger.info(f"Final Scaled Test RMSE:     {rmse_scaled:.4f}")
	logger.info(f"Final Scaled Test R2:       {r2_scaled:.4f}")
	logger.info(f"Final Denormalized MAE:     {mae_real:.4f}")
	logger.info(f"Final Denormalized MSE:     {mse_real:.4f}")
	logger.info(f"Final Denormalized RMSE:    {rmse_real:.4f}")
	logger.info(f"Mean Peak Amplitude Error:  {mean_peak_amplitude_error:.2f} sunspots")
	logger.info(f"Mean Peak Timing Error:     {mean_peak_timing_error:.2f} months")
	logger.info(f"Total Trainable Parameters: {trainable_param}")

	# Create a nice summary dataframe containing ALL information
	metrics_df = pd.DataFrame([{
		"model": args.model,
		"dataset": getattr(args, 'dataset', 'unknown'),
		"seq_len": getattr(args, 'window_len', 'unknown'),
		"horizon": getattr(args, 'horizon', 1),
		"seed": getattr(args, 'seed', 'N/A'),
		"trainable_parameters": trainable_param,
		"scaled_mae": mae_scaled,
		"scaled_mse": mse_scaled,
		"scaled_rmse": rmse_scaled,
		"scaled_r2": r2_scaled,
		"denorm_mae": mae_real,
		"denorm_mse": mse_real,
		"denorm_rmse": rmse_real,
		"peak_error": mean_peak_amplitude_error,
		"timing_error": mean_peak_timing_error,
		"train_wall_time_seconds": train_wall_time_seconds,
		"mean_batch_time_ms": mean_batch_time_ms,
		"num_epochs": args.epochs,
	}])

	metrics_csv_path = result_path / f"{args.model}_final_metrics_summary.csv"
	metrics_df.to_csv(metrics_csv_path, index=False)
	logger.info(f"Saved complete metrics summary to {metrics_csv_path}")
