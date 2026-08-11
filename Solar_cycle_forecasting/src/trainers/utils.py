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

"""Logging, prediction dumping, and figure generation for training runs."""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Column names of the shipped SILSO monthly-mean sunspot CSV.
TARGET_COL_NAME_REAL = 'Monthly Mean Total Sunspot Number' 
DATE_COL_NAME_DECIMAL = 'Decimal Date' # Standard Decimal Year format (e.g., 2023.95). Each step is roughly 1/12.


def log_epoch(epoch, train_loss, test_loss, path):
	import os, csv

	# file_exists = os.path.isfile(path)
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


def plot_losses(train_losses, test_losses, epoch, save_path):
	"""Plot and save the train/test loss curves."""
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
            # We slice [:, 0] to only grab the 1st future step for a clean continuous plot.
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

            # Convert back to lists for CSV writing and plotting
            yhat_list = yhat_np.tolist()
            ytrue_list = ytrue_np.tolist()

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
        
        # Make the y-axis label dynamically reflect if it was denormalized
        y_label = "real value" if hasattr(base_dataset, 'inverse_transform') else "scaled value"
        plt.xlabel("timestep (index)")
        plt.ylabel(y_label)
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(debug_path, dpi=200)
        plt.close()
        



def plot_sampled_multistep_forecasts(
    args,
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    csv_path: Union[str, os.PathLike],
    epoch: int,
    debug_path: Union[str, os.PathLike],
    num_samples: int = 5,
) -> None:
    """
    Grabs evenly spaced sequences from the test dataset, predicts the full horizon,
    and plots them in stacked subplots to evaluate different points in time.
    """
    model.eval()

    # The loader's dataset is a Subset for the test split
    dataset = loader.dataset
    # Safely unwrap to access inverse_transform later
    base_dataset = dataset.dataset if hasattr(dataset, 'dataset') else dataset

    # 1. Pick evenly spaced indices across the test set
    indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)
    
    xs, ys = [], []
    for idx in indices:
        x, y = dataset[idx]
        xs.append(x)
        ys.append(y)
        
    # Stack them into a custom mini-batch
    xb = torch.stack(xs).to(args.device, non_blocking=True)
    yb = torch.stack(ys).to(args.device, non_blocking=True)

    with torch.no_grad():
        # 2. Predict the full horizon for all samples at once
        yhat = model(xb)

        # 3. Model-specific output handling
        if args.model == "gqkan_qkanfwp":
            yhat = yhat[-1]
        else:
            raise ValueError(f"Model '{args.model}' is not a valid choice.")

        yhat_np = yhat.detach().cpu().numpy()
        ytrue_np = yb.detach().cpu().numpy()

        # 4. Denormalize all predictions
        if hasattr(base_dataset, 'inverse_transform'):
            yhat_np = base_dataset.inverse_transform(yhat_np)
            ytrue_np = base_dataset.inverse_transform(ytrue_np)

    # 5. Save these specific windows to CSV
    file_exists = Path(csv_path).is_file()
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["epoch", "model", "test_start_index", "step_in_horizon", "y_true", "y_pred"])
        
        for i, start_idx in enumerate(indices):
            for step, (yt, yp) in enumerate(zip(ytrue_np[i], yhat_np[i])):
                writer.writerow([epoch, args.model, start_idx, step+1, yt, yp])

    # 6. Generate the Stacked Subplots
    fig, axes = plt.subplots(num_samples, 1, figsize=(10, 2.5 * num_samples), sharex=True)
    if num_samples == 1: axes = [axes] # Handle edge case if someone sets num_samples=1
    
    # y_label = "temperature" if hasattr(base_dataset, 'inverse_transform') else "scaled value"
    y_label = "Sunspot Number" if hasattr(base_dataset, 'inverse_transform') else "scaled value"
    axes[-1].set_xlabel(f"Future Timesteps (Months)")

    for i, (ax, start_idx) in enumerate(zip(axes, indices)):
        ax.plot(ytrue_np[i], label="Ground Truth", linewidth=1.5, color='blue')
        ax.plot(yhat_np[i], label="Prediction", linewidth=1.5, color='red')
        ax.set_title(f"{args.model} - Test Set Index {start_idx} (Epoch {epoch})")
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right")
            
    # axes[-1].set_xlabel(f"Future Timesteps (hours)")
    plt.tight_layout()
    plt.savefig(debug_path, dpi=200)
    plt.close()
    


def plot_sunspot_paper_style(
    args,
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    debug_path: Union[str, os.PathLike],
    num_samples: int = 4,
) -> None:
    """
    Recreates the 4-panel Sunspot forecast plot from the reference paper.
    Solid blue line for history, blue dots for true future, red dots for forecast.
    """
    model.eval()

    dataset = loader.dataset
    base_dataset = dataset.dataset if hasattr(dataset, 'dataset') else dataset

    # Pick evenly spaced indices
    indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)
    
    xs, ys = [], []
    for idx in indices:
        x, y = dataset[idx]
        xs.append(x)
        ys.append(y)
        
    xb = torch.stack(xs).to(args.device, non_blocking=True)
    yb = torch.stack(ys).to(args.device, non_blocking=True)

    with torch.no_grad():
        yhat = model(xb)

        if args.model == "gqkan_qkanfwp":
            yhat = yhat[-1]
        else:
            raise ValueError(f"Model '{args.model}' is not a valid choice.")

        yhat_np = yhat.detach().cpu().numpy()
        ytrue_np = yb.detach().cpu().numpy()

        # History is shape (Batch, Seq_Len, 1), we flatten the last dim
        history_np = xb.detach().cpu().numpy().squeeze(-1)

        # Denormalize everything for real-world values
        if hasattr(base_dataset, 'inverse_transform'):
            yhat_np = base_dataset.inverse_transform(yhat_np)
            ytrue_np = base_dataset.inverse_transform(ytrue_np)
            
            # History needs shape preservation trick
            orig_hist_shape = history_np.shape
            hist_flat = history_np.reshape(-1, 1)
            hist_unscaled = base_dataset.inverse_transform(hist_flat)
            history_np = hist_unscaled.reshape(orig_hist_shape)

        records = []
        x_history = np.arange(-args.window_len, 0)
        x_future = np.arange(0, args.horizon)
        
        for i, idx in enumerate(indices):
            for x, val in zip(x_history, history_np[i]):
                records.append({"sample_index": idx, "time_step": x, "type": "history", "value": val})
            for x, val in zip(x_future, ytrue_np[i]):
                records.append({"sample_index": idx, "time_step": x, "type": "true_future", "value": val})
            for x, val in zip(x_future, yhat_np[i]):
                records.append({"sample_index": idx, "time_step": x, "type": "predicted_future", "value": val})
                
        df_paper_style = pd.DataFrame(records)
        # Automatically saves a .csv with the exact same name as your .png
        csv_save_path = Path(debug_path).with_suffix('.csv') 
        df_paper_style.to_csv(csv_save_path, index=False)
    # --- PLOTTING ---
    # Create a 2x2 grid (assuming num_samples=4)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    seq_len = args.window_len
    horizon = args.horizon
    
    # X-axis arrays matching the paper (-seq_len to 0, and 0 to horizon)
    x_history = np.arange(-seq_len, 0)
    x_future = np.arange(0, horizon)

    for i, ax in enumerate(axes):
        # 1. Solid blue line for history
        ax.plot(x_history, history_np[i], color='steelblue', linewidth=1.0, label='History')
        
        # 2. Blue dots for true future
        ax.scatter(x_future, ytrue_np[i], color='blue', s=12, label='True future')
        
        # 3. Red dots for predicted future
        ax.scatter(x_future, yhat_np[i], color='red', s=12, label='Forecast')
        
        ax.set_title(f"Test Set Index {indices[i]}")
        ax.axvline(x=0, color='black', linewidth=0.8, linestyle='--') # Divider at T=0
        
        if i == 0:
            ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(debug_path, dpi=200)
    plt.close()
    
    
    
    
    
    
    
    
    
   
def generate_sunspot_reconstruction_figure_13_style(
    args,
    model: torch.nn.Module,
    unscaled_target_series: np.ndarray, # Solid Blue line: Raw physical sunspot numbers for perfect history unrolling
    dates_decimal: np.ndarray, # Dates array matching unscaled series
    sunspot_dataset_instances: object, # Unwrap instance for scalar and data extraction
    debug_path: Union[str, os.PathLike],
    epoch: int,
    train_len: int = None
):
    """
    Structural replica of image_8.png (Figure 13) using dates.
    Blue solid line for HISTORY, Contiguous overlapping Dark Orange Dots across known boundaries,
    RED dashed line extending direct Multi-Step forecast beyond boundaries (mimicking Solar Cycle 25 predict).
    """
    # logger.info("Generating Final Sunspot Cycle Reconstruction Figure (image_8.png style)...")
    model.eval()
    
    scaler = sunspot_dataset_instances.scaler # Access existing MinMaxScaler
    full_scaled_data = sunspot_dataset_instances.full_scaled_data # placeholder unwrap access
    
    # Validation step: Ensure date matching
    if len(unscaled_target_series) != len(dates_decimal):
        #  logger.warning("Unscaled data and dates array length mismatch. Date-aligned plot skipped.")
        print(("Unscaled data and dates array length mismatch. Date-aligned plot skipped."))
        return

    # 1. Contiguous Overlapping Contours across known history (Validation Orange)
    # This requires running continuous element-by-element rolling inference from starts of TRAIN knowledge boundary.
    
    overlapping_reconstructed_preds_real = []
    
    # Boundary logic for rolling sequential reconstruction
    start_window_boundary = 0
    end_rolling_boundary = len(full_scaled_data) - args.window_len
    
    # Robust element-by-element sequential inference from full scaled data
    for current_idx in range(start_window_boundary, end_rolling_boundary):
        input_scaled_unsq = full_scaled_data[current_idx : current_idx + args.window_len]
        input_tensor = torch.tensor(input_scaled_unsq).float().unsqueeze(0).unsqueeze(-1).to(args.device) # [1, Window, Feat]
        
        with torch.no_grad():
            out = model(input_tensor) # Extract multivariate, unwrap specific behavior applies...
            if args.model == "gqkan_qkanfwp":
                out = out[-1]
            else:
                raise ValueError(f"Model '{args.model}' is not a valid choice.")

            # Direct multi-step, we just grab FIRST step to create clean contiguous line reconstruction
            first_step_pred_scaled = out[0, 0].detach().cpu().numpy() # [1]
            overlapping_reconstructed_preds_real.append(first_step_pred_scaled)
    
    # Denormalize overlapping contiguous preds
    overlapping_reconstructed_preds_real = scaler.inverse_transform(np.array(overlapping_reconstructed_preds_real).reshape(-1, 1)).flatten()
    
    # Calculate perfect date alignment for continuous preds starting at window_len+1 boundary
    valid_overlap_pred_dates = dates_decimal[args.window_len + 1 : args.window_len + 1 + len(overlapping_reconstructed_preds_real)]
    
    # (Boundary alignment clamping for contiguous logging bug fix validation)
    min_len = min(len(overlapping_reconstructed_preds_real), len(valid_overlap_pred_dates))
    valid_pred_real_clamped = overlapping_reconstructed_preds_real[:min_len]
    valid_pred_dates_clamped = valid_overlap_pred_dates[:min_len]

    # 2. Extend Future Simulated Direct Horizon (mimicking extended Solar Cycle 25 orange line)
    # This takes the MOST RECENT valid input window (ends at known history boundary Dec 2023), 
    # predicts its Direct 132-step horizon into Unknown Time (Jan 2024 to Dec 2034).
    
    final_input_sim_window = full_scaled_data[-args.window_len:]
    # Add .unsqueeze(-1) here as well!
    final_future_input_tensor = torch.tensor(final_input_sim_window).float().unsqueeze(0).unsqueeze(-1).to(args.device)
    
    with torch.no_grad():
        final_horizon_simulated = model(final_future_input_tensor)
        
        # Unwrap specific behavioral output handling for final rolling simulation
        if args.model == "gqkan_qkanfwp":
            final_horizon_simulated = final_horizon_simulated[-1]
        else:
            raise ValueError(f"Model '{args.model}' is not a valid choice.")


        final_horizon_np_scaled = final_horizon_simulated[0].detach().cpu().numpy() # [Horizon]
        final_horizon_real = scaler.inverse_transform(final_horizon_np_scaled.reshape(-1, 1)).flatten()

    # Create dates array for future simulation (extending past last known Decimal Year date)
    last_decimal_date = dates_decimal[-1]
    # Convert the last known date back into a Pandas Timestamp
    last_date = pd.to_datetime(last_decimal_date)
    
    # Generate exactly 'horizon' number of future month-end dates
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1), 
        periods=args.horizon, 
        freq='ME' # 'ME' stands for Month-End (e.g., Jan 31, Feb 28)
    ).values

    base_path = Path(debug_path).parent
    
    # Save 1: The full actual history line (Blue)
    df_history = pd.DataFrame({'Date': dates_decimal, 'Actual_Value': unscaled_target_series})
    df_history.to_csv(base_path / f"fig13_data_1_history_epoch_{epoch}.csv", index=False)
    
    # Save 2: The overlapping validation predictions (Orange Dots)
    df_overlap = pd.DataFrame({'Date': valid_pred_dates_clamped, 'Predicted_Value': valid_pred_real_clamped})
    df_overlap.to_csv(base_path / f"fig13_data_2_overlap_preds_epoch_{epoch}.csv", index=False)
    
    # Save 3: The extended future horizon (Red Dashed Line)
    df_future = pd.DataFrame({'Date': future_dates, 'Predicted_Value': final_horizon_real})
    df_future.to_csv(base_path / f"fig13_data_3_future_extension_epoch_{epoch}.csv", index=False)
    # ==========================================
    # FINAL PLOTTING ASSEMBLY (mimicking image_8.png structure)
    # ==========================================
    plt.figure(figsize=(14, 6))
    
    # (Blue continuous line - UNFILTERED HISTORY PHYSICAL VALUES)
    plt.plot(dates_decimal, unscaled_target_series, color='blue', label='Actual Sunspot Number (Monthly Mean SILSO)', alpha=0.5, linewidth=1.0)
    
    # (Orange CONTIGUOUS SCATTER Contour alignment - validated overlapping history predictions across boundaries)
    # Standard monthly mean waves are jagged making contiguous lines messy and confusing.
    # Dark Orange dots align contours perfectly to original JAGGENESS while mimicking Figure 10 dot style 
    # visually clear contrast. (Matlab image_8.png used lines; Switch dots to lines here for matlab mimic preference, alpha=0.7 is key contrast).
    plt.scatter(valid_pred_dates_clamped, valid_pred_real_clamped, color='darkorange', s=3, label='Forecast Overlap (Step-1 contig.)', alpha=0.7, zorder=2)
    
    # (RED CONTIGUOUS DASHED Line extension - Direct Multi-Step Simulation into unknown time mimicking extended orange)
    plt.plot(future_dates, final_horizon_real, color='red', label='Forecast Extended Future (Solar Cycle 25 predict)', linewidth=1.8, linestyle='--', zorder=3)
    # --- NEW: Train/Test Split Line ---
    if train_len is not None and (train_len + args.window_len) < len(dates_decimal):
        split_date = dates_decimal[train_len + args.window_len]
        plt.axvline(x=split_date, color='green', linestyle='-.', alpha=0.8, linewidth=1.5)
        plt.text(split_date, plt.ylim()[1]*0.90, " Test Data Start", rotation=90, va='top', ha='right', fontsize=9, alpha=0.8, color='green')
    # ----------------------------------
    
    # Boundary Delineator line marking History End boundary Dec 2023.
    plt.axvline(x=last_decimal_date, color='black', linestyle=':', alpha=0.6, linewidth=1.2)
    plt.text(last_decimal_date, plt.ylim()[1]*0.95, "History Ends", rotation=90, va='top', ha='right', fontsize=9, alpha=0.7)

    # Aesthetic Tuning for Reconstruction View
    plt.title(f"Final Solar Cycle Number Reconstruction & Prediction - model={args.model} (Window={args.window_len}, Horizon={args.horizon}, Epoch={epoch})")
    plt.ylabel("Monthly Mean Sunspot Number")
    plt.xlabel("Year (Decimal Date)")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(debug_path, dpi=200)
    plt.close()
