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

# train.py — CLI entrypoint for the GQKAN-QKANFWP sunspot forecasting benchmark.
#
# Responsibilities of this thin driver:
#   1. parse the experiment configuration from the command line,
#   2. seed every RNG and pin deterministic cuDNN behaviour,
#   3. build the sunspot sliding-window datasets + loaders
#      (`src.data.sunspot.make_datasets` -> `src.trainers.train_loading.make_loaders`),
#   4. construct the `qqkanfwp` model (`src.models.QQKANFWP`),
#   5. snapshot args/environment into the run folder and hand off to
#      `run_training`.
#
# All training and evaluation semantics — loss, LR schedule, best-val
# checkpointing, final denormalized metrics, and the figure set — live in
# `run_training` (src/trainers/train_loading.py). Every artifact is written
# under `--save_dir`.

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

# --- data ---
from src.data.sunspot import make_datasets
from src.trainers.train_loading import make_loaders, run_training
from src.trainers.utils import setup_logger

# --- model: GQKAN-QKANFWP ---
from src.models.QQKANFWP import FWP as QKAN_QKANFWP
from src.models.QQKANFWP import FWPCell as QKAN_QKANFWPCell

from src.utils.experiment import (
    save_args_json,
    save_git_revision,
    save_environment_snapshot,
    build_result_path,
    generate_experiment_readme,
    Tee,
)


def make_model(args):
    """Create a fresh model based on ``args.model``.

    ``qqkanfwp`` (GQKAN-QKANFWP) is the only model shipped in this folder.
    """
    if args.model == "qqkanfwp":
        fwp_cell = QKAN_QKANFWPCell(
            args.input_size, args.hidden_size, args.output_size, args
        ).to(args.device).float()
        use_streaming = getattr(args, "streaming_fwp", "off") == "on"
        model = QKAN_QKANFWP(
            fwp_cell, args.device,
            output_relu=getattr(args, "output_relu", False),
            use_streaming_fwp=use_streaming,
        ).to(args.device).float()
        return model
    else:
        raise ValueError(f"Unknown model: {args.model!r} (only 'qqkanfwp' is available)")


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


def main():
    parser = argparse.ArgumentParser(
        description="Train GQKAN-QKANFWP on the monthly sunspot series."
    )

    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    parser.add_argument("--in_resize", type=int, default=8, help="Quantum input projection size")
    parser.add_argument("--out_resize", type=int, default=16, help="Quantum output projection size")
    parser.add_argument("--fast_in", type=int, default=8, help="Fast-programmer input projection size")
    parser.add_argument("--fast_out", type=int, default=16, help="Fast-programmer output projection size")
    parser.add_argument("--alpha", type=float, default=3.0, help="Peak-aware loss multiplier")
    parser.add_argument("--qkan_s_dim_1", type=int, default=16, help="Slow QKAN hidden dimension 1")
    parser.add_argument("--qkan_s_dim_2", type=int, default=16, help="Slow QKAN hidden dimension 2")

    parser.add_argument(
        "--model",
        type=str,
        choices=["qqkanfwp"],
        default="qqkanfwp",
        help="Model to build (default: qqkanfwp).",
    )

    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cpu")

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["sunspots"],
        default="sunspots",
        help="Dataset (this repository ships only 'sunspots').",
    )

    # Loss / LR-schedule overrides. The defaults reproduce the published
    # training protocol.
    parser.add_argument(
        "--loss", type=str, choices=["peak_aware_mse", "plain_mse"],
        default="peak_aware_mse",
        help="Training loss function.",
    )
    parser.add_argument(
        "--lr_schedule", type=str, choices=["keras_decay", "efc_stepwise"],
        default="keras_decay",
        help=(
            "LR schedule. keras_decay: per-step 1/(1+1e-6*step) (default). "
            "efc_stepwise: multiply by 0.9 at epochs//3 and 2*epochs//3."
        ),
    )

    parser.add_argument("--save_dir", type=str, default="best_sup", help="Output root folder")
    parser.add_argument("--exp_name", type=str, default="experiment", help="Name of the experiment")

    parser.add_argument("--window_len", type=int, default=4, help="Length of sliding window (L)")
    parser.add_argument("--horizon", type=int, default=1, help="Predicting horizon (H)")

    parser.add_argument("--input_size", type=int, default=1, help="Input size (dimension)")
    parser.add_argument("--hidden_size", type=int, default=5, help="Hidden size")
    parser.add_argument("--output_size", type=int, default=1, help="Output size")
    parser.add_argument("--qnn_depth", type=int, default=5, help="QNN depth")

    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--output_relu", action="store_true",
        help="Apply a ReLU to the output head, clamping forecasts to be "
             "non-negative. Off by default, which leaves the head unclamped.",
    )
    parser.add_argument(
        "--fast_solver",
        type=str,
        choices=["flash", "cutile"],
        default="flash",
        help="Backend for the fast-programmer QKAN layer ('flash' Triton kernels "
             "by default; 'cutile' pure-PyTorch scalar recurrence).",
    )
    parser.add_argument(
        "--streaming_fwp",
        type=str,
        choices=["off", "on"],
        default="off",
        help="Opt-in memory-saving streaming-FWP prefix-scan kernel. Default 'off' "
             "uses the legacy cumsum path (the paper-model path).",
    )
    parser.add_argument(
        "--skip_plots", action="store_true",
        help="Skip all post-training visualizations (per-epoch simulation plots and "
             "final reconstruction figures). Metrics and best_model.pth are still "
             "produced. Useful for multi-seed sweeps, where the figures cost "
             "wall-time and are not needed per run.",
    )

    args = parser.parse_args()

    # seeds (verbatim from source driver)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you use multi-GPU
    np.random.seed(args.seed)

    # Deterministic CuDNN behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # output folder
    EXPERIMENT_ROOT = Path(__file__).resolve().parent
    result_path = build_result_path(args, EXPERIMENT_ROOT)

    # save experiment config
    save_args_json(args, result_path)
    save_git_revision(result_path)
    save_environment_snapshot(result_path)
    generate_experiment_readme(args, result_path)

    # console log
    console_log_path = result_path / "console_log.txt"
    log_f = open(console_log_path, "a", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_f)
    sys.stderr = Tee(sys.__stderr__, log_f)

    # logger
    log_path = result_path / "log.txt"
    logger = setup_logger(log_path)

    # datasets + loaders
    dataset_bundle = make_datasets(args)
    loaders = make_loaders(args, dataset_bundle)

    # model
    model = make_model(args)

    # train — gradient descent, keeping the best-validation checkpoint
    run_training(args=args, model=model, loaders=loaders, result_path=result_path, logger=logger)


if __name__ == "__main__":
    main()
