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

import argparse

import numpy as np
import torch
from torch import nn
from src.datasets.registry import make_datasets
from src.trainers.train_loading import make_loaders, run_training
from src.trainers.utils import setup_logger

from pathlib import Path
import sys


# GQKAN-QFWP: QKAN slow programmer, VQC (PennyLane) fast programmer
from src.models.gqkan_qfwp import FWP as GQKAN_QFWP
from src.models.gqkan_qfwp import FWPCell as GQKAN_QFWPCell
# GQKAN-QKANFWP: QKAN slow programmer, QKAN fast programmer
from src.models.gqkan_qkanfwp import FWP as GQKAN_QKANFWP
from src.models.gqkan_qkanfwp import FWPCell as GQKAN_QKANFWPCell
# GQKANFWP: Linear slow programmer, QKAN fast programmer
from src.models.gqkanfwp import FWP as GQKANFWP
from src.models.gqkanfwp import FWPCell as GQKANFWPCell
# GQKAN-FWP: QKAN slow programmer, Linear fast programmer
from src.models.gqkan_fwp import FWP as GQKAN_FWP
from src.models.gqkan_fwp import FWPCell as GQKAN_FWPCell

from src.utils.experiment import save_args_json
from src.utils.experiment import save_git_revision
from src.utils.experiment import save_environment_snapshot
from src.utils.experiment import build_result_path
from src.utils.experiment import generate_experiment_readme
from src.utils.experiment import Tee


def make_model(args):
	"""
	Create the model selected by args.model.
	"""
	if args.model == "gqkan_qfwp":            # GQKAN-QFWP
		fwp_cell = GQKAN_QFWPCell(args.input_size, args.hidden_size, args.output_size, args.qnn_depth).to(args.device).float()
		model = GQKAN_QFWP(fwp_cell).to(args.device).float()
		return model
	elif args.model == "gqkan_qkanfwp":       # GQKAN-QKANFWP
		fwp_cell = GQKAN_QKANFWPCell(args.input_size, args.hidden_size, args.output_size, args.qnn_depth).to(args.device).float()
		model = GQKAN_QKANFWP(fwp_cell, args.device).to(args.device).float()
		return model
	elif args.model == "gqkanfwp":            # GQKANFWP
		fwp_cell = GQKANFWPCell(args.input_size, args.hidden_size, args.output_size).to(args.device).float()
		model = GQKANFWP(fwp_cell, args.device).to(args.device).float()
		return model
	elif args.model == "gqkan_fwp":           # GQKAN-FWP
		fwp_cell = GQKAN_FWPCell(args.input_size, args.hidden_size, args.output_size).to(args.device).float()
		model = GQKAN_FWP(fwp_cell, args.device).to(args.device).float()
		return model
	else:
		raise ValueError(f"Unknown model: {args.model}")

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)



def main():
	parser = argparse.ArgumentParser(description="Time-series training entrypoint.")

	parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs (default: 30)")
	parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 0.001)")

	parser.add_argument(
		"--model",
		type=str,
		choices=["gqkan_qkanfwp", "gqkanfwp", "gqkan_fwp", "gqkan_qfwp"],
		default="gqkan_qkanfwp",
		help="choose model (default: gqkan_qkanfwp, i.e. GQKAN-QKANFWP)"
	)


	parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cpu")

	parser.add_argument(
		"--dataset",
		type=str,
		choices=["bessel_j2", "damped_shm", "delayed_quantum_control", "narma_5", "narma_10", "jaynes_cummings", "transmon"],
		default="bessel_j2",
		help="Choose a dataset (default: bessel_j2)"
	)

	parser.add_argument("--save_dir", type=str, default="result_cudaq_extra", help="Directory where experiment outputs are stored")
	parser.add_argument("--exp_name", type=str, default="experiment", help="Name of the experiment")

	parser.add_argument("--window_len", type=int, default=4, help="Length of sliding window")
	parser.add_argument("--horizon", type=int, default=1, help="Predicting horizon")

	parser.add_argument("--input_size", type=int, default=1, help="Input size (dimension)")
	parser.add_argument("--hidden_size", type=int, default=5, help="Hidden state size of the FWP cell")
	parser.add_argument("--output_size", type=int, default=1, help="Output size of the FWP cell")
	parser.add_argument("--qnn_depth", type=int, default=5, help="Depth of the variational quantum circuit")

	parser.add_argument("--batch_size", type=int, default=2, help="Batch size (large batch size may also need larger learning rate)")
	parser.add_argument("--seed", type=int, default=42, help="random seed (default:42)")
	parser.add_argument("--debug", action="store_true", help="Enable debug mode")

	args = parser.parse_args()

	# seeds
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	# output folder
	EXPERIMENT_ROOT = Path(__file__).resolve().parent
	result_path = build_result_path(args, EXPERIMENT_ROOT)

	# experimental setup config
	save_args_json(args, result_path)
	save_git_revision(result_path)
	save_environment_snapshot(result_path)
	generate_experiment_readme(args, result_path)

	# Console log
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
 
	#initilize weights 
	# model.apply(init_weights)

	# train
	run_training(args=args, model=model, loaders=loaders, result_path=result_path, logger=logger)


if __name__ == "__main__":
	main()
