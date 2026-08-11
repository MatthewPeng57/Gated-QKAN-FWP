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

# src/datasets/registry.py

from __future__ import annotations

from dataclasses import dataclass

import torch

# Bessel / SHM / DQC / NARMA
from src.datasets.bessel_functions import BesselSequenceDataset
from src.datasets.damped_shm import DampedSHMSequenceDataset
from src.datasets.delayed_quantum_control import DelayedQCDataset
from src.datasets.narma_generator import make_narma_dataset

# NOTE: the two CUDA-Q datasets (jaynes_cummings / transmon) are imported lazily
# inside their own branches below. They ship as pre-simulated CSVs and never need
# cudaq at import time, but keeping the import local guarantees that importing
# this registry works without CUDA-Q installed.


@dataclass
class DatasetBundle:
	train_ds: torch.utils.data.Dataset
	test_ds: torch.utils.data.Dataset
	simulation_ds: torch.utils.data.Dataset
	train_len: int


def make_datasets(args) -> DatasetBundle:
	"""
	Create datasets based on args.dataset.
	Returns DatasetBundle(train_ds, test_ds, simulation_ds, train_len).
	"""
	train_ds = None
	test_ds = None
	simulation_ds = None
	train_len = 0

	if args.dataset == "bessel_j2":
		j2_dataset = BesselSequenceDataset(seq_len = args.window_len)

		n_total = len(j2_dataset)
		n_train = int(0.8 * n_total)
		train_len = n_train

		train_ds = torch.utils.data.Subset(j2_dataset, range(0, n_train))
		test_ds  = torch.utils.data.Subset(j2_dataset, range(n_train, n_total))
		simulation_ds = j2_dataset

	elif args.dataset == "damped_shm":
		damped_shm_dataset = DampedSHMSequenceDataset(seq_len = args.window_len)

		n_total = len(damped_shm_dataset)
		n_train = int(0.8 * n_total)
		train_len = n_train

		train_ds = torch.utils.data.Subset(damped_shm_dataset, range(0, n_train))
		test_ds  = torch.utils.data.Subset(damped_shm_dataset, range(n_train, n_total))
		simulation_ds = damped_shm_dataset

	elif args.dataset == "delayed_quantum_control":
		delayed_quantum_control_dataset = DelayedQCDataset(seq_len = args.window_len)

		n_total = len(delayed_quantum_control_dataset)
		n_train = int(0.8 * n_total)
		train_len = n_train

		train_ds = torch.utils.data.Subset(delayed_quantum_control_dataset, range(0, n_train))
		test_ds  = torch.utils.data.Subset(delayed_quantum_control_dataset, range(n_train, n_total))
		simulation_ds = delayed_quantum_control_dataset

	elif args.dataset == "narma_5":
		narma_5_dataset, _ = make_narma_dataset(n_0 = 5, seq_len = args.window_len)

		n_total = len(narma_5_dataset)
		n_train = int(0.8 * n_total)
		train_len = n_train

		train_ds = torch.utils.data.Subset(narma_5_dataset, range(0, n_train))
		test_ds  = torch.utils.data.Subset(narma_5_dataset, range(n_train, n_total))
		simulation_ds = narma_5_dataset

	elif args.dataset == "narma_10":
		narma_10_dataset, _ = make_narma_dataset(n_0 = 10, seq_len = args.window_len)

		n_total = len(narma_10_dataset)
		n_train = int(0.8 * n_total)
		train_len = n_train

		train_ds = torch.utils.data.Subset(narma_10_dataset, range(0, n_train))
		test_ds  = torch.utils.data.Subset(narma_10_dataset, range(n_train, n_total))
		simulation_ds = narma_10_dataset

	elif args.dataset == "jaynes_cummings":
		from src.datasets.jaynes_cummings import JaynesCummingsDataset

		full_dataset = JaynesCummingsDataset(
			seq_len=args.window_len,
			num_steps=3000, # Large enough for training
			decay=True      # Include noise for robustness
		)

		n_total = len(full_dataset)
		n_train = int(0.8 * n_total)
		train_len = n_train

		train_ds = torch.utils.data.Subset(full_dataset, range(0, n_train))
		test_ds  = torch.utils.data.Subset(full_dataset, range(n_train, n_total))
		simulation_ds = full_dataset

	elif args.dataset == "transmon":
		from src.datasets.transmon_resonator import TransmonDataset

		transmon_dataset = TransmonDataset(
			seq_len=args.window_len,
			num_steps=3000  # Matches the shipped transmon.csv
		)

		n_total = len(transmon_dataset)
		n_train = int(0.8 * n_total)
		train_len = n_train

		train_ds = torch.utils.data.Subset(transmon_dataset, range(0, n_train))
		test_ds  = torch.utils.data.Subset(transmon_dataset, range(n_train, n_total))
		simulation_ds = transmon_dataset

	else:
		raise ValueError(f"Unknown dataset: {args.dataset}")

	return DatasetBundle(
		train_ds=train_ds,
		test_ds=test_ds,
		simulation_ds=simulation_ds,
		train_len=train_len,
	)
