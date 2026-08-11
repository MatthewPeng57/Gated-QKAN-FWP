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
#
# The A3C scaffolding these helpers belong to is derived from
# https://github.com/MorvanZhou/pytorch-A3C (MIT License), by Morvan Zhou.

"""Small helpers shared by the A3C runners: array wrapping, weight init, logging."""

from torch import nn
import torch
import numpy as np


def v_wrap(np_array, dtype=np.float32):
	if np_array.dtype != dtype:
		np_array = np_array.astype(dtype)
	return torch.from_numpy(np_array)


def set_init(modules):
	"""Recursively apply N(0, 0.1) weight init and zero bias to every nn.Linear."""
	if isinstance(modules, (list, tuple)):
		for module in modules:
			set_init(module)  # recurse
	else:
		for m in modules.modules():  # recursive
			if isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, mean=0., std=0.1)
				nn.init.constant_(m.bias, 0.)


def record(global_ep, global_ep_r, ep_r, res_queue, name):
	with global_ep.get_lock():
		global_ep.value += 1
	with global_ep_r.get_lock():
		# the raw episode score is recorded, not a running average, so that
		# smoothing stays a post-processing choice
		global_ep_r.value = ep_r
	res_queue.put(global_ep_r.value)
	print(
		name,
		"Ep:", global_ep.value,
		"| Ep_r: {}".format(ep_r),
	)
