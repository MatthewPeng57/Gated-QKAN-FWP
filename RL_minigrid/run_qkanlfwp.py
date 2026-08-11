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
# The A3C scaffolding (worker processes, shared optimizer, push_and_pull update)
# is derived from https://github.com/MorvanZhou/pytorch-A3C (MIT License),
# by Morvan Zhou.

"""Train the QKAN-linear fast-weight programmer policy on MiniGrid with A3C.

Runs asynchronous advantage actor-critic (Mnih et al., 2016) with one worker per
available CPU core, over the seeds in ``RANDOM_SEEDS``. Per seed it writes the
config, a reward CSV, a reward plot, the final global weights and the raw
episode-reward list under ``results/<env>/<model>/seed_<n>/``.

The A3C scaffolding follows Morvan Zhou's PyTorch A3C implementation
(https://github.com/MorvanZhou/pytorch-A3C, MIT License).
"""

import torch
from utils import v_wrap, record
from util.a3c_update import push_and_pull
from plot_functions import full_plotting

import torch.multiprocessing as mp
from shared_adam import SharedAdam
import gymnasium as gym
import minigrid  # noqa: F401  -- registers the MiniGrid environment ids with gymnasium
import os

from os.path import abspath, dirname, join

import pickle

from util.qkanlfwp_utils import QuantumFWPNet as Net

from MiniGridWrappers import ImgObsFlatWrapper
import random
import numpy as np

import csv
import json

os.environ["OMP_NUM_THREADS"] = "1"
# Setting OMP_NUM_THREADS from inside Python does not take effect on some Linux
# systems. Export OMP_NUM_THREADS=1 before launching this script if the worker
# processes still oversubscribe the CPU.

UPDATE_GLOBAL_ITER = 5
GAMMA = 0.9
MAX_EP = 10000


ENV_NAME = 'MiniGrid-Empty-16x16-v0'
env = gym.make(ENV_NAME)
env = ImgObsFlatWrapper(env)


# ImgObsFlatWrapper flattens the 7x7x3 image observation, but leaves
# observation_space describing the unflattened image, so the flat size is set
# explicitly here rather than read off observation_space.shape.
N_S = 147
N_A = env.action_space.n

print("OBSERVATION SPACE: ", N_S)
print("ACTION SPACE: ", N_A)

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    return total_params, trainable_params

class Worker(mp.Process):
	def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name):
		super(Worker, self).__init__()
		self.name = 'w%02i' % name
		self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
		self.gnet, self.opt = gnet, opt
		self.lnet = Net(s_dim = N_S, a_dim = N_A, rnn_as_reservoir = False).double()           # local network

		self.env = ImgObsFlatWrapper(gym.make(ENV_NAME))

	def run(self):
		total_step = 1
		while self.g_ep.value < MAX_EP:
			s, _ = self.env.reset()  # gymnasium reset returns (state, info)
			fast_weight, fast_bias = self.lnet.fwp.fwp_cell.initial_fast_params(batch_size = 1)
			buffer_s, buffer_a, buffer_r = [], [], []
			ep_r = 0.
			while True:
				a, fast_weight, fast_bias = self.lnet.choose_action(v_wrap(s[None, :]).double(), fast_weight, fast_bias)
				s_, r, terminated,truncated , _ = self.env.step(a)  # gymnasium returns terminated/truncated instead of done
				done = terminated or truncated
				ep_r += r
				buffer_a.append(a)
				buffer_s.append(s)
				buffer_r.append(r)

				if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
					# sync
					# Bootstraps on `done` (terminated or truncated). The four runners
					# deliberately differ on this and the asymmetry is preserved to stay
					# consistent with the reported results.
					push_and_pull(self.opt, self.lnet, self.gnet, done, s_, (fast_weight, fast_bias), buffer_s, buffer_a, buffer_r, GAMMA)
					buffer_s, buffer_a, buffer_r = [], [], []

					if done:  # done and print information
						record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
						break
				s = s_
				total_step += 1
		self.res_queue.put(None)


if __name__ == "__main__":
	MODEL_NAME = "QKANLFWP"

	base_results_dir = join(dirname(abspath(__file__)), "results", ENV_NAME, MODEL_NAME)

	RANDOM_SEEDS = [0,1,2,3,4]
	LR = 1e-4
	for random_seed in RANDOM_SEEDS:
		print(f"\n===== Running with random_seed={random_seed} =====")

		results_dir = join(base_results_dir, f"seed_{random_seed}")

		os.makedirs(results_dir, exist_ok=True)
		config = {
			"env": ENV_NAME,
			"model": MODEL_NAME,
			"gamma": GAMMA,
			"update_iter": UPDATE_GLOBAL_ITER,
			"max_ep": MAX_EP,
			"learning_rate": LR,
			"random_seeds": RANDOM_SEEDS,
		}

		with open(join(results_dir, "config.json"), "w") as f:
			json.dump(config, f, indent=4)
		csv_path = join(results_dir, f"seed_{random_seed}.csv")

		set_global_seed(random_seed)
		gnet = Net(s_dim = N_S, a_dim = N_A, rnn_as_reservoir = False).double()        # global network
		count_parameters(gnet)
		gnet.share_memory()         # share the global parameters in multiprocessing
		opt = SharedAdam(gnet.parameters(), lr=LR, betas=(0.92, 0.999))      # global optimizer
		global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

		# parallel training
		workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(min(mp.cpu_count(),80))]
		[w.start() for w in workers]
		res = []                    # record episode reward to plot
		while True:
			r = res_queue.get()
			if r is not None:
				res.append(r)
			else:
				break
		[w.join() for w in workers]

		# Plot and save
		na_raw, na_mu, na_sigma = full_plotting(_fileTitle=f"seed_{random_seed}", _trainingLength = len(res), _currentRewardList = res, save_dir=results_dir)
		with open(csv_path, "w", newline="") as f:
			writer = csv.writer(f)
			writer.writerow(["episode", "reward", "avg100", "std100"])

			for i in range(len(na_raw)):
				writer.writerow([i, na_raw[i], na_mu[i], na_sigma[i]])

		path_model = join(results_dir, f"seed_{random_seed}_model.pth")
		torch.save(gnet.state_dict(), path_model)
		# Save the raw data of res
		path_episode_reward = join(results_dir, f"seed_{random_seed}_raw_rewards.pkl")

		with open(path_episode_reward, "wb") as fp:
			pickle.dump(res, fp)
