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

"""Rolling reward statistics for the A3C runs.

This repository ships no plotting code. Runs emit the raw per-episode rewards
plus the rolling summary computed here; plotting is left to the reader.
"""

from collections import deque

import numpy as np


def rolling_reward_stats(reward_list, window=100):
	"""Summarise an episode-reward sequence with a trailing rolling window.

	For each episode the reward is appended to a ``window``-length trailing
	buffer, and the mean and population standard deviation of that buffer are
	recorded. Before ``window`` episodes have elapsed the statistics are over
	however many episodes have been seen so far.

	Args:
		reward_list: per-episode rewards, in order.
		window: length of the trailing window (100 episodes by default).

	Returns:
		``(na_raw, na_mu, na_sigma)`` as float arrays, each the same length as
		``reward_list``: the raw rewards, the rolling mean, and the rolling
		standard deviation. These are the ``reward``, ``avg100`` and ``std100``
		columns of the per-seed CSV.
	"""
	scores = []                           # list containing scores from each episode
	scores_std = []                       # rolling std dev over the last `window`
	scores_avg = []                       # rolling mean over the last `window`
	scores_window = deque(maxlen=window)  # last `window` scores

	for score in reward_list:
		scores_window.append(score)       # save most recent score
		scores.append(score)              # save most recent score
		scores_std.append(np.std(scores_window)) # rolling std dev of the last `window`
		scores_avg.append(np.mean(scores_window)) # rolling mean of the last `window`

	na_raw = np.array(scores)
	na_mu = np.array(scores_avg)
	na_sigma = np.array(scores_std)

	return na_raw, na_mu, na_sigma
