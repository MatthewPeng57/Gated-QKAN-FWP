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
# ImgObsFlatWrapper is adapted from MiniGrid's ImgObsWrapper
# (https://github.com/Farama-Foundation/Minigrid, MIT License,
# Farama Foundation), extended to flatten the image observation.

"""Observation wrapper that reduces MiniGrid observations to a flat image vector."""

import gymnasium as gym
import minigrid  # noqa: F401  -- registers the MiniGrid environment ids with gymnasium
import torch

from gymnasium.core import ObservationWrapper


class ImgObsFlatWrapper(ObservationWrapper):
    """
    Use the image as the only observation output, no language/mission,
    flattened to a 1-D vector.

    Example:
        >>> import gymnasium as gym
        >>> import minigrid
        >>> from MiniGridWrappers import ImgObsFlatWrapper
        >>> env = gym.make("MiniGrid-Empty-5x5-v0")
        >>> obs, _ = env.reset()
        >>> sorted(obs.keys())
        ['direction', 'image', 'mission']
        >>> env = ImgObsFlatWrapper(env)
        >>> obs, _ = env.reset()
        >>> obs.shape
        (147,)

    Note:
        ``observation_space`` is left describing the unflattened ``(7, 7, 3)``
        image, so it does not match the flattened observation this wrapper
        returns. Callers therefore set the input size explicitly (147) instead
        of reading ``observation_space.shape``.
    """

    def __init__(self, env):
        """A wrapper that makes image the only observation.

        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)
        self.observation_space = env.observation_space.spaces["image"]

    def observation(self, obs):
        return obs["image"].flatten()


def main():
	env = gym.make('MiniGrid-Empty-5x5-v0')
	env = ImgObsFlatWrapper(env)
	init_obs, _ = env.reset()
	print(init_obs)
	print(init_obs.shape)

	obs = torch.from_numpy(init_obs).unsqueeze(0).unsqueeze(0)

	print(obs)
	print(obs.shape)


	print("OBS SPACE: ", env.observation_space.shape)
	print("ACT SPACE: ", env.action_space.n)

	return


if __name__ == '__main__':
	main()
