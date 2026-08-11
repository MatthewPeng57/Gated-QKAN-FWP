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
# Derived from https://github.com/MorvanZhou/pytorch-A3C (MIT License),
# by Morvan Zhou.

"""Shared A3C update step for the fast-weight-programmer policies.

This module consolidates the previously duplicated ``push_and_pull``
implementations from ``utils_qqkan.py``, ``utils_classical_fwp.py`` and
``utils_quantum_fwp.py``. Those three copies were identical apart from how many
fast-state tensors they forwarded to ``lnet.forward_action(...)``; that single
difference is now expressed through the ``fast_state`` tuple.
"""

import numpy as np

from utils import v_wrap


def push_and_pull(opt, lnet, gnet, done, s_, fast_state, bs, ba, br, gamma):
    """A3C update: compute the n-step return, push local grads to the global net, pull weights back.

    fast_state: model-specific tuple of fast-weight tensors, forwarded as
        ``lnet.forward_action(state, *fast_state)``:
          qqkanfwp / lqkanfwp -> (theta, base_weight)
          qkanlfwp            -> (fast_weight, fast_bias)
          qkanvfwp            -> (fast_params,)
    """
    if done:
        v_s_ = 0.               # terminal
    else:
        # bootstrap from the critic's value of the next state
        v_s_ = lnet.forward_action(v_wrap(s_[None, :]).double(), *fast_state)[1].data.numpy()[0]

    buffer_v_target = []
    for r in br[::-1]:    # reverse buffer r
        v_s_ = r + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()

    loss = lnet.loss_func(
        v_wrap(np.vstack(bs)).double(),
        v_wrap(np.array(ba), dtype=np.int64) if ba[0].dtype == np.int64 else v_wrap(np.vstack(ba)),
        v_wrap(np.array(buffer_v_target)[:, None]).double())

    # calculate local gradients and push local parameters to global.
    # set_to_none=False is required: torch > 2.0 defaults to True, and setting
    # the grads to None kills the shared memory backing them, which breaks the
    # multiprocessing hand-off below.
    opt.zero_grad(set_to_none=False)
    loss.backward()
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        gp._grad = lp.grad
    opt.step()

    # pull global parameters
    lnet.load_state_dict(gnet.state_dict())
