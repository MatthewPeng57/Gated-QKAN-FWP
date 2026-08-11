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

"""GQKAN-QKANFWP: a quantum-inspired KAN fast-weight programmer.

``FWPCell`` holds the slow programmer (a linear -> QKAN -> linear encoder plus
two low-rank heads that emit the fast layer's ``theta``) and the fast QKAN
layer it programs. ``FWP`` wraps the cell for direct multi-step forecasting:
it runs the slow programmer over the whole input window at once, folds the
per-step ``theta`` updates into a single final ``theta`` (either with the
streaming prefix-scan kernel or the legacy cumsum path), evaluates the fast
QKAN layer once, and projects the result to the forecast horizon.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from qkan import QKAN

from .utils.qkan_fast import QKAN as fast_QKAN

torch.set_default_dtype(torch.float32)


### FWP Cell Module
class FWPCell(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, args):
        super().__init__()

        self.qkan_s_dim_1 = args.qkan_s_dim_1
        self.qkan_s_dim_2 = args.qkan_s_dim_2
        in_resize = args.in_resize
        # Both are read back by FWP to size its projection head.
        self.out_resize  = args.out_resize
        self.output_size = output_size

        self.qkan_layer = fast_QKAN(
            width=[in_resize, self.out_resize],
            solver=getattr(args, "fast_solver", "flash"),
            # ba_trainable=True,
            # preact_trainable=True,
            c_dtype=torch.float32,
            reps=2,
            seed=args.seed
        )
  
        layer = self.qkan_layer.layers[0]

        self.theta_shape = layer.theta.shape      
        self.base_shape  = layer.base_weight.shape 
        
        self.slow_program_encoder = nn.Sequential(
            nn.Linear(input_size, self.qkan_s_dim_1),  
            QKAN(
                width=[self.qkan_s_dim_1, self.qkan_s_dim_2],
                reps=1,
                ba_trainable=True,
                solver="cutile",
                preact_trainable=True,
                c_dtype=torch.float32,
                seed=args.seed
            ), 
            nn.Linear(self.qkan_s_dim_2, hidden_size)
        )
  
        # low-rank theta generator
        self.theta_head_A = nn.Linear(hidden_size, self.theta_shape[0]*self.theta_shape[2])
        self.theta_head_B = nn.Linear(hidden_size, self.theta_shape[1]*self.theta_shape[3])

        self.prelinear = torch.nn.Linear(input_size, in_resize)

        self.fast_gate = torch.nn.Linear(hidden_size, 1)
        self.fast_gate.bias.data.fill_(2.0)

    # NOTE: FWPCell owns the submodules only. The recurrence itself lives in
    # FWP.forward, which runs the slow programmer over the whole window at
    # once rather than stepping this cell per timestep.


### FWP Module


class FWP(nn.Module):
    def __init__(self, qfwp_cell, device, output_relu: bool = False,
                 use_streaming_fwp: bool = False):
        super().__init__()
        self.fwp_cell = qfwp_cell
        self.device = device
        self.batch_norm = nn.BatchNorm1d(self.fwp_cell.out_resize)
        self.dropout = nn.Dropout(0.3)
        self.output_layer = nn.Linear(self.fwp_cell.out_resize, self.fwp_cell.output_size)
        # Runtime toggle for the final output ReLU, which clamps forecasts to
        # be non-negative. Default False leaves the head unclamped.
        self.output_relu = output_relu
        self.use_streaming_fwp = use_streaming_fwp

    def forward(self, x):
        # x shape: [Batch, Seq_Len, Input_Dim]
        batch_size, seq_len, _ = x.size()

        # 1. Run Slow Programmer on the entire sequence at once
        # Flatten batch and seq to process in parallel
        x_flat = x.view(batch_size * seq_len, -1)
        
        # Get slow program features for all time steps
        res = self.fwp_cell.slow_program_encoder(x_flat) 
        gates = torch.sigmoid(self.fwp_cell.fast_gate(res))
        
        # Generate all delta thetas [B*L, C, H, W...]
        A = self.fwp_cell.theta_head_A(res).view(batch_size * seq_len, self.fwp_cell.theta_shape[0], self.fwp_cell.theta_shape[2])
        B = self.fwp_cell.theta_head_B(res).view(batch_size * seq_len, self.fwp_cell.theta_shape[1], self.fwp_cell.theta_shape[3])

        # Reshape A and B back to (B, L, ...) for the streaming kernel.
        A_bl = A.view(batch_size, seq_len, self.fwp_cell.theta_shape[0], self.fwp_cell.theta_shape[2])
        B_bl = B.view(batch_size, seq_len, self.fwp_cell.theta_shape[1], self.fwp_cell.theta_shape[3])
        gates_bl = gates.view(batch_size, seq_len)

        # Streaming FWP recurrence (replaces materialize + cumsum).
        if getattr(self, "use_streaming_fwp", False):
            from .utils.streaming_fwp import streaming_fwp_final_theta
            final_theta = streaming_fwp_final_theta(A_bl, B_bl, gates_bl)
        else:
            # Legacy cumsum path — kept for A/B debugging via use_streaming_fwp=False.
            delta_theta = torch.einsum("bld,bqe->blqde", A, B)
            delta_theta = delta_theta.view(batch_size, seq_len, *self.fwp_cell.theta_shape)
            gates_r = gates_bl.view(batch_size, seq_len, 1, 1, 1, 1)
            log_gates = torch.log(gates_r + 1e-12)
            reverse_gates_cumsum = torch.cumsum(log_gates.flip(dims=[1]), dim=1).flip(dims=[1])
            decay_to_end = torch.exp(reverse_gates_cumsum - log_gates)
            weighted_delta = (1 - gates_r) * delta_theta * decay_to_end
            final_theta = weighted_delta.sum(dim=1)

        # 3. Final Forward Pass
        # Use the input from the last time step
        last_input = self.fwp_cell.prelinear(x[:, -1, :])
        final_hidden_state = self.fwp_cell.qkan_layer(last_input, final_theta, None)
        
        # 4. Projection
        regularized = self.dropout(self.batch_norm(final_hidden_state))
        final_predictions = self.output_layer(regularized)
        if self.output_relu:
            final_predictions = F.relu(final_predictions)

        return final_predictions.unsqueeze(0)
