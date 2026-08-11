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

# Jaynes-Cummings cavity-QED dynamics generated with NVIDIA CUDA-Q.
#
# The evolution setup (boson operators, Schedule, collapse operators and the
# ScipyZvodeIntegrator call) is adapted from the NVIDIA CUDA-Q dynamics
# documentation examples, which are distributed under Apache-2.0:
#   https://nvidia.github.io/cuda-quantum/latest/using/backends/dynamics.html
#   https://github.com/NVIDIA/cuda-quantum  (Apache License 2.0)

import cudaq
from cudaq import boson, Schedule, ScipyZvodeIntegrator
import numpy as np
import cupy as cp

def generate_jc_dynamics(num_steps=1000, t_max=50, decay=True):
    cudaq.set_target("dynamics")
    
    # System: qubit (2-level) and cavity, truncated at 5 Fock levels
    dimensions = {0: 2, 1: 5}
    
    # Operators
    a = boson.annihilate(1)
    a_dag = boson.create(1)
    sm = boson.annihilate(0)
    sm_dag = boson.create(0)
    
    # Hamiltonian
    H_cav = 2 * np.pi * boson.number(1)
    H_qubit = 2 * np.pi * boson.number(0)
    H_int = 2 * np.pi * 0.5 * (sm * a_dag + sm_dag * a)
    hamiltonian = H_cav + H_qubit + H_int
    
    # Initial State: Qubit ground, Cavity 1 photon
    qubit_state = cp.array([[1.0, 0.0], [0.0, 0.0]], dtype=cp.complex128)
    cavity_state = cp.zeros((5, 5), dtype=cp.complex128)
    cavity_state[1][1] = 1.0
    rho0 = cudaq.State.from_data(cp.kron(cavity_state, qubit_state))
    
    steps = np.linspace(0, t_max, num_steps)
    schedule = Schedule(steps, ["time"])
    
    # Define collapse operators for noise (makes the task harder/realistic)
    collapses = [np.sqrt(0.05) * a] if decay else []
    
    result = cudaq.evolve(
        hamiltonian, dimensions, schedule, rho0,
        observables=[boson.number(0)], # Tracking Qubit Excitation
        collapse_operators=collapses,
        store_intermediate_results=cudaq.IntermediateResultSave.EXPECTATION_VALUE,
        integrator=ScipyZvodeIntegrator()
    )
    
    # Extract expectation values
    data = np.array([exp[0].expectation() for exp in result.expectation_values()])
    return data
