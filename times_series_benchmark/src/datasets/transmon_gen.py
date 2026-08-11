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

# Dispersive transmon-resonator dynamics generated with NVIDIA CUDA-Q.
#
# The dispersive Hamiltonian, the coherent-state preparation via
# operators.displace, and the ScipyZvodeIntegrator evolution are adapted from
# the NVIDIA CUDA-Q dynamics documentation examples, which are distributed
# under Apache-2.0:
#   https://nvidia.github.io/cuda-quantum/latest/using/backends/dynamics.html
#   https://github.com/NVIDIA/cuda-quantum  (Apache License 2.0)

import cudaq
from cudaq import operators, spin, Schedule, ScipyZvodeIntegrator
import numpy as np
import cupy as cp

def coherent_state(num_levels, amplitude):
    displace_mat = operators.displace(0).to_matrix({0: num_levels}, displacement=amplitude)
    return cp.array(np.transpose(displace_mat)[0])

def generate_transmon_dynamics(num_steps=3000, t_max=25.0):
    cudaq.set_target("dynamics")

    N = 20
    dimensions = {0: 2, 1: N}

    # System parameters in GHz
    omega_01 = 3.0 * 2 * np.pi  
    omega_r = 2.0 * 2 * np.pi  
    chi_01 = 0.025 * 2 * np.pi
    chi_12 = 0.0

    omega_01_prime = omega_01 + chi_01
    omega_r_prime = omega_r - chi_12 / 2.0
    chi = chi_01 - chi_12 / 2.0

    # Dispersive Hamiltonian
    hamiltonian = 0.5 * omega_01_prime * spin.z(0) + (
        omega_r_prime + chi * spin.z(0)) * operators.number(1)

    # Initial states
    transmon_state = cp.array([1. / np.sqrt(2.), 1. / np.sqrt(2.)], dtype=cp.complex128)
    cavity_state = coherent_state(N, 2.0)
    psi0 = cudaq.State.from_data(cp.kron(transmon_state, cavity_state))

    # Evolve using the shorter, highly-resolved time schedule
    steps = np.linspace(0, t_max, num_steps)
    schedule = Schedule(steps, ["time"])

    print(f"Evolving Transmon-Resonator for {num_steps} steps up to t={t_max} ns...")
    evolution_result = cudaq.evolve(
        hamiltonian,
        dimensions,
        schedule,
        psi0,
        observables=[operators.position(1)], # We only extract the dynamic resonator quadrature (x)
        collapse_operators=[],
        store_intermediate_results=cudaq.IntermediateResultSave.EXPECTATION_VALUE,
        integrator=ScipyZvodeIntegrator()
    )

    exp_vals = evolution_result.expectation_values()
    if not exp_vals or len(exp_vals) == 0:
        raise RuntimeError("CUDA-Q Evolve returned NO data.")

    # Extract the array
    data = np.array([val[0].expectation() for val in exp_vals])
    print(f"Successfully generated {len(data)} Transmon data points.")
    return data
