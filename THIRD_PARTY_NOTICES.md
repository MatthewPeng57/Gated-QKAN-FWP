# Third-Party Notices & Provenance

This repository's own source code is licensed **Apache-2.0** (see `LICENSE`). It additionally
contains, derives from, or depends on the third-party work inventoried below. Where a file is a
modified copy of upstream code, that file retains the upstream copyright header **and** carries a
notice that it was changed, per Apache-2.0 §4(b).

> **Data files are not covered by the repository's code license.** See §8.

---

## 0. Provenance of the training pipeline

The training pipeline, experiment scaffolding, trainer utilities and dataset generators in this
repository are **adapted from a private research repository authored by Samuel Yen-Chi Chen**, used
with permission. He is a co-author of the accompanying paper and is named in the copyright notice of
this repository (see `LICENSE` and `NOTICE`). The affected code includes, in each task folder,
`src/trainers/`, `src/utils/experiment.py`, the dataset modules under `src/datasets/`, and the model
scaffolding that the four gated QKAN-FWP variants are built on.

Note also that **Jiun-Cheng Jiang, the author of QKAN (§1), is a co-author of the same paper**. The
QKAN code is nonetheless treated here as third-party software and carries its own upstream Apache-2.0
headers and modification notices, exactly as it would from any outside project.

---

## 1. QKAN — Apache-2.0 *(vendored and modified)*

- **Upstream:** <https://github.com/Jim137/qkan>
- **Copyright:** (c) 2024, 2026 Jiun-Cheng Jiang. All rights reserved.
- **License:** Apache License, Version 2.0
- **Paper:** *Quantum Variational Activation Functions Empower Kolmogorov-Arnold Networks*,
  [arXiv:2509.14026](https://arxiv.org/abs/2509.14026)
- **Also a runtime dependency:** `qkan==0.2.1` is installed from PyPI; the vendored files below are in
  addition to, not a replacement for, that package.

### Derived files in this repository

| File | Upstream counterpart |
|---|---|
| `RL_minigrid/qkan_fast.py` | `qkan/qkan.py` |
| `Solar_cycle_forecasting/src/models/utils/qkan_fast.py` | `qkan/qkan.py` |
| `Solar_cycle_forecasting/src/models/utils/fast_solver.py` | `qkan/solver/` |
| `Solar_cycle_forecasting/src/models/utils/fast_fused_ops.py` | `qkan/fused_ops.py` |
| `Solar_cycle_forecasting/src/models/utils/cutile_batched_ops.py` | `qkan/cutile_ops.py` |
| `times_series_benchmark/src/models/utils/qkan_fast.py` | `qkan/qkan.py` |
| `times_series_benchmark/src/models/utils/fast_solver.py` | `qkan/solver/` |

### Statement of modifications (Apache-2.0 §4(b))

These files were changed by the authors of this repository. The substantive modifications are:

1. **`theta` supplied per call, and optionally per sample.** Upstream `QKAN` treats `theta` and
   `base_weight` as shared `nn.Parameter`s. In a fast-weight programmer they are produced by the slow
   programmer at call time instead, so `QKANLayer.forward`, `QKANLayer.forward_no_sum` and
   `QKAN.forward` were extended to accept `theta` and `base_weight` as arguments, falling back to the
   stored parameters when omitted.
   In `Solar_cycle_forecasting` and `times_series_benchmark` the supplied `theta` additionally carries
   a leading batch axis `(B, out_dim, in_dim, reps+1, 2)` — one distinct `theta` per sample — and the
   solver and fused-kernel paths were extended to index that axis (in the Triton kernels this appears
   as an added `stride_t_b` parameter; in the cuTile kernels as a leading `b_offs` index on every
   gather). In `RL_minigrid` the caller reduces `theta` before the call, so only the per-call
   substitution is exercised there.
2. **Solver rewiring.** `from qkan.solver import (...)` was replaced by `from .fast_solver import (...)`
   so the repo-local batched-theta solvers are used. The original import is retained as a comment at
   the substitution site.
3. **Added `cutile` batched solver path** (`cutile_batched_solver`) and its registration in the
   accepted-solver lists.
4. **Optional hardware-backend hooks** guarded by try/except (`qiskit`, `braket`/`cudaq`), inert when
   the corresponding module is absent.

Copyright in these modifications is held by the authors of this repository; the underlying work
remains Copyright (c) Jiun-Cheng Jiang under Apache-2.0.

### Onward attribution: pykan

Several routines inside the upstream QKAN source carry the annotation `Adapted from "pykan"`
(<https://github.com/KindXiaoming/pykan>, Liu et al., *KAN: Kolmogorov-Arnold Networks*,
[arXiv:2404.19756](https://arxiv.org/abs/2404.19756)). Those annotations are preserved verbatim in the
vendored files. Consult the pykan project for its license terms.

---

## 2. MorvanZhou / pytorch-A3C — MIT

- **Upstream:** <https://github.com/MorvanZhou/pytorch-A3C>
- **Author:** Morvan Zhou

The A3C training scaffolding in `RL_minigrid/` derives from this project — specifically
`utils.py` (`v_wrap`, `set_init`, `record`), `shared_adam.py` (`SharedAdam`), the consolidated
`util/a3c_update.py` (`push_and_pull`), and the `Worker(mp.Process)` structure inside each
`run_*.py`. The upstream docstring credit has been rewritten into a third-person citation.

---

## 3. Farama Foundation — MiniGrid & Gymnasium — MIT

- **Upstream:** <https://github.com/Farama-Foundation/Minigrid>, <https://github.com/Farama-Foundation/Gymnasium>

`RL_minigrid/MiniGridWrappers/obs_wrappers.py` contains `ImgObsFlatWrapper`, which is adapted from
MiniGrid's `ImgObsWrapper` (adding a flatten of the image observation). Both packages are also runtime
dependencies (`minigrid`, `gymnasium`).

---

## 4. PennyLane — Apache-2.0

- **Upstream:** <https://github.com/PennyLaneAI/pennylane>

Runtime dependency, used for the variational-quantum-circuit paths
(`qkanvfwp`, `vqc_components.py`, and the `qml` solver). The layer idiom in `vqc_components.py`
(`H_layer` / `RX_layer` / `RY_layer` / `RZ_layer` / `entangling_layer`) follows the standard PennyLane
tutorial pattern.

---

## 5. NVIDIA CUDA-Q — Apache-2.0

- **Upstream:** <https://github.com/NVIDIA/cuda-quantum>

The two quantum-dynamics dataset generators in `times_series_benchmark/src/datasets/`
(`jaynes_cummings_gen.py` — a two-level atom coupled to a cavity mode; `transmon_gen.py` —
transmon qubit dispersively coupled to a resonator) are derived from the CUDA-Q dynamics documentation
examples. `cudaq` is an optional dependency, required only to *regenerate* the shipped CSV data.

---

## 6. NARMA dataset generator

`times_series_benchmark/src/datasets/narma_generator.py` credits **Samuel Yen-Chi Chen** and follows
the NARMA construction described in
*Quantum reservoir computing* — <https://www.nature.com/articles/s41598-022-05061-w>.

---

## 7. Damped simple harmonic motion generator

`times_series_benchmark/src/datasets/damped_shm.py` is adapted from the Skill-Lync tutorial
*Solving 2nd order ODE for a simple pendulum using Python*
(<https://skill-lync.com/projects/Solving-2nd-order-ODE-for-a-simple-pendulum-using-python-40080>).

---

## 8. Data files — NOT covered by the repository code license

### 8.1 SILSO sunspot numbers — **CC BY-NC 4.0 (NonCommercial)**

- **File:** `Solar_cycle_forecasting/data/Sunspots.csv`
- **sha256:** `15c3d116ad6c5a5427837ae4cec39aa9b4b2e4a0d8e374d501a4ac760fc50b35`
- **Contents:** monthly mean total sunspot number, 3265 rows, 1749-01-31 … 2021-01-31
- **Source:** WDC-SILSO, Royal Observatory of Belgium, Brussels — <https://www.sidc.be/SILSO/>
- **License:** Creative Commons Attribution-NonCommercial 4.0 International

> **NonCommercial.** This dataset may not be used for commercial purposes. Cite SILSO in any work
> that uses it, and respect the Royal Observatory of Belgium's redistribution terms. This restriction
> applies to the data only, not to the Apache-2.0 source code in this repository.

### 8.2 Quantum-dynamics CSVs — generated

- **Files:** `times_series_benchmark/cuda_q_data/{jaynes_cummings,transmon}.csv`
- Produced by `times_series_benchmark/make_cudaq_data.py` using the CUDA-Q generators in §5. Covered by
  this repository's Apache-2.0 licence. Shipped so the repository runs without NVIDIA CUDA-Q;
  regenerate with that script on a CUDA-Q machine.

### 8.3 Analytic datasets — generated at runtime

`bessel_j2`, `damped_shm`, `delayed_quantum_control`, `narma_5`, `narma_10` are computed on the fly and
ship no data files.

---

## 9. Runtime dependencies and their licenses

| Package | License |
|---|---|
| `torch` | BSD-3-Clause |
| `numpy`, `pandas`, `scipy`, `scikit-learn` | BSD-3-Clause |
| `matplotlib` | Matplotlib License (PSF-based) |
| `qkan` | Apache-2.0 |
| `pennylane`, `pennylane-lightning` | Apache-2.0 |
| `triton` | MIT |
| `gymnasium`, `minigrid` | MIT |
| `tqdm` | MPL-2.0 / MIT |
| `cudaq`, `cupy`, `cuquantum` *(optional)* | Apache-2.0 / MIT / NVIDIA terms |
| `opt-einsum` *(optional)* | MIT |
