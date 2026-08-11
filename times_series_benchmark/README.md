# times_series_benchmark — QKAN-FWP time-series suite

All four quantum-inspired KAN fast-weight programmers on seven time-series forecasting tasks: five
analytic signals generated on the fly, and two quantum-dynamics traces shipped as CSVs.

## Models

| `--model` | Slow programmer | Fast programmer |
|---|---|---|
| `qqkanfwp` | QKAN | QKAN |
| `lqkanfwp` | Linear | QKAN |
| `qkanlfwp` | QKAN | Linear |
| `qkanvfwp` | QKAN | VQC (PennyLane) |

## Datasets

| `--dataset` | Kind | Source |
|---|---|---|
| `bessel_j2` | analytic | Bessel function J₂, generated at runtime |
| `damped_shm` | analytic | damped pendulum, `scipy.integrate.odeint` |
| `delayed_quantum_control` | analytic | Gaussian pulse train |
| `narma_5` | analytic | NARMA-5 |
| `narma_10` | analytic | NARMA-10 |
| `jaynes_cummings` | quantum dynamics | shipped CSV (CUDA-Q simulation) |
| `transmon` | quantum dynamics | shipped CSV (CUDA-Q simulation) |

Every dataset uses a chronological **80 / 20** train/test split (`registry.py`).

The two quantum datasets load from `cuda_q_data/*.csv` by default, so **NVIDIA CUDA-Q is not required
to train** — it is needed only to regenerate them. See [`cuda_q_data/README.md`](cuda_q_data/README.md).

## Run

```bash
cd times_series_benchmark

# one model, one dataset
PYTHONPATH=. python train.py --model qqkanfwp --dataset bessel_j2 --epochs 100

# the full sweep: 4 models x 3 seeds on jaynes_cummings
bash run_all_dataset.sh
```

Everything here runs on CPU (`--device cpu` is what the sweep uses). `qkanvfwp` is much the slowest —
its variational circuit is a PennyLane state-vector simulation.

## Defaults

`--epochs 100 --lr 1e-3 --batch_size 4 --window_len 64 --input_size 1 --hidden_size 8 --qnn_depth 2
--seed 0`, matching `run_all_dataset.sh` (which sweeps seeds 0–2).

### ⚠️ Three flags do not do what their names suggest

These are long-standing quirks of the model definitions. They are documented rather than "fixed",
because changing them would change published results.

| Flag | Model | Actual behaviour |
|---|---|---|
| `--hidden_size` | `qkanlfwp` | **Ignored.** `QKANLFWP.py` sets `hidden_size = 4` unconditionally. |
| `--hidden_size` | `lqkanfwp` | **Off by one.** `LQKANFWP.py` uses `hidden_size - 1`. |
| `--qnn_depth` | `qqkanfwp`, `lqkanfwp`, `qkanlfwp` | **No-op.** Only `qkanvfwp` consumes it (as its VQC circuit depth). |

## Outputs

Each run writes a self-describing directory under `--save_dir` (relative paths only — absolute paths
and `..` are rejected), containing the resolved `args.json`, an environment snapshot, logs, a metrics
CSV, and figures. Output directories are gitignored.

## Attribution

The NARMA generator is credited to Samuel Yen-Chi Chen; `damped_shm.py` adapts a Skill-Lync tutorial;
the two quantum-dynamics generators derive from NVIDIA CUDA-Q documentation examples;
`qkan_fast.py` and `fast_solver.py` are modified copies of [QKAN](https://github.com/Jim137/qkan)
(Apache-2.0). Full details in [`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).
