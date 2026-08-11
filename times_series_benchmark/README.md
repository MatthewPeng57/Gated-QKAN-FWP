# times_series_benchmark — GQKAN-FWP time-series suite

All four quantum-inspired KAN fast-weight programmers on seven time-series forecasting tasks: five
analytic signals generated on the fly, and two quantum-dynamics traces shipped as CSVs.

## Models

Names follow the paper, *Gated QKAN-FWP: Scalable Quantum-inspired Sequence Learning*
([arXiv:2605.06734](https://arxiv.org/abs/2605.06734)).

| `--model` | Paper name | Slow programmer | Fast programmer |
|---|---|---|---|
| `gqkan_qkanfwp` | GQKAN-QKANFWP | QKAN | QKAN |
| `gqkanfwp` | GQKANFWP | Linear | QKAN |
| `gqkan_fwp` | GQKAN-FWP | QKAN | Linear |
| `gqkan_qfwp` | GQKAN-QFWP | QKAN | VQC (PennyLane) |

> **Note:** `gqkanfwp` (GQKANFWP) and `gqkan_fwp` (GQKAN-FWP) are different models and differ by a
> single underscore. `gqkanfwp` has a **Linear** slow programmer and a QKAN fast programmer;
> `gqkan_fwp` is the reverse — a **QKAN** slow programmer and a Linear fast programmer.

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
PYTHONPATH=. python train.py --model gqkan_qkanfwp --dataset bessel_j2 --epochs 100

# the full sweep: 4 models x 3 seeds on jaynes_cummings
bash run_all_dataset.sh
```

Everything here runs on CPU (`--device cpu` is what the sweep uses). `gqkan_qfwp` (GQKAN-QFWP) is
much the slowest — its variational circuit is a PennyLane state-vector simulation.

## Settings

The benchmark configuration is the one `run_all_dataset.sh` passes explicitly:

```
--epochs 100 --lr 1e-3 --batch_size 4 --window_len 64 --input_size 1 --hidden_size 8
--qnn_depth 2 --device cpu     # sweeping seeds 0, 1, 2
```

These are **not** `train.py`'s built-in argparse defaults, which are lighter (`--epochs 30
--batch_size 2 --window_len 4 --hidden_size 5 --qnn_depth 5 --seed 42`). Pass the flags above — or
use the sweep script — to reproduce the paper configuration.

### ⚠️ Six flags do not do what their names suggest

These are long-standing quirks of the model definitions and the CLI. They are documented rather than
"fixed", because changing them would change published results or rename every output directory.

| Flag | Model | Actual behaviour |
|---|---|---|
| `--hidden_size` | `gqkan_fwp` (GQKAN-FWP) | **Ignored.** `gqkan_fwp.py` overwrites the argument with `hidden_size = 4` unconditionally. |
| `--hidden_size` | `gqkanfwp` (GQKANFWP) | **Off by one.** `gqkanfwp.py` uses `hidden_size - 1`, so `--hidden_size 8` builds a width-7 encoder. |
| `--qnn_depth` | `gqkan_qkanfwp`, `gqkanfwp`, `gqkan_fwp` | **No-op.** Only `gqkan_qfwp` (GQKAN-QFWP) consumes it, as its VQC circuit depth. `gqkan_qkanfwp` accepts the value as its `vqc_depth` argument but never reads it; the other two are not passed it at all. |
| `--horizon` | all | **Ignored by the models.** Every model predicts exactly one step ahead. The value is only written to the `horizon` column of `prediction_log.csv`, so `--horizon 5` silently mislabels a 1-step model. |
| `--exp_name` | all | **Ignored for output paths.** The run directory is built from dataset/model/hyperparameters plus a timestamp; the name survives only inside `args.json`. Two runs with different `--exp_name` and identical hyperparameters are indistinguishable by path. |
| `--debug` | all | **Never read.** Accepted and recorded in `args.json`, but no code path consults it. |

### On `solver="cutn"`

Three of the models construct their QKAN layers with `solver="cutn"`. Despite the name, this does
**not** select a cuQuantum backend and needs no GPU. It selects the whole-circuit einsum contraction
path in `fast_solver.py`, which expresses the entire quantum circuit as a single tensor network and
evaluates it with `torch.einsum`.

`cuquantum` and `opt_einsum` are consulted only to choose a good *contraction order*: if either is
installed, `_get_plan` precomputes an ordering and `_execute_plan` walks it as a sequence of pairwise
`torch.einsum` calls; if neither is installed, the code falls back to a single
`torch.einsum(expression, *operands)` over the same expression. The numerical result is the same
either way — the optional packages only affect how fast it is contracted, never what is computed.

## Outputs

Each run writes a self-describing directory under `--save_dir` (relative paths only — absolute paths
and `..` are rejected), containing the resolved `args.json`, an environment snapshot, logs,
`train_log.csv` (per-epoch train/test loss), `prediction_log.csv` (every ground-truth/prediction
pair) and model checkpoints. Output directories are gitignored.

**This repository deliberately ships no plotting code and no figures.** Runs emit CSVs only; plot
them however you like. Keeping the numbers and the presentation separate means a figure can never
silently disagree with the data behind it.

## Attribution

The NARMA generator is credited to Samuel Yen-Chi Chen; `damped_shm.py` adapts a Skill-Lync tutorial;
the two quantum-dynamics generators derive from NVIDIA CUDA-Q documentation examples;
`qkan_fast.py` and `fast_solver.py` are modified copies of [QKAN](https://github.com/Jim137/qkan)
(Apache-2.0). Full details in [`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).
