# Solar_cycle_forecasting — sunspot forecasting with GQKAN-QKANFWP

Long-horizon forecasting of the SILSO monthly mean total sunspot number: given **528 months** of
history (≈ 4.5 solar cycles), predict the **next 132 months** (11 years, ≈ one full cycle) in a single
shot.

The model is **GQKAN-QKANFWP** (CLI key `gqkan_qkanfwp`): a QKAN slow programmer reads the input window
and emits the parameters of a QKAN fast programmer, which produces the forecast. **12,474 trainable
parameters.**

> Gated QKAN-FWP: Scalable Quantum-inspired Sequence Learning — [arXiv:2605.06734](https://arxiv.org/abs/2605.06734)

## Run

```bash
cd Solar_cycle_forecasting
bash run.sh                 # 5 seeds (0-4), 100 epochs, 3-way parallel
EPOCHS=1 bash run.sh        # quick smoke
```

or a single run directly:

```bash
PYTHONPATH=. python train.py \
    --epochs 100 --lr 2.5e-3 --lr_schedule keras_decay --loss peak_aware_mse \
    --model gqkan_qkanfwp --dataset sunspots --exp_name my_run --save_dir results \
    --window_len 528 --horizon 132 --input_size 1 --output_size 132 \
    --hidden_size 48 --batch_size 32 --seed 0 --device cuda --alpha 1.0 --qnn_depth 2 \
    --in_resize 10 --out_resize 20 --qkan_s_dim_1 25 --qkan_s_dim_2 25 \
    --fast_in 16 --fast_out 16
```

Check the data pipeline without training (CPU, seconds):

```bash
PYTHONPATH=. python tests/test_stage0_data_gate.py    # -> STAGE-0 DATA GATE: PASS
```

## Protocol

| | |
|---|---|
| Input window `L` | 528 months |
| Forecast horizon `H` | 132 months |
| Windows | stride 1 → **N = 2606** |
| Split | chronological, window-level **80 / 10 / 10** → 2084 / 260 / 262 |
| Scaling | `MinMaxScaler` to [0, 1] — see the note below |
| Loss | peak-aware MSE, `(ŷ−y)²·(1+α·y)` with α = 1.0 |
| Optimiser | Adam, lr 2.5e-3, per-step decay `1/(1+1e-6·step)` |
| Batch / epochs | 32 / 100, no early stopping |
| Checkpoint | best **validation** loss |
| Seeds | 0–4 |
| Metrics | scaled MSE / MAE / R², plus denormalised peak-amplitude and peak-timing error |

> **Normalisation note.** The `MinMaxScaler` is fit on the **full series**, not on the training split
> alone. This is a deliberate, frozen choice: it reproduces the protocol under which the released
> results were produced, so new models compare against them on identical inputs. The practical effect
> is small here — the global min and max both fall inside the training region — but it is a form of
> normalisation leakage and is documented rather than hidden. If you are building a new benchmark
> rather than comparing against these numbers, fit the scaler on the training split.

## Solver options

| Flag | Default | Meaning |
|---|---|---|
| `--fast_solver {flash,cutile,cute}` | `flash` | Backend for the fast QKAN layer. `flash` uses fused Triton kernels (GPU); `cutile` is the pure-PyTorch scalar recurrence and runs on CPU; `cute` is an opt-in JIT-compiled CUDA/CuTe kernel (see below). |
| `--streaming_fwp {off,on}` | `off` | `on` replaces the materialise-and-cumsum block with a left-to-right prefix-scan recurrence that never materialises the `(B, L, O, I, R+1, 2)` delta tensor — same result, less memory. |

The two `--streaming_fwp` paths compute the same quantity: `off` sums
`Σ_t (1−g_t)·δ_t·Π_{s>t} g_s`, `on` runs the equivalent recurrence
`θ_t = (1−g_t)·δ_t + g_t·θ_{t−1}`. Measured agreement at `L=528` in fp32 is ~1e-7 relative on both the
forward pass and the gradients — pure floating-point accumulation drift.

To run without a GPU: `--device cpu --fast_solver cutile`.

### The CuTe backend (opt-in)

`--fast_solver cute` uses a hand-written CUDA kernel (`src/models/utils/csrc/cute_batched_kernels.cu`,
built on NVIDIA CuTe) for the pz readout with per-sample `theta`. It is **opt-in**: `flash` remains the
default and needs none of the machinery below.

Prerequisites, in the shell that first runs it:

| Requirement | Notes |
|---|---|
| NVIDIA GPU | the kernel is CUDA-only; there is no CPU fallback |
| CUDA toolchain | `nvcc` on `PATH`, or `CUDA_HOME` set |
| `ninja` | `pip install ninja` |
| CUTLASS checkout | `git clone --depth 1 https://github.com/NVIDIA/cutlass` then `export CUTLASS_PATH="$PWD/cutlass"` (the loader appends `include/`) |
| *(optional)* `TORCH_EXTENSIONS_DIR` | where to cache the build |

The first run JIT-compiles the kernel — roughly **60 seconds** — and caches it; later runs load it
almost instantly. The kernel implements **only the pz ansatz**; anything else raises rather than
silently falling back, so you can never end up on a different numerical path without being told. If the
toolchain is missing, importing the package still works and `flash`/`cutile` still train — only
`--fast_solver cute` fails, with a message telling you what to install.

### All four paths agree

One epoch at the protocol above (seed 0), identical except for the solver flag. All four report the
same **12,474** trainable parameters.

| Run | Flags | Train loss | Val loss |
|---|---|---|---|
| default | `--fast_solver flash --streaming_fwp off` | 0.236043 ± 1e-6 | 0.075677 ± 2e-5 |
| pure PyTorch | `--fast_solver cutile` | 0.236050 ± 1e-6 | 0.075594 ± 5e-5 |
| streaming | `--streaming_fwp on` | 0.236045 | 0.075655 |
| CuTe | `--fast_solver cute` | 0.236051 ± 2e-6 | 0.075581 ± 3e-5 |

The four agree to **~3e-5 relative** on the training loss — fp32 accumulation-order drift between a
fused Triton kernel, a pure-PyTorch scalar recurrence, a prefix-scan recurrence and a CuTe kernel, not a
behavioural difference. Validation is the looser figure because it is measured *after* an epoch of
training has folded the per-batch drift into the weights.

Measured directly at the solver boundary (`B=8, in_dim=10, out_dim=20, reps=2`, CUDA fp32, `ansatz="pz"`,
`fast_measure=True`) — maximum absolute deviation on the forward pass and on `∂L/∂θ`:

| Pair | Forward | Backward (`∂θ`) |
|---|---|---|
| flash vs cutile | 3.0e-07 | 4.8e-07 |
| flash vs CuTe | 1.5e-06 | 3.1e-06 |
| cutile vs CuTe | 1.6e-06 | 3.2e-06 |

> **These runs are not bitwise reproducible.** Repeating the *same* configuration moves the training
> loss by ~1.4e-6 absolute (the Triton kernels are non-deterministic). Quote and compare these numbers
> with a tolerance, never by exact string match: `cutile` and `cute` in particular sit almost exactly on
> the 4-decimal rounding boundary (~0.236050), so their *displayed* value flips between `0.2360` and
> `0.2361` between runs while the underlying value is unchanged.

## Outputs

Each run creates a self-describing directory under
`results/DATASET_sunspots/MODEL_gqkan_qkanfwp/HIDDEN_SIZE_48/.../SEED_<n>/RUN_<timestamp>/` containing the
resolved `args.json`, an environment snapshot (`python_info.txt`, `requirements.txt`,
`environment.yaml`), `console_log.txt`, `train_log.csv`, `prediction_log.csv`,
`gqkan_qkanfwp_final_metrics_summary.csv`, `best_model.pth`, and the figure set (loss curves, multistep
forecast snapshots, sunspot-cycle reconstruction, and the `fig13_data_*.csv` behind them).

`results/` and `logs/` are gitignored.

## Data

`data/Sunspots.csv` — SILSO monthly mean total sunspot number, 3265 rows, 1749-01-31 … 2021-01-31.
**CC BY-NC 4.0 (NonCommercial)** — not covered by this repository's Apache-2.0 code licence. See
[`data/README.md`](data/README.md).
