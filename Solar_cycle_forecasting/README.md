# Solar_cycle_forecasting — sunspot forecasting with GQKAN-QKANFWP

Long-horizon forecasting of the SILSO monthly mean total sunspot number: given **528 months** of
history (≈ 4.5 solar cycles), predict the **next 132 months** (11 years, ≈ one full cycle) in a single
shot.

The model is **`qqkanfwp`** (GQKAN-QKANFWP): a QKAN slow programmer reads the input window and emits
the parameters of a QKAN fast programmer, which produces the forecast. **12,474 trainable parameters.**

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
    --model qqkanfwp --dataset sunspots --exp_name my_run --save_dir results \
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
| `--fast_solver {flash,cutile}` | `flash` | Backend for the fast QKAN layer. `flash` uses fused Triton kernels (GPU); `cutile` is the pure-PyTorch scalar recurrence and runs on CPU. |
| `--streaming_fwp {off,on}` | `off` | `on` replaces the materialise-and-cumsum block with a left-to-right prefix-scan recurrence that never materialises the `(B, L, O, I, R+1, 2)` delta tensor — same result, less memory. |

The two `--streaming_fwp` paths compute the same quantity: `off` sums
`Σ_t (1−g_t)·δ_t·Π_{s>t} g_s`, `on` runs the equivalent recurrence
`θ_t = (1−g_t)·δ_t + g_t·θ_{t−1}`. Measured agreement at `L=528` in fp32 is ~1e-7 relative on both the
forward pass and the gradients — pure floating-point accumulation drift.

To run without a GPU: `--device cpu --fast_solver cutile`.

## Outputs

Each run creates a self-describing directory under
`results/DATASET_sunspots/MODEL_qqkanfwp/HIDDEN_SIZE_48/.../SEED_<n>/RUN_<timestamp>/` containing the
resolved `args.json`, an environment snapshot (`python_info.txt`, `requirements.txt`,
`environment.yaml`), `console_log.txt`, `train_log.csv`, `prediction_log.csv`,
`qqkanfwp_final_metrics_summary.csv`, `best_model.pth`, and the figure set (loss curves, multistep
forecast snapshots, sunspot-cycle reconstruction, and the `fig13_data_*.csv` behind them).

`results/` and `logs/` are gitignored.

## Data

`data/Sunspots.csv` — SILSO monthly mean total sunspot number, 3265 rows, 1749-01-31 … 2021-01-31.
**CC BY-NC 4.0 (NonCommercial)** — not covered by this repository's Apache-2.0 code licence. See
[`data/README.md`](data/README.md).
