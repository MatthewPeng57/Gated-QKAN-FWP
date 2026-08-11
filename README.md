# Gated-QKANFWPs

Reference implementations and training pipelines for the **gated QKAN-FWP** family of
quantum-inspired Kolmogorov–Arnold Network fast-weight programmers, evaluated on three independent
tasks.

> **Paper:** Gated QKAN-FWP: Scalable Quantum-inspired Sequence Learning —
> [arXiv:2605.06734](https://arxiv.org/abs/2605.06734)

A *fast-weight programmer* (FWP) splits a model in two: a **slow programmer** reads the input sequence
and emits the parameters of a **fast programmer**, which then produces the actual output. These four
models vary which network plays each role — a QKAN, a plain linear map, or a variational quantum
circuit.

## The four models

| Model | `--model` key | Slow programmer | Fast programmer | Fast state emitted |
|---|---|---|---|---|
| **GQKAN-QKANFWP** | `gqkan_qkanfwp` | QKAN | QKAN | per-sample `theta` |
| **GQKANFWP** | `gqkanfwp` | Linear | QKAN | per-sample `theta` |
| **GQKAN-FWP** | `gqkan_fwp` | QKAN | Linear | `fast_weight`, `fast_bias` |
| **GQKAN-QFWP** | `gqkan_qfwp` | QKAN | VQC (PennyLane) | variational circuit parameters |

The two QKAN-fast models program only `theta`; the fast layer's `base_weight` stays the shared
trained parameter and is not generated per sample.

CLI keys are lowercase with underscores so they are shell-safe and unambiguous — note that the paper
names **GQKANFWP** and **GQKAN-FWP** differ only by a hyphen but are different models.

The QKAN layers use [QKAN](https://github.com/Jim137/qkan) (Jiang, [arXiv:2509.14026](https://arxiv.org/abs/2509.14026)),
extended here so `theta` can carry a leading batch axis — required because in an FWP the fast
programmer's parameters are a *per-sample input*, not a shared trainable weight. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) §1.

## The three tasks

| Directory | Task | Models | Data |
|---|---|---|---|
| [`RL_minigrid/`](RL_minigrid/) | A3C reinforcement learning on `MiniGrid-Empty-16x16-v0` | all four | procedural (MiniGrid) |
| [`Solar_cycle_forecasting/`](Solar_cycle_forecasting/) | Solar-cycle (sunspot) forecasting, 528 → 132 months | GQKAN-QKANFWP | SILSO monthly sunspot number |
| [`times_series_benchmark/`](times_series_benchmark/) | 7 forecasting tasks — 5 analytic signals + 2 quantum-dynamics traces | all four | generated at runtime / shipped CSVs |

Each directory has its own README with the exact commands, hyperparameters, and outputs.

## Install

One environment covers all three tasks.

```bash
conda env create -f environment.yml
conda activate gated-qkanfwps
```

or with pip (Python 3.11):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**GPU.** `Solar_cycle_forecasting` **requires an NVIDIA GPU.** Its slow programmer is built with QKAN
`solver="cutile"`, which needs the `cuda-tile` package and runs CUDA-only, so there is no CPU path
regardless of `--device` or `--fast_solver`. (`train.py` itself defaults to `--device cpu`, but
`run.sh` — the intended entry point — passes `--device cuda`.) `RL_minigrid` and
`times_series_benchmark` run on CPU and need no GPU at all. NVIDIA CUDA-Q is **optional** — needed only
to regenerate the quantum-dynamics CSVs, which ship in the repository.

## Quickstart

```bash
# Reinforcement learning — A3C, 4 models, MiniGrid
cd RL_minigrid && python run_gqkan_qkanfwp.py

# Solar-cycle forecasting — the paper configuration, 5 seeds
cd Solar_cycle_forecasting && bash run.sh

# Time-series benchmark — one model, one dataset
cd times_series_benchmark && PYTHONPATH=. python train.py \
    --model gqkan_qkanfwp --dataset bessel_j2 --epochs 100
```

Each task writes a self-describing output directory, but they differ in what they keep:

- `Solar_cycle_forecasting` and `times_series_benchmark` record the resolved `args.json`, an
  environment snapshot, logs, metrics CSVs and figures.
- Only `Solar_cycle_forecasting` keeps a **best-validation** checkpoint. `times_series_benchmark`
  snapshots at fixed epochs `{1, 15, 30, 50, 100}`, so a run with `--epochs 60` gets nothing after
  epoch 50.
- `RL_minigrid` writes `config.json`, a per-episode reward CSV, a reward-curve PDF and the **final**
  (not best) network — no environment snapshot.

## Solvers and how the sequence is processed

Two things vary between the three tasks and are easy to conflate: **how the fast weights are
accumulated over the sequence**, and **which QKAN solver evaluates the circuit**.

### Sequence accumulation: prefix scan vs. sequential recurrence

An FWP updates its fast weights at every timestep, `θ_t = (1−g_t)·δ_t + g_t·θ_{t−1}`. Whether that
recurrence can be parallelised depends on whether the task needs the output at every step.

| Task | Accumulation | Why |
|---|---|---|
| **Solar_cycle_forecasting** | **Parallel prefix scan.** One `einsum` builds every per-step update at once, then a `cumsum` over the gate log-decays collapses them straight to the final `θ_L`. No Python loop over the sequence. | The task is a *single-shot* 132-month forecast produced from the **final** fast weights only. Intermediate states are never read, so the recurrence collapses into a scan. |
| **times_series_benchmark** | **Sequential.** `for t in range(seq_len)`, calling the FWP cell once per step and stacking the outputs. | An output is required at **every** timestep, so the fast weights must be materialised at each step. |
| **RL_minigrid** | **Sequential.** `for t in range(T)`, same shape. | The policy and value heads are read at every timestep. |

This applies to all four models within each task, and it is the dominant cost factor:

- The prefix-scan path is parallel in time but materialises a `(B, L, O, I, R+1, 2)` tensor, so its
  **memory** grows with sequence length. That is what `--streaming_fwp on` addresses (below).
- The sequential path uses negligible extra memory but its **wall-clock grows linearly with the
  sequence length**. This is why GQKAN-QFWP is much slower at `--window_len 64` than at `16`: its
  variational circuit is evaluated once per timestep per sample.

### Solver selection

| Task | Flags | Default | What actually runs |
|---|---|---|---|
| **Solar_cycle_forecasting** | `--fast_solver {flash,cutile,cute}`<br>`--streaming_fwp {off,on}` | `flash`, `off` | Fused Triton kernels. See that folder's README for the four-way agreement table and the CuTe prerequisites. |
| **times_series_benchmark** | *(none — not exposed)* | — | GQKAN-QKANFWP, GQKANFWP and GQKAN-FWP request `solver="cutn"`: a tensor-network contraction executed by `torch.einsum`, where the contraction **path** comes from cuQuantum if installed, else `opt-einsum`, else PyTorch's default. All three give the same result — only path quality differs — so this runs anywhere, with or without cuQuantum. |
| **RL_minigrid** | *(none — not exposed)* | — | The QKAN default, `solver="exact"` on `device="cpu"` — an exact state-vector simulation. These models are small and CPU-only by design. |

GQKAN-QFWP is the exception in both tables: its **fast** programmer is a PennyLane variational circuit
(`qml.QNode` on `default.qubit`), not a QKAN, so no QKAN solver applies to it. Only its **slow**
programmer is a QKAN, using the `exact` default.

Only the solar task exposes solver switches, because it is the only one with a GPU-resident fast path
worth swapping. The other two are CPU-bound and use a single fixed solver each.

### `--streaming_fwp` (solar only)

`off` (default) is the prefix scan described above. `on` replaces it with a left-to-right recurrence
that never materialises the big delta tensor — trading the scan's parallelism for much lower memory.
The two compute the same quantity: `off` sums `Σ_t (1−g_t)·δ_t·Π_{s>t} g_s`, `on` runs the equivalent
recurrence. Measured agreement at `L=528` in fp32 is ~1e-7 relative, forward and backward.

## Repository layout

```
├── LICENSE                     Apache-2.0 (this repository's code)
├── NOTICE                      Apache-2.0 notice
├── THIRD_PARTY_NOTICES.md      full third-party inventory + data licenses
├── CITATION.cff
├── environment.yml             one environment for all three tasks
├── requirements.txt
├── RL_minigrid/
├── Solar_cycle_forecasting/
└── times_series_benchmark/
```

## Data and licensing

The **source code** is licensed [Apache-2.0](LICENSE). **Data files are not.** In particular,
`Solar_cycle_forecasting/data/Sunspots.csv` is SILSO data (WDC-SILSO, Royal Observatory of Belgium)
distributed under **CC BY-NC 4.0 — NonCommercial**. Cite SILSO when you use it, and do not use it for
commercial purposes.

Files derived from [QKAN](https://github.com/Jim137/qkan) retain their upstream Apache-2.0 headers and
carry a statement of the modifications, per Apache-2.0 §4(b). The full inventory — including MorvanZhou's
A3C scaffolding (MIT), Farama MiniGrid (MIT), PennyLane, NVIDIA CUDA-Q examples, and the dataset
generators — is in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Citation

See [`CITATION.cff`](CITATION.cff).

## Acknowledgements

Built on [QKAN](https://github.com/Jim137/qkan) by Jiun-Cheng Jiang, the A3C implementation by
[MorvanZhou](https://github.com/MorvanZhou/pytorch-A3C), [PennyLane](https://pennylane.ai),
[MiniGrid](https://github.com/Farama-Foundation/Minigrid), and [NVIDIA CUDA-Q](https://github.com/NVIDIA/cuda-quantum).
Sunspot data courtesy of [WDC-SILSO](https://www.sidc.be/SILSO/), Royal Observatory of Belgium, Brussels.
