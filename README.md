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
| **GQKAN-QKANFWP** | `gqkan_qkanfwp` | QKAN | QKAN | per-sample `theta`, `base_weight` |
| **GQKANFWP** | `gqkanfwp` | Linear | QKAN | per-sample `theta`, `base_weight` |
| **GQKAN-FWP** | `gqkan_fwp` | QKAN | Linear | `fast_weight`, `fast_bias` |
| **GQKAN-QFWP** | `gqkan_qfwp` | QKAN | VQC (PennyLane) | variational circuit parameters |

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

**GPU.** `Solar_cycle_forecasting` defaults to `--device cuda` with the Triton "flash" solver; pass
`--device cpu --fast_solver cutile` to run without one. `RL_minigrid` and `times_series_benchmark` run
on CPU. NVIDIA CUDA-Q is **optional** — it is needed only to regenerate the quantum-dynamics CSVs, which
ship in the repository.

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

Every run writes a self-describing output directory containing the resolved `args.json`, an
environment snapshot, logs, metrics CSVs, figures, and the best checkpoint.

## Does it actually learn? (pipeline sanity check)

These are **not** the paper's results — they are single-seed checks that each pipeline trains, so you
can tell quickly whether your environment is working. Measured on an RTX 5090 / 32-core CPU.

| Task | Model | Budget | Metric | Start → end |
|---|---|---|---|---|
| Solar | GQKAN-QKANFWP | 100 epochs | train loss | 0.2360 → **0.0205** |
| | | | val loss | 0.0757 → **0.0233** |
| times_series (`bessel_j2`) | GQKAN-QKANFWP | 100 epochs | train loss | 0.0458 → **5.3e-06** |
| | GQKANFWP | 100 epochs | train loss | 0.2763 → **0.0059** |
| | GQKAN-FWP | 100 epochs | train loss | 0.0779 → **2.7e-05** |
| | GQKAN-QFWP | 11 epochs¹ | train loss | 0.0322 → **0.0082** |
| RL (`MiniGrid-Empty-16x16`) | GQKAN-QKANFWP | 600 episodes | mean reward | 0.065 → **0.599** |
| | | | goal-reached rate | 16% → **86%** |

¹ at `--window_len 16` to keep the check short; all other rows use the defaults.

**GQKAN-QFWP is the slowest model** — its variational circuit is a PennyLane `default.qubit`
state-vector simulation evaluated one sample at a time. Measured on 32 CPU cores: **~36 s/epoch** at
`--window_len 64` (so ~1 hour for the 100-epoch default), and ~28 s/epoch at `--window_len 16`. Cost
scales with the sequence length, since the circuit is evaluated once per timestep per sample.

**Run these tasks one at a time.** They are CPU-bound and heavily threaded (GQKAN-QFWP alone uses
~1300% CPU). Running several concurrently oversubscribes the machine badly — we measured load average
87 on 32 cores doing exactly that, which inflated the apparent per-epoch cost by more than an order of
magnitude.

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
