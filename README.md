# Gated-QKANFWPs

Reference implementations and training pipelines for four **quantum-inspired Kolmogorov–Arnold
Network fast-weight programmers**, evaluated on three independent tasks.

A *fast-weight programmer* (FWP) splits a model in two: a **slow programmer** reads the input sequence
and emits the parameters of a **fast programmer**, which then produces the actual output. These four
models vary which network plays each role — a QKAN, a plain linear map, or a variational quantum
circuit.

## The four models

| Key | Slow programmer | Fast programmer | Fast state emitted |
|---|---|---|---|
| `qqkanfwp` | QKAN | QKAN | per-sample `theta`, `base_weight` |
| `lqkanfwp` | Linear | QKAN | per-sample `theta`, `base_weight` |
| `qkanlfwp` | QKAN | Linear | `fast_weight`, `fast_bias` |
| `qkanvfwp` | QKAN | VQC (PennyLane) | variational circuit parameters |

The QKAN layers use [QKAN](https://github.com/Jim137/qkan) (Jiang, [arXiv:2509.14026](https://arxiv.org/abs/2509.14026)),
extended here so `theta` can carry a leading batch axis — required because in an FWP the fast
programmer's parameters are a *per-sample input*, not a shared trainable weight. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) §1.

## The three tasks

| Directory | Task | Models | Data |
|---|---|---|---|
| [`RL_minigrid/`](RL_minigrid/) | A3C reinforcement learning on `MiniGrid-Empty-16x16-v0` | all four | procedural (MiniGrid) |
| [`Solar_cycle_forecasting/`](Solar_cycle_forecasting/) | Solar-cycle (sunspot) forecasting, 528 → 132 months | `qqkanfwp` | SILSO monthly sunspot number |
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
cd RL_minigrid && python run_qqkanfwp.py

# Solar-cycle forecasting — the paper configuration, 5 seeds
cd Solar_cycle_forecasting && bash run.sh

# Time-series benchmark — one model, one dataset
cd times_series_benchmark && PYTHONPATH=. python train.py \
    --model qqkanfwp --dataset bessel_j2 --epochs 100
```

Every run writes a self-describing output directory containing the resolved `args.json`, an
environment snapshot, logs, metrics CSVs, figures, and the best checkpoint.

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
