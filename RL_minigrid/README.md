# RL_minigrid — A3C with gated QKAN fast-weight-programmer policies

Asynchronous Advantage Actor-Critic (A3C) on `MiniGrid-Empty-16x16-v0`, with the four
gated quantum-inspired KAN fast-weight-programmer (FWP) policies from the paper
**“Gated QKAN-FWP: Scalable Quantum-inspired Sequence Learning”** ([arXiv:2605.06734](https://arxiv.org/abs/2605.06734)).

The environment is procedural — nothing is downloaded and no data files ship with this task.
Observations are the 7×7×3 egocentric image, flattened to 147 dims; the action space has 7 actions.

## Models

Each model pairs a *slow programmer* (reads the observation, emits weights) with a *fast* layer
that those weights reprogram at every timestep. A sigmoid gate blends the newly generated fast
weights with the ones carried over from the previous step.

| Model | Runner | Slow programmer | Fast programmer | Trainable params |
|---|---|---|---|---|
| **GQKAN-QKANFWP** | `run_gqkan_qkanfwp.py` | QKAN | QKAN | 1,150 |
| **GQKANFWP** | `run_gqkanfwp.py` | Linear | QKAN | 1,822 |
| **GQKAN-FWP** | `run_gqkan_fwp.py` | QKAN | Linear | 2,394 |
| **GQKAN-QFWP** | `run_gqkan_qfwp.py` | QKAN | VQC (PennyLane) | 1,801 |

> **Note the two similar names.** `GQKANFWP` (`gqkanfwp`, linear slow path) and `GQKAN-FWP`
> (`gqkan_fwp`, QKAN slow path) are different models and differ by one character in their
> module names.

Each model spans three files, all sharing the same identifier stem:

| Model | Runner | Policy wrapper | FWP cell |
|---|---|---|---|
| GQKAN-QKANFWP | `run_gqkan_qkanfwp.py` | `util/gqkan_qkanfwp_utils.py` | `gqkan_qkanfwp.py` |
| GQKANFWP | `run_gqkanfwp.py` | `util/gqkanfwp_utils.py` | `gqkanfwp.py` |
| GQKAN-FWP | `run_gqkan_fwp.py` | `util/gqkan_fwp_utils.py` | `gqkan_fwp.py` |
| GQKAN-QFWP | `run_gqkan_qfwp.py` | `util/gqkan_qfwp_utils.py` | `gqkan_qfwp.py` |

**GQKAN-QFWP is by far the slowest**: its variational circuit is a PennyLane `default.qubit`
state-vector simulation evaluated one sample at a time.

## Run

```bash
cd RL_minigrid
python run_gqkan_qkanfwp.py     # or run_gqkanfwp.py / run_gqkan_fwp.py / run_gqkan_qfwp.py
```

The commands above assume this directory, but running by path from anywhere works too
(`python path/to/RL_minigrid/run_gqkanfwp.py`): Python puts the *script's* directory on `sys.path`, so
`util/` and the sibling modules resolve regardless of the working directory. Note that `results/` is
written next to the runner, not in the current directory.

**CPU only.** These models never move to GPU (the QKAN layers default to `device="cpu"`,
`solver="exact"`). A3C spawns `min(cpu_count(), 80)` worker processes, so a machine with many cores
finishes proportionally faster.

### Short runs

`MAX_EP` and `RANDOM_SEEDS` can be overridden from the environment, so a quick check needs no
source edit. The defaults are the full training configuration.

| Variable | Default | Effect |
|---|---|---|
| `QFWP_MAX_EP` | `10000` | Total episodes per seed |
| `QFWP_SEEDS` | `0,1,2,3,4` | Comma-separated seeds to loop over |

```bash
# three episodes, one seed — finishes in well under a minute
QFWP_MAX_EP=3 QFWP_SEEDS=0 python run_gqkan_fwp.py
```

The overridden values are recorded in the run's `config.json`, so a short run is never mistaken
for a full one.

## Configuration

The remaining hyperparameters are module-level constants — edit the runner to change them.

| Constant | Value |
|---|---|
| `ENV_NAME` | `MiniGrid-Empty-16x16-v0` |
| `MODEL_NAME` | paper name, e.g. `GQKAN-QKANFWP` (used as the results directory) |
| `UPDATE_GLOBAL_ITER` | 5 |
| `GAMMA` | 0.9 |
| `LR` | 1e-4 (shared Adam) |
| `N_S`, `N_A` | read from the environment (147 and 7) |

With the defaults, a full runner executes five sequential 10,000-episode training runs.

## Outputs

```
results/<ENV_NAME>/<MODEL_NAME>/seed_<N>/
├── config.json                 resolved hyperparameters (reflects any env overrides)
├── seed_<N>.csv                episode, reward, avg100, std100
├── seed_<N>_raw_rewards.pkl    raw episode rewards
└── seed_<N>_model.pth          trained global network
```

`results/` is gitignored.

**This repository deliberately ships no figures and no plotting code.** Runs emit machine-readable
artifacts only: `seed_<N>.csv` carries the per-episode reward alongside a 100-episode rolling mean and
standard deviation (computed in `reward_stats.py`), and `seed_<N>_raw_rewards.pkl` carries the raw
rewards. Plot them however you like.

## Known asymmetry in the A3C bootstrap

The four runners do **not** treat episode truncation identically, and this is preserved deliberately
so that previously produced results remain reproducible from this code:

| Runner | value passed as `done` to `push_and_pull` |
|---|---|
| `run_gqkan_qkanfwp.py`, `run_gqkanfwp.py` | `terminated` |
| `run_gqkan_fwp.py`, `run_gqkan_qfwp.py` | `done` (= `terminated or truncated`) |

`MiniGrid-Empty-16x16-v0` has a step limit, and early-training episodes reach it frequently. When an
episode ends by **truncation**, the first pair bootstraps the return with `V(s')` — the standard
treatment of a time limit — while the second pair sets the bootstrap value to 0, treating the step
limit as a genuine terminal state. Readers comparing the four models should be aware that this differs
across them. Harmonizing it (passing `terminated` everywhere) would change results and so has not been
done silently. Each call site carries a comment saying the same thing.

## Observation space

`MiniGridWrappers.ImgObsFlatWrapper` returns the 7×7×3 image flattened to `(147,)` and declares an
`observation_space` of matching shape, so the runners read `N_S = env.observation_space.shape[0]`
rather than hardcoding 147.

## Attribution

The A3C scaffolding — `utils.py`, `shared_adam.py`, `util/a3c_update.py`, and the `Worker` structure in
each runner — derives from [MorvanZhou/pytorch-A3C](https://github.com/MorvanZhou/pytorch-A3C) (MIT).
`MiniGridWrappers/obs_wrappers.py` adapts MiniGrid's `ImgObsWrapper` (MIT). `qkan_fast.py` is a modified
copy of [QKAN](https://github.com/Jim137/qkan) (Apache-2.0, © 2024 Jiun-Cheng Jiang); it carries an
Apache-2.0 §4(b) notice describing the modification. The upstream `qkan` package is also a runtime
dependency — `qkan_fast.py` imports `qkan.info` and `qkan.solver`, and three of the four FWP cells
import `qkan.QKAN` directly. Full details in
[`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

Our own code is © 2026 Kuo-Chung Peng and Samuel Yen-Chi Chen, Apache-2.0.
