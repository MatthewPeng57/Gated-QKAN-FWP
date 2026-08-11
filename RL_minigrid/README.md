# RL_minigrid — A3C with fast-weight-programmer policies

Asynchronous Advantage Actor-Critic (A3C) on `MiniGrid-Empty-16x16-v0`, with all four
quantum-inspired KAN fast-weight-programmer policies.

The environment is procedural — nothing is downloaded and no data files ship with this task.
Observations are the 7×7×3 egocentric image, flattened to 147 dims; the action space has 7 actions.

## Models

| Runner | `MODEL_NAME` | Slow programmer | Fast programmer | Trainable params |
|---|---|---|---|---|
| `run_qqkanfwp.py` | `QQKANFWP_with_fast_bias` | QKAN | QKAN | 1,150 |
| `run_lqkanfwp.py` | `LQKANFWP` | Linear | QKAN | 1,822 |
| `run_qkanlfwp.py` | `QKANLFWP` | QKAN | Linear | 2,394 |
| `run_qkanvfwp.py` | `QKANVFWP` | QKAN | VQC (PennyLane) | 1,801 |

`qkanvfwp` is by far the slowest: its variational circuit is a PennyLane `default.qubit`
state-vector simulation evaluated one sample at a time.

## Run

```bash
cd RL_minigrid
python run_qqkanfwp.py        # or run_lqkanfwp.py / run_qkanlfwp.py / run_qkanvfwp.py
```

Run from **this directory** — the runners resolve `util/` and their sibling modules relative to the
working directory.

**CPU only.** These models never move to GPU (the QKAN layers default to `device="cpu"`,
`solver="exact"`). A3C spawns `min(cpu_count(), 80)` worker processes, so a machine with many cores
finishes proportionally faster.

## Configuration

All hyperparameters are module-level constants — edit the runner to change them.

| Constant | Value |
|---|---|
| `ENV_NAME` | `MiniGrid-Empty-16x16-v0` |
| `MAX_EP` | 10000 episodes |
| `UPDATE_GLOBAL_ITER` | 5 |
| `GAMMA` | 0.9 |
| `LR` | 1e-4 (shared Adam) |
| `RANDOM_SEEDS` | `[0, 1, 2, 3, 4]` — each runner loops over all five |

A full runner therefore executes five sequential 10,000-episode training runs.

## Outputs

```
results/<ENV_NAME>/<MODEL_NAME>/seed_<N>/
├── config.json                 resolved hyperparameters
├── seed_<N>.csv                per-episode reward
├── seed_<N>_raw_rewards.pkl    raw episode rewards
├── seed_<N>_model.pth          trained global network
└── seed_<N>_full.pdf           reward curve
```

`results/` is gitignored.

## Known asymmetry in the A3C bootstrap

The four runners do **not** treat episode truncation identically, and this is preserved deliberately
so that previously produced results remain reproducible from this code:

| Runner | value passed as `done` to `push_and_pull` |
|---|---|
| `run_qqkanfwp.py`, `run_lqkanfwp.py` | `terminated` |
| `run_qkanlfwp.py`, `run_qkanvfwp.py` | `done` (= `terminated or truncated`) |

`MiniGrid-Empty-16x16-v0` has a step limit, and early-training episodes reach it frequently. When an
episode ends by **truncation**, the first pair bootstraps the return with `V(s')` — the standard
treatment of a time limit — while the second pair sets the bootstrap value to 0, treating the step
limit as a genuine terminal state. Readers comparing the four models should be aware that this differs
across them. Harmonizing it (passing `terminated` everywhere) would change results and so has not been
done silently.

## Attribution

The A3C scaffolding — `utils.py`, `shared_adam.py`, `util/a3c_update.py`, and the `Worker` structure in
each runner — derives from [MorvanZhou/pytorch-A3C](https://github.com/MorvanZhou/pytorch-A3C) (MIT).
`MiniGridWrappers/obs_wrappers.py` adapts MiniGrid's `ImgObsWrapper` (MIT). `qkan_fast.py` is a modified
copy of [QKAN](https://github.com/Jim137/qkan) (Apache-2.0). Full details in
[`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).
