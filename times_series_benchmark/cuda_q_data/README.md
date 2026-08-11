# Quantum-dynamics data

Two pre-simulated quantum-dynamics traces. These ship in the repository so the `jaynes_cummings` and
`transmon` datasets train **without** requiring NVIDIA CUDA-Q.

| File | Rows | sha256 (first 16) | System |
|---|---|---|---|
| `jaynes_cummings.csv` | 3000 | `ab8844af0a2f3758…` | Jaynes–Cummings model — a two-level atom coupled to a cavity mode, with decay |
| `transmon.csv` | 3000 | `18848ef15fb5aa9e…` | Transmon qubit dispersively coupled to a resonator, evolved to `t_max = 25 ns` |

## Licence

Both files are **generated data**, produced by `../make_cudaq_data.py` using the simulation code in
`../src/datasets/`. They are covered by this repository's Apache-2.0 licence.

The generators themselves are derived from **NVIDIA CUDA-Q dynamics documentation examples**
(Apache-2.0) — see [`../../THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md) §5.

## Regenerating

Requires an NVIDIA GPU with `cudaq` and `cupy` installed (both are commented out in the repository's
`environment.yml`; uncomment them first).

```bash
cd times_series_benchmark
PYTHONPATH=. python make_cudaq_data.py
```

To re-simulate on the fly during training instead of loading the CSV:

```bash
QFWP_REGEN_CUDAQ=1 PYTHONPATH=. python train.py --dataset jaynes_cummings ...
```

The loader (`src/datasets/cudaq_data.py`) reads the CSV by default and warns if a requested
`num_steps` disagrees with the shipped file's length. The shipped CSVs and a fresh simulation were
verified to agree exactly (max absolute difference `0.0` on both inputs and targets for both datasets).
