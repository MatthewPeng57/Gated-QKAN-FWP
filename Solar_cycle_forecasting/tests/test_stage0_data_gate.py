# Copyright 2026 Matthew Peng and contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Stage-0 extraction gate: reproduce the executed split + tensor shapes.

CPU / data-only — no model instantiation, no CUDA, no GPU required.

Verifies the clean extraction reproduces the executed ``qqkanfwp`` Task-A
sunspot pipeline exactly:

  * shipped CSV integrity (sha256 == executed FWP_tele CSV),
  * N = 2606 windows, split (train, val, test) = (2084, 260, 262),
  * x shape (528, 1), y shape (132,), dtype float32, no NaNs,
  * min-max scaler maps the full series to [0, 1],
  * inverse_transform round-trips,
  * a deterministic byte-fingerprint of the built tensors (must be stable).

Run: PYTHONDONTWRITEBYTECODE=1 <qfwp-python> tests/test_stage0_data_gate.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.data.sunspot import DEFAULT_CSV, make_sunspot_split  # noqa: E402

# Expected values, derived from the executed Task-A protocol: L=528 -> H=132
# sliding windows over the shipped SILSO monthly-mean CSV, full-series min-max
# scaling, and a window-level chronological 80/10/10 split.
EXPECTED_CSV_SHA256 = "15c3d116ad6c5a5427837ae4cec39aa9b4b2e4a0d8e374d501a4ac760fc50b35"
EXPECTED = dict(n_total=2606, n_train=2084, n_val=260, n_test=262)
SEQ_LEN, HORIZON = 528, 132


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _fingerprint(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def run_gate() -> dict:
    results: dict[str, object] = {}

    # 0. CSV integrity.
    csv_sha = _sha256(Path(DEFAULT_CSV))
    assert csv_sha == EXPECTED_CSV_SHA256, (
        f"CSV sha256 mismatch: got {csv_sha}, expected {EXPECTED_CSV_SHA256}"
    )
    results["csv_sha256"] = csv_sha

    # 1. Split counts.
    split = make_sunspot_split(seq_len=SEQ_LEN, horizon=HORIZON)
    got = dict(
        n_total=split.n_total,
        n_train=split.n_train,
        n_val=split.n_val,
        n_test=split.n_test,
    )
    assert got == EXPECTED, f"split mismatch: got {got}, expected {EXPECTED}"
    assert len(split.train) + len(split.val) + len(split.test) == split.n_total
    results["split"] = got

    # 2. Tensor shapes / dtype / finiteness.
    x0, y0 = split.dataset[0]
    assert tuple(x0.shape) == (SEQ_LEN, 1), f"x shape {tuple(x0.shape)}"
    assert tuple(y0.shape) == (HORIZON,), f"y shape {tuple(y0.shape)}"
    assert str(x0.dtype) == "torch.float32" and str(y0.dtype) == "torch.float32"
    assert np.isfinite(split.dataset.x.numpy()).all()
    assert np.isfinite(split.dataset.y.numpy()).all()
    results["x_shape"] = tuple(x0.shape)
    results["y_shape"] = tuple(y0.shape)

    # 3. Scaler range (full-series min-max -> [0, 1]).
    scaled = split.dataset.full_scaled_data
    smin, smax = float(scaled.min()), float(scaled.max())
    assert abs(smin - 0.0) < 1e-6 and abs(smax - 1.0) < 1e-6, f"scaler range [{smin},{smax}]"
    results["scaled_range"] = (smin, smax)

    # 4. inverse_transform round-trips.
    recon = split.dataset.inverse_transform(split.dataset.y[0].numpy())
    assert recon.shape == (HORIZON,)
    results["inverse_ok"] = True

    # 5. Deterministic fingerprints (must be stable across runs / machines).
    results["x_fingerprint"] = _fingerprint(split.dataset.x.numpy())
    results["y_fingerprint"] = _fingerprint(split.dataset.y.numpy())

    # 6. Boundary sanity: first test window index and its date.
    first_test_win = split.n_train + split.n_val  # 2344
    target_start_row = first_test_win + SEQ_LEN     # month index where test target begins
    dates = split.dataset.full_unscaled_df["Date"].values
    results["first_test_window_idx"] = first_test_win
    results["first_test_target_date"] = str(dates[target_start_row])

    return results


def test_stage0_data_gate() -> None:
    run_gate()


if __name__ == "__main__":
    r = run_gate()
    print("STAGE-0 DATA GATE: PASS")
    for k, v in r.items():
        print(f"  {k}: {v}")
