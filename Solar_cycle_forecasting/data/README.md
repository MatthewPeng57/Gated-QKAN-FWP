# Data — SILSO sunspot numbers

## ⚠️ Licence: CC BY-NC 4.0 — NonCommercial

This file is **not** covered by the Apache-2.0 licence that applies to this repository's source code.

| | |
|---|---|
| File | `Sunspots.csv` |
| Source | **WDC-SILSO, Royal Observatory of Belgium, Brussels** — <https://www.sidc.be/SILSO/> |
| Licence | [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) |
| Contents | Monthly mean total sunspot number |
| Rows | 3265 data rows (plus header) |
| Range | 1749-01-31 … 2021-01-31 |
| Columns | *(index)*, `Date`, `Monthly Mean Total Sunspot Number` |
| sha256 | `15c3d116ad6c5a5427837ae4cec39aa9b4b2e4a0d8e374d501a4ac760fc50b35` |

Verify with:

```bash
sha256sum Sunspots.csv
# or, from the Solar_cycle_forecasting/ directory, run the data gate — it checks
# the hash, the split sizes and the tensor fingerprints:
python tests/test_stage0_data_gate.py
```

## Terms of use

Using this dataset means you must:

- **Attribute** — credit WDC-SILSO, Royal Observatory of Belgium, Brussels.
- **Not use it commercially** — the NonCommercial clause binds you as a downstream user, independently
  of this repository's code licence.

Please cite SILSO in any publication that uses this series. The Royal Observatory of Belgium requests
acknowledgement in the form:

> Source: WDC-SILSO, Royal Observatory of Belgium, Brussels

The authoritative, continuously updated series is available from
<https://www.sidc.be/SILSO/datafiles>. The copy here is frozen at the version used to produce this
repository's results, so that they remain reproducible.
