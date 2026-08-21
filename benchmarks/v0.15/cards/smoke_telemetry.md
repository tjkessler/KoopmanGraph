# Smoke telemetry stand-in

* **Scope:** Protocol-locked METR-LA / PEMS-style split (12-step history,
  0.7 / 0.1 / 0.2, z-score) for CI `benchmark verify`
* **Size:** 149-byte UTF-8 payload (not a traffic tensor)
* **Format:** Tracked text file `benchmarks/v0.15/data/smoke_telemetry.txt`
* **Source:** Generated in-tree for KoopmanGraph smoke CI
* **License:** Apache-2.0 (same as the package)
* **Limitations:** This is **not** METR-LA HDF5, not a LibCity download,
  and not a forecast result. Default CI hashes protocol identity
  (`executed=False`); it does not train models or invent MAE / RMSE
* **Version:** `smoke-telemetry` / `1`
