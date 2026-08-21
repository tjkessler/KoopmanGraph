# Smoke multiphysics stand-in

* **Scope:** Protocol-locked networked diffusion stand-in (path-graph
  Laplacian step) for CI `benchmark verify`
* **Size:** 148-byte UTF-8 payload (not a PDE field)
* **Format:** Tracked text file `benchmarks/v0.15/data/smoke_multiphysics.txt`
* **Source:** Generated in-tree for KoopmanGraph smoke CI
* **License:** Apache-2.0 (same as the package)
* **Limitations:** This is **not** a measured continuum field and does
  not execute irregular \(\Delta t\). Default CI hashes protocol
  identity (`executed=False`); it does not train models or invent
  MAE / RMSE
* **Version:** `smoke-multiphysics` / `1`
