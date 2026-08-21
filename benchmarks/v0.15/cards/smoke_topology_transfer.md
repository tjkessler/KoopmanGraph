# Smoke topology-transfer stand-in

* **Scope:** Protocol-locked unseen-\(N\) / rewiring stand-in with
  mandatory `hold_last`, `pernode`, and `joint_ls` controls
* **Size:** 177-byte UTF-8 payload (not a multi-graph corpus)
* **Format:** Tracked text file
  `benchmarks/v0.15/data/smoke_topology_transfer.txt`
* **Source:** Generated in-tree for KoopmanGraph smoke CI
* **License:** Apache-2.0 (same as the package)
* **Limitations:** This is **not** a transfer experiment result.
  Negative transfer remains allowed when a later runner executes the
  protocol. Default CI hashes protocol identity (`executed=False`)
* **Version:** `smoke-topology-transfer` / `1`
