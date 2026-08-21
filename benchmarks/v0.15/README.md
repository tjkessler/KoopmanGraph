# v0.15 benchmark smoke fixtures

Tracked YAML manifests, hashed UTF-8 stand-ins, FAIR cards, and
identity-bound `summary.json` files for the three evidence tracks
(telemetry-like, multiphysics, topology transfer).

These files are **not** METR-LA / PEMS HDF5, not LibCity / BasicTS
leaderboards, and not trained forecast metrics. `koopman-graph benchmark
run` / `verify` check dataset SHA-256 and a canonical summary digest
(`executed=False`).

## Layout

| Path | Role |
|------|------|
| `smoke_*.yaml` | `benchmark_manifest_v1` (PyYAML / `[cli]`) |
| `data/smoke_*.txt` | Payload whose SHA-256 is `dataset.sha256` |
| `cards/smoke_*.md` | FAIR cards referenced by `dataset.card` |
| `summaries/smoke_*.json` | Checked-in `benchmark_summary_v1` |

Controls: telemetry / multiphysics list `pernode` and/or `joint_ls`;
topology transfer lists `hold_last`, `pernode`, and `joint_ls`. Methods
are `graph_koopman` plus named `control` roles (no teaching GNN ports).

## Verify

With `pip install "koopman-graph[cli]"` (PyYAML):

```bash
koopman-graph benchmark verify \
  --manifest benchmarks/v0.15/smoke_telemetry.yaml \
  --against benchmarks/v0.15/summaries/smoke_telemetry.json
```

Repeat for `smoke_multiphysics` and `smoke_topology_transfer`. `run`
needs `--data` pointing at the matching `data/smoke_*.txt` file.

## Regenerating summaries

After changing a manifest or payload, recompute `dataset.sha256` from
the payload bytes and rewrite `summaries/smoke_*.json` with
`run_manifest` (or `build_summary`) so `summary_sha256` matches.
`verify` does not require `package_version` to equal the live
`koopman_graph.__version__`.
