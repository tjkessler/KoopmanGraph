# v0.16 reaction-network benchmark manifests

Tracked YAML manifests, compact hashed payloads, FAIR cards, and
identity-bound `summary.json` files for three reaction-network datasets.
They follow the `benchmarks/v0.15/` conventions. The summaries record
protocol identity (`executed=False`), not trained forecast metrics.

## Layout

| Path | Role |
|------|------|
| `reaction_*.yaml` | `benchmark_manifest_v1` (PyYAML / `[cli]`) |
| `data/reaction_michaelis_menten_v1.pt` | Michaelis-Menten payload (`ReactionDataset.save`) |
| `data/reaction_van_de_vusse_dilution_v1.pt` | Isothermal CSTR payload with a dilution-rate input |
| `data/reaction_random_family_v1/` | Twelve random-network payloads and their `index.json` |
| `cards/reaction_*.md` | FAIR cards referenced by `dataset.card` |
| `summaries/reaction_*.json` | Checked-in `benchmark_summary_v1` |

| Manifest | Track | Dataset | Shifts |
|----------|-------|---------|--------|
| `reaction_michaelis_menten.yaml` | `multiphysics` | `reaction_michaelis_menten_v1` | `ic_shift`, `parameter_shift` (`k_cat`) |
| `reaction_van_de_vusse_dilution.yaml` | `multiphysics` | `reaction_van_de_vusse_dilution_v1` | `ic_shift` |
| `reaction_random_family.yaml` | `topology_transfer` | `reaction_random_family_v1` | `unseen_n`, `rewire`, `parameter_shift` |

Every manifest has three seeds, horizons `[1, 5, 20, 100]`, and the
controls `hold_last`, `pernode`, and `joint_ls`.

## Identity

`dataset.sha256` is the SHA-256 of the committed payload file. For the
random family it is the SHA-256 of `index.json`, which lists the SHA-256
of every member payload. Load a payload with
`koopman_graph.datasets.reaction.ReactionDataset.load`.

## Verify

With `pip install "koopman-graph[cli]"` (PyYAML):

```bash
koopman-graph benchmark verify \
  --manifest benchmarks/v0.16/reaction_michaelis_menten.yaml \
  --against benchmarks/v0.16/summaries/reaction_michaelis_menten.json
```

Repeat for `reaction_van_de_vusse_dilution` and `reaction_random_family`.
`run` needs `--data` pointing at the payload (`index.json` for the family).

## Regenerating payloads

```bash
python scripts/build_reaction_payloads.py --check
```

`--check` regenerates every payload and compares it with the committed
file within `rtol=1e-5`; `torch.save` bytes and the last floating-point
digits can differ across platforms, so the manifests bind the committed
bytes rather than a regeneration. Without `--check` the script rewrites the
payloads. After that, update `dataset.sha256` in the manifests and cards,
and rewrite the summaries with `run_manifest` (or `build_summary`) so
`summary_sha256` matches. `verify` does not require `package_version` to
equal the live `koopman_graph.__version__`.
