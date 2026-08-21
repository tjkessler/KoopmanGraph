# CLI examples

Config-driven entry points for the `koopman-graph` console script. These
configs are small enough for local smoke runs and for the package CLI test
suite. They are **not** collected by `nbmake` CI.

Install a normal editable or wheel build so the console script is on your
`PATH`. YAML configs additionally need the optional `[cli]` extra
(`pip install "koopman-graph[cli]"`).

## Synthetic path train (`synthetic_train.json`)

Tiny GCN train on a seeded path-graph decay trajectory (`data.kind =
synthetic_path`), three CPU epochs, writing a safetensors checkpoint
(`.kgckpt`):

```bash
koopman-graph train \
  --config examples/cli/synthetic_train.json \
  --out /tmp/kg-cli
```

Relative `checkpoint.path` values resolve under `--out`, so this writes
`/tmp/kg-cli/model.kgckpt`.

Forecast from the first snapshot of the same synthetic sequence:

```bash
koopman-graph predict \
  --checkpoint /tmp/kg-cli/model.kgckpt \
  --data examples/cli/synthetic_train.json \
  --steps 5 \
  --out /tmp/kg-cli/forecast.pt
```

`--data` may be a train/data JSON (or YAML) file, or a trusted `.pt`
`GraphSnapshotSequence` cache. The forecast file is a `torch.save` payload
with `steps`, `forecasts`, and `summary`.

See `koopman-graph train --help`, `koopman-graph predict --help`, and
`koopman-graph benchmark --help` for flags and exit-code conventions.
Identity-bound `benchmark run` / `verify` hash a frozen manifest; they
do not train a model. The Jupyter walkthrough is
[`examples/47_benchmark_manifest.ipynb`](../47_benchmark_manifest.ipynb)
(collected by `nbmake`; YAML still needs `[cli]`).
