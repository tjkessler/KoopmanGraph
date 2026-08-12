# Example scripts

Runnable Python entry points that complement the notebook suite under
`examples/`. These scripts are **not** collected by `nbmake` CI.

## Distributed DDP fit (`torchrun`)

Train a tiny `GraphKoopmanModel` with native PyTorch DDP via
`fit(..., strategy="ddp")`. The demo uses windowed training so a single
trajectory can be sharded across ranks:

```bash
torchrun --standalone --nproc_per_node=2 examples/scripts/ddp_fit_torchrun.py
```

If local process-group init stalls, try `MASTER_ADDR=127.0.0.1` before
`torchrun`.

Single-process smoke (DDP wrapping skipped when no process group is active):

```bash
python examples/scripts/ddp_fit_torchrun.py --epochs 2
```

Optional rank-0 checkpoint:

```bash
torchrun --standalone --nproc_per_node=2 \
  examples/scripts/ddp_fit_torchrun.py --checkpoint /tmp/ddp_demo.pt
```

Requires a normal KoopmanGraph install (core dependencies only; no Lightning
or Ray). See `python examples/scripts/ddp_fit_torchrun.py --help` for flags.

## Ray Tune HPO (examples-only)

Tiny learning-rate search with Ray Tune. The script uses
`koopman_graph.tuning` helpers (`fit_history_metrics`, `run_ray_tune`, and an
*example-only* `example_lr_loguniform_space` scaffold). The **search
configuration still lives in the script** — those scaffolds are smoke ranges,
not scientific defaults, and KoopmanGraph is not an AutoML product. Requires
the `[ray]` extra:

```bash
pip install "koopman-graph[ray]"
python examples/scripts/ray_tune_koopman_example.py --epochs 1 --num-samples 2
```

Optional Tune storage directory:

```bash
python examples/scripts/ray_tune_koopman_example.py \
  --epochs 1 --num-samples 2 --storage-path /tmp/koopman_tune
```

See `python examples/scripts/ray_tune_koopman_example.py --help`. Use native
DDP / Fabric for multi-GPU *model* training; use Ray for trial / ensemble
member parallelism.
