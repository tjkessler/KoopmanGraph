<div align="center">

<a href="https://koopmangraph.readthedocs.io/">
  <img src="https://raw.githubusercontent.com/tjkessler/KoopmanGraph/main/docs/source/_static/koopmangraph_logo.png" alt="KoopmanGraph logo" width="420"/>
</a>

# KoopmanGraph

**Graph Neural Networks with Koopman operator theory for spatiotemporal graph dynamics**

[![Tests](https://github.com/tjkessler/KoopmanGraph/actions/workflows/ci.yml/badge.svg)](https://github.com/tjkessler/KoopmanGraph/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21326273.svg)](https://doi.org/10.5281/zenodo.21326273)
[![PyPI version](https://img.shields.io/pypi/v/koopman-graph)](https://pypi.org/project/koopman-graph/)
[![codecov](https://codecov.io/gh/tjkessler/KoopmanGraph/graph/badge.svg)](https://codecov.io/gh/tjkessler/KoopmanGraph)
[![Documentation Status](https://readthedocs.org/projects/koopmangraph/badge/?version=latest)](https://koopmangraph.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyTorch_Geometric-3C2179?logo=pytorch&logoColor=white)](https://pytorch-geometric.readthedocs.io/)

**[Documentation](https://koopmangraph.readthedocs.io/)** |
**[Quickstart](#quickstart)** |
**[Examples](https://github.com/tjkessler/KoopmanGraph/tree/main/examples)** |
**[Contributing](https://github.com/tjkessler/KoopmanGraph/blob/main/CONTRIBUTING.md)**

</div>

---

**KoopmanGraph** is an open-source PyTorch library that combines **Graph Neural Networks (GNNs)** with **Koopman operator theory** to model spatiotemporal dynamics on graphs. Instead of treating node states as flat vectors, KoopmanGraph lifts features into a latent space with topology-aware encoders, advances them via a learned linear Koopman operator, and decodes predictions back to physical node features.

The result is a **topology-aware alternative to vector-based Koopman methods** — well suited for smart grids, traffic networks, epidemic modeling, and other networked dynamical systems.

## Why KoopmanGraph?

Koopman theory encodes nonlinear dynamics into a linear domain where evolution is simple matrix multiplication and spectral analysis reveals system behavior. Existing deep Koopman packages often ignore graph structure, while GNN forecasting methods typically lack explicit linear latent dynamics.

KoopmanGraph bridges that gap:

- **Topology-aware lifting** — GCN and GAT encoders propagate information along edges before Koopman evolution.
- **Explicit linear dynamics** — A learnable finite-dimensional Koopman matrix **K** governs latent evolution.
- **Multi-step forecasting** — Roll out future graph snapshots from a single initial state.
- **Spectral interpretability** — Eigendecomposition of the learned operator with continuous-time growth rates and spatial mode shapes.
- **Built on PyTorch Geometric** — Native `Data` objects, standard GNN layers, and familiar training APIs.



## Key Features


| Feature | Description |
| --- | --- |
| **GraphKoopmanModel** | End-to-end encode → Koopman advance → decode pipeline with `fit`, `predict`, `evaluate`, and preferred `encode` (deprecated `encode_latent` alias) |
| **GNNEncoder / GATEncoder** | Topology-aware latent lifting with GCN or multi-head attention |
| **GNNDecoder / GATDecoder** | Symmetric GCN or GAT reconstruction paired with the matching encoder |
| **KoopmanOperator** | Learnable linear propagator; soft modes (`dense`, `odo` + eigenloss) or structural guarantees (`schur`, `dissipative`, `lyapunov`) |
| **Spectral analysis** | Root: `KoopmanSpectrum`, `compute_spectrum`. Mode decoding and continuous helpers via `koopman_graph.analysis` |
| **Dynamical similarity** | `spectrum_distance`, `koopman_std`, `dynamical_similarity`, `detect_anomaly`, and `calibrate_anomaly_threshold` via `koopman_graph.analysis` |
| **Model persistence** | `save` / `load` checkpoints with architecture config; optional best-epoch restoration in `fit` |
| **Evaluation metrics** | Temporal train/val/test splits and per-horizon MAE, RMSE, and MAPE via root `evaluate_forecast`; low-level `mae`/`rmse`/`mape` via `koopman_graph.metrics` |
| **Consistency losses** | Forward and backward latent linearity constraints plus optional eigenvalue stability regularization |
| **Classical baselines** | `DMDBaseline`, `EDMDBaseline`, and `DMDcBaseline` for topology-agnostic comparison |
| **Control inputs** | Koopman-with-control dynamics (`z_{t+1} = K z_t + B u_t`) for driven systems |
| **Dynamic topology** | Per-snapshot `edge_index` support for rewiring contact networks |
| **Edge weights** | End-to-end `edge_weight` propagation through GCN encoder/decoder and METR-LA benchmark |
| **Advanced training** | LR schedulers, per-term loss history, explicit root `MultiTrajectory` fit (`as_multi_trajectory` via `koopman_graph.data`), and windowed mini-batching |
| **Structural stability** | Guaranteed-stable parameterizations (`schur`, `dissipative`, `lyapunov`) for 200+ step rollouts — distinct from soft `odo`/eigenloss regularization |
| **Continuous-time dynamics** | `ContinuousKoopmanOperator` with `dynamics_mode="continuous"`, irregular timestamps, and `predict_at` |
| **Online adaptation** | `RecursiveKoopmanAdapter` and `adapt_step` for RLS updates to a frozen encoder |
| **Physics-informed observables** | Hybrid `koopman_graph.observables.graph_laplacian_features` concatenated with GNN latents before linear propagation |
| **RL environment** | `GraphKoopmanEnv` and `to_latent_env` for Gymnasium / Stable-Baselines3 closed-loop control |
| **GraphSnapshotSequence** | Time-ordered container for PyG graph snapshots with optional controls and weights |
| **Benchmark datasets** | Synthetic, grid, IEEE 118-bus, and METR-LA traffic benchmarks |
| **Jupyter tutorials** | Sixteen end-to-end notebooks with real networked datasets |
| **Tested & documented** | ≥95% coverage enforced in CI, Sphinx docs on Read the Docs (see [architecture](https://koopmangraph.readthedocs.io/en/latest/architecture.html) for public vs power-user API layers, shared rollout, optional `koopman=` injection, and `ForecastModel` call-site contracts) |

**Stability mode selection:** use `dense` or `odo` when you want a soft prior (`odo` bounds `ρ(K)` via the operator 2-norm but lacks a strict ε-interior certificate; continuous `odo` needs eigenvalue loss on the true spectrum); choose `schur`, `dissipative`, or `lyapunov` when you need eigenvalues mathematically forced inside the unit disk (see [`11_long_horizon_stability.ipynb`](examples/11_long_horizon_stability.ipynb) vs [`08_loss_stability.ipynb`](examples/08_loss_stability.ipynb)).




## Architecture

Each prediction step follows three stages:

```
  Node features x_t          Latent state z_t           Predicted x_{t+1}
  (N × F, on graph)    →    (N × d, on graph)     →    (N × F, on graph)

       ┌──────────┐              ┌──────────┐              ┌──────────┐
  x_t  │  GNN     │  z_t         │ Koopman  │  z_{t+1}     │  GNN     │  x_{t+1}
  ───► │ Encoder  │ ───►   ───►  │    K     │ ───►   ───►  │ Decoder  │ ───►
       └──────────┘              └──────────┘              └──────────┘
         (lifting)              (linear step)              (reconstruction)
```

During training, the model minimizes:

1. **Reconstruction** — Autoencoder fidelity between input and decoded node features.
2. **Forward consistency** — Latent states should satisfy z_{t+1} \approx K z_t.
3. **Backward consistency** — Inverse linear evolution in latent space.



## Installation

KoopmanGraph requires **Python 3.10+**, [PyTorch](https://pytorch.org/get-started/locally/), and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html). Install those first, then install KoopmanGraph:

```bash
pip install koopman-graph
```

For development from source:

```bash
git clone https://github.com/tjkessler/KoopmanGraph.git
cd KoopmanGraph
pip install -e ".[dev]"
```

For documentation builds:

```bash
pip install -e ".[docs]"
cd docs && make html
```

See the [installation guide](https://koopmangraph.readthedocs.io/en/latest/installation.html) for platform-specific PyTorch/PyG wheels and verification steps. Release workflow and version policy are documented in [CONTRIBUTING.md](CONTRIBUTING.md#releasing).

## Quickstart

Train a model on a synthetic spatiotemporal graph and predict five future snapshots:

```python
import torch
from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
from koopman_graph.datasets import SyntheticDynamicGraphBenchmark

data_sequence = SyntheticDynamicGraphBenchmark.generate(
    num_nodes=20,
    num_timesteps=30,
    in_channels=3,
    seed=42,
    noise_std=0.01,
)

encoder = GNNEncoder(3, 64, 64)
decoder = GNNDecoder(64, 64, 3)
model = GraphKoopmanModel(
    encoder=encoder,
    decoder=decoder,
    latent_dim=64,
    time_step=0.1,
)

torch.manual_seed(0)
history = model.fit(data_sequence, epochs=20, lr=1e-3)
future_graphs = model.predict(data_sequence[0], steps=5)

print(f"Final loss: {history.loss[-1]:.6f}")
print(f"Predicted {len(future_graphs)} snapshots, shape: {future_graphs[0].x.shape}")
```

Expected output:

```text
Final loss: <float>
Predicted 5 snapshots, shape: torch.Size([20, 3])
```

More detail: [Quickstart guide](https://koopmangraph.readthedocs.io/en/latest/quickstart.html) · [Architecture](https://koopmangraph.readthedocs.io/en/latest/architecture.html) · [API reference](https://koopmangraph.readthedocs.io/en/latest/api.html)

## Built-in Datasets


| Benchmark                           | Domain        | Description                                                |
| ----------------------------------- | ------------- | ---------------------------------------------------------- |
| `SyntheticDynamicGraphBenchmark`    | Synthetic     | Laplacian diffusion on path/ring graphs                    |
| `GridDynamicGraphBenchmark`         | Synthetic     | Laplacian diffusion on a 4-connected 2D lattice            |
| `AnisotropicAdvectionGridBenchmark` | Synthetic     | Directional advection with asymmetric edge weights         |
| `IEEE118DynamicBenchmark`           | Power systems | IEEE 118-bus topology with simulated voltage/load dynamics |
| `MetrLaTrafficBenchmark`            | Traffic       | METR-LA sensor graph with cached speed snapshots           |




## Examples

Jupyter tutorials in the [`examples/`](https://github.com/tjkessler/KoopmanGraph/tree/main/examples) directory cover training, evaluation, and analysis workflows:

| Notebook | Topic |
| --- | --- |
| [`01_synthetic_graph.ipynb`](https://github.com/tjkessler/KoopmanGraph/blob/main/examples/01_synthetic_graph.ipynb) | End-to-end synthetic graph dynamics |
| [`02_ieee118_bus.ipynb`](https://github.com/tjkessler/KoopmanGraph/blob/main/examples/02_ieee118_bus.ipynb) | IEEE 118-bus Vm forecasting (chronological split; held-out RMSE scale + bus ranking; honest DMDc comparison) |
| [`03_traffic_network.ipynb`](https://github.com/tjkessler/KoopmanGraph/blob/main/examples/03_traffic_network.ipynb) | METR-LA weekday cache: chronological split, trained graph vs DMD/EDMD (multi-origin RMSE) |
| [`04_grid_attention.ipynb`](https://github.com/tjkessler/KoopmanGraph/blob/main/examples/04_grid_attention.ipynb) | GAT encoder on grid graphs |
| [`05_custom_data.ipynb`](https://github.com/tjkessler/KoopmanGraph/blob/main/examples/05_custom_data.ipynb) | Bring your own graph sequences |
| [`06_epidemic_ring.ipynb`](https://github.com/tjkessler/KoopmanGraph/blob/main/examples/06_epidemic_ring.ipynb) | SIR ring wave showcase with Schur-stable spectrum (truth vs forecast) |
| [`07_koopman_spectrum.ipynb`](https://github.com/tjkessler/KoopmanGraph/blob/main/examples/07_koopman_spectrum.ipynb) | Koopman eigenvalue analysis |
| [`08_loss_stability.ipynb`](https://github.com/tjkessler/KoopmanGraph/blob/main/examples/08_loss_stability.ipynb) | Loss weighting and training stability |
| [`09_topology_ablation.ipynb`](https://github.com/tjkessler/KoopmanGraph/blob/main/examples/09_topology_ablation.ipynb) | Topology ablation study |
| [`10_advanced_training.ipynb`](https://github.com/tjkessler/KoopmanGraph/blob/main/examples/10_advanced_training.ipynb) | LR schedulers, rollout origins, multi-trajectory `fit` |
| [`11_long_horizon_stability.ipynb`](https://github.com/tjkessler/KoopmanGraph/blob/main/examples/11_long_horizon_stability.ipynb) | Structural stability parameterizations, 200-step IEEE 118 rollout |
| [`12_irregular_sampling_continuous_time.ipynb`](https://github.com/tjkessler/KoopmanGraph/blob/main/examples/12_irregular_sampling_continuous_time.ipynb) | Synthetic continuous-time demo: generator recovery, irregular Δt comparison, `predict_at` (METR-LA forecasting → notebook 03) |
| [`13_online_adaptation_traffic_drift.ipynb`](https://github.com/tjkessler/KoopmanGraph/blob/main/examples/13_online_adaptation_traffic_drift.ipynb) | Recursive least-squares online Koopman adaptation |
| [`14_physics_informed_diffusion.ipynb`](https://github.com/tjkessler/KoopmanGraph/blob/main/examples/14_physics_informed_diffusion.ipynb) | Hybrid physics observables API (cautionary matched-capacity RMSE; custom `physics_lifting_fn` save/load) |
| [`15_closed_loop_voltage_control_rl.ipynb`](https://github.com/tjkessler/KoopmanGraph/blob/main/examples/15_closed_loop_voltage_control_rl.ipynb) | Latent PPO regulates IEEE 118 Vm surrogate near 1.0 p.u. |
| [`16_spectral_similarity_anomalies.ipynb`](https://github.com/tjkessler/KoopmanGraph/blob/main/examples/16_spectral_similarity_anomalies.ipynb) | Spectral distance clustering and anomaly detection on IEEE 118 |




## Development

Run the test suite and coverage check locally:

```bash
pytest tests/ -v --cov=koopman_graph --cov-report=term-missing --cov-fail-under=95
```

Lint and format:

```bash
ruff check src/ tests/
ruff format --check src/ tests/
```

See [CONTRIBUTING.md](https://github.com/tjkessler/KoopmanGraph/blob/main/CONTRIBUTING.md) for the full development workflow, pre-commit hooks, and pull request guidelines.

## Citation

If you use KoopmanGraph in your research, please cite the repository:

```bibtex
@software{koopmangraph2026,
  author       = {Travis Kessler},
  title        = {KoopmanGraph: Graph Neural Networks with Koopman Operator Theory},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.21326273},
  url          = {https://github.com/tjkessler/KoopmanGraph},
  version      = {0.2.0},
}
```



## License

KoopmanGraph is released under the [Apache License 2.0](https://github.com/tjkessler/KoopmanGraph/blob/main/LICENSE).