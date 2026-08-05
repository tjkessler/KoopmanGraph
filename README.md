<div align="center">

<a href="https://koopmangraph.readthedocs.io/">
  <img src="https://raw.githubusercontent.com/tjkessler/KoopmanGraph/main/docs/source/_static/koopmangraph_logo.png" alt="KoopmanGraph logo" width="420"/>
</a>

# KoopmanGraph

**Topology-aware Koopman autoencoders for forecasting and analyzing networked dynamics**

[![Tests](https://github.com/tjkessler/KoopmanGraph/actions/workflows/ci.yml/badge.svg)](https://github.com/tjkessler/KoopmanGraph/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21811756.svg)](https://doi.org/10.5281/zenodo.21811756)
[![PyPI version](https://img.shields.io/pypi/v/koopman-graph)](https://pypi.org/project/koopman-graph/)
[![codecov](https://codecov.io/gh/tjkessler/KoopmanGraph/graph/badge.svg)](https://codecov.io/gh/tjkessler/KoopmanGraph)
[![Documentation Status](https://readthedocs.org/projects/koopmangraph/badge/?version=latest)](https://koopmangraph.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**[Documentation](https://koopmangraph.readthedocs.io/)** |
**[Tutorials](https://koopmangraph.readthedocs.io/en/latest/tutorials.html)** |
**[API](https://koopmangraph.readthedocs.io/en/latest/api.html)** |
**[Contributing](https://github.com/tjkessler/KoopmanGraph/blob/main/CONTRIBUTING.md)** |
**[Citation](#community-and-citation)**

</div>

---

**KoopmanGraph** is an open-source PyTorch Geometric library for learning **topology-aware Koopman autoencoders** on graphs. GNN encoders lift node features into a latent space, a learned linear operator advances those states, and a matching decoder reconstructs physical node features for multi-step forecasting and spectral analysis.

It is aimed at researchers studying **networked dynamical systems**—smart grids, traffic networks, epidemic contact graphs, and similar domains—who want an inspectable linear latent operator instead of a flat-vector Koopman model or a purely nonlinear GNN forecaster.

## Why KoopmanGraph?

Koopman theory maps nonlinear dynamics into a linear latent domain where multi-step forecasting and spectral analysis are natural. Existing deep Koopman packages typically ignore graph structure, while spatiotemporal GNN forecasters usually lack an explicit linear latent operator.

KoopmanGraph bridges that gap with GNN lifting/decoding, an inspectable Koopman matrix **K**, row-state latent advance $z \leftarrow z K^{\top}$, and a PyTorch Geometric-native `fit` / `predict` workflow.

The library sits in the consistent Koopman autoencoder lineage and is **not claimed as a new theoretical contribution**; it packages topology-aware lifting, linear latent evolution, and analysis tooling for networked dynamical systems.

<p align="center">
  <img src="https://raw.githubusercontent.com/tjkessler/KoopmanGraph/main/docs/source/_static/architecture-overview.svg" alt="Encode → linear Koopman advance → decode architecture" width="820"/>
</p>

## Highlights

- **Topology-aware learning** — GCN/GAT/hypergraph encoders and decoders, delay embeddings, dynamic topology, optional self-adaptive edges, and edge weights
- **Flexible dynamics** — discrete, continuous-time, networked (`koopman="graph"`), hypergraph, multiplex / typed hetero (`koopman="hetero_graph"` with RelGraph), global/local, and continuous-graph operators, with soft or structural stability modes
- **Forecasting stack** — multi-step rollout, consistency losses, temporal evaluation metrics, and checkpointing
- **Spectral analysis** — eigendecomposition, mode shapes, dynamical similarity, anomaly helpers, SINDy, and spectral clustering
- **Control and adaptation** — additive/bilinear control, Koopman-MPC (`[mpc]`), online RLS adaptation, Kalman observation, and a Gymnasium RL wrapper
- **Research tooling** — classical DMD-family baselines, lightweight GNN teaching baselines, conformal UQ, and reproducible graph benchmarks
- **Optional distributed trainers** — native DDP / `torchrun`, Lightning Fabric, and Ray ensemble helpers under `koopman_graph.distributed` (power-user; compose with homo or hetero models; see [installation](https://koopmangraph.readthedocs.io/en/latest/installation.html) extras `lightning` / `ray` / `distributed`)

Full inventory: [Capabilities](https://koopmangraph.readthedocs.io/en/latest/capabilities.html) · [Architecture](https://koopmangraph.readthedocs.io/en/latest/architecture.html)

**Scope.** KoopmanGraph targets topology-aware Koopman autoencoders on graphs and hypergraphs, not traffic-forecasting leaderboards or full simplicial/Hodge / TopologicX parity. Optional GraphVAMP and synthetic molecular helpers are teaching / diagnostic paths — not Folding@home-scale MD or a PyEMMA replacement. Measured limits (transfer, factorization cost, residual diagnostics, UQ assumptions) are consolidated in [Scope and limitations](https://koopmangraph.readthedocs.io/en/latest/limitations.html).

## Installation

Requires **Python 3.10+**, [PyTorch](https://pytorch.org/get-started/locally/), and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html). Install those first, then:

```bash
pip install koopman-graph
# or: uv pip install koopman-graph
```

See the [installation guide](https://koopmangraph.readthedocs.io/en/latest/installation.html) for editable installs, `uv` workflows, docs builds, and platform-specific wheels. Release notes: [CHANGELOG.md](CHANGELOG.md).

## Quickstart

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
spectrum = model.spectrum()

print(f"Final loss: {history.loss[-1]:.6f}")
print(f"Predicted {len(future_graphs)} snapshots, shape: {future_graphs[0].x.shape}")
print(f"K eigenvalues: {tuple(spectrum.eigenvalues.shape)}")
print(f"Top |λ|: {spectrum.magnitudes[:3].tolist()}")
```

Expected output:

```text
Final loss: <float>
Predicted 5 snapshots, shape: torch.Size([20, 3])
K eigenvalues: (64,)
Top |λ|: [<float>, <float>, <float>]
```

More detail: [Quickstart guide](https://koopmangraph.readthedocs.io/en/latest/quickstart.html) · [API reference](https://koopmangraph.readthedocs.io/en/latest/api.html)

## See it in action

<p align="center">
  <img src="https://raw.githubusercontent.com/tjkessler/KoopmanGraph/main/docs/source/_static/epidemic-forecast.png" alt="Epidemic truth versus KoopmanGraph forecast on a ring graph" width="820"/>
</p>

<p align="center"><em>SIR epidemic on a ring: truth vs forecast from <a href="https://github.com/tjkessler/KoopmanGraph/blob/main/examples/06_epidemic_ring.ipynb">examples/06_epidemic_ring.ipynb</a>.</em></p>

<p align="center">
  <img src="https://raw.githubusercontent.com/tjkessler/KoopmanGraph/main/docs/source/_static/metrla-gnn-baselines.png" alt="METR-LA aggregate RMSE for GraphKoopman versus STGCN, DCRNN, and Graph WaveNet teaching baselines" width="640"/>
</p>

<p align="center"><em>METR-LA aggregate RMSE vs in-repo STGCN / DCRNN / Graph WaveNet <strong>teaching baselines</strong> (not dedicated-library SOTA) from <a href="https://github.com/tjkessler/KoopmanGraph/blob/main/examples/22_gnn_forecaster_comparison.ipynb">examples/22_gnn_forecaster_comparison.ipynb</a>.</em></p>

Featured tutorials: [01 synthetic](https://github.com/tjkessler/KoopmanGraph/blob/main/examples/01_synthetic_graph.ipynb) · [03 traffic](https://github.com/tjkessler/KoopmanGraph/blob/main/examples/03_traffic_network.ipynb) · [06 epidemic](https://github.com/tjkessler/KoopmanGraph/blob/main/examples/06_epidemic_ring.ipynb) · [22 GNN baselines](https://github.com/tjkessler/KoopmanGraph/blob/main/examples/22_gnn_forecaster_comparison.ipynb) · [37 topology transfer](https://github.com/tjkessler/KoopmanGraph/blob/main/examples/37_cross_topology_transfer.ipynb) · [39 hetero RelGraph](https://github.com/tjkessler/KoopmanGraph/blob/main/examples/39_heterogeneous_relational_koopman.ipynb) · [42 teaching baselines](https://github.com/tjkessler/KoopmanGraph/blob/main/examples/42_traffic_teaching_baselines.ipynb) · [full gallery](https://koopmangraph.readthedocs.io/en/latest/tutorials.html)

## Learn more

- [Quickstart](https://koopmangraph.readthedocs.io/en/latest/quickstart.html) — train / predict walkthrough
- [Capabilities](https://koopmangraph.readthedocs.io/en/latest/capabilities.html) — feature inventory and datasets
- [Scope and limitations](https://koopmangraph.readthedocs.io/en/latest/limitations.html) — when not to use; measured boundaries
- [Architecture](https://koopmangraph.readthedocs.io/en/latest/architecture.html) — public vs power-user API layers
- [FAQ / troubleshooting](https://koopmangraph.readthedocs.io/en/latest/faq.html) — install, imports, checkpoints
- [Installation](https://koopmangraph.readthedocs.io/en/latest/installation.html) — dependencies and install paths
- What’s new in 0.12.0: Kronecker-sum exact spectrum for eligible discrete
  and continuous graph operators (three-path spectrum contract), Scale /
  architecture honesty for residual dense inverse and eig-reg ceilings, and
  docs-only topology-criticality honesty — see [CHANGELOG.md](CHANGELOG.md).

## Related software

- [PyKoopman](https://pykoopman.readthedocs.io/) and [DLKoopman](https://github.com/GaloisInc/dlkoopman) target vector-valued Koopman / deep-Koopman workflows; they treat the state as a flat vector rather than propagating information along graph edges.
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) provides mature GNN infrastructure on irregular graphs; KoopmanGraph adds an explicit linear latent operator, consistency losses, and a documented `fit` / `predict` forecasting stack on that substrate.
- Spatiotemporal GNN forecasters such as STGCN, DCRNN, and Graph WaveNet typically learn nonlinear convolutional or recurrent maps on graphs; KoopmanGraph instead advances an inspectable linear Koopman matrix **K** (see in-repo teaching baselines in [examples/22](https://github.com/tjkessler/KoopmanGraph/blob/main/examples/22_gnn_forecaster_comparison.ipynb)).

## Community and citation

- Contribute, report issues, or seek support: [CONTRIBUTING.md](https://github.com/tjkessler/KoopmanGraph/blob/main/CONTRIBUTING.md) · [Support](https://github.com/tjkessler/KoopmanGraph/blob/main/CONTRIBUTING.md#support) · [Code of Conduct](CODE_OF_CONDUCT.md)
- Install / runtime troubleshooting: [FAQ](https://koopmangraph.readthedocs.io/en/latest/faq.html)
- Security vulnerabilities (private): [SECURITY.md](SECURITY.md)
- Development checks and release process: [CONTRIBUTING.md](https://github.com/tjkessler/KoopmanGraph/blob/main/CONTRIBUTING.md#running-checks-locally)

If you use KoopmanGraph in research, please cite:

```bibtex
@software{koopmangraph2026,
  author       = {Travis Kessler},
  title        = {KoopmanGraph: Topology-Aware Koopman Autoencoders for Networked Dynamics},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.21811756},
  url          = {https://github.com/tjkessler/KoopmanGraph},
  version      = {0.12.0},
}
```

## License

KoopmanGraph is released under the [Apache License 2.0](https://github.com/tjkessler/KoopmanGraph/blob/main/LICENSE).
