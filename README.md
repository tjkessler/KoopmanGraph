<div align="center">

<a href="https://koopmangraph.readthedocs.io/">
  <img src="https://raw.githubusercontent.com/tjkessler/KoopmanGraph/main/docs/source/_static/koopmangraph_logo.png" alt="KoopmanGraph logo" width="420"/>
</a>

# KoopmanGraph

**Graph Neural Networks with Koopman operator theory for spatiotemporal graph dynamics**

[![Tests](https://github.com/tjkessler/KoopmanGraph/actions/workflows/ci.yml/badge.svg)](https://github.com/tjkessler/KoopmanGraph/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21420623.svg)](https://doi.org/10.5281/zenodo.21420623)
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

**KoopmanGraph** is an open-source PyTorch library that combines **Graph Neural Networks** with **Koopman operator theory** to model spatiotemporal dynamics on graphs. Topology-aware encoders lift node features into a latent space, a learned linear Koopman operator advances latent states, and a matching decoder reconstructs physical node features.

It is a **topology-aware alternative to vector-based Koopman methods**, aimed at researchers working on smart grids, traffic networks, epidemic contact graphs, and related networked dynamical systems.

## Why KoopmanGraph?

Koopman theory maps nonlinear dynamics into a linear latent domain where multi-step forecasting and spectral analysis are natural. Existing deep Koopman packages typically ignore graph structure, while spatiotemporal GNN forecasters usually lack an explicit linear latent operator.

KoopmanGraph bridges that gap with GNN lifting/decoding, an inspectable Koopman matrix **K**, row-state latent advance $z \leftarrow z K^{\top}$, and a PyTorch Geometric-native `fit` / `predict` workflow.

The library sits in the consistent Koopman autoencoder lineage and is **not claimed as a new theoretical contribution**; it packages topology-aware lifting, linear latent evolution, and analysis tooling for networked dynamical systems.

<p align="center">
  <img src="docs/source/_static/architecture-overview.svg" alt="Encode → linear Koopman advance → decode architecture" width="820"/>
</p>

## Highlights

- **Topology-aware learning** — GCN/GAT encoders and decoders, delay embeddings, dynamic topology, and edge weights
- **Flexible dynamics** — discrete, continuous-time, and networked (`koopman="graph"`) operators, with soft or structural stability modes
- **Forecasting stack** — multi-step rollout, consistency losses, temporal evaluation metrics, and checkpointing
- **Spectral analysis** — eigendecomposition, mode shapes, dynamical similarity, and anomaly helpers
- **Control and adaptation** — additive/bilinear control, online RLS adaptation, Kalman observation, and a Gymnasium RL wrapper
- **Research tooling** — classical DMD-family baselines, lightweight GNN teaching baselines, and reproducible graph benchmarks

Full inventory: [Capabilities](https://koopmangraph.readthedocs.io/en/latest/capabilities.html) · [Architecture](https://koopmangraph.readthedocs.io/en/latest/architecture.html)

## Installation

Requires **Python 3.10+**, [PyTorch](https://pytorch.org/get-started/locally/), and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html). Install those first, then:

```bash
pip install koopman-graph
```

See the [installation guide](https://koopmangraph.readthedocs.io/en/latest/installation.html) for editable installs, docs builds, and platform-specific wheels. Release notes: [CHANGELOG.md](CHANGELOG.md).

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

Featured tutorials: [01 synthetic](https://github.com/tjkessler/KoopmanGraph/blob/main/examples/01_synthetic_graph.ipynb) · [03 traffic](https://github.com/tjkessler/KoopmanGraph/blob/main/examples/03_traffic_network.ipynb) · [06 epidemic](https://github.com/tjkessler/KoopmanGraph/blob/main/examples/06_epidemic_ring.ipynb) · [22 GNN baselines](https://github.com/tjkessler/KoopmanGraph/blob/main/examples/22_gnn_forecaster_comparison.ipynb) · [full gallery](https://koopmangraph.readthedocs.io/en/latest/tutorials.html)

## Learn more

- [Quickstart](https://koopmangraph.readthedocs.io/en/latest/quickstart.html) — train / predict walkthrough
- [Capabilities](https://koopmangraph.readthedocs.io/en/latest/capabilities.html) — feature inventory and datasets
- [Architecture](https://koopmangraph.readthedocs.io/en/latest/architecture.html) — public vs power-user API layers
- [FAQ / troubleshooting](https://koopmangraph.readthedocs.io/en/latest/faq.html) — install, imports, checkpoints
- [Installation](https://koopmangraph.readthedocs.io/en/latest/installation.html) — dependencies and install paths
- What’s new in 0.5.0: see [CHANGELOG.md](CHANGELOG.md).

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
  title        = {KoopmanGraph: Graph Neural Networks with Koopman Operator Theory},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.21420623},
  url          = {https://github.com/tjkessler/KoopmanGraph},
  version      = {0.4.0},
}
```

## License

KoopmanGraph is released under the [Apache License 2.0](https://github.com/tjkessler/KoopmanGraph/blob/main/LICENSE).
