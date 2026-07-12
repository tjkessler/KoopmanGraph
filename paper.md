---
title: 'KoopmanGraph: Graph Neural Networks with Koopman Operator Theory for Spatiotemporal Graph Dynamics'
tags:
  - Python
  - Koopman operator
  - graph neural networks
  - PyTorch Geometric
  - spatiotemporal dynamics
  - dynamical systems
authors:
  - name: Travis Kessler
    orcid: 0000-0002-7363-4050
    corresponding: true
    affiliation: 1
affiliations:
  - name: Independent Researcher, United States
    index: 1
date: 11 July 2026
bibliography: paper.bib
---

# Summary

`KoopmanGraph` is a Python library for learning and forecasting spatiotemporal dynamics on graphs. Many systems of scientific interest---including power grids, traffic sensor networks, and epidemic contact graphs---evolve on fixed topologies where node states interact along edges. The library combines Graph Neural Networks (GNNs) with Koopman operator theory to learn latent representations in which dynamics advance approximately linearly while respecting graph structure.

Each forecast step follows an encode--linear advance--decode workflow: GCN or GAT encoders lift node features into a latent space with message passing on the graph; a learnable finite-dimensional Koopman matrix propagates latent states; and a symmetric GNN decoder maps predictions back to physical node features. The `GraphKoopmanModel` class exposes `fit`, `predict`, and `evaluate` methods for training on sequences of PyTorch Geometric `Data` snapshots, rolling out multi-step forecasts, and reporting per-horizon error metrics. Spectral analysis of the learned operator, classical DMD/EDMD baselines, Koopman-with-control dynamics, and dynamic-topology sequences are supported for interpretation, comparison, and domain-specific experimentation. The software is available at \url{https://github.com/tjkessler/KoopmanGraph}.

# Statement of need

Koopman operator theory provides a principled framework for representing nonlinear dynamics through linear evolution in a lifted observable space [@koopman1931; @Mezic2021; @Lusch2018]. Mezić showed that finite-dimensional representations can be learned from data when the Koopman spectrum admits suitable structure, connecting operator-theoretic foundations to modern machine learning. Deep Koopman autoencoders learn encoders and linear propagators jointly, enabling multi-step prediction and spectral analysis of learned dynamics. Several Python libraries implement this paradigm for vector-valued state representations: `PyKoopman` [@Pan2024] focuses on data-driven Koopman approximations including dynamic mode decomposition variants, and `DLKoopman` [@Dey2023_L4DC] provides deep-learning Koopman autoencoders for general dynamical systems. These tools treat the state as a flat vector and do not model how information propagates along graph edges.

Conversely, `PyTorch Geometric` (PyG) [@Fey/Lenssen/2019] offers mature infrastructure for GNNs on irregular graph data but does not provide a dedicated, Koopman-theoretic forecasting stack with explicit linear latent dynamics and consistency losses. Spatiotemporal GNN methods in the literature typically learn nonlinear recurrent or autoregressive maps rather than enforcing linear evolution in a learned observable space.

Combining GNNs with Koopman operators has been explored in several domain-specific research studies [@Mukherjee2022; @Turja2023; @Guerra2024], but these efforts are distributed as bespoke research code targeting particular applications (power grids, functional brain networks, model explainability) rather than as reusable, general-purpose software. `KoopmanGraph` addresses this tooling gap: it packages GNN-based lifting and decoding, a learnable Koopman operator, and forward/backward consistency regularization behind a stable, documented, and tested `fit`/`predict` API. Users working on networked dynamical systems can train and evaluate topology-aware Koopman models without flattening node states or reimplementing the encode--advance--decode loop. The library is designed as a focused research tool: components are composable, documented, and tested for reproducible experimentation and tutorial-driven adoption.

# State of the field

Existing Koopman software addresses complementary but non-overlapping needs. `PyKoopman` [@Pan2024] provides classical and data-driven Koopman/DMD algorithms for continuous and discrete dynamical systems---including dynamic mode decomposition variants, kernel methods, and a neural-network DMD option---with strong support for system identification workflows on vector states. `DLKoopman` [@Dey2023_L4DC] targets deep autoencoder architectures that learn encodings and linear dynamics simultaneously, including trajectory and snapshot prediction modes. Neither library is built around PyG graph objects or message-passing encoders that respect node adjacency.

The combination of graph learning and Koopman theory has itself received research attention. @Mukherjee2022 learn a distributed geometric Koopman operator for sparse networked dynamical systems using GNN encoders and decoders, with power-grid applications; @Turja2023 propose DeepGraphDMD, an autoencoder that learns Koopman eigenfunctions of functional brain networks; and @Guerra2024 use Koopman-inspired dynamic mode decomposition to interpret temporal GNNs. These works establish the conceptual value of topology-aware Koopman modeling, but each is released as application-specific research code---tied to a particular domain, dataset, or analysis---rather than as a general, installable, and tested library. `KoopmanGraph` differs in intent: it is a reusable PyG-native package with a stable public API, documentation, benchmark datasets, and continuous integration, intended to support experimentation across domains rather than to reproduce a single study.

PyG [@Fey/Lenssen/2019] provides message-passing layers, data loaders, and benchmark datasets for graph-structured data, but leaves Koopman-specific training objectives and latent linear propagation to user code. Mezić's work on numerical approximations of the Koopman operator clarifies how finite-dimensional representations relate to the underlying infinite-dimensional operator [@Mezic2022]. Rather than truncating onto a fixed dictionary of observables, `KoopmanGraph` learns a finite-dimensional Koopman-invariant subspace directly, using GNN-defined observables on graph-structured states instead of hand-crafted functions on flattened vectors.

`KoopmanGraph` was developed as a standalone package rather than a patch to existing Koopman tools because graph topology must enter at the encoder and decoder stages---not only in the loss---through shared `edge_index` message passing. The design keeps PyG as the geometric backbone while exposing a small, stable public API (`GraphKoopmanModel`, `GNNEncoder`, `GATEncoder`, `GNNDecoder`, `KoopmanOperator`) aligned with Koopman autoencoder practice [@Mezic2021; @Lusch2018].

# Software design

`KoopmanGraph` follows a modular three-stage architecture for each forecast step:

1. **Lifting (encoder).** `GNNEncoder` (GCN) or `GATEncoder` (GAT) maps node features $\mathbf{x}_t \in \mathbb{R}^{N \times F}$ to latent node states $\mathbf{z}_t \in \mathbb{R}^{N \times d}$ using the graph's `edge_index`.
2. **Linear evolution (Koopman operator).** `KoopmanOperator` applies a learnable matrix $\mathbf{K} \in \mathbb{R}^{d \times d}$ via $\mathbf{z}_{t+1} \approx \mathbf{z}_t \mathbf{K}^\top$ at each node.
3. **Reconstruction (decoder).** `GNNDecoder` maps latent states back to physical features $\hat{\mathbf{x}}_{t+1} \in \mathbb{R}^{N \times F}$.

`GraphSnapshotSequence` validates that all timesteps share a consistent topology (with an opt-in dynamic-topology mode for rewiring graphs) and provides helpers to build sequences from NumPy or PyTorch arrays, optionally carrying shared `edge_weight` tensors and per-timestep control inputs. Training via `GraphKoopmanModel.fit` minimizes a weighted sum of reconstruction, forward consistency, backward consistency, rollout, and optional eigenvalue regularization losses. Optional gradient clipping, loss-weight schedules, validation-aware early stopping, learning-rate schedulers, multi-trajectory aggregation, and windowed mini-batching improve stability and scalability on longer sequences. `GraphKoopmanModel.predict` performs autoregressive rollout in eval mode and returns a list of PyG `Data` snapshots; `GraphKoopmanModel.evaluate` reports per-horizon MAE, RMSE, and MAPE on held-out data. `GraphKoopmanModel.save` and `GraphKoopmanModel.load` persist trained weights together with encoder/decoder architecture configuration. `model.spectrum()` exposes eigendecomposition of the learned operator with continuous-time growth rates and frequencies derived from `time_step`, and `decode_mode_shapes` maps latent Koopman modes back to node-space patterns.

The package is distributed on PyPI as `koopman-graph` [@koopmangraph2026], requires Python 3.10+, PyTorch, and PyTorch Geometric, and includes Sphinx documentation hosted on Read the Docs.

# Features

At the time of writing, `KoopmanGraph` provides the following functionality:

- **End-to-end model.** `GraphKoopmanModel` composes encoder, Koopman operator, and decoder with `fit`, `predict`, `evaluate`, `save`, `load`, and `spectrum` APIs.
- **Topology-aware encoders.** `GNNEncoder` (GCN) and `GATEncoder` (GAT) for latent lifting with message passing on `edge_index`; GCN paths propagate optional scalar `edge_weight`.
- **Learnable linear dynamics.** `KoopmanOperator` with configurable matrix initialization (`identity`, `identity_noise`, `xavier`) and an opt-in spectrally constrained orthogonal--diagonal--orthogonal parameterization for long-horizon stability.
- **Koopman with control.** Optional input matrix `B` and per-timestep control sequences for driven dynamics (`z_{t+1} = K z_t + B u_t`).
- **Consistency regularization.** `ForwardConsistencyLoss`, `BackwardConsistencyLoss`, and `EigenvalueRegularizationLoss` for latent linearity and unit-circle eigenvalue penalties during training.
- **Spectral analysis.** `KoopmanSpectrum`, `compute_spectrum`, and `decode_mode_shapes` for eigenvalues, mode amplitudes, continuous-time frequencies, and spatial mode shapes.
- **Sequence utilities.** `GraphSnapshotSequence` for validated, time-ordered PyG snapshot containers; `TemporalSplit` / `temporal_split` for train/validation/test partitioning; `WindowSampler` for fixed-length mini-batch training.
- **Evaluation metrics.** `evaluate_forecast` and horizon-wise MAE, RMSE, and MAPE for multi-step benchmark reporting.
- **Classical baselines.** `DMDBaseline`, `EDMDBaseline`, and `DMDcBaseline` for least-squares/SVD Koopman fits on flattened node states, enabling direct topology-aware versus vector-based comparisons.
- **Dynamic topology.** Opt-in per-snapshot `edge_index` support for rewiring contact networks during training and rollout.
- **Benchmark datasets.** Synthetic diffusion and grid graphs, IEEE 118-bus power network, and METR-LA traffic benchmarks (cached data with documented download scripts where needed).
- **Tutorials and tests.** Nine Jupyter notebooks, Sphinx API reference, pytest suite with an enforced 80% coverage gate, and nbmake smoke tests in CI.

Tutorial notebooks cover synthetic dynamics, the IEEE 118-bus test system (including controlled load ramps), METR-LA traffic forecasting with per-horizon evaluation, epidemic rewiring on ring graphs, Koopman spectral analysis, operator stability, and topology ablation studies. The traffic tutorial compares `GraphKoopmanModel` against packaged `DMDBaseline` and `EDMDBaseline` baselines that ignore graph structure.

# Example

The `KoopmanGraph` repository provides Jupyter notebook tutorials in `examples/`. Here we briefly demonstrate a typical workflow on the built-in synthetic benchmark.

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
```

After training, `future_graphs` contains five predicted PyG `Data` snapshots with node feature shape `(20, 3)`, matching the input dimensionality.

# Research impact statement

`KoopmanGraph` is a new research software package. At the time of submission there are no external publications or downstream adoptions to cite, and no related companion manuscripts. Impact is demonstrated through reproducible tutorials on synthetic dynamics, the IEEE 118-bus test system, and the METR-LA traffic sensor network; per-horizon evaluation on held-out traffic data; controlled-system demonstrations on the power-grid benchmark; epidemic rewiring on dynamic topologies; spectral and stability analysis notebooks; explicit topology-versus-vector comparisons using packaged DMD/EDMD baselines; and open development artifacts including PyPI releases (v0.1.0 initial release, v0.2.0 feature expansion), CI with pytest and an 80% coverage gate, nbmake notebook smoke tests, and API reference documentation.

The author uses `KoopmanGraph` as a platform for experimenting with Koopman-theoretic forecasting on networked systems. The library is intended to lower the barrier for other researchers to train, evaluate, and extend topology-aware Koopman models on standard PyG workflows.

A versioned software archive with DOI is planned via Zenodo integration upon release tagging (software archive DOI: TBD --- Zenodo deposit pending).

# Conclusion

`KoopmanGraph` provides a central, PyG-native toolkit for topology-aware Koopman modeling of spatiotemporal graph dynamics. By combining GNN encoders and decoders with a learnable finite-dimensional Koopman operator, consistency and stability losses, spectral analysis, classical baselines, and evaluation utilities, the library makes it straightforward to train, evaluate, and interpret graph-respecting forecasting models that retain explicit linear latent dynamics. We hope that researchers and practitioners will use `KoopmanGraph` as a platform for experimentation on networked dynamical systems and for extending Koopman-theoretic methods to graph-structured data.

# AI usage disclosure

Generative AI tools (including Cursor Agent and associated language models) were used during development of `KoopmanGraph` source code, documentation, tutorial notebooks, and the drafting of this manuscript. AI assistance included code generation and refactoring, test scaffolding, Sphinx documentation drafts, blueprint task planning, and copy-editing of paper text.

The author reviewed, edited, and validated all AI-assisted outputs, made core design and architectural decisions, and takes full responsibility for the accuracy and correctness of the software and this paper.

# Acknowledgements

The author thanks the maintainers of PyTorch Geometric, PyKoopman, and DLKoopman for foundational open-source tooling in geometric deep learning and Koopman operator computation.

# References
