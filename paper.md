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
date: 17 July 2026
bibliography: paper.bib
---

# Summary

`KoopmanGraph` is a Python library for learning and forecasting spatiotemporal dynamics on graphs. Many systems of scientific interest---including power grids, traffic sensor networks, and epidemic contact graphs---evolve on fixed topologies where node states interact along edges. The library combines Graph Neural Networks (GNNs) with Koopman operator theory to learn latent representations in which dynamics advance approximately linearly while respecting graph structure.

Each forecast step follows an encode--linear advance--decode workflow. GCN or GAT encoders lift node features into a latent space with message passing on the graph; a learnable finite-dimensional Koopman operator propagates latent states in discrete or continuous time; and a matching GNN decoder maps predictions back to physical node features. The `GraphKoopmanModel` class exposes `fit`, `predict`, and `evaluate` for training on sequences of PyTorch Geometric `Data` snapshots, rolling out multi-step forecasts, and reporting per-horizon error metrics. The software is available at \url{[https://github.com/tjkessler/KoopmanGraph}](https://github.com/tjkessler/KoopmanGraph}).

# Statement of need

Koopman operator theory provides a principled framework for representing nonlinear dynamics through linear evolution in a lifted observable space [@koopman1931; @Mezic2021; @Lusch2018]. Deep Koopman autoencoders learn encoders and linear propagators jointly, enabling multi-step prediction and spectral analysis of learned dynamics. Existing Python libraries implement this paradigm for vector-valued states: `PyKoopman` [@Pan2024] focuses on data-driven Koopman approximations including dynamic mode decomposition variants, and `DLKoopman` [@Dey2023_L4DC] provides deep-learning Koopman autoencoders for general dynamical systems. These tools treat the state as a flat vector and do not model how information propagates along graph edges.

Conversely, `PyTorch Geometric` (PyG) [@Fey/Lenssen/2019] offers mature infrastructure for GNNs on irregular graph data but does not provide a dedicated Koopman-theoretic forecasting stack with explicit linear latent dynamics and consistency losses. Spatiotemporal GNN methods typically learn nonlinear convolutional and/or recurrent maps on graphs rather than enforcing linear evolution in a learned observable space---including STGCN [@Yu2018STGCN] (fully convolutional gated temporal + graph conv blocks), DCRNN [@Li2018DCRNN] (diffusion-convolutional GRU), and Graph WaveNet [@Wu2019WaveNet] (dilated causal TCN with adaptive adjacency). `KoopmanGraph` ships lightweight in-repo reference implementations of these forecasters so tutorials can compare them under a shared evaluation protocol without claiming dedicated-library SOTA fidelity.

Combining GNNs with Koopman operators has been explored in domain-specific modeling studies [@Mukherjee2022; @Turja2023]. Closely related work compares GNN and Koopman models on power-grid transients [@Nandanoori2022] or uses Koopman-inspired methods to interpret temporal GNNs [@Guerra2024], but these efforts are distributed as bespoke research code rather than as reusable, general-purpose software. `KoopmanGraph` addresses this tooling gap by packaging GNN-based lifting and decoding, a learnable Koopman operator, and forward/backward consistency regularization in the spirit of consistent Koopman autoencoders [@Azencot2020] behind a stable, documented, and tested `fit`/`predict` API. Users working on networked dynamical systems can train and evaluate topology-aware Koopman models without flattening node states or reimplementing the encode--advance--decode loop.

# State of the field

Rather than truncating onto a fixed dictionary of observables, `KoopmanGraph` learns a finite-dimensional Koopman-invariant subspace using GNN-defined observables on graph-structured states. Mezić's work on numerical approximations of the Koopman operator clarifies how such finite-dimensional representations relate to the underlying infinite-dimensional operator [@Mezic2022]. The package is developed as a standalone PyG-native library because graph topology must enter at the encoder and decoder stages through shared `edge_index` message passing, exposing a small public API aligned with Koopman autoencoder practice [@Lusch2018; @Azencot2020].

# Software design

`KoopmanGraph` follows a modular three-stage architecture for each forecast step:

1. **Lifting (encoder).** `GNNEncoder` (GCN) or `GATEncoder` (GAT) maps node features $\mathbf{x}_t \in \mathbb{R}^{N \times F}$ to latent node states $\mathbf{z}_t \in \mathbb{R}^{N \times d}$ using the graph's `edge_index`. Optional hybrid observables can concatenate known structural features with learned latents.
2. **Linear evolution (Koopman operator).** Discrete `KoopmanOperator` applies a learnable matrix $\mathbf{K} \in \mathbb{R}^{d \times d}$ via $\mathbf{z}_{t+1} \approx \mathbf{z}_t \mathbf{K}^\top$ at each node. Optional networked `GraphKoopmanOperator` (`koopman="graph"`) couples neighbors inside the linear step through self/neighbor blocks aligned with the graph adjacency, following compositional and distributed geometric Koopman formulations [@Li2020CompositionalKoopman; @Mukherjee2022]. Continuous-time `ContinuousKoopmanOperator` learns a generator integrated with matrix exponentials, supporting irregular sampling through `predict_at`.
3. **Reconstruction (decoder).** Matching `GNNDecoder` or `GATDecoder` maps latent states back to physical features $\hat{\mathbf{x}}_{t+1} \in \mathbb{R}^{N \times F}$.

`GraphSnapshotSequence` validates time-ordered PyG snapshots (with an opt-in dynamic-topology mode) and supports control inputs and multi-trajectory fitting. Training via `GraphKoopmanModel.fit` minimizes reconstruction, forward/backward consistency following the consistent-autoencoder lineage [@Azencot2020], rollout, and optional eigenvalue regularization losses. `predict` performs autoregressive rollout; `evaluate` reports per-horizon MAE, RMSE, and MAPE; and `spectrum` exposes eigendecomposition of the learned operator with mode shapes mapped back to node space.

Beyond the core loop, the library includes structural stability options, Koopman-with-control linear predictors [@Korda2018; @Proctor2016DMDc], optional bilinear / control-affine state–control couplings [@Bruder2021], online operator adaptation, classical DMD/EDMD baselines with fixed dictionary observables [@Williams2015], lightweight STGCN/DCRNN/Graph WaveNet reference forecasters [@Yu2018STGCN; @Li2018DCRNN; @Wu2019WaveNet], optional delay-coordinate lifting at the encoder boundary for partial observations [@Takens1981], in the spirit of Hankel delay constructions used for Koopman / DMD analysis [@Brunton2017HAVOK; @Arbabi2017HankelDMD], and spectral similarity utilities. Dictionary-learning EDMD extensions are discussed by Li et al. [@Li2017EDMD]. Detailed APIs and workflows are documented in Sphinx and the tutorial notebooks.

The package is distributed on PyPI as `koopman-graph` [@koopmangraph2026], requires Python 3.10+, PyTorch, and PyTorch Geometric, and includes Sphinx documentation on Read the Docs.

# Example

Tutorial notebooks under `examples/` demonstrate end-to-end training and forecasting on synthetic dynamics, the IEEE 118-bus test system, the METR-LA traffic network, and epidemic contact graphs. Representative workflows include chronological evaluation against DMD/EDMD and spatiotemporal GNN forecaster baselines, dynamic-topology rollouts, spectral and stability analysis, and latent-space control demos.

# Research impact statement

`KoopmanGraph` is a new research software package. At the time of submission there are no external publications or downstream adoptions to cite, and no related companion manuscripts. Impact is demonstrated through reproducible tutorials on synthetic dynamics, the IEEE 118-bus test system, and the METR-LA traffic sensor network. These notebooks cover per-horizon evaluation, topology-versus-vector baseline comparisons in the spirit of prior GNN-versus-Koopman power-grid studies [@Nandanoori2022], dynamic-topology epidemic experiments, spectral and stability analysis, and latent-space control demos. Open development artifacts include PyPI releases (v0.1.0, v0.2.0, v0.3.0, v0.4.0), CI with pytest and a coverage gate, notebook smoke tests, and API reference documentation.

The author uses `KoopmanGraph` as a platform for experimenting with Koopman-theoretic forecasting on networked systems. The library is intended to lower the barrier for other researchers to train, evaluate, and extend topology-aware Koopman models on standard PyG workflows.

Versioned software archives are available on Zenodo (v0.3.0 DOI: \url{[https://doi.org/10.5281/zenodo.21404269}](https://doi.org/10.5281/zenodo.21404269}); v0.2.0 DOI: \url{[https://doi.org/10.5281/zenodo.21326273}](https://doi.org/10.5281/zenodo.21326273})).

# Conclusion

`KoopmanGraph` provides a PyG-native toolkit for topology-aware Koopman modeling of spatiotemporal graph dynamics behind a documented `fit`/`predict` workflow. We hope researchers will use it as a platform for experimentation on networked dynamical systems.

# AI usage disclosure

Generative AI was used via the Cursor IDE Agent (approximately July 2026 builds) with language models routed through Cursor, including Claude Fable, Claude Opus, Claude Sonnet, Composer, and Auto. These tools assisted with source code generation and refactoring, test scaffolding, Sphinx documentation and tutorial notebooks, blueprint task planning, and drafting and copy-editing of this manuscript. No other generative AI tools were used.

The author reviewed, edited, and validated all AI-assisted outputs, made core design and architectural decisions, and takes full responsibility for the accuracy and correctness of the software and this paper.

# Acknowledgements

The author thanks the maintainers of PyTorch Geometric, PyKoopman, and DLKoopman for foundational open-source tooling in geometric deep learning and Koopman operator computation. The author declares that there are no conflicts of interest. No external financial support was received for this work.

# References

