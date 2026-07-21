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

`KoopmanGraph` is a Python library for learning and forecasting spatiotemporal dynamics on graphs. Systems such as power grids, traffic sensor networks, and epidemic contact graphs evolve on fixed topologies where node states interact along edges. The library combines Graph Neural Networks (GNNs) with Koopman operator theory to learn latent representations in which dynamics advance approximately linearly while respecting graph structure.

Each forecast step follows an encode--linear advance--decode workflow: GNN encoders lift node features with message passing; a learnable finite-dimensional Koopman operator propagates latents in discrete or continuous time; and a matching decoder maps predictions back to physical features. `GraphKoopmanModel` exposes `fit`, `predict`, and `evaluate` for PyTorch Geometric snapshot sequences, multi-step rollouts, and per-horizon metrics. The software is available at \url{https://github.com/tjkessler/KoopmanGraph}.

# Statement of need

Koopman operator theory represents nonlinear dynamics via linear evolution in a lifted observable space [@koopman1931; @Mezic2021; @Lusch2018]. Deep Koopman autoencoders learn encoders and linear propagators jointly for multi-step prediction and spectral analysis. Existing libraries target vector-valued states: `PyKoopman` [@Pan2024] for data-driven approximations including DMD variants, and `DLKoopman` [@Dey2023_L4DC] for deep Koopman autoencoders. Neither models information flow along graph edges.

`PyTorch Geometric` (PyG) [@Fey/Lenssen/2019] provides GNN infrastructure but not a dedicated Koopman forecasting stack with explicit linear latent dynamics and consistency losses. Spatiotemporal GNN methods such as STGCN [@Yu2018STGCN], DCRNN [@Li2018DCRNN], and Graph WaveNet [@Wu2019WaveNet] learn nonlinear graph maps rather than linear evolution in a learned observable space. `KoopmanGraph` ships lightweight in-repo reference implementations of these forecasters for shared-protocol tutorial comparisons without claiming dedicated-library SOTA fidelity.

Domain studies combine GNNs with Koopman operators [@Mukherjee2022; @Turja2023], compare GNN and Koopman models on power-grid transients [@Nandanoori2022], or use Koopman-inspired methods to interpret temporal GNNs [@Guerra2024], but typically as bespoke research code. `KoopmanGraph` packages GNN lifting and decoding, a learnable Koopman operator, and forward/backward consistency regularization in the spirit of consistent Koopman autoencoders [@Azencot2020] behind a documented `fit`/`predict` API, so users need not flatten node states or reimplement the encode--advance--decode loop.

# State of the field

Rather than truncating onto a fixed dictionary of observables, `KoopmanGraph` learns a finite-dimensional Koopman-invariant subspace with GNN-defined observables on graph-structured states [@Mezic2022]. Topology enters at the encoder and decoder through shared `edge_index` message passing in a PyG-native API aligned with Koopman autoencoder practice [@Lusch2018; @Azencot2020].

# Software design

Each forecast step has three stages:

1. **Lifting.** GCN, GAT, GraphSAGE, DiffConv, or graph-transformer encoders map $\mathbf{x}_t \in \mathbb{R}^{N \times F}$ to $\mathbf{z}_t \in \mathbb{R}^{N \times d}$ via `edge_index`. Optional hybrid observables concatenate structural features with learned latents.
2. **Linear evolution.** Discrete `KoopmanOperator` applies $\mathbf{K} \in \mathbb{R}^{d \times d}$ via $\mathbf{z}_{t+1} \approx \mathbf{z}_t \mathbf{K}^\top$. Optional `GraphKoopmanOperator` (`koopman="graph"`) couples neighbors through self/neighbor blocks [@Li2020CompositionalKoopman; @Mukherjee2022]. Continuous `ContinuousKoopmanOperator` learns a matrix-exponential generator with irregular-time `predict_at`.
3. **Reconstruction.** A matching GNN decoder maps latents to $\hat{\mathbf{x}}_{t+1} \in \mathbb{R}^{N \times F}$.

`GraphSnapshotSequence` validates time-ordered PyG snapshots (optional dynamic topology), including control inputs and multi-trajectory fitting. `GraphKoopmanModel.fit` minimizes reconstruction, forward/backward consistency [@Azencot2020], rollout, and optional eigenvalue regularization. `predict` rollouts autoregressively; `evaluate` reports per-horizon MAE/RMSE/MAPE; `spectrum` returns operator eigendecomposition with node-space modes.

Extensions include structural stability, Koopman-with-control [@Korda2018; @Proctor2016DMDc], bilinear / control-affine couplings [@Bruder2021], online adaptation, DMD/EDMD with polynomial, RBF, and kernel dictionaries [@Williams2015; @Li2017EDMD], STGCN/DCRNN/Graph WaveNet references [@Yu2018STGCN; @Li2018DCRNN; @Wu2019WaveNet], delay-coordinate lifting [@Takens1981; @Brunton2017HAVOK; @Arbabi2017HankelDMD], spectral similarity, deep-ensemble and latent-Gaussian uncertainty, auxiliary-spectral continuous generators, physics-residual and sparsity losses, and hierarchical coarse-graph forecasting. The package is on PyPI as `koopman-graph` [@koopmangraph2026] (Python 3.10+, PyTorch, PyG) with Sphinx docs on Read the Docs.

# Example

Tutorials under `examples/` cover synthetic dynamics, IEEE 118-bus, METR-LA, and epidemic graphs, including baseline comparisons, dynamic topology, spectral/stability analysis, latent-space control, and v0.5.0 notebooks for auxiliary-spectral generators, uncertainty quantification, hierarchical forecasting, and sparse operators.

# Research impact statement

`KoopmanGraph` is new research software with no external publications or downstream adoptions to cite at submission. Impact is shown through reproducible tutorials on synthetic, IEEE 118-bus, METR-LA, and epidemic graphs---per-horizon evaluation, topology-versus-vector baselines in the spirit of prior GNN-versus-Koopman power-grid studies [@Nandanoori2022], dynamic topology, spectral/stability analysis, and latent-space control---plus v0.5.0 tooling and notebooks for deep-ensemble / latent-Gaussian uncertainty, hierarchical multi-resolution forecasting, auxiliary-spectral continuous generators, and sparsity / worst-case reconstruction losses (with an expanded GNN encoder zoo). Open artifacts include PyPI releases through v0.5.0, CI with coverage and notebook smoke tests, Sphinx docs (FAQ / troubleshooting), and a security policy.

The author uses the library to experiment with Koopman forecasting on networked systems and to lower the barrier for topology-aware Koopman models on standard PyG workflows. Zenodo archives are available through v0.5.0 (v0.5.0 DOI: \url{https://doi.org/10.5281/zenodo.21468449}; v0.4.0 DOI: \url{https://doi.org/10.5281/zenodo.21420623}; v0.3.0 DOI: \url{https://doi.org/10.5281/zenodo.21404269}; v0.2.0 DOI: \url{https://doi.org/10.5281/zenodo.21326273}).

# Conclusion

`KoopmanGraph` provides a PyG-native toolkit for topology-aware Koopman modeling of spatiotemporal graph dynamics behind a documented `fit`/`predict` workflow.

# AI usage disclosure

Generative AI tools assisted with source code generation and refactoring, test scaffolding, Sphinx documentation and tutorial notebooks, and drafting and copy-editing of this manuscript. The author reviewed, edited, and validated all AI-assisted outputs, made core design and architectural decisions, and takes full responsibility for the accuracy and correctness of the software and this paper.

# Acknowledgements

The author thanks the maintainers of PyTorch Geometric, PyKoopman, and DLKoopman for foundational open-source tooling. The author declares that there are no conflicts of interest. No external financial support was received for this work.

# References
