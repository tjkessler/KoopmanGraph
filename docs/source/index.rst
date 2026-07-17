KoopmanGraph
============

**KoopmanGraph** integrates Graph Neural Networks with Koopman operator theory
for spatiotemporal graph dynamics. The library lifts node features into a
latent space with topology-aware encoders, advances state via a learned
finite-dimensional Koopman operator, and decodes predictions back to physical
node features.

Version 0.2 adds spectral analysis, model checkpointing, temporal evaluation
metrics, operator stability options, edge-weight support, classical DMD/EDMD/DMDc
baselines, Koopman-with-control dynamics, dynamic-topology sequences, and
advanced training utilities including windowed mini-batching.

Version 0.3.0 adds structural stability parameterizations, continuous-time
generator learning, online RLS adaptation, hybrid physics observables,
dynamical similarity metrics, and a Gymnasium RL wrapper. It also standardizes
the public API surface: symmetric GCN/GAT encoder–decoder pairs, preferred
``encode``, explicit ``MultiTrajectory`` fitting, shared autoregressive
rollout primitives, optional ``koopman=`` operator injection, frozen public
result types, classical baseline scaffolding with a clear ``ForecastModel``
call-site contract, documented power-user modules (``graph_utils``,
``protocols``), capability packages (``training``, ``data``, ``operators``,
``nn``, ``analysis``, ``baselines``), a **thin** root façade (core workflow in
``__all__``; specialized helpers via capability modules), and discrete
spectrum plotting with unit-disk / data-zoom views.

Version 0.4.0 extends the forecasting stack with networked
``GraphKoopmanOperator`` coupling, delay / Hankel encoder embeddings, bilinear
control-affine terms, latent Kalman observation / imputation, nonlinear and
chaotic graph benchmarks, and lightweight STGCN / DCRNN / Graph WaveNet
reference forecasters for protocol-matched comparisons.
See :doc:`architecture` for the public vs power-user layering contract, the
thin-façade keep/demote inventory, and the package-layout nesting policy.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   quickstart
   architecture
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
