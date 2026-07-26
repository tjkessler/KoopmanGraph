KoopmanGraph
============

**KoopmanGraph** is a PyTorch Geometric library for learning topology-aware
Koopman autoencoders on graphs. GNN encoders lift node features into a latent
space, a learned linear operator advances those states, and a matching decoder
reconstructs physical node features for multi-step forecasting and spectral
analysis.

Version **0.6.0** extends the library with hypergraph encode/decode/operators,
self-adaptive topology, global/local and continuous networked operators,
symmetry-adapted orbit ties, SINDy / spectral clustering / topology-estimation
diagnostics, conformal UQ, and Koopman-MPC, on top of the v0.5.0 ensemble /
latent-Gaussian UQ, auxiliary-spectral generators, physics-residual and
sparsity losses, expanded GNN encoder zoo, and hierarchical forecasting.
See :doc:`capabilities` for the full inventory and :doc:`architecture` for
the public vs power-user API contract.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   quickstart
   faq
   capabilities
   data
   tutorials
   architecture
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
