KoopmanGraph
============

**KoopmanGraph** integrates Graph Neural Networks with Koopman operator theory
for spatiotemporal graph dynamics. The library lifts node features into a
latent space with topology-aware encoders, advances state via a learned
finite-dimensional Koopman operator, and decodes predictions back to physical
node features.

Version **0.5.0** adds ensemble and latent-Gaussian uncertainty, auxiliary
state-dependent continuous generators, physics-residual and sparsity losses,
SAGE/DiffConv/Transformer encoder-decoder pairs, kernel EDMD dictionaries,
and hierarchical graph forecasting. These extend the existing
discrete/continuous/networked operators, stability and control tools,
baselines, and graph benchmarks. See :doc:`capabilities` for the full
inventory and :doc:`architecture` for the public vs power-user API contract.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   quickstart
   faq
   capabilities
   tutorials
   architecture
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
