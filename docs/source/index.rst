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

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   quickstart
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
