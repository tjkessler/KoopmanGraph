KoopmanGraph
============

**KoopmanGraph** is a PyTorch Geometric library for learning topology-aware
Koopman autoencoders on graphs. GNN encoders lift node features into a latent
space, a learned linear operator advances those states, and a matching decoder
reconstructs physical node features for multi-step forecasting and spectral
analysis.

Version **0.10.0** extends the library with continuous heterogeneous /
multiplex operators (opt-in per-type latent widths), finite-dictionary
ResDMD and resolvent-grid analysis, classical DMD-family peers and a
topology-blind VAMP-2 precursor, simplicial-1 encode/decode, Tier A
invariant geometry from ``Data.pos``, Bayesian Laplace UQ over operator
factors, and ``dask_prep`` materialize helpers — on top of the 0.9 hetero
RelGraph stack, 0.8 distributed trainers, and earlier hypergraph /
stability / conformal / MPC surfaces. See :doc:`capabilities` for the
full inventory, :doc:`limitations` for scope boundaries and the 0.11
roadmap, and :doc:`architecture` for the public vs power-user API contract.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   quickstart
   faq
   capabilities
   limitations
   data
   tutorials
   architecture
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
