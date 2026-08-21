KoopmanGraph
============

**KoopmanGraph** is a PyTorch Geometric library for learning topology-aware
Koopman autoencoders on graphs. GNN encoders lift node features into a latent
space, a learned linear operator advances those states, and a matching decoder
reconstructs physical node features for multi-step forecasting and spectral
analysis.

Version **0.15.0** is an integration increment on the 0.14 stack: opt-in
closed-form identification, identity-bound benchmark manifests,
polynomial graph filters, Nyquist / conditioning diagnostics, and
labeled research MVPs (graph-state closure, cochain dynamics,
matrix-free algebra, residual-tube MPC). Homogeneous scientific
defaults and the linear latent operator contract are unchanged
relative to 0.14.0 (factory ``koopman=None`` still selects
``"pernode"``; ``sparsity="dense"``; AMP off). ``FORMAT_VERSION``
stays 1. See :doc:`capabilities` for the full inventory,
:doc:`limitations` for the honesty ceilings, and :doc:`architecture`
for the public vs power-user API contract.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   quickstart
   cli
   faq
   capabilities
   limitations
   data
   tutorials
   architecture
   graphon
   benchmarks
   identification
   spectral_diagnostics
   graph_dynamics
   matrix_free
   criticality
   time_conditioning
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
