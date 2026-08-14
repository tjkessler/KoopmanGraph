KoopmanGraph
============

**KoopmanGraph** is a PyTorch Geometric library for learning topology-aware
Koopman autoencoders on graphs. GNN encoders lift node features into a latent
space, a learned linear operator advances those states, and a matching decoder
reconstructs physical node features for multi-step forecasting and spectral
analysis.

Version **0.14.0** is a remaining-limits close-out: opt-in structure-preserving
operators, switched / mixture maps, equivariant and Hodge-structured
:math:`K`, protocol-matched leaderboard adapters, restricted portable
export, and analysis extras (wired finite ResDMD, dispersion, TDA).
Homogeneous scientific defaults and the linear latent operator contract
are unchanged relative to 0.13.0 (factory ``koopman=None`` still selects
``"pernode"``; ``sparsity="dense"``; AMP off). See :doc:`capabilities`
for the full inventory, :doc:`limitations` for the new honesty ceilings,
and :doc:`architecture` for the public vs power-user API contract.

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
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
