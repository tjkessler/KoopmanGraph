KoopmanGraph
============

**KoopmanGraph** is a PyTorch Geometric library for learning topology-aware
Koopman autoencoders on graphs. GNN encoders lift node features into a latent
space, a learned linear operator advances those states, and a matching decoder
reconstructs physical node features for multi-step forecasting and spectral
analysis.

Version **0.11.0** adds measured cross-topology transfer evaluation, directed
hypergraph incidence modes, Ray Train model-DDP fitting, fixed-union
presence-mask node churn, teaching traffic forecaster ports (AGCRN / MTGNN /
STGODE / GraphCast), in-repo sheaf and cell-complex encode/decode MVPs, a
GraphVAMP + synthetic molecular teaching path, and exact-automorphism
isotypic self-block ties — on top of the 0.10 hetero / ResDMD / VAMP-2 stack
and earlier distributed, hypergraph, stability, conformal, and MPC surfaces.
Homogeneous scientific defaults and the linear latent operator contract are
unchanged. See :doc:`capabilities` for the full inventory,
:doc:`limitations` for current scope boundaries, and :doc:`architecture`
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
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
