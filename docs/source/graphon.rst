Graphon sampling and continuum limits
=====================================

A **graphon** is a symmetric measurable kernel
:math:`W:[0,1]^2\to[0,1]` that represents a limit object for dense graph
sequences (Lovász and Szegedy, 2006; DOI
`10.1016/j.jctb.2006.05.001 <https://doi.org/10.1016/j.jctb.2006.05.001>`_).
KoopmanGraph uses that viewpoint for **transfer experiments at multiple**
:math:`N`, not as theorems proved in this repository.

What ships
----------

:func:`~koopman_graph.operators.sample_graphon_adjacency` draws a
symmetric 0/1 adjacency from a small family of kernels:

* ``kernel="constant"`` — independent edges with probability ``density``
* ``kernel="product"`` — :math:`W(u,v)=u v` after sorting uniform latent
  positions

The helper returns an undirected ``edge_index`` (both orientations). A
shared :math:`K_{\mathrm{self}}` with graphon-sampled :math:`A` is the
intended transfer setup: train at one :math:`N`, evaluate at another,
and report measured error. The library does **not** claim a proven
continuum-limit theorem, a unique graphon estimator, or sparse-graph
graphon theory (Borgs–Chayes–Lovász and related sparse limits).

How to use it
-------------

Pair the sampled adjacency with the factory default per-node operator or
with ``koopman="graph"``:

.. code-block:: python

   import torch
   from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
   from koopman_graph.operators import sample_graphon_adjacency
   from torch_geometric.data import Data

   edge_index = sample_graphon_adjacency(16, kernel="constant", density=0.3)
   x = torch.randn(16, 3)
   graph = Data(x=x, edge_index=edge_index)
   model = GraphKoopmanModel(
       encoder=GNNEncoder(3, 16, 8),
       decoder=GNNDecoder(8, 16, 3),
       latent_dim=8,
       koopman="graph",
   )

Ceilings
--------

* Sampling is dense :math:`O(N^2)` and is a teaching / diagnostic path.
* User-supplied injective remaps
  (:func:`~koopman_graph.data.remap_node_features`) grow a fixed union;
  they are **not** entity resolution across unrelated universes.
* Theory bounds in Lovász–Szegedy (2006) are **cited**, not re-proved.

See :doc:`limitations` for the remaining continuum-limit honesty boundary
and :doc:`capabilities` for the operator inventory.
