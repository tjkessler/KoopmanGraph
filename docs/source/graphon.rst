Graphon sampling and continuum limits
=====================================

A **graphon** is a symmetric measurable kernel
:math:`W:[0,1]^2\to[0,1]` that represents a limit object for dense graph
sequences (Lovász and Szegedy, 2006; DOI
`10.1016/j.jctb.2006.05.001 <https://doi.org/10.1016/j.jctb.2006.05.001>`_).
KoopmanGraph uses that viewpoint for **transfer experiments at multiple**
:math:`N` and for a dense teaching estimator of two named kernels. Bounds
in the citations are **cited**, not proved in this repository.

What ships
----------

:func:`~koopman_graph.operators.sample_graphon_adjacency` draws a
symmetric 0/1 adjacency from a small family of kernels:

* ``kernel="constant"`` — independent edges with probability ``density``
  (dimensionless, in :math:`[0, 1]`)
* ``kernel="product"`` — :math:`W(u,v)=u v` at uniform latent positions,
  or at caller-supplied ``positions`` in :math:`[0, 1]`

The helper returns an undirected ``edge_index`` (both orientations).
Latent coordinates are **not** sorted before sampling.

:func:`~koopman_graph.operators.estimate_graphon` fits a kernel on
**aligned** homogeneous graphs that share a finite node count
:math:`N`. The mean loopless adjacency is the sufficient statistic:

* ``kernel_family="constant"`` — mean off-diagonal edge probability
* ``kernel_family="product"`` — degree scores
  :math:`\hat u_i = 2 d_i / (N-1)`, clipped to :math:`[0, 1]`
* ``kernel_family="low_rank"`` — truncated SVD of the mean adjacency,
  clipped to :math:`[0, 1]` (a sketch; **not** an oracle-quality or
  USVT consistency claim)

A shared :math:`K_{\mathrm{self}}` with graphon-sampled :math:`A` is the
intended transfer setup: train at one :math:`N`, evaluate at another,
and report measured error. Recovering a teaching kernel at one
:math:`N` is **not** a transferability certificate on arbitrary sparse
sensor graphs (Ruiz, Chamon, and Ribeiro, 2023; DOI
`10.1109/TSP.2023.3297848 <https://doi.org/10.1109/TSP.2023.3297848>`_).
The library does **not** claim a proven continuum-limit theorem, a
unique graphon identifier, or sparse-graph graphon theory
(Borgs–Chayes–Lovász and related sparse limits).

How to use it
-------------

Pair the sampled adjacency with the factory default per-node operator or
with ``koopman="graph"``:

.. code-block:: python

   import torch
   from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
   from koopman_graph.operators import (
       estimate_graphon,
       sample_graphon_adjacency,
   )
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
   fit = estimate_graphon([graph], kernel_family="constant")
   density = fit.density  # dimensionless mean edge probability

Ceilings
--------

* Sampling and estimation are dense :math:`O(N^2)` teaching / diagnostic
  paths. :func:`~koopman_graph.operators.estimate_graphon` refuses
  :math:`N > 256` (``MAX_GRAPHON_NODES``) and refuses mixed or unbounded
  node counts.
* Graphs must share :math:`N` (aligned teaching estimator). Unaligned
  multi-:math:`N` histogram estimators are out of scope.
* User-supplied injective remaps
  (:class:`~koopman_graph.data.EntityRemap` /
  :func:`~koopman_graph.data.remap_node_features`) grow a fixed union;
  they are **not** entity resolution across unrelated universes.
* Theory bounds in Lovász–Szegedy (2006) are **cited**, not re-proved.
  Sparse-graph limits are refused. Size-transfer remains “sample at two
  :math:`N`”; estimation quality is not GNN/graphon transferability
  (Ruiz, Chamon, and Ribeiro, 2023).

See :doc:`limitations` for the remaining continuum-limit honesty boundary
and :doc:`capabilities` for the operator inventory. Related guides:
:doc:`graph_dynamics` (fixed-union remap), :doc:`time_conditioning`
(parameter records), and :doc:`benchmarks` (identity-bound fixtures).
