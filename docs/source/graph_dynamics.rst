Predicted graph dynamics
========================

Default ``graph_dynamics=None`` keeps the 0.14 encode / decode /
advance path: omitted future edges **hold last**. Opt-in
:class:`~koopman_graph.data.GraphDynamicsConfig` attaches a topology
head so a recursive loop can consume
:math:`\widehat{A}_{t+1}=g_{\phi}(z_t)`.

What ships
----------

When a config is passed, ``topology_head`` defaults to
``sparse_candidate`` (at most ``candidate_k`` destinations per node;
default 8) via
:class:`~koopman_graph.nn.SparseCandidateTopologyHead`.
``dense_mlp`` keeps
:class:`~koopman_graph.nn.PredictedTopologyHead` with an :math:`N`
ceiling of 64. The two heads are distinct from
``learn_topology="self_adaptive"`` /
:class:`~koopman_graph.nn.AdaptiveAdjacency` (static Graph WaveNet)
and cannot be combined when ``topology_head`` is not ``none``.

If a head is attached and ``recursive_training`` is true,
``fit`` / ``predict`` / ``evaluate`` / UQ use predicted sigmoid
weights on the candidate COO unless ``future_topologies`` or
``topology_policy="hold_last"`` is set. Evaluate on dynamic sequences
injects oracle futures only when that recursive path is off.

:class:`~koopman_graph.data.GraphStateSnapshot` is a supervision
record (features, edges, presence). It does not replace
:class:`~koopman_graph.data.GraphSnapshotSequence`. Changing node
count still requires :class:`~koopman_graph.data.EntityRemap` into a
finite :math:`N_{\max}` (injective placement; not automatic entity
resolution; no unbounded growth).

How to use it
-------------

.. code-block:: python

   from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
   from koopman_graph.data import GraphDynamicsConfig

   config = GraphDynamicsConfig(
       topology_head="sparse_candidate",
       recursive_training=True,
       candidate_k=8,
   )
   model = GraphKoopmanModel(
       encoder=GNNEncoder(3, 16, 8),
       decoder=GNNDecoder(8, 16, 3),
       latent_dim=8,
       graph_dynamics=config,
   )

``examples/50_graph_state_closure.ipynb`` is a wiring check versus
hold-last on a synthetic structural event. It is **not** a
learned-forecast claim.

Ceilings
--------

* Default ``graph_dynamics=None`` is unchanged 0.14 behavior.
* Dense logits have an :math:`N\le 64` ceiling. Sparse-candidate is
  the default for a reason.
* Presence BCE uses a linear per-node head when masks exist. Losses
  ignore inactive nodes; matvecs stay at capacity.
* Additive checkpoint key ``graph_dynamics``; ``FORMAT_VERSION``
  remains 1.

See :doc:`limitations` (graph-state fixed-union), :doc:`data`, and
:doc:`tutorials`.
