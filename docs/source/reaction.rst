Reaction networks
=================

:mod:`koopman_graph.datasets.reaction` turns mass-action reaction
networks into graph dynamical systems with a known answer key. Species
are nodes, stoichiometry sets the coupling, and each card stores the
Jacobian spectrum at equilibrium, so a learned operator can be checked
against true kinetic timescales and conservation laws as well as
forecast error. The cards are **diagnostic / teaching** datasets of a
few species, not a production kinetics benchmark.

What ships
----------

:class:`~koopman_graph.datasets.ReactionNetworkBenchmark` lists six
cards (:meth:`~koopman_graph.datasets.ReactionNetworkBenchmark.dataset_ids`):
a linear chain, Michaelis-Menten, the Brusselator, the van de Vusse
CSTR, a seeded random network, and an H\ :sub:`2`/O\ :sub:`2` mechanism
imported with the optional ``[cantera]`` extra. Each card is a JSON
resource (:meth:`~koopman_graph.datasets.ReactionNetworkBenchmark.load_card`)
with the generator, simulator, time step, initial ranges, feature
transform, graph encoding, units, license, limitations, and an
``oracle`` block.

* :meth:`~koopman_graph.datasets.ReactionNetworkBenchmark.generate`
  draws seeded initial conditions and integrates every trajectory on
  the card's grid (``float64``). ``simulator="ssa"`` gives Gillespie
  trajectories instead.
* :meth:`~koopman_graph.datasets.ReactionNetworkBenchmark.split` splits
  by contiguous trajectory blocks (70/10/20 by default).
* :meth:`~koopman_graph.datasets.reaction.ReactionDataset.sequence`
  returns one trajectory as a species-node
  :class:`~koopman_graph.GraphSnapshotSequence`.
  ``graph_encoding="species_influence"`` gives a directed edge
  :math:`s \to t` when :math:`s` is a reactant of a reaction that
  changes :math:`t`; ``"reaction_hyperedge"`` adds one hyperedge per
  reaction. :func:`~koopman_graph.datasets.reaction.species_reaction_bipartite`
  builds the species-reaction bipartite ``HeteroData``.
* :class:`~koopman_graph.datasets.reaction.LogConcentrationTransform`
  maps concentrations to :math:`\log \max(c, \varepsilon)`. Four cards
  use it (floor :math:`10^{-8}`, or :math:`10^{-12}` for Cantera); the
  Brusselator and van de Vusse cards keep raw concentrations.

Oracles
-------

For a closed system :math:`\dot c = N\,r(c)` the card stores the
equilibrium reached from the middle of the initial ranges, the
conservation laws :math:`W` with :math:`W N = 0`, and the eigenvalues of
:math:`J = N\,\partial r/\partial c` there. A discrete operator with step
:math:`\Delta t` should have :math:`e^{\lambda \Delta t}`, and each
conservation law adds the eigenvalue :math:`1`. For nonlinear networks
the oracle is a **linearization** at one equilibrium: it describes the
late-time approach only, and each trajectory's conserved totals select
its own equilibrium. Recompute it per trajectory with
:func:`~koopman_graph.datasets.reaction.equilibrium` and
:func:`~koopman_graph.datasets.reaction.oracle_spectrum`.

:func:`~koopman_graph.datasets.reaction.element_balance_violation`
returns the element drift :math:`E (c_t - c_0)` of a decoded rollout.
In log coordinates the floor alone adds :math:`E\,\varepsilon\mathbf{1}`,
so drift below that bias is not model error.

Recommended workflow
--------------------

1. Fit a graph-coupled model (``koopman="graph"`` on the
   ``species_influence`` graph). Shared encoder weights make the model
   equivariant under graph automorphisms; appending a one-hot species
   index to the node features breaks a symmetry the kinetics do not
   have.
2. Report forecast error at several horizons next to ``hold_last``,
   flat DMD, and EDMD fitted on pairs pooled over trajectories. The
   0.16.0 classical baselines fit one sequence; concatenating
   trajectories adds pairs across their boundaries.
3. Refit the operator on frozen encodings by ridge regression. A much
   better refit means the encoder is adequate and the end-to-end
   operator is not.
4. Gate the spectrum with :func:`~koopman_graph.analysis.resdmd` on
   the frozen encodings, drop Nyquist-adjacent modes
   (:func:`~koopman_graph.analysis.implied_timescales`), and compare
   timescales inside :math:`[2\Delta t, T\Delta t]` with the oracle.
5. Enforce atom balance, if it must hold exactly, in raw concentration
   coordinates with
   :class:`~koopman_graph.nn.LinearConservingDecoder` and a shared atom
   budget.

Limitations
-----------

See also :doc:`limitations`.

* Log features saturate at the floor. A decaying species falls linearly
  in log coordinates until it stops at the floor, which a linear
  operator can only approximate with a near-unit eigenvalue. Gated
  spectra can therefore keep slow modes with no kinetic counterpart.
* A fixed operator encodes the training rate constants. Shifts in rate
  constants or temperature need the rate as an input.
* Fast modes that decay within one sample are not resolved at the
  card's :math:`\Delta t`.
* In 0.16.0, ``project_linear_conservation`` detaches its input, so a
  :class:`~koopman_graph.nn.LinearConservingDecoder` inside
  :meth:`~koopman_graph.GraphKoopmanModel.fit` passes no gradient.
  Apply its :meth:`~koopman_graph.nn.LinearConservingDecoder.project`
  to decoded states after training instead.

Tutorials
---------

* `55_reaction_network_oracles.ipynb <https://github.com/tjkessler/KoopmanGraph/blob/main/examples/55_reaction_network_oracles.ipynb>`_:
  the card oracles, exact DMD on the linear chain, and the
  Michaelis-Menten slow mode.
* `56_reaction_network_koopman.ipynb <https://github.com/tjkessler/KoopmanGraph/blob/main/examples/56_reaction_network_koopman.ipynb>`_:
  species-graph, hypergraph, and bipartite encodings on
  Michaelis-Menten, ResDMD-gated timescales against the oracle, atom
  balance against the floor bias, a conserving projection, and a
  :math:`k_\mathrm{cat}` shift.
