Time and parameter conditioning
===============================

Homogeneous sequences may carry a ``parameter_trajectory`` and
optional timestamps / controls. :class:`~koopman_graph.data.ConditioningContext`
is a **data record** for one index :math:`(\mu, t, u)`. It does not
select a factory operator.

What ships
----------

:func:`~koopman_graph.data.conditioning_at` reads those fields from a
:class:`~koopman_graph.data.GraphSnapshotSequence`. Optional
``period`` / ``num_phases`` fill ``phase_index`` via
:func:`~koopman_graph.data.diurnal_phase_index`.

:func:`~koopman_graph.data.diurnal_control_features` returns Fourier
sine/cosine columns for existing additive / bilinear
``control_inputs``. That recipe is not a calendar serializer and not
a checkpoint key.

Factory kinds stay distinct:

* ``koopman="switched"`` / ``"mixture"`` — latent-gated or piecewise
  LTI maps, **not** :math:`K(\mu)`
* ``koopman="parametric"`` —
  :class:`~koopman_graph.operators.ParametricKoopmanOperator`, the
  interpolant :math:`K(\mu)=\sum_j\alpha_j(\mu)K_j` (RBF or simplex
  weights). Convex combinations preserve dense / row-stochastic /
  doubly-stochastic factors; symplectic and other structural mixes
  raise. Export refuses the interpolant
* Default ``koopman=None`` still selects ``"pernode"``. Carrying
  :math:`\mu` does not change that default

Discrete sequences still require uniform :math:`\Delta t`
(``validate_uniform_discrete_increments``).
Gaps raise. Use ``dynamics_mode="continuous"`` and ``predict_at``,
or supply derivatives to
:class:`~koopman_graph.baselines.GEDMDBaseline`, for irregular
sampling. Irregular timestamps on a gEDMD sequence do not create
:math:`L`.

Related literature for the interpolant (cited, not re-proved):
Maćešić, Črnjarić-Žic, and Mezić, *SIAM J. Appl. Dyn. Syst.*, 2018
(DOI
`10.1137/17M1133610 <https://doi.org/10.1137/17M1133610>`_;
``Macesic2018Nonautonomous``).

How to use it
-------------

.. code-block:: python

   from koopman_graph.data import (
       conditioning_at,
       diurnal_control_features,
   )

   ctx = conditioning_at(sequence, index=3)
   mu = ctx.parameters  # shape (d_mu,) or None
   clock = diurnal_control_features(timestamps, period=24.0)

:func:`~koopman_graph.operators.leave_one_regime_out` compares the
parametric interpolant to pooled LTI on latent pairs. Heterogeneous
sequences have no calendar helper.

Ceilings
--------

* Discrete ``fit`` / ``predict_at`` reject non-uniform increments.
  Time-of-day helpers sit on existing control and switched APIs.
* ``parameter_trajectory`` is homogeneous ``(T, d_\mu)``. Units are
  caller-defined (dimensionless if unspecified).
* Types stay in :mod:`koopman_graph.data` (off root ``__all__``).
  ``data`` must not import ``nn``.

See :doc:`faq` (irregular :math:`\Delta t`; time of day),
:doc:`limitations`, :doc:`data`, and
``examples/12_irregular_sampling_continuous_time.ipynb``.
