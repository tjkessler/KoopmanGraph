Operator identification
=======================

Identification is an **opt-in** closed-form update of a discrete dense
per-node :math:`K`. Default ``fit(..., identification=None)`` remains
the Adam path and does not import
:mod:`koopman_graph.identification`.

What ships
----------

:class:`~koopman_graph.identification.IdentificationConfig` selects
ridge, TLS, or constrained least squares. Pass it to
:meth:`~koopman_graph.model.GraphKoopmanModel.fit` to alternate a
frozen-encoder :math:`K` write with encoder/decoder Adam steps.
:class:`~koopman_graph.identification.IdentificationReport` records
latent one-step / short-rollout mean squared error (MSE) and
:math:`\rho(K)`. Those scalars are not Haseli–Cortés, certified
ResDMD, or stability certificates.

Related helpers (all off root ``__all__``):

* :func:`~koopman_graph.identification.identify_operator` — closed-form
  solve on frozen :class:`~koopman_graph.identification.LatentPairs`
* :func:`~koopman_graph.identification.subspace_invariance_report` —
  finite-sample projection leakage :math:`\eta` (not a Haseli–Cortés
  certificate). ``fit`` does not populate the report
  ``invariance`` block; use
  ``evaluate(..., include_invariance=True)`` or the helper
* :func:`~koopman_graph.identification.select_resdmd_gated` — drop
  pre-scored dictionaries whose max finite-dictionary ResDMD residual
  exceeds :data:`~koopman_graph.identification.DEFAULT_RESDMD_GATE_TOLERANCE`
  (:math:`10^{-2}`). ``IdentificationConfig.gate_resdmd=True`` fills
  the report ``spectral`` block and does **not** abort ``fit``.
  :class:`~koopman_graph.training.ResDMDFitCallback` ``mode="gate"``
  may raise at fit end
* :func:`~koopman_graph.identification.identify_sparse_graph_factors`
  — sparse :math:`K_{\mathrm{self}}` / :math:`K_{\mathrm{nbr}}` on
  frozen encodings (not latent SINDy; not
  :class:`~koopman_graph.losses.KoopmanSparsityLoss`)
* :func:`~koopman_graph.identification.select_latent_rank` —
  truncated-SVD rank grid on frozen encodings (VAMP-2, ResDMD elbow,
  or stability-penalized held-out MSE). **Not** Ray Tune AutoML for
  encoder ``latent_dim``

Graph, hetero, continuous, and controlled operators raise.
``solver="varpro"`` is not implemented.

How to use it
-------------

.. code-block:: python

   from koopman_graph.identification import (
       IdentificationConfig,
       LatentPairs,
       build_identification_report,
       identify_operator,
       select_resdmd_gated,
   )

   pairs = LatentPairs(z_t=z_t, z_next=z_next)
   snapshot = identify_operator(
       pairs, IdentificationConfig(solver="ridge", ridge=0.0)
   )
   report = build_identification_report(pairs, snapshot, gate_resdmd=True)
   residual_max = report.spectral.residual_max

A teaching walkthrough that populates report fields, rejects a
polluted RMSE-only dictionary, and reads finite-sample leakage
:math:`\eta` on a rank-deficient line is
``examples/48_identification_invariance.ipynb``. Rank selection on a
linear Gaussian oracle is ``examples/53_latent_rank_selection.ipynb``.

Ceilings
--------

* Discrete dense per-node :math:`K` only. Networked / continuous maps
  stay on Adam (or a later increment).
* Closed-form ridge / TLS / constrained least squares is **not**
  ResKoopNet residual training (``Xu2025ResKoopNet``).
* Reports are finite-sample MSE and :math:`\rho(K)`, not certificates.
* :func:`~koopman_graph.identification.select_latent_rank` does not
  train an encoder per candidate and does not set
  :class:`~koopman_graph.model.GraphKoopmanModel` ``latent_dim``.
  :mod:`koopman_graph.tuning` remains caller-owned HPO.

See :doc:`faq` (identification versus Adam; rank versus Ray Tune),
:doc:`limitations` (Identification), and :doc:`api`.
