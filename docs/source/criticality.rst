Spectral-gap criticality monitor
================================

:func:`~koopman_graph.analysis.monitor_critical_transition` scores a
**sequence** of already-computed
:class:`~koopman_graph.spectrum_types.KoopmanSpectrum` objects. A
positive windowed rate means the closest-eigenvalue gap shrank. That
heuristic is **not** a topology-criticality certificate and not an
infrastructure early-warning score.

What ships
----------

For each spectrum the helper records

.. math::

   \gamma_t = \min_{i\neq j}\lvert\lambda_i(t)-\lambda_j(t)\rvert

and the windowed closure rate
:math:`(\gamma_{t-w+1}-\gamma_t)/(\tau_t-\tau_{t-w+1})`.
:class:`~koopman_graph.analysis.CriticalityReport` stores the gap
trajectory, the rate series, a near-defectivity flag
(:math:`\kappa(V)` non-finite or above
:data:`~koopman_graph.spectrum_types.CONDITION_WARN`), and
``max_gap_closure_rate`` (the scalar “score”).

Near-defectivity is a flag on the eigenbasis, not a Schur form and
not :class:`~koopman_graph.spectrum_types.DefectiveSpectrumError`.
The monitor does not raise on a singular stored basis; it sets the
flag.

This helper does **not** implement Ghosh, *Intelligent Systems with
Applications*, 2025 (DOI
`10.1016/j.iswa.2025.200575 <https://doi.org/10.1016/j.iswa.2025.200575>`_;
``Ghosh2025``). That paper is related literature and the honesty
ceiling.

How to use it
-------------

.. code-block:: python

   from koopman_graph.analysis import monitor_critical_transition
   from koopman_graph.spectrum_types import compute_spectrum

   spectra = [compute_spectrum(k_t, time_step=1.0) for k_t in operators]
   report = monitor_critical_transition(spectra, window=5)
   score = report.max_gap_closure_rate  # eigenvalue units / step

``examples/54_criticality_monitor.ipynb`` closes a two-eigenvalue gap
on a diagonal toy and shows the score rising versus a constant-gap
control. It is not a certificate.

Ceilings
--------

* Callers supply spectra. The helper does not fit a model or infer
  edges.
* Closest-pair distance need not be adjacent after magnitude sort.
* This is not
  :meth:`~koopman_graph.operators.KoopmanOperator.stability_certificate`
  and not :func:`~koopman_graph.analysis.resolvent_norm_grid`.
* Types stay on ``analysis.__all__``, off the root façade.

See :doc:`spectral_diagnostics`, :doc:`limitations`, and :doc:`api`.
