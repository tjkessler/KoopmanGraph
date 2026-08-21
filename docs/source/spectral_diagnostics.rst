Spectral diagnostics
====================

:class:`~koopman_graph.spectrum_types.SpectralDiagnostics` attaches
conditioning and sampling fields to a
:class:`~koopman_graph.spectrum_types.KoopmanSpectrum`. The numbers
describe the **computed eigenbasis**. They are not a finite-horizon
bound on :math:`\|K^{k}\|` and not a sampling-theorem identification
result.

What ships
----------

:func:`~koopman_graph.spectrum_types.compute_spectrum` (and
:func:`~koopman_graph.spectrum_types.compute_generator_spectrum`)
always populate optional ``diagnostics``:

* :math:`\kappa(V)` after column normalization
* per-mode Wilkinson sensitivities :math:`\kappa_i`
* Frobenius departure from normality and a scale-free relative variant
* discrete Nyquist frequency :math:`1/(2\Delta t)` in **cycles per
  unit time** (``None`` on generator spectra)
* per-mode aliasing flags and :math:`\operatorname{sign}(\operatorname{Re}\lambda)`

Discrete assembly emits one ``UserWarning`` when any mode is
Nyquist-adjacent. Frequencies remain principal values; phase
unwrapping does not recover aliases.

:meth:`~koopman_graph.spectrum_types.KoopmanSpectrum.mode_amplitudes`
warns when :math:`\kappa(V)` exceeds
:data:`~koopman_graph.spectrum_types.CONDITION_WARN` (:math:`10^{6}`)
and still solves :math:`Va=z^{\top}`. Singular :math:`V` raises
:class:`~koopman_graph.spectrum_types.DefectiveSpectrumError` (a
``LinAlgError``) with a Schur-subspace hint.

mpEDMD is a **baseline**
(:class:`~koopman_graph.baselines.MpEDMDBaseline`). It does not
replace these Euclidean conditioning fields on a general directed
:math:`K_{\mathrm{eff}}`.

How to use it
-------------

.. code-block:: python

   import torch
   from koopman_graph.spectrum_types import compute_spectrum

   operator = torch.tensor([[1.0, 10.0], [0.0, 2.0]], dtype=torch.float64)
   spectrum = compute_spectrum(operator, time_step=1.0)
   kappa = spectrum.diagnostics.eigenvector_condition
   nyquist = spectrum.diagnostics.nyquist_frequency  # 0.5 cycles / unit time

A non-normal / Nyquist teaching notebook is
``examples/51_spectral_diagnostics.ipynb``.

Ceilings
--------

* Diagnostics are optional on manually constructed
  :class:`~koopman_graph.spectrum_types.KoopmanSpectrum` objects
  (backward compatible).
* Eligible Kronecker graph spectra may omit ``diagnostics``.
* A ``CONDITION_WARN`` notice is not a transient-growth certificate.
* Generator spectra have no Nyquist frequency: the placeholder
  :math:`\Delta t` is not a sampling interval.

See :doc:`limitations` (Nyquist / non-normal), :doc:`criticality`,
and :doc:`api`.
