"""Shared typing contracts for forecasting and trainable façades.

Power-user module: importable as ``koopman_graph.protocols``, documented in
architecture docs, and **not** re-exported in package ``__all__``.

See :class:`ForecastModel` for the loose classical baseline / model forecasting
façade, :class:`SpectrumProvider` for spectrum-only comparison surfaces,
:class:`UncontrolledForecastModel` for Data-only autonomous ``predict`` peers,
and :class:`TrainableKoopmanModel` for the training and metrics duck-typing
contract.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from typing import Any, Protocol, runtime_checkable

from torch import Tensor, nn
from torch_geometric.data import Data

from koopman_graph.operators import DynamicsMode, KoopmanOperatorContract
from koopman_graph.spectrum_types import KoopmanSpectrum

# ``DynamicsMode`` is defined next to ``Parameterization`` in
# :mod:`koopman_graph.operators.contract` (also available via
# :mod:`koopman_graph.operators`) and re-exported here so models, losses, and
# adaptation keep importing the canonical alias from protocols
# (``AdaptationMode`` aliases this type).


@runtime_checkable
class ForecastModel(Protocol):
    """Loose ``fit`` / ``predict`` / ``spectrum`` forecasting façade.

    Structural contract for classical baselines
    (:class:`~koopman_graph.baselines.DMDBaseline`,
    :class:`~koopman_graph.baselines.DMDcBaseline`,
    :class:`~koopman_graph.baselines.EDMDBaseline`) and
    :class:`~koopman_graph.model.GraphKoopmanModel`.

    **Not drop-in interchangeable at call sites.** Implementer signatures diverge
    (DMDc ``predict`` requires ``controls``; classical baselines accept ``Data``
    only; ``GraphKoopmanModel.predict`` also accepts tensors, optional topology /
    control kwargs; continuous ``spectrum`` may take ``delta_t``).
    ``fit`` return types also diverge: baselines return ``self``;
    ``GraphKoopmanModel.fit`` returns ``FitHistory``.
    ``@runtime_checkable`` verifies method presence only, not call signatures.

    For autonomous peers that share ``predict(initial_graph: Data, steps: int)``,
    prefer :class:`UncontrolledForecastModel` and
    :func:`accepts_uncontrolled_data_predict`. See the call-site matrix in
    :doc:`architecture`.

    This module is a power-user typing surface and is intentionally omitted from
    ``koopman_graph.__all__``.

    Notes
    -----
    Prefer importing from :mod:`koopman_graph.protocols` for type annotations.
    Training and multi-horizon evaluation that need encode / Module APIs use
    :class:`TrainableKoopmanModel`.
    """

    def fit(self, *args: Any, **kwargs: Any) -> Any:
        """Fit the forecasting model from snapshot data.

        Parameters
        ----------
        *args, **kwargs
            Implementer-specific training inputs and options.

        Returns
        -------
        Any
            **Not interchangeable across peers.** Classical baselines return
            ``self`` (sklearn chaining);
            :meth:`~koopman_graph.model.GraphKoopmanModel.fit` returns
            :class:`~koopman_graph.training.FitHistory`. See the call-site
            matrix in :doc:`architecture`.
        """
        ...

    def predict(self, *args: Any, **kwargs: Any) -> list[Data]:
        """Autoregressively predict future graph snapshots.

        Parameters
        ----------
        *args, **kwargs
            Implementer-specific rollout inputs. Common pattern:
            ``(initial_graph, steps, ...)``. Call sites are **not** portable
            across all :class:`ForecastModel` implementers; see
            :class:`UncontrolledForecastModel` for the autonomous Data-only peer
            set.

        Returns
        -------
        list of Data
            Predicted graph snapshots.
        """
        ...

    def spectrum(self, *args: Any, **kwargs: Any) -> KoopmanSpectrum:
        """Return the learned operator (or baseline) spectrum.

        Parameters
        ----------
        *args, **kwargs
            Implementer-specific options (for example continuous-mode
            ``delta_t`` on :class:`~koopman_graph.model.GraphKoopmanModel`).

        Returns
        -------
        KoopmanSpectrum
            Eigendecomposition and continuous-time mode characteristics.
        """
        ...


@runtime_checkable
class SpectrumProvider(Protocol):
    """Narrow façade for anything that can yield a :class:`KoopmanSpectrum`.

    Satisfied by classical baselines and
    :class:`~koopman_graph.model.GraphKoopmanModel`. Used by
    :func:`~koopman_graph.analysis.dynamical_similarity` so comparisons do not
    require a concrete neural model type. Call-site kwargs still diverge:
    continuous neural ``spectrum`` may accept ``delta_t``; baselines take none.

    Prefer precomputed :class:`KoopmanSpectrum` values when comparing offline
    spectra; use this Protocol when comparing fitted forecasting façades.

    Notes
    -----
    This module is a power-user typing surface and is intentionally omitted from
    ``koopman_graph.__all__``.
    """

    def spectrum(self, *args: Any, **kwargs: Any) -> KoopmanSpectrum:
        """Return a Koopman spectrum for the fitted operator.

        Parameters
        ----------
        *args, **kwargs
            Implementer-specific options (for example continuous-mode
            ``delta_t``).

        Returns
        -------
        KoopmanSpectrum
            Eigendecomposition and continuous-time mode characteristics.
        """
        ...


@runtime_checkable
class UncontrolledForecastModel(Protocol):
    """Narrow forecasting peer for autonomous Data-only ``predict``.

    Documents the interchangeable call site::

        predictions = model.predict(initial_graph, steps)

    where ``initial_graph`` is a PyG ``Data`` snapshot. Satisfied (as a
    signature superset or exact match) by
    :class:`~koopman_graph.baselines.DMDBaseline`,
    :class:`~koopman_graph.baselines.EDMDBaseline`, and
    :class:`~koopman_graph.model.GraphKoopmanModel` when called with
    ``Data`` + ``steps`` only.

    :class:`~koopman_graph.baselines.DMDcBaseline` is **not** in this peer set:
    its ``predict`` requires a ``controls`` sequence. Use loose
    :class:`ForecastModel` for DMDc, or call with explicit controls.

    Notes
    -----
    ``@runtime_checkable`` still checks method presence only. Prefer
    :func:`accepts_uncontrolled_data_predict` when excluding controlled-only
    implementers at runtime.
    """

    def fit(self, *args: Any, **kwargs: Any) -> Any:
        """Fit the forecasting model from snapshot data.

        Parameters
        ----------
        *args, **kwargs
            Implementer-specific training inputs and options.

        Returns
        -------
        Any
            **Not interchangeable across peers.** Classical baselines return
            ``self``;
            :meth:`~koopman_graph.model.GraphKoopmanModel.fit` returns
            :class:`~koopman_graph.training.FitHistory`. See the call-site
            matrix in :doc:`architecture`.
        """
        ...

    def predict(self, initial_graph: Data, steps: int) -> list[Data]:
        """Autoregressively predict future graph snapshots without controls.

        Parameters
        ----------
        initial_graph : Data
            Initial PyG graph snapshot.
        steps : int
            Number of future snapshots to predict.

        Returns
        -------
        list of Data
            Predicted graph snapshots.
        """
        ...

    def spectrum(self, *args: Any, **kwargs: Any) -> KoopmanSpectrum:
        """Return the learned operator (or baseline) spectrum.

        Parameters
        ----------
        *args, **kwargs
            Implementer-specific options. Classical baselines take no kwargs;
            continuous :class:`~koopman_graph.model.GraphKoopmanModel` may accept
            ``delta_t``.

        Returns
        -------
        KoopmanSpectrum
            Eigendecomposition and continuous-time mode characteristics.
        """
        ...


def accepts_uncontrolled_data_predict(model: object) -> bool:
    """Return whether ``model.predict(data, steps)`` is a valid call site.

    Combines :class:`ForecastModel` method presence with signature inspection
    and optional ``control_dim`` metadata:

    * implementers whose ``predict`` requires ``controls`` (no default) are
      excluded (e.g. DMDc);
    * models with ``control_dim > 0`` are excluded even when ``controls`` is
      optional in the signature (controlled GraphKoopmanModel).

    Use this instead of ``isinstance(..., UncontrolledForecastModel)`` when
    controlled-only implementers must be rejected at runtime.

    Parameters
    ----------
    model : object
        Candidate forecasting façade.

    Returns
    -------
    bool
        ``True`` when the uncontrolled Data-only ``predict`` peer contract holds.
    """
    if not isinstance(model, ForecastModel):
        return False
    try:
        signature = inspect.signature(model.predict)
    except (TypeError, ValueError):
        return False
    controls = signature.parameters.get("controls")
    if controls is not None and controls.default is inspect.Parameter.empty:
        return False
    control_dim = getattr(model, "control_dim", None)
    return not (isinstance(control_dim, int) and control_dim > 0)


class TrainableKoopmanModel(ForecastModel, Protocol):
    """Contract for duck-typed training and forecast evaluation.

    Documents the members :mod:`koopman_graph.training` and
    :func:`~koopman_graph.metrics.evaluate_forecast` rely on. Satisfied by
    :class:`~koopman_graph.model.GraphKoopmanModel`. Classical baselines satisfy
    only :class:`ForecastModel` (they are not ``nn.Module`` training targets).

    Required members
    ----------------
    * Forecasting façade from :class:`ForecastModel` (``fit``, ``predict``,
      ``spectrum``)
    * ``encode`` — primary latent-lifting API (no encoder-only fallback)
    * ``resolve_delta_t`` — model-backed continuous interval policy
    * ``encoder``, ``decoder``, ``koopman`` — composed submodules / operator
    * ``time_step``, ``dynamics_mode``, ``control_dim`` — training metadata
    * ``train`` / ``eval`` / ``training`` / ``parameters`` / ``__call__`` —
      ``nn.Module`` façade used by epoch loops and evaluation

    Runtime checking
    ----------------
    Runtime ``isinstance`` is **disabled** for this Protocol (see
    ``_is_runtime_protocol = False`` below). Submodule attributes
    (``encoder`` / ``decoder`` / ``koopman``) live in ``nn.Module._modules`` and
    are invisible to ``getattr_static``-based checks inherited from
    ``@runtime_checkable`` parents. Prefer structural smoke tests
    (``hasattr`` / ``callable``) or static type checkers. :class:`ForecastModel`
    remains ``@runtime_checkable`` for method-only façade checks.

    Notes
    -----
    :class:`~koopman_graph.env.GraphKoopmanEnv` and checkpoint serialization
    remain hard-typed to :class:`~koopman_graph.model.GraphKoopmanModel`
    because they reconstruct architecture configs and reach encoder/decoder
    internals. This Protocol is intentionally **not** in ``__all__``.
    """

    encoder: nn.Module
    decoder: nn.Module
    koopman: KoopmanOperatorContract
    time_step: float
    dynamics_mode: DynamicsMode
    control_dim: int
    training: bool

    def encode(
        self,
        x_or_data: Tensor | Data,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        """Lift graph node features into the Koopman latent space.

        Parameters
        ----------
        x_or_data : Tensor or Data
            Node features or a PyG ``Data`` snapshot.
        edge_index : Tensor or None, optional
            Edge index required when ``x_or_data`` is a tensor.
        edge_weight : Tensor or None, optional
            Optional scalar edge weights for tensor input.

        Returns
        -------
        Tensor
            Latent node features.
        """
        ...

    def resolve_delta_t(
        self,
        delta_t: float | Tensor | None = None,
    ) -> float | Tensor:
        """Resolve the continuous integration interval for this model.

        Parameters
        ----------
        delta_t : float, Tensor, or None, optional
            Explicit interval. When ``None``, returns the model default.

        Returns
        -------
        float or Tensor
            Resolved integration interval.
        """
        ...

    def train(self, mode: bool = True) -> Any:
        """Set training mode (``nn.Module`` façade).

        Parameters
        ----------
        mode : bool, optional
            If ``True``, enable training mode. Default is ``True``.

        Returns
        -------
        Any
            Typically ``self``.
        """
        ...

    def eval(self) -> Any:
        """Set evaluation mode (``nn.Module`` façade).

        Returns
        -------
        Any
            Typically ``self``.
        """
        ...

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        """Yield trainable parameters (``nn.Module`` façade).

        Parameters
        ----------
        recurse : bool, optional
            If ``True``, include submodule parameters. Default is ``True``.

        Returns
        -------
        Iterator of Parameter
            Trainable parameter iterator.
        """
        ...

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Single-step forward pass used by reconstruction losses.

        Parameters
        ----------
        *args, **kwargs
            Implementer-specific forward inputs (snapshot, control, delta_t).

        Returns
        -------
        Any
            Typically decoded node features.
        """
        ...


# Subclassing ``@runtime_checkable`` ForecastModel would otherwise keep
# isinstance enabled; Module-stored submodules make those checks unreliable.
TrainableKoopmanModel._is_runtime_protocol = False  # type: ignore[attr-defined]
