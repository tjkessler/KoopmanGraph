"""Tests for shared typing Protocols."""

from __future__ import annotations

from collections.abc import Callable
from typing import get_args, get_type_hints

import torch
from torch_geometric.data import Data

from koopman_graph import DMDBaseline, DMDcBaseline, EDMDBaseline, GraphKoopmanModel
from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.losses import rollout_multi_start_loss, rollout_sequence_loss
from koopman_graph.nn import GNNDecoder, GNNEncoder
from koopman_graph.protocols import (
    DynamicsMode,
    ForecastModel,
    SpectrumProvider,
    TrainableKoopmanModel,
    UncontrolledForecastModel,
    accepts_uncontrolled_data_predict,
)
from koopman_graph.training import FitHistory


def test_dynamics_mode_is_single_source_of_truth() -> None:
    """Verify DynamicsMode is shared via operator → protocols re-export."""
    from koopman_graph import adaptation, model
    from koopman_graph.adaptation import AdaptationMode
    from koopman_graph.operators import DynamicsMode as OperatorDynamicsMode

    assert AdaptationMode is DynamicsMode
    assert OperatorDynamicsMode is DynamicsMode
    assert model.DynamicsMode is DynamicsMode
    assert adaptation.AdaptationMode is DynamicsMode
    assert get_args(DynamicsMode) == ("discrete", "continuous")


def _small_graph_koopman_model() -> GraphKoopmanModel:
    """Build a tiny GraphKoopmanModel for Protocol conformance checks."""
    encoder = GNNEncoder(
        in_channels=2,
        hidden_channels=4,
        latent_dim=4,
        num_layers=1,
    )
    decoder = GNNDecoder(
        latent_dim=4,
        hidden_channels=4,
        out_channels=2,
        num_layers=1,
    )
    return GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=4,
        time_step=0.1,
    )


def _assert_trainable_members(model: object) -> None:
    """Assert structural TrainableKoopmanModel members without isinstance.

    ``TrainableKoopmanModel`` is not ``@runtime_checkable`` because submodule
    attributes live in ``nn.Module._modules`` and fail ``getattr_static``.
    """
    for name in (
        "fit",
        "predict",
        "spectrum",
        "encode",
        "resolve_delta_t",
        "train",
        "eval",
        "parameters",
    ):
        assert callable(getattr(model, name))
    for name in (
        "encoder",
        "decoder",
        "koopman",
        "time_step",
        "dynamics_mode",
        "control_dim",
        "training",
    ):
        assert hasattr(model, name)
    assert isinstance(model.training, bool)  # type: ignore[attr-defined]


def test_baselines_satisfy_forecast_model_protocol() -> None:
    """Verify classical baselines structurally implement ForecastModel."""
    for baseline in (
        DMDBaseline(time_step=0.1),
        DMDcBaseline(time_step=0.1),
        EDMDBaseline(time_step=0.1, polynomial_degree=1),
    ):
        assert isinstance(baseline, ForecastModel)
        assert isinstance(baseline, SpectrumProvider)
        assert callable(baseline.fit)
        assert callable(baseline.predict)
        assert callable(baseline.spectrum)


def test_graph_koopman_model_satisfies_spectrum_provider() -> None:
    """Neural model is a SpectrumProvider for dynamical_similarity."""
    model = _small_graph_koopman_model()
    assert isinstance(model, SpectrumProvider)
    assert isinstance(model, ForecastModel)


def test_uncontrolled_forecast_peers() -> None:
    """Verify uncontrolled Data-only peers vs controlled-only implementers."""
    dmd = DMDBaseline(time_step=0.1)
    edmd = EDMDBaseline(time_step=0.1, polynomial_degree=1)
    dmdc = DMDcBaseline(time_step=0.1)
    model = _small_graph_koopman_model()
    controlled = GraphKoopmanModel(
        encoder=GNNEncoder(
            in_channels=2,
            hidden_channels=4,
            latent_dim=4,
            num_layers=1,
        ),
        decoder=GNNDecoder(
            latent_dim=4,
            hidden_channels=4,
            out_channels=2,
            num_layers=1,
        ),
        latent_dim=4,
        time_step=0.1,
        control_dim=1,
    )

    for peer in (dmd, edmd, model):
        assert isinstance(peer, ForecastModel)
        assert isinstance(peer, UncontrolledForecastModel)
        assert accepts_uncontrolled_data_predict(peer)

    # Loose ForecastModel still holds; helper rejects controlled-only peers.
    assert isinstance(dmdc, ForecastModel)
    assert isinstance(dmdc, UncontrolledForecastModel)  # method presence only
    assert not accepts_uncontrolled_data_predict(dmdc)
    assert isinstance(controlled, ForecastModel)
    assert not accepts_uncontrolled_data_predict(controlled)


def test_baselines_are_forecast_only_not_trainable() -> None:
    """Verify classical baselines expose ForecastModel, not trainable members."""
    for baseline in (
        DMDBaseline(time_step=0.1),
        DMDcBaseline(time_step=0.1),
        EDMDBaseline(time_step=0.1, polynomial_degree=1),
    ):
        assert isinstance(baseline, ForecastModel)
        assert not hasattr(baseline, "encode")
        assert not hasattr(baseline, "koopman")
        assert not hasattr(baseline, "resolve_delta_t")


def test_graph_koopman_model_satisfies_forecast_model_protocol() -> None:
    """Verify GraphKoopmanModel structurally implements ForecastModel."""
    model = _small_graph_koopman_model()
    assert isinstance(model, ForecastModel)
    assert isinstance(model, UncontrolledForecastModel)
    assert accepts_uncontrolled_data_predict(model)
    assert callable(model.fit)
    assert callable(model.predict)
    assert callable(model.spectrum)


def test_graph_koopman_model_satisfies_trainable_protocol() -> None:
    """Verify GraphKoopmanModel exposes TrainableKoopmanModel members."""
    model = _small_graph_koopman_model()
    assert isinstance(model, ForecastModel)
    _assert_trainable_members(model)
    # Static typing surface remains importable for annotations.
    _: type[TrainableKoopmanModel] = TrainableKoopmanModel


def test_forecast_model_rejects_incomplete_objects() -> None:
    """Verify runtime_checkable rejects objects missing the façade."""

    class _Partial:
        def fit(self) -> None:
            return None

        def predict(self) -> list:
            return []

    assert not isinstance(_Partial(), ForecastModel)
    assert not isinstance(torch.nn.Linear(2, 2), ForecastModel)


def test_trainable_protocol_not_runtime_checkable() -> None:
    """Document that TrainableKoopmanModel rejects isinstance usage."""
    model = _small_graph_koopman_model()
    assert TrainableKoopmanModel._is_runtime_protocol is False  # type: ignore[attr-defined]
    try:
        isinstance(model, TrainableKoopmanModel)
    except TypeError:
        _assert_trainable_members(model)
        return
    raise AssertionError(
        "isinstance(TrainableKoopmanModel) should raise TypeError; "
        "use structural hasattr/callable checks instead"
    )


def test_rollout_losses_annotate_trainable_protocol() -> None:
    """Verify rollout losses type against TrainableKoopmanModel.

    :class:`~koopman_graph.model.GraphKoopmanModel` remains the intended
    implementer for these helpers (structural members, no encoder-only path).
    """
    for fn in (rollout_sequence_loss, rollout_multi_start_loss):
        hints = get_type_hints(fn)
        assert hints["model"] is TrainableKoopmanModel

    model = _small_graph_koopman_model()
    _assert_trainable_members(model)
    assert callable(model.encode)
    assert callable(model.resolve_delta_t)
    assert hasattr(model, "koopman")
    assert hasattr(model, "decoder")


def test_forecast_model_fit_return_divergence(
    make_snapshots: Callable[..., list[Data]],
) -> None:
    """Document fit returns: baselines → self; GraphKoopmanModel → FitHistory."""
    sequence = GraphSnapshotSequence(
        make_snapshots(num_timesteps=4, num_nodes=5, in_channels=2)
    )
    dmd = DMDBaseline(time_step=0.1)
    assert dmd.fit(sequence) is dmd
    edmd = EDMDBaseline(time_step=0.1, polynomial_degree=1)
    assert edmd.fit(sequence) is edmd

    controls = torch.randn(sequence.num_timesteps, 1)
    controlled = GraphSnapshotSequence(sequence.snapshots, control_inputs=controls)
    dmdc = DMDcBaseline(time_step=0.1)
    assert dmdc.fit(controlled) is dmdc

    model = _small_graph_koopman_model()
    history = model.fit(sequence, epochs=1)
    assert isinstance(history, FitHistory)
    assert history is not model
