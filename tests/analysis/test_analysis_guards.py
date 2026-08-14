"""Coverage and error-path tests for :mod:`koopman_graph.analysis`."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph import (
    GNNDecoder,
    GNNEncoder,
    GraphKoopmanModel,
)
from koopman_graph.analysis import (
    calibrate_anomaly_threshold,
    compute_generator_spectrum,
    detect_anomaly,
    discrete_spectrum_at_delta_t,
    plot_spectrum,
)
from koopman_graph.analysis.anomaly import AnomalyDetectionResult
from koopman_graph.analysis.similarity import (
    _resolve_num_modes,
    _subspace_angle_distance,
    _wasserstein_magnitude_distance,
    resolve_spectrum,
)
from koopman_graph.baselines import DMDcBaseline
from koopman_graph.data import GraphSnapshotSequence, MultiTrajectory
from koopman_graph.datasets.topology import TopologyPayload
from koopman_graph.metrics import (
    evaluate_forecast,
    masked_mae,
    masked_mape,
    masked_rmse,
)
from koopman_graph.protocols import accepts_uncontrolled_data_predict
from koopman_graph.spectrum_types import KoopmanSpectrum
from koopman_graph.training.inputs import (
    _classify_trajectory_items,
    resolve_validation_sequences,
)


def _edge_index() -> torch.Tensor:
    return torch.tensor([[0, 1], [1, 0]], dtype=torch.long)


def _tiny_model(
    *,
    latent_dim: int = 4,
    control_dim: int = 0,
    dynamics_mode: str = "discrete",
    parameterization: str = "dense",
    physics_dim: int = 0,
    physics_preset: str | None = None,
) -> GraphKoopmanModel:
    gnn_dim = latent_dim - physics_dim
    encoder = GNNEncoder(2, 8, gnn_dim)
    decoder = GNNDecoder(latent_dim, 8, 2)
    return GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=latent_dim,
        time_step=0.1,
        control_dim=control_dim,
        dynamics_mode=dynamics_mode,
        koopman_parameterization=parameterization,
        physics_dim=physics_dim,
        physics_preset=physics_preset,
    )


def _sequence(
    *,
    num_timesteps: int = 4,
    with_weights: bool = False,
    with_masks: bool = False,
    with_timestamps: bool = False,
    control_dim: int = 0,
) -> GraphSnapshotSequence:
    edge = _edge_index()
    weight = torch.ones(edge.shape[1]) if with_weights else None
    snapshots = [
        Data(
            x=torch.randn(2, 2),
            edge_index=edge,
            **({"edge_weight": weight} if weight is not None else {}),
        )
        for _ in range(num_timesteps)
    ]
    kwargs: dict[str, Any] = {}
    if with_masks:
        kwargs["observation_masks"] = torch.ones(num_timesteps, 2, dtype=torch.bool)
    if with_timestamps:
        kwargs["timestamps"] = torch.arange(num_timesteps, dtype=torch.float32)
    if control_dim > 0:
        kwargs["control_inputs"] = torch.randn(num_timesteps, control_dim)
    return GraphSnapshotSequence(snapshots, **kwargs)


def test_analysis_metrics_protocol_and_training_gaps() -> None:
    with pytest.raises(ValueError, match="square matrix"):
        compute_generator_spectrum(torch.randn(2, 3))
    with pytest.raises(ValueError, match="non-empty"):
        compute_generator_spectrum(torch.empty(0, 0))
    with pytest.raises(TypeError, match="floating-point or complex"):
        compute_generator_spectrum(torch.ones(2, 2, dtype=torch.int64))
    with pytest.raises(ValueError, match="delta_t must be positive"):
        discrete_spectrum_at_delta_t(torch.eye(2), 0.0)

    spectrum = compute_generator_spectrum(torch.eye(2) * -0.5)
    spectrum_b = compute_generator_spectrum(torch.eye(2) * -0.4)
    threshold = calibrate_anomaly_threshold(
        [spectrum, spectrum_b],
        method="mean_std",
        k=1.0,
    )
    result = detect_anomaly([spectrum], spectrum, threshold=threshold)
    assert isinstance(result, AnomalyDetectionResult)

    class NoSig:
        def spectrum(self, *args: Any, **kwargs: Any) -> KoopmanSpectrum:
            return spectrum

    import inspect

    monkey_sig = pytest.MonkeyPatch()
    monkey_sig.setattr(
        inspect,
        "signature",
        lambda *_a, **_k: (_ for _ in ()).throw(TypeError("no signature")),
    )
    try:
        assert resolve_spectrum(NoSig(), delta_t=0.1).eigenvalues.numel() == 2
    finally:
        monkey_sig.undo()
    with pytest.raises(ValueError, match="num_modes must be in"):
        _resolve_num_modes(0, 2, 2)
    assert (
        float(_wasserstein_magnitude_distance(torch.tensor([]), torch.tensor([])))
        == 0.0
    )
    assert float(_subspace_angle_distance(torch.empty(2, 0), torch.empty(2, 0))) == 0.0
    plot_spectrum(torch.tensor([0.1 + 0.2j, 0.3 - 0.1j]), limits="data")
    mismatched = KoopmanSpectrum(
        eigenvalues=torch.tensor([0.1 + 0.0j, 0.2 + 0.0j]),
        eigenvectors=torch.eye(2, dtype=torch.complex64),
        magnitudes=torch.tensor([0.1]),
        growth_rates=torch.zeros(2),
        frequencies=torch.zeros(2),
        time_step=1.0,
    )
    plot_spectrum(mismatched, limits="unit_disk")

    values = torch.randn(3)
    assert masked_mae(values, values, torch.zeros(3, dtype=torch.bool)).item() == 0.0
    assert (
        masked_rmse(
            torch.randn(3, 2),
            torch.randn(3, 2),
            torch.zeros(3, dtype=torch.bool),
        ).item()
        == 0.0
    )
    assert (
        masked_mape(
            torch.randn(3, 2),
            torch.randn(3, 2),
            torch.zeros(3, dtype=torch.bool),
        ).item()
        == 0.0
    )

    model = _tiny_model()
    sequence = _sequence(with_masks=True, num_timesteps=5)
    metrics = evaluate_forecast(model, sequence, horizons=(1, 2))
    assert [item.horizon for item in metrics.horizons] == [1, 2]

    payload = TopologyPayload(edge_index=_edge_index(), num_nodes=2)
    assert list(payload) == ["edge_index", "num_nodes"]
    assert len(payload) == 2
    assert TopologyPayload.from_mapping(payload) is payload

    class BadPredict:
        control_dim = 0

        def predict(self, data: Data, steps: int, controls: Any) -> list[Data]:
            return []

        def fit(self, *args: Any, **kwargs: Any) -> Any:
            return self

    assert accepts_uncontrolled_data_predict(BadPredict()) is False

    class BrokenPredict:
        control_dim = 0
        predict = 123

        def fit(self, *args: Any, **kwargs: Any) -> Any:
            return self

    assert accepts_uncontrolled_data_predict(BrokenPredict()) is False

    baseline = DMDcBaseline()
    with pytest.raises(RuntimeError):
        baseline._require_control_matrix()
    with pytest.raises(RuntimeError):
        baseline._require_control_dim()

    seq = _sequence()
    with pytest.raises(ValueError, match="cannot mix"):
        _classify_trajectory_items(
            [seq, Data(x=torch.randn(2, 2), edge_index=_edge_index())],
            empty_message="empty",
        )
    with pytest.raises(TypeError, match="MultiTrajectory"):
        _classify_trajectory_items([seq, seq], empty_message="empty")
    with pytest.raises(TypeError, match="GraphSnapshotSequence"):
        _classify_trajectory_items([seq, "bad"], empty_message="empty")  # type: ignore[list-item]
    with pytest.raises(ValueError, match="validation_sequence list length"):
        resolve_validation_sequences(
            MultiTrajectory([seq]),
            num_training_sequences=2,
        )
