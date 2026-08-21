"""Coverage and error-path tests for :mod:`koopman_graph.serialization`."""

from __future__ import annotations

from pathlib import Path

from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
from koopman_graph.serialization import FORMAT_VERSION, build_checkpoint


def _tiny_model(*, koopman: str = "pernode", parameterization: str = "dense", **kwargs):
    """Build a tiny homogeneous GCN model."""
    return GraphKoopmanModel(
        encoder=GNNEncoder(2, 4, 4, num_layers=1),
        decoder=GNNDecoder(4, 4, 2, num_layers=1),
        latent_dim=4,
        time_step=0.1,
        koopman=koopman,
        koopman_parameterization=parameterization,
        **kwargs,
    )


def test_switched_mixture_checkpoint_roundtrip(tmp_path: Path) -> None:
    """Format-1 save/load preserves switched kind and row-stochastic ``K``."""
    model = _tiny_model(koopman="switched")
    path = tmp_path / "switched.pt"
    model.save(path)
    loaded = GraphKoopmanModel.load(path)
    assert loaded.koopman_kind == "switched"

    mixture = _tiny_model(koopman="mixture")
    path_m = tmp_path / "mixture.pt"
    mixture.save(path_m)
    loaded_m = GraphKoopmanModel.load(path_m)
    assert loaded_m.koopman_kind == "mixture"

    hodge = _tiny_model(koopman="hodge")
    path_h = tmp_path / "hodge.pt"
    hodge.save(path_h)
    loaded_h = GraphKoopmanModel.load(path_h)
    assert loaded_h.koopman_kind == "hodge"

    parametric = _tiny_model(koopman="parametric", koopman_parameter_dim=2)
    path_p = tmp_path / "parametric.pt"
    parametric.save(path_p)
    loaded_p = GraphKoopmanModel.load(path_p)
    assert loaded_p.koopman_kind == "parametric"
    assert loaded_p.koopman.parameter_dim == 2
    assert loaded_p.koopman.weight_kind == "rbf"
    checkpoint = build_checkpoint(parametric)
    assert checkpoint["format_version"] == FORMAT_VERSION == 1
    assert checkpoint["config"]["koopman_kind"] == "parametric"
    assert checkpoint["config"]["koopman_parameter_dim"] == 2
    assert checkpoint["config"]["koopman_weight_kind"] == "rbf"
    assert checkpoint["config"]["koopman_num_modes"] == 2

    stochastic = _tiny_model(parameterization="row_stochastic")
    path2 = tmp_path / "row_stoch.pt"
    stochastic.save(path2)
    loaded_stoch = GraphKoopmanModel.load(path2)
    assert loaded_stoch.koopman.parameterization == "row_stochastic"
