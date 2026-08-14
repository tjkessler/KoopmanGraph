"""Coverage and error-path tests for :mod:`koopman_graph.serialization`."""

from __future__ import annotations

from pathlib import Path

from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel


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

    stochastic = _tiny_model(parameterization="row_stochastic")
    path2 = tmp_path / "row_stoch.pt"
    stochastic.save(path2)
    loaded_stoch = GraphKoopmanModel.load(path2)
    assert loaded_stoch.koopman.parameterization == "row_stochastic"
