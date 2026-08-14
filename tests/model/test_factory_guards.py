"""Coverage and error-path tests for :mod:`koopman_graph.model`."""

from __future__ import annotations

from typing import Any

import pytest

from koopman_graph import (
    GNNDecoder,
    GNNEncoder,
    GraphKoopmanModel,
)
from koopman_graph.model.factory import (
    _align_relgraph_latent_dims,
    _relgraph_edge_types_match,
    build_koopman,
    resolve_injected_koopman,
    resolve_model_components,
    validate_typed_relgraph_peers,
)
from koopman_graph.nn.heterogeneous import (
    RelGraphDecoder,
    RelGraphEncoder,
)
from koopman_graph.operators import (
    ContinuousGraphKoopmanOperator,
    GlobalLocalKoopmanOperator,
    GraphKoopmanOperator,
    HeteroGraphKoopmanOperator,
    HypergraphKoopmanOperator,
)

_TYPES = ("a", "b")

_EDGE_TYPES = (("a", "r", "b"),)

_LATENT_DIMS = {"a": 2, "b": 3}


def _typed_relgraph_peers(
    *,
    num_relations: int = 1,
    edge_types: tuple[tuple[str, str, str], ...] = _EDGE_TYPES,
    encoder_dims: dict[str, int] | None = None,
    decoder_dims: dict[str, int] | None = None,
) -> tuple[RelGraphEncoder, RelGraphDecoder]:
    encoder = RelGraphEncoder(
        {"a": 2, "b": 2},
        4,
        4,
        num_relations,
        num_layers=1,
        node_types=_TYPES,
        edge_types=edge_types,
        latent_dims=encoder_dims,
    )
    decoder = RelGraphDecoder(
        4,
        4,
        {"a": 2, "b": 2},
        num_relations,
        num_layers=1,
        node_types=_TYPES,
        edge_types=edge_types,
        latent_dims=decoder_dims,
    )
    return encoder, decoder


def _resolve_kwargs(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "latent_dim": 2,
        "control_dim": 0,
        "control_mode": "additive",
        "bilinear_rank": None,
        "dynamics_mode": "discrete",
        "koopman_init_mode": "identity_noise",
        "koopman_init_scale": 0.01,
        "koopman_parameterization": "dense",
        "koopman_max_spectral_radius": 1.0,
        "koopman_auxiliary_hidden_dims": None,
    }
    kwargs.update(overrides)
    return kwargs


def _build_kwargs(**overrides: Any) -> dict[str, Any]:
    kwargs = _resolve_kwargs(koopman="pernode")
    kwargs.update(overrides)
    return kwargs


def test_factory_relgraph_latent_mismatch_guards() -> None:
    """Factory helpers reject inconsistent RelGraph per-type latent widths."""
    encoder, decoder = _typed_relgraph_peers(
        encoder_dims=_LATENT_DIMS,
        decoder_dims={"a": 2, "b": 4},
    )
    with pytest.raises(ValueError, match="must match RelGraphDecoder.latent_dims"):
        _align_relgraph_latent_dims(encoder, decoder, _LATENT_DIMS)

    multiplex_encoder = RelGraphEncoder(2, 4, 2, 1, num_layers=1)
    multiplex_decoder = RelGraphDecoder(2, 4, 2, 1, num_layers=1)
    with pytest.raises(ValueError, match="explicit edge_types"):
        _align_relgraph_latent_dims(multiplex_encoder, multiplex_decoder, {"node": 2})
    assert _relgraph_edge_types_match(multiplex_encoder, _EDGE_TYPES)

    encoder, decoder = _typed_relgraph_peers()
    operator = HeteroGraphKoopmanOperator(
        4,
        1,
        node_types=_TYPES,
        edge_types=_EDGE_TYPES,
        latent_dims=_LATENT_DIMS,
    )
    with pytest.raises(ValueError, match="latent_dims must match"):
        validate_typed_relgraph_peers(encoder, decoder, operator)
    encoder.latent_dims = _LATENT_DIMS
    decoder.latent_dims = _LATENT_DIMS
    encoder.is_rectangular = False
    decoder.is_rectangular = True
    with pytest.raises(ValueError, match="is_rectangular flags"):
        validate_typed_relgraph_peers(encoder, decoder, operator)


def test_factory_component_reverse_and_latent_rejects() -> None:
    """Component resolution rejects misplaced latent and reverse-relation options."""
    gnn_encoder = GNNEncoder(2, 4, 2, num_layers=1)
    gnn_decoder = GNNDecoder(2, 4, 2, num_layers=1)
    with pytest.raises(ValueError, match="requires koopman='hetero_graph'"):
        resolve_model_components(
            gnn_encoder,
            gnn_decoder,
            2,
            1.0,
            koopman="graph",
            physics_position="concat",
            koopman_latent_dims={"node": 2},
        )
    with pytest.raises(ValueError, match="requires koopman='hetero_graph'"):
        resolve_model_components(
            gnn_encoder,
            gnn_decoder,
            2,
            1.0,
            koopman="graph",
            physics_position="concat",
            koopman_synthesize_reverse_relations=True,
            koopman_edge_types=(("node", "r", "node"),),
        )

    encoder = RelGraphEncoder(2, 4, 2, 1, num_layers=1)
    decoder = RelGraphDecoder(2, 4, 2, 1, num_layers=1)
    with pytest.raises(ValueError, match="requires koopman_edge_types"):
        resolve_model_components(
            encoder,
            decoder,
            2,
            1.0,
            koopman="hetero_graph",
            physics_position="concat",
            koopman_synthesize_reverse_relations=True,
        )

    expanded = (("node", "r", "node"), ("node", "rev_r", "node"))
    bad_encoder = RelGraphEncoder(
        2,
        4,
        2,
        2,
        num_layers=1,
        edge_types=(("node", "wrong", "node"), expanded[1]),
    )
    bad_decoder = RelGraphDecoder(2, 4, 2, 2, num_layers=1, edge_types=expanded)
    with pytest.raises(ValueError, match="RelGraphEncoder.edge_types"):
        resolve_model_components(
            bad_encoder,
            bad_decoder,
            2,
            1.0,
            koopman="hetero_graph",
            physics_position="concat",
            koopman_edge_types=(expanded[0],),
            koopman_synthesize_reverse_relations=True,
        )

    good_encoder = RelGraphEncoder(2, 4, 2, 2, num_layers=1, edge_types=expanded)
    bad_decoder = RelGraphDecoder(
        2,
        4,
        2,
        2,
        num_layers=1,
        edge_types=(("node", "wrong", "node"), expanded[1]),
    )
    with pytest.raises(ValueError, match="RelGraphDecoder.edge_types"):
        resolve_model_components(
            good_encoder,
            bad_decoder,
            2,
            1.0,
            koopman="hetero_graph",
            physics_position="concat",
            koopman_edge_types=(expanded[0],),
            koopman_synthesize_reverse_relations=True,
        )

    three_encoder = RelGraphEncoder(2, 4, 2, 3, num_layers=1)
    three_decoder = RelGraphDecoder(2, 4, 2, 3, num_layers=1)
    with pytest.raises(ValueError, match="num_relations equal"):
        resolve_model_components(
            three_encoder,
            three_decoder,
            2,
            1.0,
            koopman="hetero_graph",
            physics_position="concat",
            koopman_edge_types=(expanded[0],),
            koopman_synthesize_reverse_relations=True,
        )


def test_factory_stochastic_attachment_and_continuous_hetero_guards() -> None:
    """Factory attaches process noise and rejects unsupported continuous options."""
    graph = GraphKoopmanOperator(2)
    resolved = resolve_injected_koopman(
        graph, **_resolve_kwargs(dynamics_mode="stochastic")
    )
    assert getattr(resolved, "stochastic", False)

    with pytest.raises(ValueError, match="auxiliary_hidden_dims"):
        build_koopman(
            **_build_kwargs(
                koopman="hetero_graph",
                dynamics_mode="continuous",
                num_relations=1,
                koopman_auxiliary_hidden_dims=(4,),
            )
        )
    with pytest.raises(ValueError, match="num_relations can be resolved"):
        build_koopman(
            **_build_kwargs(koopman="hetero_graph", dynamics_mode="continuous")
        )


def test_factory_stochastic_and_continuous_hetero_extra_guards() -> None:
    """Factory rejects unsupported stochastic kinds and continuous hetero orbits."""
    with pytest.raises(ValueError, match="supports koopman='pernode'"):
        build_koopman(**_build_kwargs(koopman="hypergraph", dynamics_mode="stochastic"))
    with pytest.raises(ValueError, match="does not support rectangular"):
        build_koopman(
            **_build_kwargs(
                koopman="hetero_graph",
                dynamics_mode="stochastic",
                num_relations=1,
                node_types=_TYPES,
                edge_types=_EDGE_TYPES,
                latent_dims=_LATENT_DIMS,
            )
        )
    with pytest.raises(ValueError, match="unsupported for continuous hetero"):
        build_koopman(
            **_build_kwargs(
                koopman="hetero_graph",
                dynamics_mode="continuous",
                num_relations=1,
                koopman_auto_orbits=True,
            )
        )
    with pytest.raises(ValueError, match="requires dynamics_mode='discrete' or"):
        resolve_injected_koopman(
            GraphKoopmanOperator(2),
            **_resolve_kwargs(dynamics_mode="continuous"),
        )


def test_factory_relgraph_latent_dims_mismatch_raises() -> None:
    """Injected RelGraph peers with disagreeing latent_dims are rejected."""
    encoder, decoder = _typed_relgraph_peers(
        encoder_dims={"a": 2, "b": 3},
        decoder_dims={"a": 2, "b": 4},
    )
    with pytest.raises(ValueError, match="must match RelGraphDecoder.latent_dims"):
        resolve_model_components(
            encoder,
            decoder,
            4,
            0.1,
            physics_position="concat",
            dynamics_mode="discrete",
            koopman="hetero_graph",
        )


def test_factory_isotypic_and_hypergraph_incidence_guards() -> None:
    """Isotypic factory conflicts and hypergraph incidence-mode validation."""
    from koopman_graph.model.factory import (
        build_encoder_peers,
        build_koopman,
        resolve_injected_koopman,
    )
    from koopman_graph.operators import GraphKoopmanOperator

    with pytest.raises(ValueError, match="Unknown encoder"):
        build_encoder_peers(
            "mlp",  # type: ignore[arg-type]
            in_channels=2,
            hidden_channels=4,
            latent_dim=2,
            out_channels=2,
        )
    with pytest.raises(ValueError, match="omit koopman_orbit_method"):
        GraphKoopmanModel(
            encoder=GNNEncoder(2, 4, 2, num_layers=1),
            decoder=GNNDecoder(2, 4, 2, num_layers=1),
            latent_dim=2,
            time_step=1.0,
            koopman="graph",
            koopman_symmetry="isotypic",
            koopman_orbit_method="exact",
        )
    with pytest.raises(ValueError, match="unsupported for dynamics_mode"):
        GraphKoopmanModel(
            encoder=GNNEncoder(2, 4, 2, num_layers=1),
            decoder=GNNDecoder(2, 4, 2, num_layers=1),
            latent_dim=2,
            time_step=1.0,
            koopman="graph",
            dynamics_mode="continuous",
            koopman_symmetry="isotypic",
        )
    with pytest.raises(ValueError, match="incidence_mode must be one of"):
        build_koopman(
            koopman="hypergraph",
            latent_dim=2,
            control_dim=0,
            control_mode="additive",
            bilinear_rank=None,
            dynamics_mode="discrete",
            koopman_init_mode="identity_noise",
            koopman_init_scale=0.01,
            koopman_parameterization="dense",
            koopman_max_spectral_radius=1.0,
            koopman_auxiliary_hidden_dims=None,
            koopman_hypergraph_incidence_mode="not_a_mode",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="only meaningful for koopman='hypergraph'"):
        build_koopman(
            koopman="graph",
            latent_dim=2,
            control_dim=0,
            control_mode="additive",
            bilinear_rank=None,
            dynamics_mode="discrete",
            koopman_init_mode="identity_noise",
            koopman_init_scale=0.01,
            koopman_parameterization="dense",
            koopman_max_spectral_radius=1.0,
            koopman_auxiliary_hidden_dims=None,
            koopman_hypergraph_incidence_mode="forward_random_walk",
        )
    with pytest.raises(ValueError, match="mutually exclusive with non-default"):
        resolve_injected_koopman(
            GraphKoopmanOperator(2, init_mode="identity"),
            latent_dim=2,
            control_dim=0,
            control_mode="additive",
            bilinear_rank=None,
            dynamics_mode="discrete",
            koopman_init_mode="identity_noise",
            koopman_init_scale=0.01,
            koopman_parameterization="dense",
            koopman_max_spectral_radius=1.0,
            koopman_auxiliary_hidden_dims=None,
            koopman_symmetry="isotypic",
        )


def test_factory_patch_coverage_gaps() -> None:
    """Cover factory injection, validation, and build_koopman reject paths."""
    from unittest.mock import patch

    from koopman_graph.model.factory import (
        build_koopman,
        resolve_injected_koopman,
        resolve_model_components,
    )
    from koopman_graph.operators import (
        GraphKoopmanOperator,
    )

    enc = GNNEncoder(2, 8, 2)
    dec = GNNDecoder(2, 8, 2)
    base_kwargs = dict(
        encoder=enc,
        decoder=dec,
        latent_dim=2,
        time_step=0.1,
        control_dim=0,
        control_mode="additive",
        bilinear_rank=None,
        dynamics_mode="discrete",
        physics_dim=0,
        physics_lifting_fn=None,
        physics_position="post",
        n_delays=1,
        koopman=None,
    )

    with (
        patch(
            "koopman_graph.model.factory.resolve_physics_lifting_fn",
            return_value=None,
        ),
        pytest.raises(ValueError, match="physics_preset requires"),
    ):
        resolve_model_components(**base_kwargs, physics_preset="missing")

    injected_global_local = GlobalLocalKoopmanOperator(2)
    injected_graph = GraphKoopmanOperator(2)
    for extra in (
        {"koopman_local_window": 8},
        {"koopman_local_rank": 4},
        {"koopman_local_hidden_dims": (16,)},
    ):
        with pytest.raises(ValueError, match="mutually exclusive"):
            GraphKoopmanModel(
                enc,
                dec,
                latent_dim=2,
                time_step=0.1,
                koopman=injected_global_local,
                **extra,
            )
    with pytest.raises(ValueError, match="mutually exclusive"):
        GraphKoopmanModel(
            enc,
            dec,
            latent_dim=2,
            time_step=0.1,
            koopman=injected_graph,
            koopman_orbit_method="exact",
        )
    resolve_common = dict(
        koopman=injected_graph,
        latent_dim=2,
        control_dim=0,
        control_mode="additive",
        bilinear_rank=None,
        dynamics_mode="discrete",
        koopman_init_mode="identity_noise",
        koopman_init_scale=1e-2,
        koopman_parameterization="dense",
        koopman_max_spectral_radius=1.0,
        koopman_auxiliary_hidden_dims=None,
        koopman_local_window=4,
        koopman_local_rank=2,
        koopman_local_hidden_dims=None,
        koopman_orbit_partition=None,
        koopman_auto_orbits=False,
        koopman_orbit_method="auto",
    )
    for key, value in (
        ("koopman_orbit_partition", ((0, 1),)),
        ("koopman_auto_orbits", True),
        ("koopman_orbit_method", "exact"),
    ):
        kwargs = dict(resolve_common)
        kwargs[key] = value
        with pytest.raises(ValueError, match="mutually exclusive"):
            resolve_injected_koopman(**kwargs)
    with pytest.raises(ValueError, match="mutually exclusive"):
        aux_kwargs = dict(resolve_common)
        aux_kwargs["koopman_auxiliary_hidden_dims"] = (8,)
        resolve_injected_koopman(**aux_kwargs)

    dynamics_mismatch = [
        (GraphKoopmanOperator(2), "continuous"),
        (HypergraphKoopmanOperator(2), "continuous"),
        (GlobalLocalKoopmanOperator(2), "continuous"),
        (ContinuousGraphKoopmanOperator(2), "discrete"),
    ]
    for operator, dynamics_mode in dynamics_mismatch:
        with pytest.raises(ValueError, match="dynamics_mode"):
            GraphKoopmanModel(
                enc,
                dec,
                latent_dim=2,
                time_step=0.1,
                dynamics_mode=dynamics_mode,  # type: ignore[arg-type]
                koopman=operator,
            )

    kind_cases = [
        (GraphKoopmanOperator(2), "discrete", "graph"),
        (HypergraphKoopmanOperator(2), "discrete", "hypergraph"),
        (GlobalLocalKoopmanOperator(2), "discrete", "global_local"),
        (ContinuousGraphKoopmanOperator(2), "continuous", "continuous_graph"),
    ]
    for operator, dynamics_mode, expected_kind in kind_cases:
        model = GraphKoopmanModel(
            enc,
            dec,
            latent_dim=2,
            time_step=0.1,
            dynamics_mode=dynamics_mode,  # type: ignore[arg-type]
            koopman=operator,
        )
        assert model.koopman_kind == expected_kind

    common_build = dict(
        latent_dim=2,
        control_dim=0,
        control_mode="additive",
        bilinear_rank=None,
        dynamics_mode="discrete",
        koopman_init_mode="identity_noise",
        koopman_init_scale=1e-2,
        koopman_parameterization="dense",
        koopman_max_spectral_radius=1.0,
        koopman_auxiliary_hidden_dims=None,
    )
    with pytest.raises(ValueError, match="koopman_sparsity is only meaningful"):
        build_koopman(
            koopman="pernode",
            koopman_sparsity="block_diagonal",
            **common_build,
        )
    with pytest.raises(ValueError, match="koopman_orbit_partition"):
        build_koopman(
            koopman="pernode",
            koopman_orbit_partition=((0, 1),),
            **common_build,
        )
    with pytest.raises(ValueError, match="koopman_orbit_method must be"):
        build_koopman(
            koopman="graph",
            koopman_orbit_method="bogus",  # type: ignore[arg-type]
            **common_build,
        )

    with pytest.raises(
        ValueError,
        match="koopman_auxiliary_hidden_dims is not supported",
    ):
        GraphKoopmanModel(
            enc,
            dec,
            latent_dim=2,
            time_step=0.1,
            dynamics_mode="continuous",
            koopman="graph",
            koopman_parameterization="auxiliary_spectral",
            koopman_auxiliary_hidden_dims=(8,),
        )
    with pytest.raises(ValueError, match="koopman_auxiliary_hidden_dims requires"):
        GraphKoopmanModel(
            enc,
            dec,
            latent_dim=2,
            time_step=0.1,
            koopman="hypergraph",
            koopman_parameterization="auxiliary_spectral",
            koopman_auxiliary_hidden_dims=(8,),
        )
    with pytest.raises(ValueError, match="koopman_auxiliary_hidden_dims requires"):
        GraphKoopmanModel(
            enc,
            dec,
            latent_dim=2,
            time_step=0.1,
            dynamics_mode="continuous",
            koopman="graph",
            koopman_parameterization="dense",
            koopman_auxiliary_hidden_dims=(8,),
        )
    with pytest.raises(ValueError, match="koopman_auxiliary_hidden_dims requires"):
        GraphKoopmanModel(
            enc,
            dec,
            latent_dim=2,
            time_step=0.1,
            koopman="global_local",
            koopman_parameterization="auxiliary_spectral",
            koopman_auxiliary_hidden_dims=(8,),
        )
