"""Patch-coverage guards for the 0.10.0 Codecov patch gate."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn
from torch_geometric.data import Data, HeteroData

from koopman_graph import GNNDecoder, GNNEncoder, GraphSnapshotSequence
from koopman_graph.baselines.base import (
    fit_fb_row_operator,
    fit_opt_row_operator,
    fit_tls_row_operator,
    streaming_gram_init,
    streaming_gram_update,
)
from koopman_graph.data import HeteroGraphSnapshotSequence
from koopman_graph.graph_utils.topology import (
    materialize_reverse_relation_edges,
    synthesize_reverse_edge_types,
)
from koopman_graph.model.factory import (
    _align_relgraph_latent_dims,
    _relgraph_edge_types_match,
    build_koopman,
    resolve_injected_koopman,
    resolve_model_components,
    validate_typed_relgraph_peers,
)
from koopman_graph.mpc.controller import _validate_mpc_model
from koopman_graph.nn.heterogeneous import (
    RelGraphDecoder,
    RelGraphEncoder,
    _pack_relgraph_latents,
    _unpack_relgraph_latents,
)
from koopman_graph.operators import (
    ContinuousGraphKoopmanOperator,
    ContinuousHeteroGraphKoopmanOperator,
    GlobalLocalKoopmanOperator,
    GraphKoopmanOperator,
    HeteroGraphKoopmanOperator,
    HypergraphKoopmanOperator,
)
from koopman_graph.operators.graph import _koopman_spectrum_from_eigenvalues
from koopman_graph.serialization import (
    _require_hetero_schema,
    _state_dict_has_rectangular_hetero_markers,
    _validate_hetero_latent_dims_vs_state,
)
from koopman_graph.training.extra_objectives import compute_vamp2_loss
from koopman_graph.training.loop import bind_pending_orbit_ties
from koopman_graph.training.objectives import compute_eigenvalue_regularization_loss
from koopman_graph.uq.conformal import (
    ConformalKoopmanUQ,
    _is_hetero_sequence,
    _stack_hetero_features,
    _union_relation_edge_index,
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


def _hetero_schema(**overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "latent_dim": 4,
        "node_types": ["a", "b"],
        "edge_types": [["a", "r", "b"]],
        "relation_tying": "independent",
        "basis_size": None,
        "relation_normalization": "rgcn_in_degree",
    }
    config.update(overrides)
    return config


def _empty_multiplex_snapshot() -> HeteroData:
    snapshot = HeteroData()
    snapshot["node"].x = torch.zeros(2, 2)
    snapshot["node", "r", "node"].edge_index = torch.empty(2, 0, dtype=torch.long)
    return snapshot


def test_mpc_and_continuous_hetero_objective_guards() -> None:
    """MPC and continuous hetero regularization reject incompatible surfaces."""
    model = SimpleNamespace(koopman=nn.Linear(2, 2), dynamics_mode="continuous")
    with pytest.raises(ValueError, match="dynamics_mode='discrete'"):
        _validate_mpc_model(model)  # type: ignore[arg-type]

    operator = ContinuousHeteroGraphKoopmanOperator(2, 1, parameterization="dense")
    continuous = SimpleNamespace(koopman=operator, dynamics_mode="continuous")
    with pytest.raises(ValueError, match="sequence is required"):
        compute_eigenvalue_regularization_loss(continuous, None)  # type: ignore[arg-type]
    sequence = GraphSnapshotSequence(
        [Data(x=torch.zeros(2, 2), edge_index=torch.empty(2, 0, dtype=torch.long))]
    )
    with pytest.raises(ValueError, match="HeteroGraphSnapshotSequence"):
        compute_eigenvalue_regularization_loss(continuous, sequence)  # type: ignore[arg-type]


def test_hypergraph_distributed_inverse_paths() -> None:
    """Distributed hypergraph inversion rejects overrides and assembles its inverse."""
    operator = HypergraphKoopmanOperator(
        2,
        init_mode="identity",
        sparsity="distributed",
    )
    z = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    hyperedges = torch.tensor([[0, 1, 2], [0, 0, 0]], dtype=torch.long)
    with pytest.raises(ValueError, match="sparsity='dense'"):
        operator.inverse_advance(
            z,
            hyperedge_index=hyperedges,
            inverse_matrix=torch.eye(6),
        )
    recovered = operator.inverse_advance(z, hyperedge_index=hyperedges)
    torch.testing.assert_close(recovered, z)


def test_relgraph_latent_dimension_and_shape_guards() -> None:
    """RelGraph rectangular metadata and flat/block shapes are validated."""
    with pytest.raises(ValueError, match="single node type"):
        RelGraphEncoder(
            2,
            4,
            3,
            1,
            node_types=("a", "b"),
            latent_dims={"a": 3, "b": 3},
        )
    with pytest.raises(ValueError, match="must equal latent_dim"):
        RelGraphEncoder(2, 4, 3, 1, latent_dims={"node": 2})

    counts = {"a": 2, "b": 1}
    with pytest.raises(ValueError, match="latent block"):
        _pack_relgraph_latents(
            _TYPES,
            {"a": torch.zeros(2, 2), "b": torch.zeros(2, 3)},
            counts,
            _LATENT_DIMS,
        )
    with pytest.raises(ValueError, match="z_flat must have shape"):
        _unpack_relgraph_latents(torch.zeros(7, 1), _TYPES, counts, _LATENT_DIMS)

    decoder = RelGraphDecoder(
        4,
        4,
        {"a": 2, "b": 2},
        1,
        num_layers=1,
        node_types=_TYPES,
        edge_types=_EDGE_TYPES,
        latent_dims=_LATENT_DIMS,
    )
    edge_bank = [torch.tensor([[0], [2]], dtype=torch.long)]
    with pytest.raises(ValueError, match="num_nodes_dict is required"):
        decoder(torch.zeros(7), edge_bank)
    with pytest.raises(TypeError, match="flat latent Tensor"):
        decoder(HeteroData(), edge_bank, num_nodes_dict=counts)
    with pytest.raises(ValueError, match="edge_index relation banks"):
        decoder(torch.zeros(7), num_nodes_dict=counts)


def test_reverse_relation_validation_and_materialization_guards() -> None:
    """Reverse-relation helpers reject malformed schemas and absent forwards."""
    for bad_prefix in ("",):
        with pytest.raises(ValueError, match="non-empty"):
            synthesize_reverse_edge_types(_EDGE_TYPES, reverse_prefix=bad_prefix)
        with pytest.raises(ValueError, match="non-empty"):
            materialize_reverse_relation_edges(
                HeteroData(), _EDGE_TYPES, reverse_prefix=bad_prefix
            )

    with pytest.raises(ValueError, match="triples"):
        synthesize_reverse_edge_types((("a", "r"),))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty strings"):
        synthesize_reverse_edge_types((("a", "", "b"),))
    with pytest.raises(ValueError, match="unique"):
        synthesize_reverse_edge_types((_EDGE_TYPES[0], _EDGE_TYPES[0]))
    with pytest.raises(ValueError, match="not the geometric reverse"):
        synthesize_reverse_edge_types((("a", "r", "b"), ("a", "rev_r", "a")))

    with pytest.raises(ValueError, match="triples"):
        materialize_reverse_relation_edges(
            HeteroData(),
            (("a", "r"),),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="non-empty strings"):
        materialize_reverse_relation_edges(HeteroData(), (("a", "", "b"),))
    schema = (("a", "r", "b"), ("b", "rev_r", "a"))
    with pytest.raises(ValueError, match="missing forward edge type"):
        materialize_reverse_relation_edges(HeteroData(), schema)

    snapshot = HeteroData()
    snapshot["a"].x = torch.zeros(1, 1)
    snapshot["b"].x = torch.zeros(1, 1)
    snapshot["a", "r", "b"].edge_index = torch.tensor([[0], [0]])
    snapshot["b", "rev_r", "a"].edge_index = torch.tensor([[0], [0]])
    preserved = materialize_reverse_relation_edges(snapshot, schema)
    torch.testing.assert_close(
        preserved["b", "rev_r", "a"].edge_index,
        snapshot["b", "rev_r", "a"].edge_index,
    )


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


@pytest.mark.parametrize(
    ("operator", "mode", "message"),
    [
        (HypergraphKoopmanOperator(2), "stochastic", "does not support hypergraph"),
        (HypergraphKoopmanOperator(2), "continuous", "requires dynamics_mode"),
        (HeteroGraphKoopmanOperator(2, 1), "continuous", "discrete.*stochastic"),
        (
            HeteroGraphKoopmanOperator(
                2,
                1,
                node_types=_TYPES,
                edge_types=_EDGE_TYPES,
                latent_dims=_LATENT_DIMS,
            ),
            "stochastic",
            "rectangular",
        ),
        (
            ContinuousHeteroGraphKoopmanOperator(2, 1),
            "discrete",
            "requires dynamics_mode",
        ),
        (GlobalLocalKoopmanOperator(2), "stochastic", "global_local"),
        (GlobalLocalKoopmanOperator(2), "continuous", "requires dynamics_mode"),
        (ContinuousGraphKoopmanOperator(2), "discrete", "requires dynamics_mode"),
    ],
)
def test_injected_operator_dynamics_guards(
    operator: nn.Module, mode: str, message: str
) -> None:
    """Injected operator families enforce their supported dynamics modes."""
    with pytest.raises(ValueError, match=message):
        resolve_injected_koopman(
            operator,  # type: ignore[arg-type]
            **_resolve_kwargs(dynamics_mode=mode),
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


def test_baseline_validation_and_dead_eigenvalue_branch() -> None:
    """Optimized and streaming DMD validate iterations and vector ranks."""
    left = torch.zeros(3, 2)
    right = torch.zeros(3, 2)
    with pytest.raises(ValueError, match="max_iter must be >= 1"):
        fit_opt_row_operator(left, right, None, max_iter=0)
    operator = fit_opt_row_operator(left, right, None, max_iter=1)
    assert operator.shape == (2, 2)

    gram, cross = streaming_gram_init(
        2, dtype=torch.float32, device=torch.device("cpu")
    )
    with pytest.raises(ValueError, match="1-D state vectors"):
        streaming_gram_update(gram, cross, torch.zeros(1, 2), torch.zeros(2))


class _BadHeteroSequence(HeteroGraphSnapshotSequence):
    def __init__(self) -> None:
        """Bypass container validation to exercise the defensive bind guard."""

    def __getitem__(self, _index: int) -> Data:
        return Data(x=torch.zeros(2, 2))


def test_orbit_binding_errors_and_empty_union() -> None:
    """Hetero orbit binding rejects bad snapshots and accepts empty bank unions."""
    calls: list[torch.Tensor] = []
    koopman = SimpleNamespace(
        auto_orbits=True,
        orbit_partition=None,
        is_typed=False,
        num_relations=1,
        ensure_orbit_binding=lambda _n, *, edge_index: calls.append(edge_index),
    )
    model = SimpleNamespace(koopman=koopman)
    with pytest.raises(ValueError, match="requires HeteroData"):
        bind_pending_orbit_ties(model, [_BadHeteroSequence()])  # type: ignore[list-item]

    typed = SimpleNamespace(
        auto_orbits=True,
        orbit_partition=None,
        is_typed=True,
        ensure_orbit_binding=lambda *_a, **_k: None,
    )
    with pytest.raises(ValueError, match="ensure_typed_orbit_binding"):
        bind_pending_orbit_ties(
            SimpleNamespace(koopman=typed),
            [HeteroGraphSnapshotSequence([_empty_multiplex_snapshot()])],
        )

    bind_pending_orbit_ties(
        model,
        [HeteroGraphSnapshotSequence([_empty_multiplex_snapshot()])],
    )
    assert calls[0].shape == (2, 0)


def test_graph_spectrum_certificate_and_distributed_inverse_guards() -> None:
    """Graph helpers validate time, assemble certificates, and reject tied inverse."""
    with pytest.raises(ValueError, match="time_step must be positive"):
        _koopman_spectrum_from_eigenvalues(torch.ones(1, dtype=torch.complex64), 0.0)

    graph = GraphKoopmanOperator(2, init_mode="identity")
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    certificate = graph.stability_certificate(edge_index, 2, kind="schur")
    assert torch.isfinite(certificate.bound)

    distributed = GraphKoopmanOperator(
        2,
        init_mode="identity",
        sparsity="distributed",
        orbit_partition=((0, 1),),
    )
    z = torch.zeros(2, 2)
    with pytest.raises(ValueError, match="requires a shared K_self"):
        distributed.inverse_advance(z, edge_index=edge_index)
    with pytest.raises(ValueError, match="sparsity='dense'"):
        GraphKoopmanOperator(2, sparsity="distributed").inverse_advance(
            z,
            edge_index=edge_index,
            inverse_matrix=torch.eye(4),
        )


def test_vamp2_short_sequence_guard() -> None:
    """VAMP-2 requires at least one lagged snapshot pair."""
    model = nn.Linear(2, 2)
    sequence = GraphSnapshotSequence(
        [Data(x=torch.zeros(2, 2), edge_index=torch.empty(2, 0, dtype=torch.long))]
    )
    with pytest.raises(ValueError, match="at least 2 snapshots"):
        compute_vamp2_loss(model, sequence, weight=1.0)  # type: ignore[arg-type]


def test_hetero_conformal_helper_guards() -> None:
    """Heterogeneous conformal helpers validate sequences, features, and origins."""
    hetero = HeteroData()
    hetero["a"].x = torch.zeros(1, 2)
    assert _is_hetero_sequence([hetero])
    assert not _is_hetero_sequence([])

    with pytest.raises(ValueError, match="missing node type"):
        _stack_hetero_features(hetero, _TYPES)

    class _MissingFeatures:
        node_types = _TYPES

        def __getitem__(self, name: str) -> SimpleNamespace:
            features = torch.zeros(1, 2) if name == "a" else None
            return SimpleNamespace(x=features)

    with pytest.raises(ValueError, match="missing feature matrix"):
        _stack_hetero_features(_MissingFeatures(), _TYPES)  # type: ignore[arg-type]

    empty = HeteroData()
    empty["a"].x = torch.zeros(1, 2)
    empty["b"].x = torch.zeros(1, 2)
    empty["a", "r", "b"].edge_index = torch.empty(2, 0, dtype=torch.long)
    with pytest.raises(ValueError, match="non-empty relation"):
        _union_relation_edge_index(empty, _EDGE_TYPES, _TYPES)

    bad_uq = ConformalKoopmanUQ(
        SimpleNamespace(koopman=nn.Linear(2, 2))  # type: ignore[arg-type]
    )
    with pytest.raises(TypeError, match="HeteroGraphKoopmanOperator"):
        bad_uq._hetero_operator()

    hetero_model = SimpleNamespace(
        koopman=HeteroGraphKoopmanOperator(2, 1),
        uses_hetero_koopman=True,
    )
    uq = ConformalKoopmanUQ(hetero_model)  # type: ignore[arg-type]
    uq._quantiles = torch.ones(1)
    uq._alpha = 0.1
    uq._calibrated_steps = 1
    with pytest.raises(TypeError, match="requires a HeteroData origin"):
        uq.predict_interval(Data(x=torch.zeros(2, 2)), steps=1, level=0.9)


def test_serialization_reverse_latent_and_rectangular_guards() -> None:
    """Checkpoint validation rejects malformed additive hetero metadata."""
    with pytest.raises(ValueError, match="synthesize_reverse_relations must be a bool"):
        _require_hetero_schema(_hetero_schema(synthesize_reverse_relations=1))
    with pytest.raises(ValueError, match="latent_dims must be a mapping"):
        _require_hetero_schema(_hetero_schema(latent_dims=[]))
    with pytest.raises(ValueError, match="latent_dims is incomplete or invalid"):
        _require_hetero_schema(_hetero_schema(latent_dims={"a": 2}))

    assert _state_dict_has_rectangular_hetero_markers(
        {"koopman._rel_rect.0.K": torch.eye(2)}
    )
    assert _state_dict_has_rectangular_hetero_markers(
        {"encoder.type_latent.a.weight": torch.zeros(2, 2)}
    )
    assert not _state_dict_has_rectangular_hetero_markers({"other": torch.zeros(1)})

    with pytest.raises(ValueError, match="config.latent_dims is missing"):
        _validate_hetero_latent_dims_vs_state(
            _hetero_schema(),
            {"decoder.type_latent_in.a.weight": torch.zeros(2, 2)},
        )
    with pytest.raises(ValueError, match="has no koopman._rel_rect"):
        _validate_hetero_latent_dims_vs_state(
            _hetero_schema(latent_dims=_LATENT_DIMS),
            {},
        )
    state = {
        "koopman._rel_rect.0.K": torch.zeros(3, 2),
        "koopman._selves.a.K": torch.zeros(3, 3),
    }
    with pytest.raises(ValueError, match="expects \\(2, 2\\)"):
        _validate_hetero_latent_dims_vs_state(
            _hetero_schema(latent_dims=_LATENT_DIMS),
            state,
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


def test_fb_and_tls_row_operator_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward-backward invertibility and TLS shape/rank guards."""
    left = torch.randn(6, 3)
    right = torch.randn(6, 3)
    with pytest.raises(ValueError, match="share shape"):
        fit_tls_row_operator(left, right[:, :2], rank=None)
    with pytest.raises(ValueError, match="must be 2-D"):
        fit_tls_row_operator(left[0], right[0], rank=None)

    def boom(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise RuntimeError("singular")

    monkeypatch.setattr(torch.linalg, "inv", boom)
    with pytest.raises(ValueError, match="invertible backward operator"):
        fit_fb_row_operator(left, right, rank=None)

    import koopman_graph.baselines.base as base_mod

    monkeypatch.setattr(base_mod, "resolve_fit_rank", lambda *_a, **_k: 0)
    with pytest.raises(ValueError, match="truncation rank must be >= 1"):
        fit_tls_row_operator(left, right, rank=1)


def test_bind_typed_and_nonempty_multiplex_auto_orbits() -> None:
    """Typed auto_orbits binding and nonempty multiplex union succeed."""
    typed_calls: list[object] = []
    typed = SimpleNamespace(
        auto_orbits=True,
        orbit_partition=None,
        is_typed=True,
        node_types=_TYPES,
        edge_types=_EDGE_TYPES,
        ensure_orbit_binding=lambda *_a, **_k: None,
        ensure_typed_orbit_binding=lambda banks, counts: typed_calls.append(
            (banks, counts)
        ),
    )
    snap = HeteroData()
    snap["a"].x = torch.randn(2, 2)
    snap["b"].x = torch.randn(2, 2)
    snap["a", "r", "b"].edge_index = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    bind_pending_orbit_ties(
        SimpleNamespace(koopman=typed),
        [HeteroGraphSnapshotSequence([snap, snap])],
    )
    assert typed_calls

    calls: list[torch.Tensor] = []
    multiplex = SimpleNamespace(
        auto_orbits=True,
        orbit_partition=None,
        is_typed=False,
        num_relations=1,
        ensure_orbit_binding=lambda _n, *, edge_index: calls.append(edge_index),
    )
    multi = HeteroData()
    multi["node"].x = torch.randn(3, 2)
    multi["node", "r0", "node"].edge_index = torch.tensor(
        [[0, 1], [1, 2]],
        dtype=torch.long,
    )
    bind_pending_orbit_ties(
        SimpleNamespace(koopman=multiplex),
        [HeteroGraphSnapshotSequence([multi, multi])],
    )
    assert calls and calls[0].numel() > 0
