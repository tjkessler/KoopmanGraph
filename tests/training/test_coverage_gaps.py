"""Coverage for batched-objective guards and fit-loop error paths."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import Data, HeteroData

from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
from koopman_graph.data import (
    GraphDynamicsConfig,
    GraphSnapshotSequence,
    HeteroGraphSnapshotSequence,
    MultiTrajectory,
    WindowSampler,
)
from koopman_graph.identification import IdentificationConfig
from koopman_graph.nn import DelayEmbeddingEncoder, HypergraphEncoder, RelGraphEncoder
from koopman_graph.operators import KoopmanOperator
from koopman_graph.training import (
    LossWeights,
    NoOpFitCallback,
    compute_batched_training_loss,
    run_fit_loop,
    validate_graph_batching_request,
)
from koopman_graph.training.loop import (
    _collect_identification_pairs,
    _encode_sequence_latents,
    _validate_identification_fit_request,
    bind_pending_orbit_ties,
)


def _path_edge_index(num_nodes: int) -> torch.Tensor:
    """Return a bidirectional path graph.

    Parameters
    ----------
    num_nodes : int
        Node count (at least 2).

    Returns
    -------
    Tensor
        COO index with shape ``(2, 2 * (num_nodes - 1))``.
    """
    src = list(range(num_nodes - 1)) + list(range(1, num_nodes))
    dst = list(range(1, num_nodes)) + list(range(num_nodes - 1))
    return torch.tensor([src, dst], dtype=torch.long)


def _sequence(
    *,
    num_nodes: int = 3,
    num_timesteps: int = 4,
    in_channels: int = 3,
    seed: int = 0,
    control_inputs: torch.Tensor | None = None,
    timestamps: torch.Tensor | None = None,
    observation_masks: torch.Tensor | None = None,
    presence_masks: torch.Tensor | None = None,
    allow_node_churn: bool = False,
    hyperedges: bool = False,
) -> GraphSnapshotSequence:
    """Build a static-topology homogeneous sequence.

    Parameters
    ----------
    num_nodes, num_timesteps, in_channels, seed
        Snapshot geometry and RNG.
    control_inputs, timestamps, observation_masks, presence_masks
        Optional sequence metadata.
    allow_node_churn : bool, optional
        Fixed-union churn flag. Default is ``False``.
    hyperedges : bool, optional
        Attach a static bipartite incidence. Default is ``False``.

    Returns
    -------
    GraphSnapshotSequence
        Random snapshots on a path graph.
    """
    torch.manual_seed(seed)
    edge_index = _path_edge_index(num_nodes)
    hyperedge_index = (
        torch.tensor([[0, 1, 2], [0, 0, 0]], dtype=torch.long) if hyperedges else None
    )
    snapshots = []
    for _ in range(num_timesteps):
        kwargs: dict[str, torch.Tensor] = {
            "x": torch.randn(num_nodes, in_channels),
            "edge_index": edge_index,
        }
        if hyperedge_index is not None:
            kwargs["hyperedge_index"] = hyperedge_index
        snapshots.append(Data(**kwargs))
    return GraphSnapshotSequence(
        snapshots,
        control_inputs=control_inputs,
        timestamps=timestamps,
        observation_masks=observation_masks,
        presence_masks=presence_masks,
        allow_node_churn=allow_node_churn,
    )


def _make_model(*, seed: int = 0, **kwargs: object) -> GraphKoopmanModel:
    """Construct a small hop-matched per-node model.

    Parameters
    ----------
    seed : int, optional
        RNG seed. Default is ``0``.
    **kwargs
        Forwarded to :class:`~koopman_graph.model.GraphKoopmanModel`.

    Returns
    -------
    GraphKoopmanModel
        One-layer GCN stack.
    """
    torch.manual_seed(seed)
    encoder = GNNEncoder(in_channels=3, hidden_channels=8, latent_dim=4, num_layers=1)
    decoder = GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=3, num_layers=1)
    return GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=4,
        time_step=0.1,
        **kwargs,  # type: ignore[arg-type]
    )


def _ok_koopman(**overrides: object) -> SimpleNamespace:
    """Return a discrete-operator stub that batching accepts.

    Parameters
    ----------
    **overrides
        Attribute replacements.

    Returns
    -------
    SimpleNamespace
        Orbit / family flags.
    """
    payload: dict[str, object] = {
        "auto_orbits": False,
        "isotypic_symmetry": False,
        "orbit_partition": None,
        "uses_orbit_selves": False,
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


def _batching_stub(
    *,
    encoder: nn.Module | None = None,
    **flags: object,
) -> SimpleNamespace:
    """Return a model stub for :func:`validate_graph_batching_request`.

    Parameters
    ----------
    encoder : nn.Module or None, optional
        Encoder to inspect. Default is a one-layer GCN.
    **flags
        Model attributes (``n_delays``, ``control_dim``, topology flags).

    Returns
    -------
    SimpleNamespace
        Minimal trainable-model stand-in.
    """
    defaults: dict[str, object] = {
        "encoder": encoder
        if encoder is not None
        else GNNEncoder(3, 8, 4, num_layers=1),
        "n_delays": 1,
        "control_dim": 0,
        "learns_pairwise_topology": False,
        "adaptive_topology": None,
        "predicted_topology": None,
        "uses_hypergraph_koopman": False,
        "graph_dynamics": None,
        "koopman": _ok_koopman(),
    }
    defaults.update(flags)
    return SimpleNamespace(**defaults)


def test_batch_graphs_rejects_empty_and_short_sequences() -> None:
    """Empty batches and single-snapshot trajectories raise ``ValueError``."""
    model = _make_model()
    with pytest.raises(ValueError, match="at least one GraphSnapshotSequence"):
        validate_graph_batching_request(model, [])
    short = _sequence(num_timesteps=1, seed=3)
    with pytest.raises(ValueError, match="at least 2 snapshots"):
        compute_batched_training_loss(model, (short,), LossWeights())


def test_batch_graphs_rejects_controls_hyperedges_and_timestamps() -> None:
    """Sequence-level control, incidence, and timestamp records are refused."""
    model = _make_model()
    controlled = _sequence(control_inputs=torch.randn(4, 1), seed=4)
    with pytest.raises(ValueError, match="control_inputs"):
        validate_graph_batching_request(model, (controlled,))
    hyper = _sequence(hyperedges=True, seed=5)
    with pytest.raises(ValueError, match="hyperedge"):
        validate_graph_batching_request(model, (hyper,))
    stamped = _sequence(timestamps=torch.arange(4, dtype=torch.float32), seed=6)
    with pytest.raises(ValueError, match="timestamps"):
        validate_graph_batching_request(model, (stamped,))


def test_batch_graphs_rejects_control_dim_adaptive_and_hypergraph_flags() -> None:
    """Model-level control, adaptive topology, and hypergraph flags raise."""
    sequence = _sequence(seed=7)
    with pytest.raises(ValueError, match="control_dim"):
        validate_graph_batching_request(
            _batching_stub(control_dim=1),
            (sequence,),
        )
    with pytest.raises(ValueError, match="adaptive topology"):
        validate_graph_batching_request(
            _batching_stub(adaptive_topology=object()),
            (sequence,),
        )
    with pytest.raises(ValueError, match="hypergraph Koopman"):
        validate_graph_batching_request(
            _batching_stub(uses_hypergraph_koopman=True),
            (sequence,),
        )


def test_batch_graphs_unwraps_delay_then_rejects_relgraph_encoder() -> None:
    """Delay wrappers unwrap before RelGraph / hypergraph encoder guards."""
    sequence = _sequence(seed=8)
    delayed = DelayEmbeddingEncoder(
        RelGraphEncoder(3, 8, 4, 2, num_layers=1),
        n_delays=1,
    )
    with pytest.raises(ValueError, match="RelGraph"):
        validate_graph_batching_request(_batching_stub(encoder=delayed), (sequence,))
    with pytest.raises(ValueError, match="hypergraph"):
        validate_graph_batching_request(
            _batching_stub(encoder=HypergraphEncoder(3, 8, 4, num_layers=1)),
            (sequence,),
        )


def test_batch_graphs_rejects_orbit_and_isotypic_operator_flags() -> None:
    """Orbit-tied and isotypic operator flags are refused."""
    sequence = _sequence(seed=9)
    with pytest.raises(ValueError, match="orbit-tied or isotypic"):
        validate_graph_batching_request(
            _batching_stub(koopman=_ok_koopman(auto_orbits=True)),
            (sequence,),
        )
    with pytest.raises(ValueError, match="orbit-tied or isotypic"):
        validate_graph_batching_request(
            _batching_stub(koopman=_ok_koopman(isotypic_symmetry=True)),
            (sequence,),
        )
    with pytest.raises(ValueError, match="orbit-tied"):
        validate_graph_batching_request(
            _batching_stub(koopman=_ok_koopman(uses_orbit_selves=True)),
            (sequence,),
        )


def test_batch_graphs_allows_nonrecursive_predicted_topology() -> None:
    """A topology head without recursive training is not a batching refusal."""
    model = _make_model(
        seed=26,
        graph_dynamics=GraphDynamicsConfig(
            topology_head="sparse_candidate",
            recursive_training=False,
            topology_loss_weight=0.0,
            presence_loss_weight=0.0,
        ),
    )
    sequence = _sequence(seed=27)
    validate_graph_batching_request(model, (sequence,))


def test_batched_loss_skips_inactive_recon_or_forward_terms() -> None:
    """Zero reconstruction / forward weights skip the matching vectorized loop."""
    model = _make_model(seed=28)
    model.eval()
    sequence = _sequence(seed=29)
    forward_only = compute_batched_training_loss(
        model,
        (sequence,),
        LossWeights(reconstruction=0.0, forward=1.0),
    )
    assert torch.isfinite(forward_only.forward)
    recon_only = compute_batched_training_loss(
        model,
        (sequence,),
        LossWeights(reconstruction=1.0, forward=0.0),
    )
    assert torch.isfinite(recon_only.reconstruction)
    inactive = compute_batched_training_loss(
        model,
        (sequence,),
        LossWeights(reconstruction=0.0, forward=0.0),
    )
    assert float(inactive.total) == pytest.approx(0.0, abs=1e-8)


def test_batched_loss_uses_observation_masks_and_graph_state_weights() -> None:
    """Masked reconstruction and leftover graph-state weights stay finite."""
    model = _make_model(
        seed=10,
        graph_dynamics=GraphDynamicsConfig(
            topology_head="none",
            recursive_training=False,
            topology_loss_weight=1.0,
            presence_loss_weight=1.0,
        ),
    )
    model.eval()
    masks = torch.ones(4, 3, dtype=torch.bool)
    masks[1, 0] = False
    sequence = _sequence(observation_masks=masks, seed=11)
    breakdown = compute_batched_training_loss(model, (sequence,), LossWeights())
    assert torch.isfinite(breakdown.reconstruction)
    assert torch.isfinite(breakdown.total)


def test_identification_fit_request_rejects_unsupported_layouts() -> None:
    """Identification refuses non-config objects and unsupported families."""
    model = _make_model()
    sequence = _sequence(seed=12)
    config = IdentificationConfig()
    with pytest.raises(TypeError, match="IdentificationConfig"):
        _validate_identification_fit_request(
            model,
            (sequence,),
            "ridge",
            window_sampler=None,
            batch_graphs=False,
        )
    with pytest.raises(ValueError, match="batch_graphs"):
        _validate_identification_fit_request(
            model,
            (sequence,),
            config,
            window_sampler=None,
            batch_graphs=True,
        )
    hetero = HeteroData()
    hetero["node"].x = torch.randn(3, 3)
    hetero["node", "r", "node"].edge_index = torch.tensor(
        [[0, 1], [1, 2]], dtype=torch.long
    )
    hetero_seq = HeteroGraphSnapshotSequence([hetero, hetero.clone()])
    with pytest.raises(ValueError, match="HeteroGraphSnapshotSequence"):
        _validate_identification_fit_request(
            model,
            (hetero_seq,),
            config,
            window_sampler=None,
            batch_graphs=False,
        )
    presence = torch.ones(4, 3, dtype=torch.bool)
    churn = _sequence(presence_masks=presence, allow_node_churn=True, seed=13)
    with pytest.raises(ValueError, match="allow_node_churn"):
        _validate_identification_fit_request(
            model,
            (churn,),
            config,
            window_sampler=None,
            batch_graphs=False,
        )


def test_identification_fit_request_rejects_operator_and_control_flags() -> None:
    """Continuous, delay, controlled, adaptive, and non-dense maps raise."""
    sequence = _sequence(seed=14)
    config = IdentificationConfig()
    continuous = SimpleNamespace(
        koopman_kind="pernode",
        dynamics_mode="continuous",
        n_delays=1,
        control_dim=0,
        adaptive_topology=None,
        koopman=KoopmanOperator(4),
    )
    with pytest.raises(ValueError, match="dynamics_mode"):
        _validate_identification_fit_request(
            continuous,
            (sequence,),
            config,
            window_sampler=None,
            batch_graphs=False,
        )
    delayed = SimpleNamespace(
        koopman_kind="pernode",
        dynamics_mode="discrete",
        n_delays=2,
        control_dim=0,
        adaptive_topology=None,
        koopman=KoopmanOperator(4),
    )
    with pytest.raises(ValueError, match="n_delays"):
        _validate_identification_fit_request(
            delayed,
            (sequence,),
            config,
            window_sampler=None,
            batch_graphs=False,
        )
    controlled = SimpleNamespace(
        koopman_kind="pernode",
        dynamics_mode="discrete",
        n_delays=1,
        control_dim=1,
        adaptive_topology=None,
        koopman=KoopmanOperator(4),
    )
    with pytest.raises(ValueError, match="control_dim"):
        _validate_identification_fit_request(
            controlled,
            (sequence,),
            config,
            window_sampler=None,
            batch_graphs=False,
        )
    adaptive = SimpleNamespace(
        koopman_kind="pernode",
        dynamics_mode="discrete",
        n_delays=1,
        control_dim=0,
        adaptive_topology=object(),
        koopman=KoopmanOperator(4),
    )
    with pytest.raises(ValueError, match="pairwise topology"):
        _validate_identification_fit_request(
            adaptive,
            (sequence,),
            config,
            window_sampler=None,
            batch_graphs=False,
        )
    injected = SimpleNamespace(
        koopman_kind="pernode",
        dynamics_mode="discrete",
        n_delays=1,
        control_dim=0,
        adaptive_topology=None,
        koopman=object(),
    )
    with pytest.raises(ValueError, match="KoopmanOperator only"):
        _validate_identification_fit_request(
            injected,
            (sequence,),
            config,
            window_sampler=None,
            batch_graphs=False,
        )


def test_identification_pairs_require_two_snapshots_and_encode_fallback() -> None:
    """Pair collection rejects short series; encode is used without ``encode_at``."""
    model = _make_model(seed=15)
    short = _sequence(num_timesteps=1, seed=16)
    with pytest.raises(ValueError, match="at least two snapshots"):
        _collect_identification_pairs(model, (short,))

    sequence = _sequence(num_timesteps=3, seed=17)

    class _EncodeOnly:
        """Minimal encode surface without ``encode_at``."""

        encode_at = None

        @staticmethod
        def encode(snapshot: Data) -> torch.Tensor:
            """Return node features.

            Parameters
            ----------
            snapshot : Data
                Graph snapshot.

            Returns
            -------
            Tensor
                Node feature matrix.
            """
            return snapshot.x

    stacked = _encode_sequence_latents(_EncodeOnly(), sequence)  # type: ignore[arg-type]
    assert stacked.shape == (3, 3, 3)


def test_bind_pending_orbit_ties_empty_and_heterodata_on_homogeneous() -> None:
    """Empty trains are a no-op; HeteroData on a homogeneous path raises."""
    model = _make_model()
    bind_pending_orbit_ties(model, [])

    hetero = HeteroData()
    hetero["node"].x = torch.zeros(2, 2)
    hetero["node", "r", "node"].edge_index = torch.tensor(
        [[0, 1], [1, 0]], dtype=torch.long
    )

    class _HomogeneousLooking:
        """Sequence that is not a hetero container but yields HeteroData."""

        def __getitem__(self, _index: int) -> HeteroData:
            return hetero

    koopman = SimpleNamespace(
        auto_orbits=True,
        isotypic_symmetry=False,
        orbit_partition=None,
        ensure_orbit_binding=lambda *_args, **_kwargs: None,
    )
    with pytest.raises(ValueError, match="homogeneous Data"):
        bind_pending_orbit_ties(
            SimpleNamespace(koopman=koopman),
            [_HomogeneousLooking()],  # type: ignore[list-item]
        )


def test_run_fit_loop_rejects_val_monitor_and_dual_window_args() -> None:
    """Loop-level guards fire when ``fit`` resolution is bypassed."""
    model = _make_model(seed=18)
    sequence = _sequence(num_timesteps=4, seed=19)
    with pytest.raises(ValueError, match="val_sequences"):
        run_fit_loop(
            model,
            (sequence,),
            epochs=1,
            early_stopping_monitor="val",
        )
    sampler = WindowSampler((sequence,), window_length=3, batch_size=1, seed=0)
    with pytest.raises(ValueError, match="sampler or window_length"):
        run_fit_loop(
            model,
            (sequence,),
            epochs=1,
            sampler=sampler,
            window_length=3,
        )


def test_fit_batch_graphs_validates_val_sequences() -> None:
    """``batch_graphs=True`` also validates the held-out trajectories."""
    model = _make_model(koopman="graph", seed=20)
    train = _sequence(num_nodes=3, num_timesteps=3, seed=21)
    val = _sequence(num_nodes=4, num_timesteps=3, seed=22)
    history = model.fit(
        train,
        validation_sequence=val,
        epochs=1,
        lr=1e-2,
        batch_graphs=True,
    )
    assert history.epochs == 1
    assert history.val_loss is not None


def test_identification_fit_trains_encoder_or_rejects_scheduler() -> None:
    """Identification builds an encoder Adam step, or refuses a scheduler."""
    sequence = _sequence(num_timesteps=5, seed=23)
    trained = _make_model(seed=24)
    history = trained.fit(
        sequence,
        epochs=1,
        lr=1e-2,
        identification=IdentificationConfig(solver="ridge", ridge=1e-2),
    )
    assert history.epochs == 1
    assert trained.identification_report is not None

    class _IdentityCodec(nn.Module):
        """Parameter-free pass-through codec."""

        def __init__(self, dim: int) -> None:
            super().__init__()
            self.in_channels = dim
            self.latent_dim = dim
            self.out_channels = dim
            self.num_layers = 1

        def forward(
            self,
            x_or_data: torch.Tensor | Data,
            edge_index: torch.Tensor | None = None,
            edge_weight: torch.Tensor | None = None,
        ) -> torch.Tensor:
            """Return node features unchanged."""
            del edge_index, edge_weight
            if isinstance(x_or_data, Data):
                return x_or_data.x
            return x_or_data

    frozen = GraphKoopmanModel(
        encoder=_IdentityCodec(3),
        decoder=_IdentityCodec(3),
        latent_dim=3,
        time_step=0.1,
        koopman_init_mode="identity",
        koopman_init_scale=0.0,
    )
    dummy_opt = torch.optim.Adam(trained.parameters(), lr=1e-2)
    with pytest.raises(ValueError, match="lr_scheduler"):
        frozen.fit(
            sequence,
            epochs=1,
            identification=IdentificationConfig(solver="ridge"),
            lr_scheduler=StepLR(dummy_opt, step_size=1),
        )


def test_windowed_identification_without_trainable_params_raises() -> None:
    """Windowed identification with no encoder parameters is refused."""

    class _IdentityCodec(nn.Module):
        """Parameter-free pass-through codec."""

        def __init__(self, dim: int) -> None:
            super().__init__()
            self.in_channels = dim
            self.latent_dim = dim
            self.out_channels = dim
            self.num_layers = 1

        def forward(
            self,
            x_or_data: torch.Tensor | Data,
            edge_index: torch.Tensor | None = None,
            edge_weight: torch.Tensor | None = None,
        ) -> torch.Tensor:
            """Return node features unchanged."""
            del edge_index, edge_weight
            if isinstance(x_or_data, Data):
                return x_or_data.x
            return x_or_data

    sequence = _sequence(num_timesteps=5, seed=25)
    frozen = GraphKoopmanModel(
        encoder=_IdentityCodec(3),
        decoder=_IdentityCodec(3),
        latent_dim=3,
        time_step=0.1,
        koopman_init_mode="identity",
        koopman_init_scale=0.0,
    )
    with (
        patch("koopman_graph.training.loop._validate_identification_fit_request"),
        pytest.raises(ValueError, match="windowed fit requires trainable"),
    ):
        frozen.fit(
            MultiTrajectory((sequence,)),
            epochs=1,
            window_length=3,
            identification=IdentificationConfig(solver="ridge"),
        )


def test_fit_callbacks_observe_encodings_and_on_fit_end() -> None:
    """Callbacks receive a frozen latent stack and a final history hook."""
    model = _make_model(seed=30)
    sequence = _sequence(num_timesteps=3, seed=31)
    seen: list[tuple[int, ...]] = []

    class _Observer(NoOpFitCallback):
        """Record encoding shapes and fit-end epochs."""

        def observe_encodings(self, encodings: torch.Tensor) -> None:
            """Store the time-major latent shape."""
            seen.append(tuple(encodings.shape))

    history = model.fit(sequence, epochs=1, lr=1e-2, callbacks=[_Observer()])
    assert history.epochs == 1
    assert seen and seen[0][0] == sequence.num_timesteps
