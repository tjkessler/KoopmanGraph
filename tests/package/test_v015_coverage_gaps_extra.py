"""Extra release-gate coverage for 0.15 modules still under 90%.

Leftover lines after this file plus the existing scientific suites are
defensive or unreachable through public APIs:

* ``mpc/tube.py`` ``564-565`` — ``y_min`` / ``y_max`` are required kwargs.
* ``nn/predicted_topology.py`` ``152``, ``158-159``, ``559``, ``565-566``,
  ``641`` — empty candidate / bucket paths cannot occur for ``N >= 2``
  and ``k >= 1``; ``916-917`` is guarded by ``resolve_topology_policy``.
* ``analysis/causal_intervention.py`` ``241-242`` — public samplers always
  pass ``intervene_value`` when a source is set.
* ``nn/constraint_decoders.py`` ``61`` — empty ``channels`` raise in
  ``_ConstraintHead`` before ``_as_channel_index``.
* ``adaptation/joint_observer.py`` ``136-140`` — non-dense graph
  parameterization is refused at construction.
* ``baselines/gedmd.py`` ``213-217``, ``282-283`` — stacked ``dx_dt``
  always matches ``x``, and ``GraphSnapshotSequence`` refuses ``T = 0``.
* ``baselines/hankel_dmd.py`` ``285`` — ``_require_operator`` raises
  before the ``state_dim is None`` check on an unfitted baseline.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tests.adaptation.test_joint_observer import _sequence
from tests.mpc.test_mpc import _identity_plant_model, _origin
from torch import nn
from torch_geometric.data import Data

from koopman_graph import GraphKoopmanModel
from koopman_graph.adaptation import JointStateTopologyObserver
from koopman_graph.analysis.causal_intervention import (
    SyntheticInterventionReport,
    SyntheticSCM,
    recover_synthetic_interventional_edges,
    sample_synthetic_intervention,
    sample_synthetic_observational,
    teaching_three_node_scm,
)
from koopman_graph.analysis.hodge_modes import (
    HodgeModeComponents,
    hodge_decompose_modes,
)
from koopman_graph.analysis.memory import (
    LAG_UNIT,
    FiniteMemoryKoopman,
    MarkovClosureReport,
    markov_closure_report,
)
from koopman_graph.baselines import GEDMDBaseline, HankelDMDBaseline
from koopman_graph.baselines.gedmd import polynomial_observable_derivatives
from koopman_graph.baselines.hankel_dmd import assemble_delay_state, delay_embed_rows
from koopman_graph.data import (
    EntityRemap,
    GraphDynamicsConfig,
    GraphSnapshotSequence,
    remap_node_features,
)
from koopman_graph.mpc import TubeKoopmanMPC, TubeMPCReport, ensemble_residual_radii
from koopman_graph.nn import (
    LinearConservingDecoder,
    MassConservingDecoder,
    PositivityDecoder,
    PredictedTopologyHead,
    PresenceHead,
    SparseCandidateTopologyHead,
    build_candidate_index,
    build_supervision_index,
)
from koopman_graph.nn.constraint_decoders import project_linear_conservation
from koopman_graph.nn.predicted_topology import (
    candidate_edge_labels,
    decode_weighted_topology,
    resolve_rollout_topology_at,
    resolve_topology_policy,
)
from koopman_graph.nn.separable import (
    SeparableDictionaryDecoder,
    SeparableDictionaryEncoder,
)
from koopman_graph.spectrum_types import KoopmanSpectrum
from koopman_graph.uq import JointCoverageSpec
from koopman_graph.uq.common import PredictionInterval, snapshot_with_features


def _coverage() -> JointCoverageSpec:
    """Return the shipped residual-tube coverage spec."""
    return JointCoverageSpec()


def _two_node_edges() -> torch.Tensor:
    """Return a bidirectional two-node edge index."""
    return torch.tensor([[0, 1], [1, 0]], dtype=torch.long)


def _path_edges(num_nodes: int) -> torch.Tensor:
    """Return an oriented path on ``num_nodes`` vertices."""
    tails = torch.arange(num_nodes - 1, dtype=torch.long)
    return torch.stack((tails, tails + 1), dim=0)


def _interval_template() -> Data:
    """Return a two-node template snapshot."""
    return Data(x=torch.zeros(2, 2), edge_index=_two_node_edges())


def _real_spectrum(columns: torch.Tensor) -> KoopmanSpectrum:
    """Build a spectrum whose eigenvectors stay real."""
    modes = int(columns.shape[1])
    eigenvalues = torch.ones(modes)
    return KoopmanSpectrum(
        eigenvalues=eigenvalues,
        eigenvectors=columns,
        magnitudes=eigenvalues.abs(),
        growth_rates=torch.zeros(modes),
        frequencies=torch.zeros(modes),
        time_step=1.0,
    )


def _gedmd_sequence(
    states: list[torch.Tensor],
    edge_index: torch.Tensor,
    *,
    derivatives: list[torch.Tensor] | None = None,
) -> GraphSnapshotSequence:
    """Build a two-node scalar-feature sequence for gEDMD guards."""
    snapshots = []
    for index, state in enumerate(states):
        payload: dict[str, torch.Tensor] = {
            "x": state.reshape(2, 1),
            "edge_index": edge_index,
        }
        if derivatives is not None:
            payload["dx_dt"] = derivatives[index].reshape(2, 1)
        snapshots.append(Data(**payload))
    return GraphSnapshotSequence(snapshots)


class _ChannelCopyDecoder(nn.Module):
    """Copy the first two latent columns as decoded features."""

    def forward(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del edge_index, edge_weight
        return z[:, :2]


# ---------------------------------------------------------------------------
# mpc/tube.py
# ---------------------------------------------------------------------------


def test_ensemble_residual_radii_reject_empty_and_mismatched_bands() -> None:
    """Empty or length-mismatched prediction bands raise."""
    empty = PredictionInterval(
        mean=(),
        lower=(),
        upper=(),
        level=0.9,
        n_members=1,
    )
    with pytest.raises(ValueError, match="at least one forecast step"):
        ensemble_residual_radii(empty)
    template = _interval_template()
    band = snapshot_with_features(template, torch.zeros(2, 2))
    mismatched = PredictionInterval(
        mean=(band,),
        lower=(band,),
        upper=(band, band),
        level=0.9,
        n_members=2,
    )
    with pytest.raises(ValueError, match="lengths differ"):
        ensemble_residual_radii(mismatched)


def test_ensemble_residual_radii_reject_missing_features_and_negative_width() -> None:
    """Homogeneous ``x`` is required; half-widths must be non-negative."""
    template = _interval_template()
    bare = Data(edge_index=template.edge_index)
    featured = snapshot_with_features(template, torch.zeros(2, 2))
    missing = PredictionInterval(
        mean=(featured,),
        lower=(bare,),
        upper=(featured,),
        level=0.9,
        n_members=1,
    )
    with pytest.raises(TypeError, match="homogeneous node features"):
        ensemble_residual_radii(missing)
    lower = snapshot_with_features(template, torch.ones(2, 2))
    upper = snapshot_with_features(template, torch.zeros(2, 2))
    inverted = PredictionInterval(
        mean=(featured,),
        lower=(lower,),
        upper=(upper,),
        level=0.9,
        n_members=1,
    )
    with pytest.raises(ValueError, match="non-negative"):
        ensemble_residual_radii(inverted)


def test_tube_residual_source_shape_and_type_guards() -> None:
    """Scalar, short, rank-2, rank-3, negative, and typed residuals are checked."""
    pytest.importorskip("osqp")
    model = _identity_plant_model()
    y_min = torch.tensor([-2.0, -2.0])
    y_max = torch.tensor([2.0, 2.0])
    kwargs = {
        "horizon": 2,
        "Q": torch.eye(2),
        "R": torch.eye(1),
        "y_min": y_min,
        "y_max": y_max,
    }
    scalar = TubeKoopmanMPC(model, residual_source=torch.tensor(0.05), **kwargs)
    assert scalar.horizon == 2
    assert scalar.model is model
    rank2 = TubeKoopmanMPC(
        model,
        residual_source=torch.tensor([[0.05, 0.02], [0.04, 0.01]]),
        **kwargs,
    )
    assert rank2.horizon == 2
    with pytest.raises(ValueError, match="must cover horizon"):
        TubeKoopmanMPC(model, residual_source=torch.tensor([0.1]), **kwargs)
    with pytest.raises(ValueError, match="must cover horizon"):
        TubeKoopmanMPC(
            model,
            residual_source=torch.ones(1, 2),
            **kwargs,
        )
    with pytest.raises(ValueError, match="scalar, shape \\(H,\\)"):
        TubeKoopmanMPC(model, residual_source=torch.ones(2, 2, 2), **kwargs)
    with pytest.raises(ValueError, match="non-negative"):
        TubeKoopmanMPC(model, residual_source=torch.tensor([-0.1, 0.1]), **kwargs)
    with pytest.raises(TypeError, match="residual_source must be"):
        TubeKoopmanMPC(model, residual_source="radii", **kwargs)


def test_tube_report_validates_counts_and_rates() -> None:
    """``TubeMPCReport`` refuses inconsistent counts and rates."""
    coverage = _coverage()
    with pytest.raises(ValueError, match="n_steps must be >= 1"):
        TubeMPCReport(
            violation_rate=0.0,
            feasibility_rate=1.0,
            cost=0.0,
            n_steps=0,
            n_feasible=0,
            n_violations=0,
            coverage=coverage,
        )
    with pytest.raises(ValueError, match="n_feasible must lie"):
        TubeMPCReport(
            violation_rate=0.0,
            feasibility_rate=2.0,
            cost=0.0,
            n_steps=2,
            n_feasible=3,
            n_violations=0,
            coverage=coverage,
        )
    with pytest.raises(ValueError, match="n_violations must lie"):
        TubeMPCReport(
            violation_rate=1.0,
            feasibility_rate=1.0,
            cost=0.0,
            n_steps=2,
            n_feasible=2,
            n_violations=-1,
            coverage=coverage,
        )
    with pytest.raises(ValueError, match="feasibility_rate must equal"):
        TubeMPCReport(
            violation_rate=0.0,
            feasibility_rate=0.5,
            cost=0.0,
            n_steps=2,
            n_feasible=2,
            n_violations=0,
            coverage=coverage,
        )
    with pytest.raises(ValueError, match="violation_rate must equal"):
        TubeMPCReport(
            violation_rate=0.5,
            feasibility_rate=1.0,
            cost=0.0,
            n_steps=2,
            n_feasible=2,
            n_violations=0,
            coverage=coverage,
        )


def test_tube_rollout_evaluate_and_infeasible_fallback() -> None:
    """Rollout, ``steps < 1``, and clipped-zero fallback on an infeasible QP."""
    pytest.importorskip("osqp")
    model = _identity_plant_model()
    y_min = torch.tensor([-2.0, -2.0])
    y_max = torch.tensor([2.0, 2.0])
    feasible = TubeKoopmanMPC(
        model,
        horizon=2,
        Q=torch.eye(2),
        R=0.05 * torch.eye(1),
        residual_source=np.array(0.0),
        y_min=y_min,
        y_max=y_max,
        u_min=torch.tensor([-1.0]),
        u_max=torch.tensor([1.0]),
    )
    snapshots = feasible.rollout(_origin(), torch.tensor([0.0, 0.0]), steps=2)
    assert len(snapshots) == 2
    with pytest.raises(ValueError, match="steps must be >= 1"):
        feasible.evaluate(_origin(), torch.tensor([0.0, 0.0]), steps=0)
    with pytest.raises(ValueError, match="steps must be >= 1"):
        feasible.rollout(_origin(), torch.tensor([0.0, 0.0]), steps=0)
    infeasible = TubeKoopmanMPC(
        model,
        horizon=2,
        Q=torch.eye(2),
        R=torch.eye(1),
        residual_source=torch.zeros(2),
        y_min=torch.tensor([5.0, 5.0]),
        y_max=torch.tensor([6.0, 6.0]),
        u_min=torch.tensor([0.1]),
        u_max=torch.tensor([0.2]),
    )
    report = infeasible.evaluate(_origin(), torch.tensor([0.0, 0.0]), steps=2)
    assert report.feasibility_rate == pytest.approx(0.0)
    assert report.n_feasible == 0


# ---------------------------------------------------------------------------
# nn/predicted_topology.py
# ---------------------------------------------------------------------------


def test_topology_heads_cover_self_loops_empty_keep_and_presence() -> None:
    """Self-loops, empty keep, presence guards, and dense thresholding."""
    loops = torch.tensor([[0, 0, 1], [0, 1, 1]], dtype=torch.long)
    index = build_candidate_index(
        3, 2, loops, generator=torch.Generator().manual_seed(0)
    )
    assert int((index[0] == index[1]).sum()) == 0
    head = PredictedTopologyHead(3, hidden_dim=8)
    z = torch.randn(4, 3)
    kept = head.edge_index(z, threshold=-1e9)
    assert kept.shape[0] == 2
    assert kept.shape[1] > 0
    sparse = SparseCandidateTopologyHead(3, hidden_dim=8, candidate_k=2)
    candidates = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    with pytest.raises(ValueError, match="shape \\(2, E\\)"):
        sparse.pair_logits(z[:3], torch.arange(4))
    empty_keep = sparse.edge_index(z[:3], candidates, threshold=1e9)
    assert empty_keep.shape == (2, 3)
    with pytest.raises(ValueError, match="latent_dim"):
        PresenceHead(0)
    presence = PresenceHead(3)
    assert presence(z).shape == (4,)
    with pytest.raises(ValueError, match="shape \\(N, d\\)"):
        presence(torch.randn(3))
    with pytest.raises(ValueError, match="latent width"):
        presence(torch.randn(4, 2))


def test_supervision_labels_decode_and_rollout_policy_guards() -> None:
    """Supervision COO, labels, decode, and rollout-policy error paths."""
    with pytest.raises(ValueError, match="num_nodes"):
        build_supervision_index(1, 2, None, _path_edges(2))
    with pytest.raises(ValueError, match="candidate_k"):
        build_supervision_index(3, 0, None, _path_edges(3))
    with pytest.raises(ValueError, match="edge_index must have shape"):
        build_supervision_index(3, 2, torch.arange(3), _path_edges(3))
    next_index = torch.tensor([[0, 1, 2], [0, 2, 1]], dtype=torch.long)
    current = torch.tensor([[0, 1], [2, 2]], dtype=torch.long)
    overlap = build_supervision_index(
        3,
        2,
        current,
        next_index,
        device="cpu",
        generator=torch.Generator().manual_seed(2),
    )
    assert overlap.shape[0] == 2
    supervised = build_supervision_index(
        3,
        2,
        None,
        next_index,
        generator=torch.Generator().manual_seed(1),
    )
    assert supervised.shape[0] == 2
    labels = candidate_edge_labels(supervised, next_index, 3)
    assert labels.shape[0] == int(supervised.shape[1])
    with pytest.raises(ValueError, match="candidate_index must have shape"):
        candidate_edge_labels(torch.arange(3), next_index, 3)
    with pytest.raises(ValueError, match="true_next_index must have shape"):
        candidate_edge_labels(supervised, torch.arange(3), 3)
    empty = torch.empty(2, 0, dtype=torch.long)
    assert candidate_edge_labels(empty, next_index, 3).shape == (0,)
    zeros = candidate_edge_labels(supervised, empty, 3)
    assert torch.equal(zeros, torch.zeros(int(supervised.shape[1])))
    sparse = SparseCandidateTopologyHead(2, hidden_dim=4, candidate_k=2)
    dense = PredictedTopologyHead(2, hidden_dim=4)
    z = torch.randn(3, 2)
    edges, weights = decode_weighted_topology(sparse, z, _path_edges(3))
    assert edges.shape[0] == 2
    assert weights.ndim == 1
    dense_edges, dense_weights = decode_weighted_topology(dense, z, _path_edges(3))
    assert dense_edges.shape[1] == 3 * 2
    assert dense_weights.shape == (6,)
    with pytest.raises(TypeError, match="PredictedTopologyHead or"):
        decode_weighted_topology(nn.Linear(2, 1), z, _path_edges(3))
    dummy = SimpleNamespace(predicted_topology=None, graph_dynamics=None)
    with pytest.raises(ValueError, match="topology_policy must be one of"):
        resolve_topology_policy(dummy, "oracle")
    assert resolve_topology_policy(dummy, "hold_last") == "hold_last"
    with pytest.raises(ValueError, match="requires a predicted topology head"):
        resolve_topology_policy(dummy, "recursive")
    origin = _path_edges(3)
    hold = resolve_rollout_topology_at(dummy, origin, None, topology_policy="hold_last")
    held = hold(0)
    assert held[0].shape[0] == 2
    future = [Data(x=torch.zeros(3, 1), edge_index=_path_edges(3))]
    oracle = resolve_rollout_topology_at(dummy, origin, None, future_topologies=future)
    assert callable(oracle)
    with pytest.raises(ValueError, match="requires a predicted topology head"):
        resolve_rollout_topology_at(dummy, origin, None, topology_policy="recursive")
    wrong = SimpleNamespace(predicted_topology=nn.Identity(), graph_dynamics=None)
    with pytest.raises(TypeError, match="predicted_topology must be"):
        resolve_rollout_topology_at(wrong, origin, None, topology_policy="recursive")
    configured = SimpleNamespace(
        predicted_topology=sparse,
        graph_dynamics=GraphDynamicsConfig(recursive_training=True),
    )
    recursive = resolve_rollout_topology_at(
        configured,
        origin,
        None,
        topology_policy="auto",
    )
    predicted = recursive(0, z)
    assert predicted[0].shape[0] == 2
    hold_auto = SimpleNamespace(
        predicted_topology=sparse,
        graph_dynamics=GraphDynamicsConfig(recursive_training=False),
    )
    assert resolve_topology_policy(hold_auto, "auto") == "hold_last"


# ---------------------------------------------------------------------------
# analysis/causal_intervention.py
# ---------------------------------------------------------------------------


def test_synthetic_scm_and_report_reject_invalid_weights_and_flags() -> None:
    """Square-weight, noise, flag, and report guards raise."""
    with pytest.raises(ValueError, match="shape \\(N, N\\)"):
        SyntheticSCM(weights=torch.zeros(3), noise_scale=0.1, seed=0)
    with pytest.raises(ValueError, match="N >= 2"):
        SyntheticSCM(weights=torch.zeros(1, 1), noise_scale=0.1, seed=0)
    nan_weights = torch.zeros(2, 2)
    nan_weights[0, 1] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        SyntheticSCM(weights=nan_weights, noise_scale=0.1, seed=0)
    diag = torch.eye(2)
    with pytest.raises(ValueError, match="zero diagonal"):
        SyntheticSCM(weights=diag, noise_scale=0.1, seed=0)
    ok = torch.zeros(2, 2)
    ok[0, 1] = 0.5
    with pytest.raises(ValueError, match="noise_scale"):
        SyntheticSCM(weights=ok, noise_scale=0.0, seed=0)
    with pytest.raises(ValueError, match="labeled_synthetic must be True"):
        SyntheticSCM(weights=ok, noise_scale=0.1, seed=0, labeled_synthetic=False)
    scores = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="scores must have shape"):
        SyntheticInterventionReport(
            scores=scores,
            recovered_edges=(),
            true_edges=(),
            threshold=0.2,
        )
    with pytest.raises(ValueError, match="threshold must be > 0"):
        SyntheticInterventionReport(
            scores=torch.zeros(2, 2),
            recovered_edges=(),
            true_edges=(),
            threshold=0.0,
        )
    with pytest.raises(ValueError, match="labeled_synthetic must be True"):
        SyntheticInterventionReport(
            scores=torch.zeros(2, 2),
            recovered_edges=(),
            true_edges=(),
            threshold=0.2,
            labeled_synthetic=False,
        )


def test_synthetic_sampling_and_recovery_guards() -> None:
    """Sample-count, intervention index, teaching weight, and threshold guards."""
    with pytest.raises(ValueError, match="edge_weight must be nonzero"):
        teaching_three_node_scm(edge_weight=0.0)
    scm = teaching_three_node_scm(seed=2)
    observed = sample_synthetic_observational(scm, 8)
    assert observed.shape == (8, 3)
    intervened = sample_synthetic_intervention(scm, source=0, value=0.5, n_samples=8)
    assert intervened.shape == (8, 3)
    with pytest.raises(ValueError, match="n_samples must be >= 1"):
        sample_synthetic_observational(scm, 0)
    with pytest.raises(ValueError, match="outside"):
        sample_synthetic_intervention(scm, source=5, value=1.0, n_samples=4)
    with pytest.raises(ValueError, match="threshold must be > 0"):
        recover_synthetic_interventional_edges(scm, threshold=0.0)


# ---------------------------------------------------------------------------
# nn/constraint_decoders.py
# ---------------------------------------------------------------------------


def test_constraint_decoder_channel_and_table_guards() -> None:
    """Channel indices and decoded-table validation raise."""
    inner = _ChannelCopyDecoder()
    with pytest.raises(ValueError, match="non-empty sequence"):
        MassConservingDecoder(inner, channels=(), mass=1.0)
    with pytest.raises(TypeError, match="nn.Module"):
        MassConservingDecoder("decoder", channels=(0,), mass=1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be ints"):
        MassConservingDecoder(inner, channels=(True,), mass=1.0).project(  # type: ignore[arg-type]
            torch.randn(4, 2)
        )
    with pytest.raises(ValueError, match="unique"):
        MassConservingDecoder(inner, channels=(0, 0), mass=1.0).project(
            torch.randn(4, 2)
        )
    headed = MassConservingDecoder(inner, channels=(0,), mass=1.0)
    with pytest.raises(TypeError, match="must be a Tensor"):
        headed.project([1.0, 2.0])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be real"):
        headed.project(torch.ones(3, 2, dtype=torch.complex64))
    with pytest.raises(ValueError, match="floating-point"):
        headed.project(torch.ones(3, 2, dtype=torch.int64))
    with pytest.raises(ValueError, match="both axes >= 1"):
        headed.project(torch.ones(3))
    bad = torch.ones(3, 2)
    bad[0, 0] = float("nan")
    with pytest.raises(ValueError, match="must be finite"):
        headed.project(bad)
    with pytest.raises(ValueError, match="finite float"):
        MassConservingDecoder(inner, channels=(0,), mass="1.0")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="mass must be finite"):
        MassConservingDecoder(inner, channels=(0,), mass=float("inf"))


def test_project_linear_conservation_and_positivity_forward() -> None:
    """Linear conservation shape guards and positivity ``softplus`` / forward."""
    values = torch.ones(4)
    with pytest.raises(ValueError, match="shape \\(n_eq, n_nodes\\)"):
        project_linear_conservation(values, torch.ones(4), torch.ones(1))
    with pytest.raises(ValueError, match="shape \\(n_eq,\\)"):
        project_linear_conservation(values, torch.ones(1, 4), torch.ones(2))
    with pytest.raises(ValueError, match="at least one equation"):
        project_linear_conservation(values, torch.ones(0, 4), torch.ones(0))
    nan_c = torch.ones(1, 4)
    nan_c[0, 0] = float("nan")
    with pytest.raises(ValueError, match="constraint and target must be finite"):
        project_linear_conservation(values, nan_c, torch.ones(1))
    inner = _ChannelCopyDecoder()
    positive = PositivityDecoder(inner, channels=(0,), method="softplus")
    table = torch.tensor([[-1.0, 0.5], [0.0, -2.0], [2.0, 1.0], [-0.5, 0.0]])
    projected = positive.project(table)
    assert bool((projected[:, 0] >= 0).all().item())
    decoded = positive(torch.randn(4, 3), _path_edges(4))
    assert decoded.shape == (4, 2)
    with pytest.raises(TypeError, match="must be tensors"):
        LinearConservingDecoder(
            inner,
            channels=(0,),
            constraint="C",  # type: ignore[arg-type]
            target=torch.ones(1),
        )
    with pytest.raises(ValueError, match="n_eq >= 1"):
        LinearConservingDecoder(
            inner,
            channels=(0,),
            constraint=torch.ones(3),
            target=torch.ones(1),
        )
    with pytest.raises(ValueError, match="shape \\(n_eq,\\)"):
        LinearConservingDecoder(
            inner,
            channels=(0,),
            constraint=torch.ones(1, 4),
            target=torch.ones(2),
        )
    inf_c = torch.ones(1, 4)
    inf_c[0, 0] = float("inf")
    with pytest.raises(ValueError, match="constraint and target must be finite"):
        LinearConservingDecoder(
            inner,
            channels=(0,),
            constraint=inf_c,
            target=torch.ones(1),
        )


# ---------------------------------------------------------------------------
# data/remap.py
# ---------------------------------------------------------------------------


def test_entity_remap_construction_and_snapshot_edge_cases() -> None:
    """Empty union, bad index, missing ``x``, empty edges, and empty trajectories."""
    with pytest.raises(ValueError, match="non-empty"):
        EntityRemap(entity_ids=(), index=torch.tensor([0]))
    with pytest.raises(ValueError, match="1-D tensor"):
        EntityRemap(entity_ids=("a", "b"), index=torch.tensor([[0, 1]]))
    with pytest.raises(ValueError, match="at least one source row"):
        EntityRemap(entity_ids=("a",), index=torch.tensor([], dtype=torch.long))
    remap = EntityRemap(entity_ids=("a", "b", "c"), index=torch.tensor([0, 2]))
    with pytest.raises(ValueError, match="requires Data.x"):
        remap.apply_snapshot(Data(edge_index=_path_edges(2)))
    missing_edges = remap.apply_snapshot(Data(x=torch.ones(2, 1)))
    assert missing_edges.edge_index.shape == (2, 0)
    empty_edges = remap.apply_snapshot(
        Data(x=torch.ones(2, 1), edge_index=torch.empty(2, 0, dtype=torch.long))
    )
    assert empty_edges.edge_index.shape == (2, 0)
    with pytest.raises(ValueError, match="shape \\(2, E\\)"):
        remap.apply_snapshot(Data(x=torch.ones(2, 1), edge_index=torch.arange(3)))
    weighted = Data(
        x=torch.ones(2, 1),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_weight=torch.tensor([0.5]),
    )
    union = remap.apply_snapshot(weighted)
    assert union.edge_weight is not None
    with pytest.raises(ValueError, match="at least one snapshot"):
        remap.apply_snapshots([])
    features = torch.ones(2, 3)
    with pytest.raises(ValueError, match="shape \\(N, F\\)"):
        remap_node_features(
            torch.ones(3),
            old_index=torch.tensor([0, 1]),
            new_capacity=3,
        )
    with pytest.raises(ValueError, match="1-D with length"):
        remap_node_features(features, old_index=torch.tensor([0]), new_capacity=3)
    with pytest.raises(ValueError, match="at least N_old"):
        remap_node_features(features, old_index=torch.tensor([0, 1]), new_capacity=1)
    with pytest.raises(ValueError, match="outside"):
        remap_node_features(features, old_index=torch.tensor([0, 5]), new_capacity=3)


# ---------------------------------------------------------------------------
# analysis/memory.py
# ---------------------------------------------------------------------------


def test_markov_report_and_operator_type_guards() -> None:
    """Report field ranges and typed constructor / series arguments raise."""
    base = {
        "autocorrelation": torch.zeros(2),
        "ljung_box_statistic": 1.0,
        "ljung_box_pvalue": 0.5,
        "max_abs_autocorrelation": 0.1,
        "max_lag": 2,
        "n_timesteps": 20,
        "n_fit_parameters": 0,
        "alpha": 0.05,
        "white": True,
        "lag_unit": LAG_UNIT,
    }
    with pytest.raises(ValueError, match="shape \\(max_lag,\\)"):
        MarkovClosureReport(**{**base, "autocorrelation": torch.zeros(3)})
    with pytest.raises(ValueError, match="max_lag must be >= 1"):
        MarkovClosureReport(**{**base, "autocorrelation": torch.zeros(0), "max_lag": 0})
    with pytest.raises(ValueError, match="n_timesteps must exceed"):
        MarkovClosureReport(**{**base, "n_timesteps": 2})
    with pytest.raises(ValueError, match="n_fit_parameters must be >= 0"):
        MarkovClosureReport(**{**base, "n_fit_parameters": -1})
    with pytest.raises(ValueError, match="max_lag - n_fit_parameters"):
        MarkovClosureReport(**{**base, "n_fit_parameters": 2})
    with pytest.raises(ValueError, match="alpha must lie"):
        MarkovClosureReport(**{**base, "alpha": 1.0})
    with pytest.raises(ValueError, match="ljung_box_statistic must be finite"):
        MarkovClosureReport(**{**base, "ljung_box_statistic": float("nan")})
    with pytest.raises(ValueError, match="ljung_box_pvalue must lie"):
        MarkovClosureReport(**{**base, "ljung_box_pvalue": 1.5})
    with pytest.raises(ValueError, match="max_abs_autocorrelation must be finite"):
        MarkovClosureReport(**{**base, "max_abs_autocorrelation": float("inf")})
    series = torch.randn(40)
    with pytest.raises(TypeError, match="must be a Tensor"):
        markov_closure_report([1.0, 2.0, 3.0])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="floating-point"):
        markov_closure_report(torch.arange(20))
    with pytest.raises(ValueError, match="positive int"):
        markov_closure_report(series, max_lag=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-negative int"):
        markov_closure_report(series, max_lag=4, n_fit_parameters=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_fit_parameters must be >= 0"):
        markov_closure_report(series, max_lag=4, n_fit_parameters=-1)
    with pytest.raises(ValueError, match="latent_dim must be a positive int"):
        FiniteMemoryKoopman(latent_dim=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="memory_order must be a positive int"):
        FiniteMemoryKoopman(latent_dim=2, memory_order=True)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# analysis/hodge_modes.py
# ---------------------------------------------------------------------------


def test_hodge_components_empty_basis_real_modes_and_degree_types() -> None:
    """Isolated-node curl is empty; real columns skip the complex reassemble."""
    with pytest.raises(ValueError, match="n_cells, n_modes"):
        HodgeModeComponents(
            gradient=torch.zeros(3, 0),
            curl=torch.zeros(3, 0),
            harmonic=torch.zeros(3, 0),
        )
    isolated = torch.empty(2, 0, dtype=torch.long)
    real = _real_spectrum(torch.ones(3, 1, dtype=torch.float64))
    parts = hodge_decompose_modes(real, isolated, num_nodes=3, degree=0)
    assert parts.gradient.dtype == torch.float64
    assert torch.equal(parts.curl, torch.zeros_like(parts.curl))
    with pytest.raises(ValueError, match="degree must be 0 or 1"):
        hodge_decompose_modes(real, isolated, num_nodes=3, degree=True)
    with pytest.raises(ValueError, match="num_nodes must be >= 1"):
        hodge_decompose_modes(real, isolated, num_nodes=0)
    flat = KoopmanSpectrum(
        eigenvalues=torch.ones(1),
        eigenvectors=torch.ones(4),
        magnitudes=torch.ones(1),
        growth_rates=torch.zeros(1),
        frequencies=torch.zeros(1),
        time_step=1.0,
    )
    with pytest.raises(ValueError, match="n_cells, n_modes"):
        hodge_decompose_modes(flat, isolated, num_nodes=3)


# ---------------------------------------------------------------------------
# adaptation/joint_observer.py
# ---------------------------------------------------------------------------


def test_joint_observer_rejects_non_dense_and_unsupported_operators() -> None:
    """Schur graph factors and Hodge operators are refused."""
    schur = GraphKoopmanModel(
        SeparableDictionaryEncoder(2, 8, 3, num_layers=1),
        SeparableDictionaryDecoder(3, 8, 2, num_layers=1),
        latent_dim=3,
        time_step=0.1,
        koopman="graph",
        koopman_parameterization="schur",
    )
    with pytest.raises(ValueError, match="parameterization='dense'"):
        JointStateTopologyObserver(schur, claim_homomorphism=False)
    continuous_graph = GraphKoopmanModel(
        SeparableDictionaryEncoder(2, 8, 3, num_layers=1),
        SeparableDictionaryDecoder(3, 8, 2, num_layers=1),
        latent_dim=3,
        time_step=0.1,
        koopman="continuous_graph",
        dynamics_mode="continuous",
    )
    with pytest.raises(TypeError, match="GraphKoopmanOperator"):
        JointStateTopologyObserver(continuous_graph, claim_homomorphism=False)


def test_joint_observer_continuous_path_runs_rls() -> None:
    """Continuous per-node ``L`` uses RLS write-back with ``delta_t``."""
    model = GraphKoopmanModel(
        SeparableDictionaryEncoder(2, 8, 3, num_layers=1),
        SeparableDictionaryDecoder(3, 8, 2, num_layers=1),
        latent_dim=3,
        time_step=0.1,
        dynamics_mode="continuous",
    )
    observer = JointStateTopologyObserver(model, claim_homomorphism=False)
    result = observer.filter_and_adapt(_sequence())
    assert result.sparse_factors is None
    assert len(result.rls_steps) == 3


# ---------------------------------------------------------------------------
# baselines/gedmd.py and hankel_dmd.py
# ---------------------------------------------------------------------------


def test_gedmd_dictionary_derivative_and_fit_guards() -> None:
    """Polynomial derivative helpers and stacked ``dx/dt`` shapes raise."""
    states = torch.randn(6, 2)
    derivs = torch.randn(6, 2)
    with pytest.raises(ValueError, match="must share shape"):
        polynomial_observable_derivatives(states, derivs[:, :1], polynomial_degree=1)
    degree1 = polynomial_observable_derivatives(states, derivs, polynomial_degree=1)
    assert torch.equal(degree1, derivs)
    degree2 = polynomial_observable_derivatives(states, derivs, polynomial_degree=2)
    assert degree2.shape == (6, 4)
    with pytest.raises(ValueError, match="polynomial_degree must be 1 or 2"):
        polynomial_observable_derivatives(states, derivs, polynomial_degree=3)
    edges = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    sequence = _gedmd_sequence(
        [torch.randn(2) for _ in range(4)],
        edges,
        derivatives=[torch.randn(2) for _ in range(4)],
    )
    fitted = GEDMDBaseline(polynomial_degree=2).fit(sequence)
    assert fitted.K is not None
    stacked = torch.stack([snapshot.dx_dt.reshape(-1) for snapshot in sequence])
    via_kwarg = GEDMDBaseline(polynomial_degree=1).fit(sequence, derivatives=stacked)
    assert via_kwarg.K is not None
    with pytest.raises(ValueError, match="steps must be >= 1"):
        fitted.predict(sequence[0], steps=0)
    with pytest.raises(ValueError, match="\\(T, state_dim\\)"):
        GEDMDBaseline(polynomial_degree=1).fit(sequence, derivatives=torch.ones(3, 2))
    with pytest.raises(ValueError, match="\\(T, N, d\\)"):
        GEDMDBaseline(polynomial_degree=1).fit(
            sequence,
            derivatives=torch.ones(4, 3, 1),
        )
    with pytest.raises(ValueError, match="state_dim\\) or \\(T, N, d\\)"):
        GEDMDBaseline(polynomial_degree=1).fit(sequence, derivatives=torch.ones(4))
    mismatch = _gedmd_sequence(
        [torch.randn(2) for _ in range(3)],
        edges,
        derivatives=[torch.randn(2) for _ in range(3)],
    )
    mismatch[0].dx_dt = torch.randn(2, 2)
    with pytest.raises(ValueError, match="does not match x shape"):
        GEDMDBaseline(polynomial_degree=1).fit(mismatch)


def test_hankel_dmd_unfitted_short_history_and_spectrum() -> None:
    """Unfitted predict, ``steps < 1``, zero-pad history, and spectrum."""
    with pytest.raises(ValueError, match="2-D"):
        delay_embed_rows(torch.zeros(3, 2, 1), n_delays=2)
    unfitted = HankelDMDBaseline(n_delays=2)
    loop = torch.tensor([[0], [0]], dtype=torch.long)
    graph = Data(x=torch.ones(1, 1), edge_index=loop)
    with pytest.raises(RuntimeError, match="must be fit"):
        unfitted.predict(graph, steps=1)
    states = [torch.tensor([float(index)]) for index in range(6)]
    sequence = GraphSnapshotSequence(
        [Data(x=state.reshape(1, 1), edge_index=loop) for state in states]
    )
    fitted = HankelDMDBaseline(n_delays=3).fit(sequence)
    with pytest.raises(ValueError, match="steps must be >= 1"):
        fitted.predict(sequence[-1], steps=0)
    padded = assemble_delay_state(
        sequence[-1],
        history=list(sequence[:1]),
        n_delays=3,
        num_nodes=1,
        in_channels=1,
        state_dim=1,
    )
    assert padded.shape == (3,)
    forecasts = fitted.predict(sequence[-1], steps=2)
    assert len(forecasts) == 2
    spectrum = fitted.spectrum()
    assert spectrum.eigenvalues.ndim == 1
