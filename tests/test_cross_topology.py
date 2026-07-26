"""Cross-topology / cross-N transfer validation (TASK-1409).

Measures zero-shot rollout of ``koopman="graph"`` trained on a path graph
with ``N1`` nodes and evaluated on a path with ``N2 != N1``, against an
in-distribution holdout and a ``koopman="pernode"`` control.

Naming note
-----------
The v0.7.0 design text says ``koopman="dense"`` for the per-node control.
The public factory kind is ``"pernode"`` (``koopman_parameterization="dense"``
is a separate matrix parameterization). Tests use ``"pernode"``.

Measured outcome (seeded, 2026-07-26)
-------------------------------------
On ``SyntheticDynamicGraphBenchmark`` path diffusion with
``initial_state="ones"``, ``N1=4``, ``N2=6``, ``epochs=100``:

* graph in-distribution MSE ≈ 0.26; transfer MSE ≈ 0.26 (comparable)
* per-node in-distribution MSE ≈ 0.21; transfer MSE ≈ 0.21

Zero-shot transfer does **not** catastrophically fail on this surrogate.
It also does **not** outperform the per-node control. Findings feed
``limitations.rst`` / ``paper.md`` citation scoping in TASK-1414 / 1415 —
``Li2020CompositionalKoopman`` motivates the factorized operator; this
measurement does not demonstrate a transfer advantage.
"""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
from koopman_graph.datasets.synthetic import SyntheticDynamicGraphBenchmark

_N1 = 4
_N2 = 6
_FEATURES = 2
_LATENT = 4
_STEPS = 8
_EPOCHS = 100
_LR = 5e-3
# Documented loose upper bound from the measured ~0.26 MSE (seeded).
_MSE_UPPER = 0.75
# Transfer may not match in-distribution exactly; require it stay within 50%.
_TRANSFER_REL_TOL = 0.5


def _diffusion_params() -> dict:
    return {
        "in_channels": _FEATURES,
        "topology": "path",
        "diffusion_rate": 0.35,
        "decay_rate": 0.92,
        "noise_std": 0.0,
        "initial_state": "ones",
    }


def _make_model(
    koopman: str,
    *,
    learn_topology: str | None = None,
    **kwargs,
) -> GraphKoopmanModel:
    return GraphKoopmanModel(
        GNNEncoder(_FEATURES, 32, _LATENT, num_layers=2),
        GNNDecoder(_LATENT, 32, _FEATURES, num_layers=2),
        latent_dim=_LATENT,
        time_step=1.0,
        koopman=koopman,  # type: ignore[arg-type]
        learn_topology=learn_topology,
        **kwargs,
    )


def _rollout_mse(model: GraphKoopmanModel, sequence, *, steps: int = _STEPS) -> float:
    preds = model.predict(sequence[0], steps)
    errors = []
    for horizon in range(steps):
        target = sequence[horizon + 1].x
        pred = preds[horizon].x
        assert pred is not None and target is not None
        errors.append(torch.mean((pred - target) ** 2))
    return float(torch.stack(errors).mean().item())


def test_graph_operator_zero_shot_transfer_across_node_counts() -> None:
    """Train on N1; predict on N2 without shape errors; transfer ≈ in-dist.

    Assertions encode the measured outcome (comparable transfer error), not
    an aspirational claim that graph transfer beats per-node.
    """
    params = _diffusion_params()
    train = SyntheticDynamicGraphBenchmark.generate(
        num_nodes=_N1, num_timesteps=50, seed=0, **params
    )
    hold_n1 = SyntheticDynamicGraphBenchmark.generate(
        num_nodes=_N1, num_timesteps=20, seed=1, **params
    )
    hold_n2 = SyntheticDynamicGraphBenchmark.generate(
        num_nodes=_N2, num_timesteps=20, seed=2, **params
    )

    torch.manual_seed(0)
    graph_model = _make_model("graph")
    graph_model.fit(train, epochs=_EPOCHS, lr=_LR, device="cpu")

    preds = graph_model.predict(hold_n2[0], steps=_STEPS)
    assert len(preds) == _STEPS
    assert preds[-1].x is not None
    assert preds[-1].x.shape == (_N2, _FEATURES)

    in_dist = _rollout_mse(graph_model, hold_n1)
    transfer = _rollout_mse(graph_model, hold_n2)
    assert in_dist < _MSE_UPPER
    assert transfer < _MSE_UPPER
    # Measured: transfer ≈ in-distribution on this seeded diffusion surrogate.
    rel = abs(transfer - in_dist) / max(in_dist, 1e-8)
    assert rel <= _TRANSFER_REL_TOL, (
        f"unexpected transfer gap: in_dist={in_dist:.4f}, transfer={transfer:.4f}, "
        f"rel={rel:.4f} (tol={_TRANSFER_REL_TOL})"
    )


def test_transfer_compared_to_pernode_control_and_in_distribution() -> None:
    """Compare graph transfer to per-node control; record measured ordering."""
    params = _diffusion_params()
    train = SyntheticDynamicGraphBenchmark.generate(
        num_nodes=_N1, num_timesteps=50, seed=0, **params
    )
    hold_n1 = SyntheticDynamicGraphBenchmark.generate(
        num_nodes=_N1, num_timesteps=20, seed=1, **params
    )
    hold_n2 = SyntheticDynamicGraphBenchmark.generate(
        num_nodes=_N2, num_timesteps=20, seed=2, **params
    )

    metrics: dict[str, dict[str, float]] = {}
    for kind in ("graph", "pernode"):
        torch.manual_seed(0)
        model = _make_model(kind)
        model.fit(train, epochs=_EPOCHS, lr=_LR, device="cpu")
        metrics[kind] = {
            "in_dist": _rollout_mse(model, hold_n1),
            "transfer": _rollout_mse(model, hold_n2),
        }

    graph_xfer = metrics["graph"]["transfer"]
    pernode_xfer = metrics["pernode"]["transfer"]
    # Measured: per-node is competitive; do not require graph to win.
    assert graph_xfer < _MSE_UPPER
    assert pernode_xfer < _MSE_UPPER
    # Documented observed ordering on this seed: per-node ≤ graph × 1.25.
    assert pernode_xfer <= graph_xfer * 1.25 + 1e-6, (
        f"unexpected control ordering: graph_xfer={graph_xfer:.4f}, "
        f"pernode_xfer={pernode_xfer:.4f}; metrics={metrics}"
    )


def test_self_adaptive_topology_raises_on_node_count_change() -> None:
    """AdaptiveAdjacency binds N at first use; different N is a hard boundary."""
    params = _diffusion_params()
    train = SyntheticDynamicGraphBenchmark.generate(
        num_nodes=_N1, num_timesteps=12, seed=0, **params
    )
    other = SyntheticDynamicGraphBenchmark.generate(
        num_nodes=_N2, num_timesteps=8, seed=1, **params
    )
    model = _make_model("graph", learn_topology="self_adaptive")
    model.fit(train, epochs=2, lr=_LR, device="cpu")
    with pytest.raises(ValueError, match="static per fit"):
        model.predict(other[0], steps=2)


def test_orbit_partition_raises_on_node_count_change() -> None:
    """Orbit ties bound at construction for N1 cannot evaluate at N2."""
    params = _diffusion_params()
    train = SyntheticDynamicGraphBenchmark.generate(
        num_nodes=_N1, num_timesteps=12, seed=0, **params
    )
    other = SyntheticDynamicGraphBenchmark.generate(
        num_nodes=_N2, num_timesteps=8, seed=1, **params
    )
    model = _make_model(
        "graph",
        koopman_orbit_partition=((0, 1), (2, 3)),
    )
    model.fit(train, epochs=2, lr=_LR, device="cpu")
    with pytest.raises(ValueError, match="orbit partition was bound"):
        model.predict(other[0], steps=2)


def test_cross_topology_predict_preserves_eval_edge_index() -> None:
    """Zero-shot predict on N2 copies the evaluation topology onto forecasts."""
    params = _diffusion_params()
    train = SyntheticDynamicGraphBenchmark.generate(
        num_nodes=_N1, num_timesteps=20, seed=0, **params
    )
    hold_n2 = SyntheticDynamicGraphBenchmark.generate(
        num_nodes=_N2, num_timesteps=10, seed=2, **params
    )
    torch.manual_seed(0)
    model = _make_model("graph")
    model.fit(train, epochs=20, lr=_LR, device="cpu")
    origin = hold_n2[0]
    assert isinstance(origin, Data)
    preds = model.predict(origin, steps=3)
    for snap in preds:
        assert torch.equal(snap.edge_index, origin.edge_index)
        assert snap.x is not None
        assert snap.x.shape[0] == _N2
