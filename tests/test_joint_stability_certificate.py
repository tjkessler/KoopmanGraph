"""Gershgorin / Schur / Lyapunov joint stability_certificate (TASK-1822–1824).

Honesty: the Gershgorin bound is sufficient, not tight (DESIGN R4). Opt-in
``kind="schur"|"lyapunov"`` are post-hoc on assembled ``K_eff`` under size
ceilings. Factor :class:`~koopman_graph.operators.StabilityCertificate`
objects never certify joint ``ρ(K_eff)``. Soft assembled eig-reg is a
separate training hinge.
"""

from __future__ import annotations

import pytest
import torch

from koopman_graph.analysis import (
    MAX_JOINT_LYAPUNOV_SIZE,
    MAX_JOINT_SCHUR_SIZE,
    JointStabilityCertificate,
    gershgorin_radius_bound,
    lyapunov_joint_bound,
    schur_radius_bound,
)
from koopman_graph.operators import GraphKoopmanOperator, HeteroGraphKoopmanOperator

_ATOL = 1e-5


def test_gershgorin_bound_dominates_spectral_radius_on_dense_matrix() -> None:
    """Gershgorin upper bound is always ≥ ρ(A) for a random square matrix."""
    torch.manual_seed(0)
    matrix = torch.randn(5, 5)
    bound = gershgorin_radius_bound(matrix)
    rho = torch.linalg.eigvals(matrix).abs().max().real
    assert float(bound) >= float(rho) - _ATOL


def test_gershgorin_known_loose_bound_case() -> None:
    """Documented loose case: bound can strictly exceed true ρ (R4)."""
    # Non-normal / off-diagonal mass: ρ = 0 for [[0, 2], [0, 0]], Gershgorin = 2.
    matrix = torch.tensor([[0.0, 2.0], [0.0, 0.0]])
    bound = float(gershgorin_radius_bound(matrix))
    rho = float(torch.linalg.eigvals(matrix).abs().max().real)
    assert rho == pytest.approx(0.0, abs=_ATOL)
    assert bound == pytest.approx(2.0, abs=_ATOL)
    assert bound > rho + 0.5


def test_gershgorin_rejects_nonsquare() -> None:
    """Non-square inputs raise clearly."""
    with pytest.raises(ValueError, match="square 2-D"):
        gershgorin_radius_bound(torch.zeros(2, 3))


def test_hetero_multiplex_joint_bound_dominates_rho() -> None:
    """Multiplex joint_bound_metric ≥ assembled ρ(K_eff)."""
    edge_indices = [
        torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        torch.tensor([[3, 2, 0], [2, 0, 1]], dtype=torch.long),
    ]
    op = HeteroGraphKoopmanOperator(2, num_relations=2, init_mode="identity")
    op.set_dense_matrices(
        torch.tensor([[0.7, 0.1], [0.0, 0.6]]),
        [
            torch.tensor([[0.2, 0.0], [0.05, 0.15]]),
            torch.tensor([[0.0, 0.1], [0.1, 0.0]]),
        ],
    )
    num_nodes = 4
    bound = op.joint_bound_metric(edge_indices, num_nodes).detach()
    rho = op.spectral_radius(edge_indices, num_nodes).detach()
    assert float(bound) >= float(rho) - _ATOL


def test_hetero_joint_bound_requires_topology() -> None:
    """joint_bound_metric does not silently fall back to factor bound_metric."""
    op = HeteroGraphKoopmanOperator(2, num_relations=1, init_mode="identity")
    with pytest.raises(ValueError, match="edge_indices and num_nodes"):
        op.joint_bound_metric()


def test_hetero_typed_and_rectangular_joint_bound() -> None:
    """Typed shared-d and rectangular d_τ fixtures expose joint_bound_metric."""
    node_types = ("a", "b")
    edge_types = (("a", "to_b", "b"), ("b", "to_a", "a"))
    num_nodes_dict = {"a": 2, "b": 3}
    num_nodes = 5
    banks = [
        torch.tensor([[0, 1], [2, 4]], dtype=torch.long),
        torch.tensor([[2, 3], [0, 1]], dtype=torch.long),
    ]

    shared = HeteroGraphKoopmanOperator(
        4,
        num_relations=2,
        node_types=node_types,
        edge_types=edge_types,
        init_mode="identity",
    )
    shared_bound = shared.joint_bound_metric(
        banks,
        num_nodes,
        num_nodes_dict=num_nodes_dict,
    ).detach()
    shared_rho = shared.spectral_radius(
        banks,
        num_nodes,
        num_nodes_dict=num_nodes_dict,
    ).detach()
    assert float(shared_bound) >= float(shared_rho) - _ATOL

    latent_dims = {"a": 2, "b": 3}
    rect = HeteroGraphKoopmanOperator(
        4,
        num_relations=2,
        node_types=node_types,
        edge_types=edge_types,
        latent_dims=latent_dims,
        init_mode="identity",
        parameterization="dense",
        sparsity="dense",
    )
    k_self = {
        "a": torch.tensor([[0.7, 0.1], [0.0, 0.8]]),
        "b": torch.tensor(
            [
                [0.6, 0.0, 0.1],
                [0.0, 0.5, 0.0],
                [0.2, 0.0, 0.4],
            ]
        ),
    }
    k_rels = [
        torch.tensor([[0.1, 0.0, 0.2], [0.0, 0.3, 0.0]]),
        torch.tensor([[0.2, 0.0], [0.0, 0.1], [0.1, 0.2]]),
    ]
    rect.set_dense_matrices(k_self, k_rels)
    rect_bound = rect.joint_bound_metric(
        banks,
        num_nodes,
        num_nodes_dict=num_nodes_dict,
    ).detach()
    rect_rho = rect.spectral_radius(
        banks,
        num_nodes,
        num_nodes_dict=num_nodes_dict,
    ).detach()
    assert float(rect_bound) >= float(rect_rho) - _ATOL


def test_factor_stable_joint_rho_unstable_bound_flags_instability() -> None:
    """G7 fixture: factor bound_metric ≤ 1 while ρ and joint cert bound > 1."""
    edge_indices = [torch.tensor([[0, 1], [1, 0]], dtype=torch.long)]
    op = HeteroGraphKoopmanOperator(
        2,
        num_relations=1,
        init_mode="identity",
        parameterization="dense",
    )
    k_self = 0.9 * torch.eye(2)
    k_rel = 0.9 * torch.eye(2)
    op.set_dense_matrices(k_self, [k_rel])
    # Dense has no factor StabilityCertificate; monitoring uses bound_metric.
    assert op.factor_stability_certificate() is None
    assert float(op.bound_metric().detach()) <= 1.0 + 1e-6
    rho = float(op.spectral_radius(edge_indices, num_nodes=2).detach())
    cert = op.stability_certificate(edge_indices, num_nodes=2)
    assert isinstance(cert, JointStabilityCertificate)
    assert cert.kind == "gershgorin"
    joint = float(cert.bound.detach())
    assert rho > 1.0
    assert joint >= rho - _ATOL
    assert joint > 1.0
    assert float(cert.margin.detach()) < 0.0


def test_hetero_stability_certificate_requires_topology() -> None:
    """Joint stability_certificate does not silently return a factor proxy."""
    op = HeteroGraphKoopmanOperator(2, num_relations=1, init_mode="identity")
    with pytest.raises(ValueError, match="edge_indices and num_nodes"):
        op.stability_certificate()


def test_schur_factor_certificate_is_not_joint_rho_cert() -> None:
    """Schur factor certificate stays distinct from joint Gershgorin object."""
    edge_indices = [torch.tensor([[0, 1], [1, 0]], dtype=torch.long)]
    op = HeteroGraphKoopmanOperator(
        2,
        num_relations=1,
        parameterization="schur",
        max_spectral_radius=0.95,
        init_mode="identity",
    )
    factor = op.factor_stability_certificate()
    assert factor is not None
    assert float(factor.margin.detach()) > 0.0
    joint = op.stability_certificate(edge_indices, num_nodes=2)
    assert isinstance(joint, JointStabilityCertificate)
    # Factor margin monitors factors; joint margin monitors Gershgorin(K_eff).
    assert type(factor).__name__ == "StabilityCertificate"
    assert type(joint).__name__ == "JointStabilityCertificate"


def test_graph_operator_joint_bound_dominates_assembled_rho() -> None:
    """Homogeneous GraphKoopmanOperator joint_bound_metric ≥ ρ(K_eff)."""
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    op = GraphKoopmanOperator(2, init_mode="identity")
    op.set_dense_matrices(0.5 * torch.eye(2), 2.0 * torch.eye(2))
    bound = op.joint_bound_metric(edge_index, num_nodes=2).detach()
    rho = (
        torch.linalg.eigvals(op.effective_matrix(edge_index, 2))
        .abs()
        .max()
        .real.detach()
    )
    assert float(bound) >= float(rho) - _ATOL
    # Distinct from factor-only spectral_radius / bound_metric.
    assert float(op.spectral_radius().detach()) != pytest.approx(
        float(bound),
        abs=1e-3,
    )
    cert = op.stability_certificate(edge_index, num_nodes=2)
    assert isinstance(cert, JointStabilityCertificate)
    assert cert.kind == "gershgorin"
    assert float(cert.bound.detach()) == pytest.approx(float(bound), abs=_ATOL)
    assert op.factor_stability_certificate() is None


def test_schur_radius_bound_matches_eigvals() -> None:
    """Schur joint bound equals true spectral radius via eigvals."""
    torch.manual_seed(1)
    matrix = torch.randn(4, 4)
    bound = float(schur_radius_bound(matrix))
    rho = float(torch.linalg.eigvals(matrix).abs().max().real)
    assert bound == pytest.approx(rho, abs=_ATOL)


def test_hetero_schur_certificate_matches_assembled_rho() -> None:
    """Opt-in kind='schur' tracks ρ(K_eff); default remains Gershgorin."""
    edge_indices = [torch.tensor([[0, 1], [1, 0]], dtype=torch.long)]
    op = HeteroGraphKoopmanOperator(2, num_relations=1, init_mode="identity")
    op.set_dense_matrices(0.4 * torch.eye(2), [0.1 * torch.eye(2)])
    default = op.stability_certificate(edge_indices, num_nodes=2)
    schur = op.stability_certificate(edge_indices, num_nodes=2, kind="schur")
    rho = float(op.spectral_radius(edge_indices, num_nodes=2).detach())
    assert default.kind == "gershgorin"
    assert schur.kind == "schur"
    assert float(schur.bound.detach()) == pytest.approx(rho, abs=_ATOL)
    assert float(default.bound.detach()) >= float(schur.bound.detach()) - _ATOL


def test_lyapunov_joint_bound_contractive_has_pd_matrix() -> None:
    """Contractive matrix yields ρ < 1 and a positive-definite P."""
    matrix = 0.3 * torch.eye(3)
    rho, lyap = lyapunov_joint_bound(matrix)
    assert float(rho) < 1.0
    assert lyap is not None
    evals = torch.linalg.eigvalsh(lyap)
    assert float(evals.min()) > 0.0
    # Residual of discrete Lyapunov: P - K^T P K ≈ I
    residual = lyap - matrix.T @ lyap @ matrix
    assert torch.allclose(residual, torch.eye(3), atol=1e-4)


def test_hetero_lyapunov_certificate_contractive() -> None:
    """Multiplex contractive fixture exposes PD Lyapunov joint certificate."""
    edge_indices = [torch.tensor([[0, 1], [1, 0]], dtype=torch.long)]
    op = HeteroGraphKoopmanOperator(2, num_relations=1, init_mode="identity")
    op.set_dense_matrices(0.2 * torch.eye(2), [0.05 * torch.eye(2)])
    cert = op.stability_certificate(edge_indices, num_nodes=2, kind="lyapunov")
    assert cert.kind == "lyapunov"
    assert float(cert.bound.detach()) < 1.0
    assert float(cert.margin.detach()) > 0.0
    assert cert.lyapunov_matrix is not None
    assert float(torch.linalg.eigvalsh(cert.lyapunov_matrix).min().detach()) > 0.0


def test_lyapunov_joint_bound_unstable_omits_matrix() -> None:
    """When ρ ≥ 1, Lyapunov kind returns bound without P."""
    matrix = 1.5 * torch.eye(2)
    rho, lyap = lyapunov_joint_bound(matrix)
    assert float(rho) >= 1.0
    assert lyap is None
    edge_indices = [torch.tensor([[0, 1], [1, 0]], dtype=torch.long)]
    op = HeteroGraphKoopmanOperator(2, num_relations=1, init_mode="identity")
    op.set_dense_matrices(0.9 * torch.eye(2), [0.9 * torch.eye(2)])
    cert = op.stability_certificate(edge_indices, num_nodes=2, kind="lyapunov")
    assert cert.kind == "lyapunov"
    assert float(cert.bound.detach()) >= 1.0 - _ATOL
    assert float(cert.margin.detach()) <= 0.0 + _ATOL
    assert cert.lyapunov_matrix is None


def test_joint_certificate_size_ceilings() -> None:
    """Schur / Lyapunov kinds reject oversized assembled operators."""
    over_lyap = torch.eye(MAX_JOINT_LYAPUNOV_SIZE + 1)
    with pytest.raises(ValueError, match="N·d ≤"):
        lyapunov_joint_bound(over_lyap)
    # Schur ceiling is large; probe via require path on a synthetic size.
    from koopman_graph.analysis.joint_stability import require_joint_assembled_size

    with pytest.raises(ValueError, match="N·d ≤"):
        require_joint_assembled_size(MAX_JOINT_SCHUR_SIZE + 1, kind="schur")
    require_joint_assembled_size(MAX_JOINT_SCHUR_SIZE, kind="schur")
    require_joint_assembled_size(8, kind="gershgorin")


def test_stability_certificate_rejects_unknown_kind() -> None:
    """Unknown kind strings raise clearly."""
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    op = GraphKoopmanOperator(2, init_mode="identity")
    with pytest.raises(ValueError, match="kind must be one of"):
        op.stability_certificate(edge_index, num_nodes=2, kind="bogus")
