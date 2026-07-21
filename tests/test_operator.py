"""Unit tests for KoopmanOperator."""

import pytest
import torch

from koopman_graph.operators import (
    ContinuousKoopmanOperator,
    KoopmanOperator,
    cayley_orthogonal,
    resolve_factory_stability_bound,
)


def test_cayley_orthogonal_produces_orthogonal_matrix() -> None:
    """Cayley map of a random matrix should yield Q with Q^T Q = I."""
    torch.manual_seed(0)
    skew_params = torch.randn(5, 5)
    orthogonal = cayley_orthogonal(skew_params)
    identity = torch.eye(5)
    assert torch.allclose(orthogonal.T @ orthogonal, identity, atol=1e-5)
    assert torch.allclose(orthogonal @ orthogonal.T, identity, atol=1e-5)
    assert torch.isclose(torch.det(orthogonal).abs(), torch.tensor(1.0), atol=1e-5)


def test_resolve_factory_stability_bound_maps_operator_fields() -> None:
    """Verify factory bound helper reads discrete/continuous attribute names."""
    discrete = KoopmanOperator(4, max_spectral_radius=0.85)
    continuous = ContinuousKoopmanOperator(4, max_real_eigenvalue=0.7)
    assert resolve_factory_stability_bound(
        discrete, dynamics_mode="discrete"
    ) == pytest.approx(0.85)
    assert resolve_factory_stability_bound(
        continuous, dynamics_mode="continuous"
    ) == pytest.approx(0.7)
    with pytest.raises(TypeError, match="max_spectral_radius"):
        resolve_factory_stability_bound(continuous, dynamics_mode="discrete")
    with pytest.raises(TypeError, match="max_real_eigenvalue"):
        resolve_factory_stability_bound(discrete, dynamics_mode="continuous")
    with pytest.raises(TypeError, match="dynamics_mode must be"):
        resolve_factory_stability_bound(discrete, dynamics_mode="bogus")  # type: ignore[arg-type]


@pytest.mark.parametrize("init_mode", ["identity", "identity_noise", "xavier"])
def test_init_modes_construct(init_mode: str) -> None:
    """Verify KoopmanOperator constructs for each init mode."""
    op = KoopmanOperator(4, init_mode=init_mode)  # type: ignore[arg-type]
    assert op.K.shape == (4, 4)
    assert torch.isfinite(op.K).all()


def test_identity_init() -> None:
    """Verify identity initialization sets ``K`` to the identity matrix."""
    op = KoopmanOperator(3, init_mode="identity")
    assert torch.allclose(op.K, torch.eye(3))


def test_forward_single_graph_shape() -> None:
    """Verify forward output shape for a single graph."""
    op = KoopmanOperator(8)
    z = torch.randn(5, 8)
    out = op(z)
    assert out.shape == (5, 8)


def test_forward_batched_shape() -> None:
    """Verify forward output shape for batched latent states."""
    op = KoopmanOperator(8)
    z = torch.randn(2, 5, 8)
    out = op(z)
    assert out.shape == (2, 5, 8)


def test_forward_matches_manual_matmul() -> None:
    """Verify forward matches ``z @ K.T``."""
    op = KoopmanOperator(4, init_mode="xavier")
    z = torch.randn(3, 4)
    expected = z @ op.K.T
    assert torch.allclose(op(z), expected)


def test_gradient_flow() -> None:
    """Verify gradients flow through the operator forward pass."""
    op = KoopmanOperator(6, init_mode="identity_noise")
    z = torch.randn(4, 6, requires_grad=True)
    loss = op(z).sum()
    loss.backward()
    assert op.K.grad is not None
    assert torch.isfinite(op.K.grad).all()
    assert z.grad is not None


def test_invalid_latent_dim_raises() -> None:
    """Verify non-positive ``latent_dim`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="latent_dim must be positive"):
        KoopmanOperator(0)


def test_shape_mismatch_raises() -> None:
    """Verify trailing dimension mismatch raises ``ValueError``."""
    op = KoopmanOperator(4)
    z = torch.randn(3, 5)
    with pytest.raises(ValueError, match="Expected trailing dimension 4"):
        op(z)


def test_exported_from_package() -> None:
    """Verify the symbol is exported from the package root."""
    from koopman_graph import KoopmanOperator as ExportedOperator

    assert ExportedOperator is KoopmanOperator


def test_invalid_init_scale_raises() -> None:
    """Verify negative ``init_scale`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="init_scale must be non-negative"):
        KoopmanOperator(4, init_scale=-0.1)


def test_reset_parameters_unknown_init_mode_raises() -> None:
    """Verify unknown ``init_mode`` in ``reset_parameters`` raises ``ValueError``."""
    op = KoopmanOperator(3)
    op.init_mode = "bogus"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="Unknown init_mode"):
        op.reset_parameters()


def test_odo_parameterization_constructs() -> None:
    """Verify ODO parameterization builds a finite assembled matrix."""
    op = KoopmanOperator(4, parameterization="odo", init_mode="identity")
    assert op.K.shape == (4, 4)
    assert torch.isfinite(op.K).all()


def test_odo_bound_metric_is_diagonal_factor() -> None:
    """Verify ODO bound_metric is the diagonal-factor bound (upper-bounds ρ).

    Orthogonal Cayley factors imply ``ρ(K) ≤ ‖K‖₂ = max|dᵢ| = bound_metric``.
    """
    torch.manual_seed(0)
    op = KoopmanOperator(6, parameterization="odo", max_spectral_radius=0.9)
    with torch.no_grad():
        op.diag_raw.copy_(torch.tensor([5.0, -5.0, 2.0, -2.0, 0.0, 1.0]))
        op.cayley_O1.copy_(torch.randn_like(op.cayley_O1) * 5.0)
        op.cayley_O2.copy_(torch.randn_like(op.cayley_O2) * 5.0)
    factor_bound = op.bound_metric()
    expected = torch.tanh(op.diag_raw).abs().max() * op.max_spectral_radius
    assert torch.allclose(factor_bound, expected)
    true_radius = op.spectral_radius()
    eig_radius = torch.linalg.eigvals(op.K).abs().max()
    assert torch.allclose(true_radius, eig_radius)
    assert true_radius.item() <= factor_bound.item() + 1e-5
    assert true_radius.item() < factor_bound.item() - 1e-3
    assert factor_bound.item() <= op.max_spectral_radius + 1e-5


def test_odo_stability_certificate_is_none() -> None:
    """Verify soft ODO mode exposes no structural stability certificate."""
    op = KoopmanOperator(4, parameterization="odo")
    assert op.stability_certificate() is None


def test_odo_inverse_recovers_forward_step() -> None:
    """Verify ODO inverse step recovers the previous latent state."""
    op = KoopmanOperator(5, parameterization="odo", init_mode="identity_noise")
    z = torch.randn(7, 5)
    z_next = op(z)
    recovered = op.inverse_step(z_next)
    assert torch.allclose(recovered, z, atol=1e-5)


def test_odo_inverse_recovers_with_negative_diagonal() -> None:
    """ODO inverse must handle negative diagonal factors (not clamp_min)."""
    op = KoopmanOperator(4, parameterization="odo", max_spectral_radius=1.0)
    with torch.no_grad():
        target = torch.tensor([-0.5, -0.3, 0.4, 0.2])
        op.diag_raw.copy_(torch.atanh(target))
        op.cayley_O1.zero_()
        op.cayley_O2.zero_()
    z = torch.randn(6, 4)
    z_next = op(z)
    recovered = op.inverse_step(z_next)
    assert torch.allclose(recovered, z, atol=1e-5)
    # Assembled spectrum includes the negative factors.
    assert (torch.linalg.eigvals(op.K).real < 0).any()


def test_lyapunov_inverse_recovers_with_negative_diagonal() -> None:
    """Lyapunov inverse must handle mixed-sign diagonal eigenvalues."""
    op = KoopmanOperator(4, parameterization="lyapunov", max_spectral_radius=1.0)
    bound = 1.0 - 1e-4
    with torch.no_grad():
        target = torch.tensor([-0.5, -0.3, 0.4, 0.2]) * bound
        op.lyap_diag_raw.copy_(torch.atanh(target / bound))
        op.cayley_Q.zero_()
    z = torch.randn(6, 4)
    z_next = op(z)
    recovered = op.inverse_step(z_next)
    assert torch.allclose(recovered, z, atol=1e-5)
    assert (op._lyapunov_diagonal() < 0).any()


def test_odo_gradient_flow() -> None:
    """Verify gradients reach ODO factor parameters."""
    op = KoopmanOperator(4, parameterization="odo", init_mode="identity_noise")
    z = torch.randn(3, 4, requires_grad=True)
    loss = op(z).sum()
    loss.backward()
    assert op.cayley_O1.grad is not None
    assert op.cayley_O2.grad is not None
    assert op.diag_raw.grad is not None


def test_invalid_max_spectral_radius_raises() -> None:
    """Verify non-positive ``max_spectral_radius`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="max_spectral_radius must be positive"):
        KoopmanOperator(4, parameterization="odo", max_spectral_radius=0.0)


def test_dense_inverse_matrix_reused() -> None:
    """Verify dense inverse matrix matches a single inverse step."""
    op = KoopmanOperator(4, init_mode="identity")
    z = torch.randn(5, 4)
    inverse = op.dense_inverse_matrix()
    assert torch.allclose(op.inverse_step(z, inverse_matrix=inverse), z, atol=1e-5)


def test_negative_control_dim_raises() -> None:
    """Verify negative ``control_dim`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="control_dim must be non-negative"):
        KoopmanOperator(4, control_dim=-1)


def test_unknown_parameterization_raises() -> None:
    """Verify unknown parameterization raises ``ValueError``."""
    with pytest.raises(ValueError, match="Unknown parameterization"):
        KoopmanOperator(4, parameterization="bogus")  # type: ignore[arg-type]


def test_reset_control_parameters_noop_without_control() -> None:
    """Verify control reset is a no-op for uncontrolled operators."""
    op = KoopmanOperator(4)
    op.reset_control_parameters()
    assert not hasattr(op, "B")


def test_control_term_requires_positive_control_dim() -> None:
    """Verify ``control_term`` rejects uncontrolled operators."""
    op = KoopmanOperator(4)
    with pytest.raises(ValueError, match="requires control_dim > 0"):
        op.control_term(torch.zeros(1))


def test_control_term_validates_shapes() -> None:
    """Verify global, per-node, and rank validation in ``control_term``."""
    op = KoopmanOperator(4, control_dim=2)
    with pytest.raises(ValueError, match="Expected global control shape"):
        op.control_term(torch.zeros(3))
    with pytest.raises(ValueError, match="Expected per-node control shape"):
        op.control_term(torch.zeros(5, 3))
    with pytest.raises(ValueError, match="Per-node control has 4 rows"):
        op.control_term(torch.zeros(4, 2), num_nodes=5)
    with pytest.raises(ValueError, match="control input must have shape"):
        op.control_term(torch.zeros(2, 2, 2))


def test_forward_rejects_control_for_uncontrolled_operator() -> None:
    """Verify controls passed to an uncontrolled operator raise."""
    op = KoopmanOperator(4)
    with pytest.raises(ValueError, match="uncontrolled operator"):
        op(torch.randn(3, 4), control=torch.zeros(1))


def test_forward_requires_control_when_controlled() -> None:
    """Verify a controlled operator requires a control input."""
    op = KoopmanOperator(4, control_dim=1)
    with pytest.raises(ValueError, match="control input is required"):
        op(torch.randn(3, 4))


def test_inverse_step_requires_control_when_controlled() -> None:
    """Verify inverse step requires control for controlled operators."""
    op = KoopmanOperator(4, control_dim=1)
    with pytest.raises(ValueError, match="control input is required"):
        op.inverse_step(torch.randn(3, 4))


def test_inverse_step_with_per_node_control() -> None:
    """Verify controlled inverse step recovers state for per-node controls."""
    op = KoopmanOperator(3, control_dim=2, init_mode="identity")
    with torch.no_grad():
        op.B.copy_(torch.randn(2, 3) * 0.1)
    z = torch.randn(5, 3)
    control = torch.randn(5, 2)
    z_next = op(z, control=control)
    recovered = op.inverse_step(z_next, control=control)
    assert torch.allclose(recovered, z, atol=1e-5)


def test_odo_xavier_init_constructs() -> None:
    """Verify xavier initialization for the ODO parameterization."""
    op = KoopmanOperator(4, parameterization="odo", init_mode="xavier")
    assert torch.isfinite(op.K).all()


def test_odo_reset_unknown_init_mode_raises() -> None:
    """Verify unknown ``init_mode`` in ODO reset raises ``ValueError``."""
    op = KoopmanOperator(3, parameterization="odo")
    op.init_mode = "bogus"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="Unknown init_mode"):
        op.reset_parameters()


def test_dense_spectral_radius_matches_eigvals() -> None:
    """Verify dense spectral radius equals the max eigenvalue magnitude."""
    op = KoopmanOperator(3, init_mode="identity")
    with torch.no_grad():
        op._parameters["K"].copy_(torch.diag(torch.tensor([0.5, -1.5, 0.9])))
    assert torch.isclose(op.spectral_radius(), torch.tensor(1.5))


def test_dense_inverse_matrix_rejects_odo() -> None:
    """Verify ``dense_inverse_matrix`` is dense-only."""
    op = KoopmanOperator(3, parameterization="odo")
    with pytest.raises(ValueError, match="only available for dense"):
        op.dense_inverse_matrix()


def test_dense_inverse_matrix_falls_back_to_pinv() -> None:
    """Verify singular dense matrices fall back to the pseudo-inverse."""
    op = KoopmanOperator(3, init_mode="identity")
    with torch.no_grad():
        op._parameters["K"].zero_()
    inverse = op.dense_inverse_matrix()
    assert torch.allclose(inverse, torch.zeros(3, 3))


@pytest.mark.parametrize("parameterization", ["schur", "dissipative", "lyapunov"])
def test_structural_parameterizations_construct(parameterization: str) -> None:
    """Verify structurally stable parameterizations build finite matrices."""
    op = KoopmanOperator(4, parameterization=parameterization, init_mode="identity")  # type: ignore[arg-type]
    assert op.K.shape == (4, 4)
    assert torch.isfinite(op.K).all()
    assert op.spectral_radius().item() < 1.0
    assert op.bound_metric().item() < 1.0


@pytest.mark.parametrize("parameterization", ["schur", "dissipative", "lyapunov"])
def test_structural_spectral_radius_bounded(parameterization: str) -> None:
    """Verify structural modes keep eigenvalues strictly inside the unit disk."""
    op = KoopmanOperator(5, parameterization=parameterization, init_mode="xavier")  # type: ignore[arg-type]
    assert op.bound_metric().item() <= 1.0 - 1e-4 + 1e-5
    assert op.spectral_radius().item() < 1.0
    eigenvalues = torch.linalg.eigvals(op.K)
    assert eigenvalues.abs().max().item() < 1.0
    assert torch.allclose(op.spectral_radius(), eigenvalues.abs().max())


@pytest.mark.parametrize("parameterization", ["schur", "dissipative", "lyapunov"])
def test_structural_inverse_recovers_forward_step(parameterization: str) -> None:
    """Verify inverse step recovers the previous latent state."""
    op = KoopmanOperator(
        5,
        parameterization=parameterization,  # type: ignore[arg-type]
        init_mode="identity_noise",
    )
    z = torch.randn(7, 5)
    z_next = op(z)
    recovered = op.inverse_step(z_next)
    assert torch.allclose(recovered, z, atol=1e-4)


@pytest.mark.parametrize("parameterization", ["schur", "dissipative", "lyapunov"])
def test_structural_gradient_flow(parameterization: str) -> None:
    """Verify gradients reach structural parameter tensors."""
    op = KoopmanOperator(
        4,
        parameterization=parameterization,  # type: ignore[arg-type]
        init_mode="identity_noise",
    )
    z = torch.randn(3, 4, requires_grad=True)
    loss = op(z).sum()
    loss.backward()
    assert any(
        param.grad is not None and torch.isfinite(param.grad).all()
        for name, param in op.named_parameters()
        if name != "B"
    )


def test_schur_bound_metric_uses_diagonal_without_eigvals() -> None:
    """Verify Schur bound_metric matches the triangular diagonal bound."""
    op = KoopmanOperator(4, parameterization="schur", init_mode="identity")
    with torch.no_grad():
        op.schur_off_raw.copy_(torch.randn_like(op.schur_off_raw))
        op.schur_off_raw.copy_(torch.triu(op.schur_off_raw, diagonal=1))
    bound = op.bound_metric()
    diag_vals = torch.tanh(op.schur_diag_raw).abs() * (1.0 - 1e-4)
    assert torch.allclose(bound, diag_vals.max())
    assert torch.allclose(op.spectral_radius(), torch.linalg.eigvals(op.K).abs().max())


def test_lyapunov_stability_certificate() -> None:
    """Verify Lyapunov certificate satisfies K^T P K - P ≺ 0."""
    op = KoopmanOperator(4, parameterization="lyapunov", init_mode="identity")
    certificate = op.stability_certificate()
    assert certificate is not None
    assert certificate.lyapunov_matrix is not None
    assert certificate.margin.item() > 0
    p = certificate.lyapunov_matrix
    residual = op.K.T @ p @ op.K - p
    eigenvalues = torch.linalg.eigvalsh(residual)
    assert eigenvalues.max().item() < 1e-5


def test_dissipative_stability_certificate() -> None:
    """Verify dissipative mode reports a positive spectral margin."""
    op = KoopmanOperator(3, parameterization="dissipative", init_mode="identity")
    certificate = op.stability_certificate()
    assert certificate is not None
    assert certificate.lyapunov_matrix is None
    assert certificate.margin.item() > 0


def test_schur_stability_certificate() -> None:
    """Verify Schur mode reports a positive unit-disk spectral margin."""
    op = KoopmanOperator(4, parameterization="schur", init_mode="identity")
    certificate = op.stability_certificate()
    assert certificate is not None
    assert certificate.lyapunov_matrix is None
    assert certificate.margin.item() > 0
    assert torch.isclose(certificate.margin, 1.0 - op.bound_metric())
    assert op.bound_metric().item() < op.max_spectral_radius
    assert op.spectral_radius().item() < op.max_spectral_radius


@pytest.mark.parametrize("parameterization", ["schur", "dissipative", "lyapunov"])
def test_structural_rejects_max_spectral_radius_above_one(
    parameterization: str,
) -> None:
    """Structural modes must reject max_spectral_radius > 1 for unit-disk guarantees."""
    with pytest.raises(ValueError, match="max_spectral_radius <= 1"):
        KoopmanOperator(
            4,
            parameterization=parameterization,  # type: ignore[arg-type]
            max_spectral_radius=1.5,
        )


def test_odo_allows_max_spectral_radius_above_one() -> None:
    """Soft ODO mode may use max_spectral_radius > 1 as an operator-norm bound."""
    op = KoopmanOperator(3, parameterization="odo", max_spectral_radius=1.5)
    assert op.max_spectral_radius == 1.5
    assert op.bound_metric().item() <= 1.5 + 1e-5


@pytest.mark.parametrize("parameterization", ["schur", "lyapunov"])
def test_structural_certificate_valid_at_subunitary_bound(
    parameterization: str,
) -> None:
    """Schur/Lyapunov certificates stay valid when max_spectral_radius < 1."""
    op = KoopmanOperator(
        4,
        parameterization=parameterization,  # type: ignore[arg-type]
        max_spectral_radius=0.5,
        init_mode="xavier",
    )
    assert op.spectral_radius().item() < 0.5
    certificate = op.stability_certificate()
    assert certificate is not None
    assert certificate.margin.item() > 0.5
    assert torch.isclose(certificate.margin, 1.0 - op.bound_metric())
    if parameterization == "lyapunov":
        assert certificate.lyapunov_matrix is not None
        residual = (
            op.K.T @ certificate.lyapunov_matrix @ op.K - certificate.lyapunov_matrix
        )
        assert torch.linalg.eigvalsh(residual).max().item() < 1e-5


def test_stability_certificate_none_for_dense() -> None:
    """Verify dense operators do not expose a stability certificate."""
    op = KoopmanOperator(3)
    assert op.stability_certificate() is None


def test_long_rollout_latent_norm_bounded_for_structural_modes(
    scaling_sequence,
) -> None:
    """Verify 200-step latent rollouts stay bounded for structural modes."""
    from koopman_graph.model import GraphKoopmanModel
    from koopman_graph.nn import GNNDecoder, GNNEncoder

    def latent_rollout_norm(model, sequence, steps: int) -> float:
        model.eval()
        with torch.no_grad():
            edge_index = sequence[0].edge_index
            z = model.encoder(sequence[0], edge_index)
            for _ in range(steps):
                z = model.koopman(z)
            return float(z.norm().item())

    encoder = GNNEncoder(in_channels=3, hidden_channels=16, latent_dim=8)
    decoder = GNNDecoder(latent_dim=8, hidden_channels=16, out_channels=3)
    stable_model = GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=8,
        time_step=0.1,
        koopman_parameterization="lyapunov",
    )
    torch.manual_seed(0)
    stable_model.fit(scaling_sequence, epochs=5, lr=1e-2)
    stable_norm = latent_rollout_norm(stable_model, scaling_sequence, steps=200)

    unstable_encoder = GNNEncoder(in_channels=3, hidden_channels=16, latent_dim=8)
    unstable_decoder = GNNDecoder(latent_dim=8, hidden_channels=16, out_channels=3)
    unstable_model = GraphKoopmanModel(
        encoder=unstable_encoder,
        decoder=unstable_decoder,
        latent_dim=8,
        time_step=0.1,
    )
    with torch.no_grad():
        unstable_eigs = torch.tensor([1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7])
        unstable_model.koopman.K.copy_(torch.diag(unstable_eigs))
    unstable_norm = latent_rollout_norm(unstable_model, scaling_sequence, steps=200)

    assert stable_norm < unstable_norm
    assert stable_norm < 1e3
    assert unstable_norm > 1e3


def test_koopman_operator_satisfies_contract() -> None:
    """Verify discrete operators implement KoopmanOperatorContract."""
    from koopman_graph.graph_utils import KoopmanPropagator
    from koopman_graph.operators import KoopmanOperatorContract

    op = KoopmanOperator(4, init_mode="identity")
    assert isinstance(op, KoopmanOperatorContract)
    assert KoopmanPropagator is KoopmanOperatorContract
    z = torch.randn(3, 4)
    assert torch.allclose(op.matrix, op.K)
    assert torch.allclose(op.bound_metric(), op.spectral_radius())
    assert torch.allclose(op.advance(z), op(z))
    assert torch.allclose(op.inverse_advance(op.advance(z)), z, atol=1e-5)


def test_operator_peers_have_no_private_cross_module_imports() -> None:
    """Operator peers must not import leading-``_`` symbols across modules."""
    import ast
    from pathlib import Path

    operators_root = (
        Path(__file__).resolve().parents[1] / "src" / "koopman_graph" / "operators"
    )
    peer_paths = [
        operators_root / "discrete.py",
        operators_root / "continuous.py",
        operators_root / "graph.py",
        operators_root / "auxiliary_spectral.py",
        operators_root / "control.py",
        operators_root / "__init__.py",
    ]
    private_imports: list[str] = []
    for path in peer_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            if not node.module.startswith("koopman_graph.operators"):
                continue
            for alias in node.names:
                if alias.name.startswith("_"):
                    private_imports.append(f"{path.name}:{node.module}.{alias.name}")
    assert private_imports == []


def test_operators_do_not_import_analysis() -> None:
    """Operators must not depend on analysis (eager or lazy)."""
    import ast
    from pathlib import Path

    operators_root = (
        Path(__file__).resolve().parents[1] / "src" / "koopman_graph" / "operators"
    )
    analysis_imports: list[str] = []
    for path in sorted(operators_root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "koopman_graph.analysis" or alias.name.startswith(
                        "koopman_graph.analysis."
                    ):
                        analysis_imports.append(f"{path.name}:{alias.name}")
            elif (
                isinstance(node, ast.ImportFrom)
                and node.module
                and (
                    node.module == "koopman_graph.analysis"
                    or node.module.startswith("koopman_graph.analysis.")
                )
            ):
                analysis_imports.append(f"{path.name}:{node.module}")
    assert analysis_imports == []


def test_contract_structural_helpers_are_exported() -> None:
    """Verify shared structural helpers are non-private package exports."""
    from koopman_graph.operators import (
        bounded_diagonal,
        safe_diagonal_inverse,
        strict_diagonal_values,
    )

    raw = torch.tensor([0.5, 1.0, -1.0])
    diag = bounded_diagonal(raw, max_radius=0.9)
    assert diag.shape == (3, 3)
    assert torch.all(diag.diag().abs() <= 0.9 + 1e-6)
    values = strict_diagonal_values(raw, max_spectral_radius=0.95)
    assert values.shape == (3,)
    inv = safe_diagonal_inverse(values)
    assert inv.shape == (3, 3)
    assert torch.allclose(inv.diag() * values, torch.ones(3), atol=1e-5)
