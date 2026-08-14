"""Coverage and error-path tests for :mod:`koopman_graph.operators`."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph import (
    ContinuousKoopmanOperator,
    GNNDecoder,
    GNNEncoder,
    GraphKoopmanModel,
)
from koopman_graph.datasets import (
    EpidemicNetworkBenchmark,
    KuramotoSivashinskyBenchmark,
)
from koopman_graph.observables import (
    resolve_physics_lifting_fn,
    validate_physics_output,
)
from koopman_graph.operators.auxiliary_spectral import (
    inverse_advance_bilinear as aux_inverse_advance_bilinear,
)
from koopman_graph.operators.auxiliary_spectral import (
    split_auxiliary_spectrum,
)
from koopman_graph.uq import (
    EnsembleGraphKoopmanModel,
    LatentGaussianKoopmanUQ,
    empirical_coverage,
)
from koopman_graph.uq.common import PredictionInterval


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


def test_residual_coverage_to_95() -> None:
    """Hit remaining high-mass validation / inverse / factory branches."""
    from koopman_graph.operators.auxiliary_spectral import (
        AuxiliarySpectralNetwork,
        reset_auxiliary_network,
    )
    from koopman_graph.operators.auxiliary_spectral import (
        inverse_advance as aux_inverse_advance,
    )
    from koopman_graph.operators.discrete_propagation import (
        inverse_matrix_for_parameterization,
    )
    from koopman_graph.uq.latent_gaussian import propagate_gaussian_covariance

    # auxiliary: bad spectral output dim, network ctor, inverse additive per-node,
    # reset modes
    with pytest.raises(ValueError, match="Expected auxiliary output"):
        split_auxiliary_spectrum(torch.zeros(3), latent_dim=4)
    with pytest.raises(ValueError, match="latent_dim must be positive"):
        AuxiliarySpectralNetwork(0)
    net = AuxiliarySpectralNetwork(4, hidden_dims=(8,))
    reset_auxiliary_network(net, init_mode="identity", init_scale=0.01)
    reset_auxiliary_network(net, init_mode="identity_noise", init_scale=0.01)
    reset_auxiliary_network(net, init_mode="xavier", init_scale=0.01)
    with pytest.raises(ValueError, match="Unknown init_mode"):
        reset_auxiliary_network(net, init_mode="nope", init_scale=0.01)  # type: ignore[arg-type]

    latent_dim = 4
    control_dim = 2
    z_nodes = torch.randn(3, latent_dim)
    u_nodes = torch.randn(3, control_dim)
    generator = (-0.1 * torch.eye(latent_dim)).expand(3, latent_dim, latent_dim).clone()
    control_matrix = torch.randn(control_dim, latent_dim) * 0.05
    coupling = torch.randn(control_dim, latent_dim, latent_dim) * 0.01
    dt = torch.tensor(0.1)

    with pytest.raises(ValueError, match="matching node"):
        aux_inverse_advance_bilinear(
            torch.randn(latent_dim),
            dt,
            u_nodes,
            generator=generator[0],
            control_matrix=control_matrix,
            coupling=coupling,
            latent_dim=latent_dim,
        )
    with pytest.raises(ValueError, match="control input must have shape"):
        aux_inverse_advance_bilinear(
            z_nodes,
            dt,
            torch.randn(2, 2, control_dim),
            generator=generator,
            control_matrix=control_matrix,
            coupling=coupling,
            latent_dim=latent_dim,
        )
    # additive inverse via inverse_advance (uncontrolled + per-node + bad ndim)
    recovered = aux_inverse_advance(
        z_nodes,
        dt,
        None,
        generator=generator,
        control_dim=0,
        control_mode="additive",
        latent_dim=latent_dim,
        control_matrix=None,
        coupling=None,
    )
    assert recovered.shape == z_nodes.shape
    recovered = aux_inverse_advance(
        z_nodes,
        dt,
        u_nodes,
        generator=generator,
        control_dim=control_dim,
        control_mode="additive",
        latent_dim=latent_dim,
        control_matrix=control_matrix,
        coupling=None,
    )
    assert recovered.shape == z_nodes.shape
    with pytest.raises(ValueError, match="matching node"):
        aux_inverse_advance(
            torch.randn(latent_dim),
            dt,
            u_nodes,
            generator=generator[0],
            control_dim=control_dim,
            control_mode="additive",
            latent_dim=latent_dim,
            control_matrix=control_matrix,
            coupling=None,
        )
    with pytest.raises(ValueError, match="control input must have shape"):
        aux_inverse_advance(
            z_nodes,
            dt,
            torch.randn(2, 2, control_dim),
            generator=generator,
            control_dim=control_dim,
            control_mode="additive",
            latent_dim=latent_dim,
            control_matrix=control_matrix,
            coupling=None,
        )
    with pytest.raises(ValueError, match="bilinear coupling"):
        aux_inverse_advance(
            z_nodes,
            dt,
            u_nodes,
            generator=generator,
            control_dim=control_dim,
            control_mode="bilinear",
            latent_dim=latent_dim,
            control_matrix=control_matrix,
            coupling=None,
        )

    # continuous parameterization bound / certificate helpers
    from koopman_graph.operators.continuous_parameterizations import (
        continuous_bound_metric,
        continuous_stability_certificate,
    )

    gen = -0.2 * torch.eye(4)
    _ = continuous_bound_metric(
        "dense",
        max_real_eigenvalue=0.1,
        assembled_generator=gen,
    )
    with pytest.raises(ValueError, match="auxiliary_spectral"):
        continuous_bound_metric(
            "auxiliary_spectral",
            max_real_eigenvalue=0.1,
        )
    with pytest.raises(ValueError, match="requires dissipative_generator"):
        continuous_bound_metric(
            "dissipative",
            max_real_eigenvalue=0.1,
        )
    _ = continuous_bound_metric(
        "dissipative",
        max_real_eigenvalue=0.1,
        dissipative_generator=torch.eye(4),
    )
    with pytest.raises(ValueError, match="diagonal raw"):
        continuous_bound_metric(
            "odo",
            max_real_eigenvalue=0.1,
        )
    _ = continuous_bound_metric(
        "odo",
        max_real_eigenvalue=0.1,
        diag_raw=torch.zeros(4),
    )
    _ = continuous_stability_certificate(
        "lyapunov",
        lyapunov_diagonal=-torch.ones(4),
        lyapunov_matrix=torch.eye(4),
    )
    with pytest.raises(ValueError, match="lyapunov_diagonal"):
        continuous_stability_certificate("lyapunov")
    _ = continuous_stability_certificate(
        "schur",
        bound_metric=torch.tensor(-0.1),
    )
    with pytest.raises(ValueError, match="bound_metric"):
        continuous_stability_certificate("dissipative")
    assert continuous_stability_certificate("dense") is None

    # discrete inverse_matrix_for_parameterization branches
    eye = torch.eye(4)
    skew = torch.zeros(4, 4)
    _ = inverse_matrix_for_parameterization(
        "odo",
        odo_left=eye,
        odo_right=eye,
        odo_diagonal=torch.ones(4) * 0.5,
    )
    _ = inverse_matrix_for_parameterization(
        "schur",
        schur_cayley_q=skew,
        schur_triangular=torch.triu(eye * 0.5),
    )
    _ = inverse_matrix_for_parameterization(
        "dissipative",
        dissipative_generator=eye,
    )
    _ = inverse_matrix_for_parameterization(
        "lyapunov",
        lyapunov_cayley_q=skew,
        lyapunov_diagonal=torch.ones(4) * 0.5,
    )
    with pytest.raises(ValueError, match="Unknown parameterization"):
        inverse_matrix_for_parameterization("nope")  # type: ignore[arg-type]

    # latent gaussian helpers / ctor guards
    a = 0.5 * torch.eye(4)
    p = torch.eye(4)
    _ = propagate_gaussian_covariance(a, p, 1e-3)
    _ = propagate_gaussian_covariance(a, p, torch.eye(4) * 1e-3)
    with pytest.raises(ValueError, match="share shape"):
        propagate_gaussian_covariance(a, torch.eye(3), 1e-3)
    with pytest.raises(ValueError, match="process_noise tensor must match"):
        propagate_gaussian_covariance(a, p, torch.eye(3))
    model = _tiny_model()
    with pytest.raises(ValueError, match="observation_noise"):
        LatentGaussianKoopmanUQ(model, observation_noise=0.0)
    with pytest.raises(ValueError, match="initial_covariance"):
        LatentGaussianKoopmanUQ(model, initial_covariance=0.0)
    with pytest.raises(ValueError, match="n_samples"):
        LatentGaussianKoopmanUQ(model, n_samples=0)

    # epidemic validation tails
    with pytest.raises(ValueError, match="num_nodes"):
        EpidemicNetworkBenchmark.generate(num_nodes=0, num_timesteps=5)
    with pytest.raises(ValueError, match="patient_zero"):
        EpidemicNetworkBenchmark.generate(patient_zero=1.5)
    with pytest.raises(ValueError, match="patient_zero \\+ neighbor_seed"):
        EpidemicNetworkBenchmark.generate(patient_zero=0.8, neighbor_seed=0.8)
    with pytest.raises(ValueError, match="small_world"):
        EpidemicNetworkBenchmark.generate(
            num_nodes=2,
            topology="small_world",
        )
    with pytest.raises(ValueError, match="even integer"):
        EpidemicNetworkBenchmark.generate(
            num_nodes=8,
            topology="small_world",
            small_world_k=3,
        )
    with pytest.raises(ValueError, match="rewire_prob"):
        EpidemicNetworkBenchmark.generate(
            num_nodes=8,
            topology="small_world",
            rewire_prob=1.5,
        )
    EpidemicNetworkBenchmark.generate(
        num_nodes=8,
        num_timesteps=5,
        topology="small_world",
        small_world_k=2,
        rewire_prob=0.5,
        seed=0,
    )
    with pytest.raises(ValueError, match="edge_index"):
        EpidemicNetworkBenchmark.generate(topology="custom", edge_index=None)
    edges = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    with pytest.raises(ValueError, match="references node"):
        EpidemicNetworkBenchmark.generate(
            num_nodes=1,
            topology="custom",
            edge_index=edges,
        )

    # KS burn_in / dt
    with pytest.raises(ValueError, match="burn_in"):
        KuramotoSivashinskyBenchmark.generate(
            num_nodes=8,
            num_timesteps=5,
            burn_in=-1,
        )
    with pytest.raises(ValueError, match="dt"):
        KuramotoSivashinskyBenchmark.generate(
            num_nodes=8,
            num_timesteps=5,
            dt=0.0,
            burn_in=0,
        )

    # ensemble empty / empirical_coverage empty / missing x
    with pytest.raises(ValueError, match="at least one member"):
        EnsembleGraphKoopmanModel([])
    with pytest.raises(ValueError, match="at least one snapshot"):
        empirical_coverage(
            [], PredictionInterval(mean=(), lower=(), upper=(), level=0.9, n_members=1)
        )
    interval = PredictionInterval(
        mean=(Data(x=torch.randn(2, 2), edge_index=_edge_index()),),
        lower=(Data(x=torch.randn(2, 2), edge_index=_edge_index()),),
        upper=(Data(x=torch.randn(2, 2), edge_index=_edge_index()),),
        level=0.9,
        n_members=1,
    )
    with pytest.raises(ValueError, match="node features"):
        empirical_coverage(
            [Data(edge_index=_edge_index())],
            interval,
        )

    # observables validation tails
    with pytest.raises(ValueError, match="physics lifting output"):
        validate_physics_output(torch.randn(2, 3), physics_dim=2, num_nodes=2)
    with pytest.raises(ValueError, match="Unknown physics_preset"):
        resolve_physics_lifting_fn(physics_preset="not_a_preset")

    # factory / model injection residual via ContinuousKoopmanOperator odd dim
    op_odd = ContinuousKoopmanOperator(
        latent_dim=3,
        parameterization="auxiliary_spectral",
    )
    z_odd = torch.randn(3)
    assert op_odd.advance(z_odd, delta_t=0.05).shape == (3,)
