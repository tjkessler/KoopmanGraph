"""Coverage and error-path tests for :mod:`koopman_graph.operators`."""

from __future__ import annotations

import pytest
import torch

from koopman_graph import (
    ContinuousKoopmanOperator,
)
from koopman_graph.operators import (
    normalize_auxiliary_hidden_dims,
)
from koopman_graph.operators.auxiliary_spectral import (
    advance_controlled as aux_advance_controlled,
)
from koopman_graph.operators.auxiliary_spectral import (
    assemble_block_diagonal_generator,
    split_auxiliary_spectrum,
)
from koopman_graph.operators.auxiliary_spectral import (
    inverse_advance_bilinear as aux_inverse_advance_bilinear,
)
from koopman_graph.operators.continuous_propagation import (
    advance_controlled as cont_advance_controlled,
)
from koopman_graph.operators.continuous_propagation import (
    advance_interval,
    inverse_advance_interval,
)
from koopman_graph.operators.continuous_propagation import (
    inverse_advance_bilinear as cont_inverse_advance_bilinear,
)


def _noop_van_loan(
    state: torch.Tensor,
    dt: torch.Tensor,
    control: torch.Tensor,
    *,
    generator: torch.Tensor,
) -> torch.Tensor:
    return state


def test_auxiliary_spectral_validation_and_assembly() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        normalize_auxiliary_hidden_dims(())
    with pytest.raises(ValueError, match="positive integers"):
        normalize_auxiliary_hidden_dims((8, 0))
    assert normalize_auxiliary_hidden_dims(None)

    mu = torch.zeros(2)
    omega = torch.zeros(3)
    with pytest.raises(ValueError, match="share shape"):
        assemble_block_diagonal_generator(mu, omega, None)
    with pytest.raises(ValueError, match="trailing dimension must be 1"):
        assemble_block_diagonal_generator(
            torch.zeros(2), torch.zeros(2), torch.zeros(2)
        )
    with pytest.raises(ValueError, match="leading shape"):
        assemble_block_diagonal_generator(
            torch.zeros(2, 2),
            torch.zeros(2, 2),
            torch.zeros(3, 1),
        )
    even = assemble_block_diagonal_generator(torch.zeros(2), torch.ones(2), None)
    assert even.shape == (4, 4)
    odd = assemble_block_diagonal_generator(
        torch.zeros(1),
        torch.ones(1),
        torch.tensor([-0.5]),
    )
    assert odd.shape == (3, 3)
    mu_s, om_s, real_s = split_auxiliary_spectrum(
        torch.tensor([0.1, 0.2, 0.3, 0.4]),
        latent_dim=4,
    )
    assert real_s is None and mu_s.shape == (2,)
    _, _, real_odd = split_auxiliary_spectrum(
        torch.tensor([0.1, 0.2, -0.3]),
        latent_dim=3,
    )
    assert real_odd is not None and real_odd.shape == (1,)


def test_auxiliary_and_continuous_controlled_propagation_branches() -> None:
    latent_dim = 4
    control_dim = 2
    z = torch.randn(latent_dim)
    z_nodes = torch.randn(3, latent_dim)
    z2 = torch.randn(2, latent_dim)
    generator = -0.1 * torch.eye(latent_dim)
    control_matrix = torch.randn(control_dim, latent_dim) * 0.05
    coupling = torch.randn(control_dim, latent_dim, latent_dim) * 0.01
    u = torch.randn(control_dim)
    u_nodes = torch.randn(3, control_dim)
    dt = torch.tensor(0.1)

    out = cont_advance_controlled(
        z,
        dt,
        u,
        generator=generator,
        control_matrix=control_matrix,
        control_mode="bilinear",
        latent_dim=latent_dim,
        coupling=coupling,
        parameterization="dense",
    )
    assert out.shape == z.shape
    out_n = cont_advance_controlled(
        z_nodes,
        dt,
        u_nodes,
        generator=generator,
        control_matrix=control_matrix,
        control_mode="bilinear",
        latent_dim=latent_dim,
        coupling=coupling,
        parameterization="dense",
    )
    assert out_n.shape == z_nodes.shape
    with pytest.raises(ValueError, match="bilinear coupling"):
        cont_advance_controlled(
            z,
            dt,
            u,
            generator=generator,
            control_matrix=control_matrix,
            control_mode="bilinear",
            latent_dim=latent_dim,
            coupling=None,
            parameterization="dense",
        )
    with pytest.raises(ValueError, match="matching node axis"):
        cont_advance_controlled(
            z,
            dt,
            u_nodes,
            generator=generator,
            control_matrix=control_matrix,
            control_mode="bilinear",
            latent_dim=latent_dim,
            coupling=coupling,
            parameterization="dense",
        )
    with pytest.raises(ValueError, match="control input must have shape"):
        cont_advance_controlled(
            z,
            dt,
            torch.randn(2, 2, control_dim),
            generator=generator,
            control_matrix=control_matrix,
            control_mode="bilinear",
            latent_dim=latent_dim,
            coupling=coupling,
            parameterization="dense",
        )

    with pytest.raises(ValueError, match="delta_t is required"):
        advance_interval(
            z,
            None,
            None,
            latent_dim=latent_dim,
            control_dim=0,
            control_mode="additive",
            parameterization="dense",
            generator=generator,
        )
    assert torch.allclose(
        advance_interval(
            z,
            0.0,
            None,
            latent_dim=latent_dim,
            control_dim=0,
            control_mode="additive",
            parameterization="dense",
            generator=generator,
        ),
        z,
    )
    with pytest.raises(ValueError, match="ignored when delta_t is zero"):
        advance_interval(
            z,
            0.0,
            u,
            latent_dim=latent_dim,
            control_dim=control_dim,
            control_mode="additive",
            parameterization="dense",
            generator=generator,
            control_matrix=control_matrix,
        )
    with pytest.raises(ValueError, match="control_matrix is required"):
        advance_interval(
            z,
            0.1,
            u,
            latent_dim=latent_dim,
            control_dim=control_dim,
            control_mode="additive",
            parameterization="dense",
            generator=generator,
            control_matrix=None,
        )

    z_adv = cont_advance_controlled(
        z,
        dt,
        u,
        generator=generator,
        control_matrix=control_matrix,
        control_mode="bilinear",
        latent_dim=latent_dim,
        coupling=coupling,
        parameterization="dense",
    )
    assert (
        cont_inverse_advance_bilinear(
            z_adv,
            dt,
            u,
            generator=generator,
            control_matrix=control_matrix,
            coupling=coupling,
            latent_dim=latent_dim,
        ).shape
        == z.shape
    )
    z_adv_n = cont_advance_controlled(
        z_nodes,
        dt,
        u_nodes,
        generator=generator,
        control_matrix=control_matrix,
        control_mode="bilinear",
        latent_dim=latent_dim,
        coupling=coupling,
        parameterization="dense",
    )
    assert (
        cont_inverse_advance_bilinear(
            z_adv_n,
            dt,
            u_nodes,
            generator=generator,
            control_matrix=control_matrix,
            coupling=coupling,
            latent_dim=latent_dim,
        ).shape
        == z_nodes.shape
    )
    with pytest.raises(ValueError, match="matching"):
        cont_inverse_advance_bilinear(
            z,
            dt,
            u_nodes,
            generator=generator,
            control_matrix=control_matrix,
            coupling=coupling,
            latent_dim=latent_dim,
        )
    with pytest.raises(ValueError, match="control input must have shape"):
        cont_inverse_advance_bilinear(
            z,
            dt,
            torch.randn(2, 2, control_dim),
            generator=generator,
            control_matrix=control_matrix,
            coupling=coupling,
            latent_dim=latent_dim,
        )

    with pytest.raises(ValueError, match="delta_t is required"):
        inverse_advance_interval(
            z,
            None,
            None,
            latent_dim=latent_dim,
            control_dim=0,
            control_mode="additive",
            parameterization="dense",
            generator=generator,
        )
    with pytest.raises(ValueError, match="control_matrix is required"):
        inverse_advance_interval(
            z,
            0.1,
            u,
            latent_dim=latent_dim,
            control_dim=control_dim,
            control_mode="additive",
            parameterization="dense",
            generator=generator,
            control_matrix=None,
        )
    with pytest.raises(ValueError, match="bilinear coupling"):
        inverse_advance_interval(
            z,
            0.1,
            u,
            latent_dim=latent_dim,
            control_dim=control_dim,
            control_mode="bilinear",
            parameterization="dense",
            generator=generator,
            control_matrix=control_matrix,
            coupling=None,
        )

    op = ContinuousKoopmanOperator(
        latent_dim=latent_dim,
        control_dim=control_dim,
        parameterization="auxiliary_spectral",
        control_mode="additive",
    )
    advanced = op.advance(z2, delta_t=0.1, control=u)
    assert op.inverse_advance(advanced, delta_t=0.1, control=u).shape == z2.shape

    op_b = ContinuousKoopmanOperator(
        latent_dim=latent_dim,
        control_dim=control_dim,
        parameterization="auxiliary_spectral",
        control_mode="bilinear",
    )
    advanced_b = op_b.advance(z, delta_t=0.1, control=u)
    assert op_b.inverse_advance(advanced_b, delta_t=0.1, control=u).shape == z.shape
    advanced_bn = op_b.advance(z_nodes, delta_t=0.1, control=u_nodes)
    assert (
        op_b.inverse_advance(advanced_bn, delta_t=0.1, control=u_nodes).shape
        == z_nodes.shape
    )

    gen_batch = generator.expand(2, latent_dim, latent_dim).clone()
    assert (
        aux_advance_controlled(
            z2,
            dt,
            u,
            generator=gen_batch,
            control_mode="additive",
            latent_dim=latent_dim,
            coupling=None,
            advance_van_loan=_noop_van_loan,
        ).shape
        == z2.shape
    )
    with pytest.raises(ValueError, match="matching node"):
        aux_advance_controlled(
            z,
            dt,
            u_nodes,
            generator=generator,
            control_mode="additive",
            latent_dim=latent_dim,
            coupling=None,
            advance_van_loan=_noop_van_loan,
        )
    with pytest.raises(ValueError, match="control input must have shape"):
        aux_advance_controlled(
            z,
            dt,
            torch.randn(2, 2, control_dim),
            generator=generator,
            control_mode="additive",
            latent_dim=latent_dim,
            coupling=None,
            advance_van_loan=_noop_van_loan,
        )
    with pytest.raises(ValueError, match="bilinear coupling"):
        aux_advance_controlled(
            z,
            dt,
            u,
            generator=generator,
            control_mode="bilinear",
            latent_dim=latent_dim,
            coupling=None,
            advance_van_loan=_noop_van_loan,
        )
    gen_nodes = generator.expand(3, latent_dim, latent_dim).clone()
    assert (
        aux_advance_controlled(
            z_nodes,
            dt,
            u_nodes,
            generator=gen_nodes,
            control_mode="additive",
            latent_dim=latent_dim,
            coupling=None,
            advance_van_loan=_noop_van_loan,
        ).shape
        == z_nodes.shape
    )
    assert (
        aux_advance_controlled(
            z2,
            dt,
            u,
            generator=gen_batch,
            control_mode="bilinear",
            latent_dim=latent_dim,
            coupling=coupling,
            advance_van_loan=_noop_van_loan,
        ).shape
        == z2.shape
    )
    assert (
        aux_advance_controlled(
            z_nodes,
            dt,
            u_nodes,
            generator=gen_nodes,
            control_mode="bilinear",
            latent_dim=latent_dim,
            coupling=coupling,
            advance_van_loan=_noop_van_loan,
        ).shape
        == z_nodes.shape
    )
    with pytest.raises(ValueError, match="matching node"):
        aux_advance_controlled(
            z,
            dt,
            u_nodes,
            generator=generator,
            control_mode="bilinear",
            latent_dim=latent_dim,
            coupling=coupling,
            advance_van_loan=_noop_van_loan,
        )
    with pytest.raises(ValueError, match="control input must have shape"):
        aux_advance_controlled(
            z,
            dt,
            torch.randn(2, 2, control_dim),
            generator=generator,
            control_mode="bilinear",
            latent_dim=latent_dim,
            coupling=coupling,
            advance_van_loan=_noop_van_loan,
        )

    assert (
        aux_inverse_advance_bilinear(
            z,
            dt,
            u,
            generator=generator,
            control_matrix=control_matrix,
            coupling=coupling,
            latent_dim=latent_dim,
        ).shape
        == z.shape
    )
    assert (
        aux_inverse_advance_bilinear(
            z2,
            dt,
            u,
            generator=gen_batch,
            control_matrix=control_matrix,
            coupling=coupling,
            latent_dim=latent_dim,
        ).shape
        == z2.shape
    )
    assert (
        aux_inverse_advance_bilinear(
            z_nodes,
            dt,
            u_nodes,
            generator=gen_nodes,
            control_matrix=control_matrix,
            coupling=coupling,
            latent_dim=latent_dim,
        ).shape
        == z_nodes.shape
    )
