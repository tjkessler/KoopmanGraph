"""Opt-in ``fit(..., identification=...)`` bind and default-Adam isolation."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.identification import IdentificationConfig, IdentificationReport
from koopman_graph.model import GraphKoopmanModel
from koopman_graph.operators import GraphKoopmanOperator, KoopmanOperator


class _IdentityCodec(nn.Module):
    """Pass-through encoder/decoder for linear latent oracles.

    Parameters
    ----------
    dim : int
        Feature and latent width.
    """

    def __init__(self, dim: int) -> None:
        """Store channel metadata.

        Parameters
        ----------
        dim : int
            Shared input / latent / output width.
        """
        super().__init__()
        self.in_channels = dim
        self.latent_dim = dim
        self.out_channels = dim
        self.num_layers = 1

    def forward(
        self,
        x_or_data: Tensor | Data,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        """Return node features unchanged.

        Parameters
        ----------
        x_or_data : Tensor or Data
            Features or a snapshot.
        edge_index, edge_weight : Tensor or None
            Ignored topology.

        Returns
        -------
        Tensor
            Node features.
        """
        del edge_index, edge_weight
        if isinstance(x_or_data, Data):
            return x_or_data.x
        return x_or_data


def _eigvals_match(
    left: Tensor,
    right: Tensor,
    *,
    rtol: float,
    atol: float,
) -> bool:
    """Greedy multiset match of complex eigenvalues.

    Parameters
    ----------
    left, right : Tensor
        Eigenvalue vectors.
    rtol, atol : float
        Construction tolerances.

    Returns
    -------
    bool
        Whether the multisets match within tolerance.
    """
    remaining = right.detach().clone()
    for value in left.detach():
        diffs = (remaining - value).abs()
        index = int(torch.argmin(diffs))
        if not torch.isclose(value, remaining[index], rtol=rtol, atol=atol):
            return False
        remaining[index] = complex(float("inf"), float("inf"))
    return True


def _linear_sequence(
    true_k: Tensor, *, n_nodes: int = 4, n_times: int = 16
) -> tuple[GraphSnapshotSequence, Tensor]:
    """Build a graph sequence whose node states follow ``z @ K.T``.

    Parameters
    ----------
    true_k : Tensor
        Shared per-node map.
    n_nodes, n_times : int
        Graph size and length.

    Returns
    -------
    tuple
        Sequence and the undirected edge index.
    """
    dim = true_k.shape[0]
    generator = torch.Generator()
    generator.manual_seed(0)
    state = torch.randn(n_nodes, dim, dtype=true_k.dtype, generator=generator)
    src = list(range(n_nodes - 1))
    dst = list(range(1, n_nodes))
    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
    snapshots = []
    for _ in range(n_times):
        snapshots.append(Data(x=state.clone(), edge_index=edge_index))
        state = state @ true_k.T
    return GraphSnapshotSequence(snapshots), edge_index


def _identity_model(dim: int, *, dtype: torch.dtype) -> GraphKoopmanModel:
    """Per-node dense model with identity lift/decode.

    Parameters
    ----------
    dim : int
        Latent width.
    dtype : torch.dtype
        Parameter dtype.

    Returns
    -------
    GraphKoopmanModel
        Untrained identity codec model.
    """
    model = GraphKoopmanModel(
        encoder=_IdentityCodec(dim),
        decoder=_IdentityCodec(dim),
        latent_dim=dim,
        time_step=1.0,
        koopman_init_mode="identity",
        koopman_init_scale=0.0,
    )
    return model.to(dtype=dtype)


def test_fit_identification_recovers_linear_gaussian_eigenvalues() -> None:
    """Identity-encoder fit recovers known eigenvalues.

    Noiseless float64 oracle; ``rtol=1e-5``, ``atol=1e-8`` from
    construction.
    """
    true_k = torch.tensor([[0.9, 0.15], [-0.05, 0.55]], dtype=torch.float64)
    sequence, _ = _linear_sequence(true_k)
    model = _identity_model(2, dtype=torch.float64)
    history = model.fit(
        sequence,
        epochs=2,
        identification=IdentificationConfig(solver="ridge", ridge=0.0),
    )
    assert history.epochs == 2
    report = model.identification_report
    assert isinstance(report, IdentificationReport)
    assert report.one_step.mse is not None
    assert report.one_step.mse == pytest.approx(0.0, abs=1e-8)
    assert report.invariance.leakage is None
    assert report.spectral.polluted is None
    assert report.spectral.residual_max is None
    assert report.stability.spectral_radius is not None
    recovered = torch.linalg.eigvals(model.koopman.K.detach())
    assert _eigvals_match(
        recovered,
        torch.linalg.eigvals(true_k),
        rtol=1e-5,
        atol=1e-8,
    )


def test_default_fit_leaves_identification_report_unset(
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Adam ``fit`` does not populate ``identification_report``."""
    from koopman_graph import GNNDecoder, GNNEncoder

    model = GraphKoopmanModel(
        encoder=GNNEncoder(
            in_channels=3, hidden_channels=8, latent_dim=4, num_layers=1
        ),
        decoder=GNNDecoder(
            latent_dim=4, hidden_channels=8, out_channels=3, num_layers=1
        ),
        latent_dim=4,
        time_step=0.1,
    )
    assert model.identification_report is None
    model.fit(scaling_sequence, epochs=1, lr=1e-2, device="cpu")
    assert model.identification_report is None
    assert isinstance(model.koopman, KoopmanOperator)


def test_identification_rejects_graph_operator() -> None:
    """Networked operators are out of scope for this increment."""
    true_k = torch.diag(torch.tensor([0.8, 0.5], dtype=torch.float64))
    sequence, _ = _linear_sequence(true_k)
    model = GraphKoopmanModel(
        encoder=_IdentityCodec(2),
        decoder=_IdentityCodec(2),
        latent_dim=2,
        time_step=1.0,
        koopman=GraphKoopmanOperator(latent_dim=2),
    )
    with pytest.raises(ValueError, match="per-node"):
        model.fit(
            sequence,
            epochs=1,
            identification=IdentificationConfig(solver="ridge"),
        )


def test_identification_rejects_windowed_fit() -> None:
    """Windowed sampling is refused with identification."""
    true_k = torch.eye(2, dtype=torch.float64)
    sequence, _ = _linear_sequence(true_k, n_times=8)
    model = _identity_model(2, dtype=torch.float64)
    with pytest.raises(ValueError, match="windowed"):
        model.fit(
            sequence,
            epochs=1,
            window_length=4,
            identification=IdentificationConfig(),
        )


def test_identification_rejects_ddp_strategy() -> None:
    """DDP is mutually exclusive with identification."""
    true_k = torch.eye(2, dtype=torch.float64)
    sequence, _ = _linear_sequence(true_k, n_times=6)
    model = _identity_model(2, dtype=torch.float64)
    with pytest.raises(ValueError, match="ddp"):
        model.fit(
            sequence,
            epochs=1,
            strategy="ddp",
            identification=IdentificationConfig(),
        )


def test_fit_identification_gate_resdmd_fills_spectral() -> None:
    """``gate_resdmd=True`` fills the spectral block and does not abort."""
    true_k = torch.diag(torch.tensor([0.8, 0.5], dtype=torch.float64))
    sequence, _ = _linear_sequence(true_k)
    model = _identity_model(2, dtype=torch.float64)
    model.fit(
        sequence,
        epochs=1,
        identification=IdentificationConfig(
            solver="ridge", ridge=0.0, gate_resdmd=True
        ),
    )
    report = model.identification_report
    assert isinstance(report, IdentificationReport)
    assert report.spectral.residual_max is not None
    assert report.spectral.residual_max == pytest.approx(0.0, abs=1e-6)
    assert report.spectral.polluted is False
