"""Coverage and error-path tests for :mod:`koopman_graph.nn`."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from koopman_graph.nn import (
    InvariantGeometryEncoder,
)
from koopman_graph.nn.equivariant import (
    E3EquivariantEncoder,
    invariant_geometry_features,
)


def _path_edges(num_nodes: int = 4) -> Tensor:
    """Undirected path ``edge_index``."""
    src = list(range(num_nodes - 1)) + list(range(1, num_nodes))
    dst = list(range(1, num_nodes)) + list(range(num_nodes - 1))
    return torch.tensor([src, dst], dtype=torch.long)


def test_e3_encoder_with_stub_o3(monkeypatch: pytest.MonkeyPatch) -> None:
    """E3 encoder constructs and validates inputs against a local e3nn stand-in."""
    import koopman_graph.nn.equivariant as equiv_mod

    class _FakeIrreps:
        def __init__(self, spec: str = "") -> None:
            self.spec = spec

        @classmethod
        def spherical_harmonics(cls, lmax: int) -> _FakeIrreps:
            return cls(f"sh{lmax}")

    def _irreps_dim(spec: object) -> int:
        text = str(getattr(spec, "spec", spec))
        dim = 0
        for part in text.replace(" ", "").split("+"):
            if "x" not in part:
                continue
            mult, irrep = part.split("x", maxsplit=1)
            count = int(mult)
            dim += count if irrep.startswith("0") else 3 * count
        return max(dim, 1)

    class _FakeLinear(nn.Module):
        def __init__(self, irreps_in: object, irreps_out: object) -> None:
            super().__init__()
            self.in_dim = _irreps_dim(irreps_in)
            self.out_dim = _irreps_dim(irreps_out)
            self.core = nn.Linear(self.in_dim, self.out_dim)

        def forward(self, features: Tensor) -> Tensor:
            if features.size(-1) < self.in_dim:
                pad = features.new_zeros(
                    *features.shape[:-1],
                    self.in_dim - features.size(-1),
                )
                features = torch.cat([features, pad], dim=-1)
            elif features.size(-1) > self.in_dim:
                features = features[..., : self.in_dim]
            return self.core(features)

    class _FakeTP(nn.Module):
        def __init__(
            self,
            irreps_in: object,
            irreps_sh: object,
            irreps_out: object,
            shared_weights: bool = False,
        ) -> None:
            super().__init__()
            del irreps_in, irreps_sh, shared_weights
            self.weight_numel = 4
            self.out_dim = _irreps_dim(irreps_out)

        def forward(self, src_features: Tensor, *_rest: Tensor) -> Tensor:
            return src_features.new_zeros(src_features.shape[0], self.out_dim)

    class _FakeFC(nn.Module):
        def __init__(self, dims: list[int], _act: Any) -> None:
            super().__init__()
            self.out = int(dims[-1])

        def forward(self, values: Tensor) -> Tensor:
            return values.new_zeros(values.shape[0], self.out)

    fake_o3 = SimpleNamespace(
        Irreps=_FakeIrreps,
        Linear=_FakeLinear,
        FullyConnectedTensorProduct=_FakeTP,
        spherical_harmonics=lambda *args, **kwargs: args[1],
    )
    fake_enn = SimpleNamespace(FullyConnectedNet=_FakeFC)
    monkeypatch.setattr(equiv_mod, "_import_e3nn_modules", lambda: (fake_o3, fake_enn))
    with pytest.raises(ValueError, match="lmax"):
        E3EquivariantEncoder(2, 4, 3, lmax=-1)
    encoder = E3EquivariantEncoder(2, 4, 3, num_layers=1, lmax=1)
    with pytest.raises(ValueError, match="requires a torch_geometric"):
        encoder(torch.randn(4, 2))
    data = Data(
        x=torch.randn(4, 2),
        edge_index=_path_edges(4),
        pos=torch.randn(4, 3),
    )
    assert encoder(data).shape[0] == 4
    missing_x = data.clone()
    missing_x.x = None
    with pytest.raises(ValueError, match="data.x is required"):
        encoder(missing_x)
    missing_edge = data.clone()
    missing_edge.edge_index = None
    with pytest.raises(ValueError, match="data.edge_index is required"):
        encoder(missing_edge)
    missing_pos = data.clone()
    del missing_pos.pos
    with pytest.raises(ValueError, match="data.pos is required"):
        encoder(missing_pos)
    bad_pos = data.clone()
    bad_pos.pos = torch.randn(4, 2)
    with pytest.raises(ValueError, match="shape"):
        encoder(bad_pos)
    nonfinite = data.clone()
    nonfinite.pos = data.pos.clone()
    nonfinite.pos[0, 0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        encoder(nonfinite)
    bad_x = data.clone()
    bad_x.x = torch.randn(4, 3)
    with pytest.raises(ValueError, match="Expected data.x"):
        encoder(bad_x)
    mismatch = data.clone()
    mismatch.x = torch.randn(3, 2)
    with pytest.raises(ValueError, match="node counts"):
        encoder(mismatch)
    empty = Data(
        x=torch.randn(3, 2),
        edge_index=torch.zeros(2, 0, dtype=torch.long),
        pos=torch.randn(3, 3),
    )
    assert encoder(empty).shape[0] == 3
    with pytest.raises(ValueError, match="n_vectors"):
        E3EquivariantEncoder(2, 4, 3, project_invariants=False, n_vectors=-1)
    vectors = E3EquivariantEncoder(
        2,
        4,
        3,
        num_layers=1,
        project_invariants=False,
        n_vectors=1,
    )
    assert vectors(data).shape[0] == 4
    geom = invariant_geometry_features(
        torch.randn(3, 3),
        torch.zeros(2, 0, dtype=torch.long),
    )
    assert geom.shape == (3, 4)
    looped = torch.tensor([[0, 0, 1], [0, 1, 0]], dtype=torch.long)
    _ = invariant_geometry_features(torch.randn(2, 3), looped)
    encoder_inv = InvariantGeometryEncoder(2, 4, 3)
    weighted = Data(
        x=torch.randn(4, 2),
        edge_index=_path_edges(4),
        pos=torch.randn(4, 3),
        edge_weight=torch.ones(6),
    )
    assert encoder_inv(weighted).shape[0] == 4
