"""Order-2 TDL teaching path and cochain-operator hook."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

import koopman_graph
from koopman_graph.model.factory import parse_koopman_arg
from koopman_graph.nn import (
    MAX_CELL_COMPLEX_DEGREE,
    Order2CochainTeaching,
    SheafGNNEncoder,
    bind_cochain_operator,
    cell_complex_boundary_nilpotency,
    order2_cochain_teaching,
    teaching_order2_triangle,
    teaching_order3_tetrahedron,
)
from koopman_graph.nn.cell_complex import boundary_operator
from koopman_graph.operators import CochainKoopmanOperator, CochainState
from koopman_graph.operators.cochain import DEFAULT_NILPOTENCY_ATOL

_ATOL = DEFAULT_NILPOTENCY_ATOL
_LATENT_DIM = 2


def test_tdl_teaching_exports_off_root() -> None:
    """Teaching helpers live on ``nn.__all__``, not the root façade."""
    assert "order2_cochain_teaching" in koopman_graph.nn.__all__
    assert "bind_cochain_operator" in koopman_graph.nn.__all__
    assert "Order2CochainTeaching" in koopman_graph.nn.__all__
    assert "teaching_order2_triangle" in koopman_graph.nn.__all__
    assert "order2_cochain_teaching" not in koopman_graph.__all__
    assert "bind_cochain_operator" not in koopman_graph.__all__
    assert "Order2CochainTeaching" not in koopman_graph.__all__


def test_operators_cochain_still_does_not_import_nn() -> None:
    """L2 cochain module must not import L3 nn (hook lives in nn)."""
    source = Path(__file__).resolve().parents[2] / (
        "src/koopman_graph/operators/cochain.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)
    offenders = [
        name
        for name in imported
        if name == "koopman_graph.nn" or name.startswith("koopman_graph.nn.")
    ]
    assert not offenders


def test_order2_teaching_binds_cochain_and_is_nilpotent() -> None:
    """Filled triangle binds ``CochainKoopmanOperator`` with ``B1 B2 = 0``."""
    path = order2_cochain_teaching(latent_dim=_LATENT_DIM)
    assert isinstance(path, Order2CochainTeaching)
    assert isinstance(path.operator, CochainKoopmanOperator)
    assert path.max_cell_degree == MAX_CELL_COMPLEX_DEGREE == 3
    assert path.complex.num_cells(0) == 3
    assert path.complex.num_cells(1) == 3
    assert path.complex.num_cells(2) == 1
    assert path.nilpotency.nilpotent is True
    assert path.nilpotency.max_abs <= _ATOL
    assert path.operator.num_nodes == 3
    assert path.operator.num_edges == 3


def test_bound_operator_copies_face_latents() -> None:
    """Face table is stored and copied; ``k=2`` is not evolved."""
    path = order2_cochain_teaching(latent_dim=_LATENT_DIM)
    face = torch.arange(4, dtype=torch.float32).reshape(1, 4)
    state = CochainState(
        node=torch.ones(3, _LATENT_DIM),
        edge=torch.full((3, _LATENT_DIM), 0.5),
        face=face,
    )
    nxt = path.operator.advance(state)
    assert nxt.face is not None
    assert torch.equal(nxt.face, face)
    assert not torch.equal(nxt.node, state.node)


def test_bind_cochain_operator_on_degree3_tetrahedron() -> None:
    """Degree-3 ceiling fixture still binds a ``k<=1`` operator."""
    complex_ = teaching_order3_tetrahedron()
    assert complex_.num_cells(3) == 1
    report = cell_complex_boundary_nilpotency(complex_)
    assert report.nilpotent is True
    assert report.max_abs <= _ATOL
    operator = bind_cochain_operator(complex_, latent_dim=_LATENT_DIM)
    assert operator.num_nodes == 4
    assert operator.num_edges == 6
    product_b2_b3 = boundary_operator(complex_, 2) @ boundary_operator(complex_, 3)
    assert torch.allclose(
        product_b2_b3,
        torch.zeros_like(product_b2_b3),
        atol=_ATOL,
        rtol=0.0,
    )
    with pytest.raises(ValueError, match="teaching MVP"):
        boundary_operator(complex_, MAX_CELL_COMPLEX_DEGREE + 1)


def test_sheaf_restriction_maps_remain_optional() -> None:
    """Sheaf peers keep diagonal restriction maps unless opted in."""
    encoder = SheafGNNEncoder(2, 4, 4, num_layers=1)
    assert encoder.restriction_maps == "diagonal"


def test_cochain_is_still_not_a_factory_kind() -> None:
    """Order-2 teaching does not add a ``koopman="cochain"`` kind."""
    kind, injected = parse_koopman_arg(None)
    assert kind == "pernode"
    assert injected is None
    with pytest.raises(ValueError, match="cochain"):
        parse_koopman_arg("cochain")


def test_teaching_helpers_reject_invalid_inputs() -> None:
    """Hook guards stay on CellComplex and require 2-cells."""
    with pytest.raises(TypeError, match="CellComplex"):
        bind_cochain_operator(object(), latent_dim=_LATENT_DIM)  # type: ignore[arg-type]
    skeleton = teaching_order2_triangle()
    object.__setattr__(
        skeleton,
        "face_index",
        torch.empty((3, 0), dtype=torch.long),
    )
    with pytest.raises(ValueError, match="2-cell"):
        cell_complex_boundary_nilpotency(skeleton)
