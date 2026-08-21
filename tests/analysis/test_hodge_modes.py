"""Combinatorial Hodge split of stored Koopman mode shapes."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

import koopman_graph
import koopman_graph.analysis as analysis
from koopman_graph.analysis.hodge_modes import (
    DEFAULT_HODGE_ATOL,
    HodgeModeComponents,
    hodge_decompose_modes,
)
from koopman_graph.observables import boundary_incidence_b1
from koopman_graph.spectrum_types import KoopmanSpectrum, compute_spectrum

# Independent construction: oriented cycle circulation is ker(B_1).
_ORACLE_ABS = DEFAULT_HODGE_ATOL
_CYCLE_N = 6
_SEED = 0


def _cycle_edge_index(num_nodes: int) -> torch.Tensor:
    """One oriented column per cycle edge ``i → i+1`` (last wraps)."""
    tails = torch.arange(num_nodes, dtype=torch.long)
    heads = torch.roll(tails, -1)
    return torch.stack((tails, heads), dim=0)


def _path_edge_index(num_nodes: int) -> torch.Tensor:
    """Oriented path ``0→1→…→N-1`` (acyclic 1-skeleton)."""
    tails = torch.arange(num_nodes - 1, dtype=torch.long)
    heads = tails + 1
    return torch.stack((tails, heads), dim=0)


def _spectrum_from_columns(columns: torch.Tensor) -> KoopmanSpectrum:
    """Minimal spectrum whose eigenvector columns are the given cochains."""
    modes = columns.to(dtype=torch.complex128)
    n_modes = int(modes.shape[1])
    eigenvalues = torch.ones(n_modes, dtype=torch.complex128)
    return KoopmanSpectrum(
        eigenvalues=eigenvalues,
        eigenvectors=modes,
        magnitudes=eigenvalues.abs(),
        growth_rates=torch.zeros(n_modes, dtype=torch.float64),
        frequencies=torch.zeros(n_modes, dtype=torch.float64),
        time_step=1.0,
    )


def _relative_energy(part: torch.Tensor, mode: torch.Tensor) -> float:
    """``||part||_2 / ||mode||_2`` for one column."""
    mode_norm = float(torch.linalg.vector_norm(mode).item())
    part_norm = float(torch.linalg.vector_norm(part).item())
    return part_norm / mode_norm


def test_package_and_analysis_export_hodge_modes_off_root() -> None:
    """Hodge split lives on ``analysis.__all__``, not the root façade."""
    assert "hodge_decompose_modes" in analysis.__all__
    assert "HodgeModeComponents" in analysis.__all__
    assert "hodge_decompose_modes" not in koopman_graph.__all__
    assert "HodgeModeComponents" not in koopman_graph.__all__
    assert analysis.hodge_decompose_modes is hodge_decompose_modes
    assert analysis.HodgeModeComponents is HodgeModeComponents


def test_hodge_modes_module_does_not_import_model() -> None:
    """L2 analysis helper must not import L4 model."""
    source = Path(__file__).resolve().parents[2] / (
        "src/koopman_graph/analysis/hodge_modes.py"
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
        if name == "koopman_graph.model"
        or name.startswith("koopman_graph.model.")
        or name == "koopman_graph.nn"
        or name.startswith("koopman_graph.nn.")
    ]
    assert not offenders


def test_cycle_circulation_is_harmonic_1_cochain() -> None:
    """All-ones 1-cochain on an oriented cycle is identified as harmonic.

    Independent oracle: column ``e`` of ``B_1`` is ``+1`` at the tail and
    ``-1`` at the head, so ``B_1 1_E = 0``. The combinatorial 1-Laplacian
    ``L_1 = B_1^T B_1`` therefore annihilates the circulation. This is
    not a physical current.
    """
    edge_index = _cycle_edge_index(_CYCLE_N)
    circulation = torch.ones(_CYCLE_N, dtype=torch.float64)
    spectrum = _spectrum_from_columns(circulation.unsqueeze(1))
    parts = hodge_decompose_modes(
        spectrum,
        edge_index,
        num_nodes=_CYCLE_N,
        degree=1,
    )
    mode = spectrum.eigenvectors[:, 0]
    assert _relative_energy(parts.harmonic[:, 0], mode) == pytest.approx(
        1.0, abs=_ORACLE_ABS
    )
    assert _relative_energy(parts.gradient[:, 0], mode) == pytest.approx(
        0.0, abs=_ORACLE_ABS
    )
    assert torch.equal(parts.curl, torch.zeros_like(parts.curl))
    incidence = boundary_incidence_b1(edge_index, num_nodes=_CYCLE_N).to(
        dtype=torch.float64
    )
    laplacian = incidence.T @ incidence
    leftover = laplacian @ parts.harmonic[:, 0].real
    assert torch.linalg.vector_norm(leftover).item() == pytest.approx(
        0.0, abs=_ORACLE_ABS
    )


def test_cycle_all_ones_without_degree_infers_k0() -> None:
    """When ``N=E``, omitted ``degree`` infers a 0-cochain split.

    The all-ones vector is harmonic as both ``1_N`` and ``1_E``. The
    inferred path is ``k=0``: gradient is structurally zero and curl
    is a least-squares residual, not the structural zero of a
    face-free ``k=1`` split.
    """
    edge_index = _cycle_edge_index(_CYCLE_N)
    spectrum = _spectrum_from_columns(torch.ones(_CYCLE_N, 1))
    inferred = hodge_decompose_modes(spectrum, edge_index, num_nodes=_CYCLE_N)
    forced = hodge_decompose_modes(
        spectrum,
        edge_index,
        num_nodes=_CYCLE_N,
        degree=1,
    )
    assert torch.equal(inferred.gradient, torch.zeros_like(inferred.gradient))
    assert not torch.equal(inferred.curl, torch.zeros_like(inferred.curl))
    assert torch.equal(forced.curl, torch.zeros_like(forced.curl))
    assert not torch.equal(forced.gradient, torch.zeros_like(forced.gradient))
    mode = spectrum.eigenvectors[:, 0]
    assert _relative_energy(inferred.harmonic[:, 0], mode) == pytest.approx(
        1.0, abs=_ORACLE_ABS
    )
    assert _relative_energy(inferred.curl[:, 0], mode) == pytest.approx(
        0.0, abs=_ORACLE_ABS
    )


def test_cycle_gradient_flow_is_not_harmonic() -> None:
    """``B_1^T φ`` on a cycle is a gradient 1-cochain, not harmonic."""
    edge_index = _cycle_edge_index(_CYCLE_N)
    incidence = boundary_incidence_b1(edge_index, num_nodes=_CYCLE_N).to(
        dtype=torch.float64
    )
    potential = torch.tensor(
        [1.0, -0.5, 0.25, 0.0, -0.75, 1.0],
        dtype=torch.float64,
    )
    flow = incidence.T @ potential
    spectrum = _spectrum_from_columns(flow.unsqueeze(1))
    parts = hodge_decompose_modes(
        spectrum,
        edge_index,
        num_nodes=_CYCLE_N,
        degree=1,
    )
    mode = spectrum.eigenvectors[:, 0]
    assert _relative_energy(parts.gradient[:, 0], mode) == pytest.approx(
        1.0, abs=_ORACLE_ABS
    )
    assert _relative_energy(parts.harmonic[:, 0], mode) == pytest.approx(
        0.0, abs=_ORACLE_ABS
    )
    reconstructed = parts.gradient + parts.curl + parts.harmonic
    assert torch.allclose(
        reconstructed,
        spectrum.eigenvectors,
        atol=_ORACLE_ABS,
        rtol=0.0,
    )


def test_compute_spectrum_cycle_mode_is_harmonic() -> None:
    """Largest-magnitude eigenmode of a cycle circulation map is harmonic."""
    torch.manual_seed(_SEED)
    edge_index = _cycle_edge_index(_CYCLE_N)
    raw = torch.randn(_CYCLE_N, _CYCLE_N, dtype=torch.float64)
    raw[:, 0] = 1.0
    basis, _ = torch.linalg.qr(raw)
    eigenvalues = torch.tensor(
        [0.9, 0.5, 0.3, 0.2, 0.1, 0.05],
        dtype=torch.float64,
    )
    operator = basis @ torch.diag(eigenvalues) @ basis.T
    spectrum = compute_spectrum(operator, time_step=1.0)
    parts = hodge_decompose_modes(
        spectrum,
        edge_index,
        num_nodes=_CYCLE_N,
        degree=1,
    )
    leading = spectrum.eigenvectors[:, 0]
    assert _relative_energy(parts.harmonic[:, 0], leading) == pytest.approx(
        1.0, abs=_ORACLE_ABS
    )
    assert _relative_energy(parts.gradient[:, 0], leading) == pytest.approx(
        0.0, abs=_ORACLE_ABS
    )


def test_path_ones_1_cochain_is_gradient() -> None:
    """An acyclic path has trivial ``ker L_1``; all-ones is gradient."""
    num_nodes = 4
    edge_index = _path_edge_index(num_nodes)
    ones = torch.ones(num_nodes - 1, dtype=torch.float64)
    spectrum = _spectrum_from_columns(ones.unsqueeze(1))
    parts = hodge_decompose_modes(
        spectrum,
        edge_index,
        num_nodes=num_nodes,
    )
    mode = spectrum.eigenvectors[:, 0]
    assert _relative_energy(parts.harmonic[:, 0], mode) == pytest.approx(
        0.0, abs=_ORACLE_ABS
    )
    assert _relative_energy(parts.gradient[:, 0], mode) == pytest.approx(
        1.0, abs=_ORACLE_ABS
    )


def test_constant_0_cochain_is_harmonic_on_connected_graph() -> None:
    """Constants span ``ker L_0``; a mean-zero node vector is curl."""
    edge_index = _cycle_edge_index(_CYCLE_N)
    constant = torch.ones(_CYCLE_N, dtype=torch.float64)
    incidence = boundary_incidence_b1(edge_index, num_nodes=_CYCLE_N).to(
        dtype=torch.float64
    )
    flow = torch.arange(_CYCLE_N, dtype=torch.float64)
    mean_zero = incidence @ flow
    spectrum = _spectrum_from_columns(torch.stack((constant, mean_zero), dim=1))
    parts = hodge_decompose_modes(
        spectrum,
        edge_index,
        num_nodes=_CYCLE_N,
        degree=0,
    )
    assert _relative_energy(
        parts.harmonic[:, 0],
        spectrum.eigenvectors[:, 0],
    ) == pytest.approx(1.0, abs=_ORACLE_ABS)
    assert _relative_energy(
        parts.curl[:, 0],
        spectrum.eigenvectors[:, 0],
    ) == pytest.approx(0.0, abs=_ORACLE_ABS)
    assert _relative_energy(
        parts.curl[:, 1],
        spectrum.eigenvectors[:, 1],
    ) == pytest.approx(1.0, abs=_ORACLE_ABS)
    assert _relative_energy(
        parts.harmonic[:, 1],
        spectrum.eigenvectors[:, 1],
    ) == pytest.approx(0.0, abs=_ORACLE_ABS)
    assert torch.equal(parts.gradient, torch.zeros_like(parts.gradient))


def test_complex_circulation_splits_real_and_imag_independently() -> None:
    """Real combinatorial Hodge is applied to each part of a complex column."""
    edge_index = _cycle_edge_index(_CYCLE_N)
    circulation = (1.0 + 2.0j) * torch.ones(_CYCLE_N, dtype=torch.complex128)
    spectrum = _spectrum_from_columns(circulation.unsqueeze(1))
    parts = hodge_decompose_modes(
        spectrum,
        edge_index,
        num_nodes=_CYCLE_N,
        degree=1,
    )
    mode = spectrum.eigenvectors[:, 0]
    assert _relative_energy(parts.harmonic[:, 0], mode) == pytest.approx(
        1.0, abs=_ORACLE_ABS
    )
    assert torch.equal(parts.curl, torch.zeros_like(parts.curl))
    assert _relative_energy(parts.gradient[:, 0], mode) == pytest.approx(
        0.0, abs=_ORACLE_ABS
    )


def test_hodge_decompose_modes_rejects_invalid_inputs() -> None:
    """Row-count, degree, and non-finite columns raise ``ValueError``."""
    edge_index = _cycle_edge_index(_CYCLE_N)
    valid = _spectrum_from_columns(torch.ones(_CYCLE_N, 1))
    with pytest.raises(TypeError, match="KoopmanSpectrum"):
        hodge_decompose_modes(
            object(),  # type: ignore[arg-type]
            edge_index,
            num_nodes=_CYCLE_N,
        )
    with pytest.raises(ValueError, match="num_nodes"):
        hodge_decompose_modes(valid, edge_index, num_nodes=True)
    with pytest.raises(ValueError, match="infer"):
        flat = _spectrum_from_columns(torch.ones(_CYCLE_N * 2, 1))
        hodge_decompose_modes(flat, edge_index, num_nodes=_CYCLE_N)
    with pytest.raises(ValueError, match="k=2"):
        hodge_decompose_modes(valid, edge_index, num_nodes=_CYCLE_N, degree=2)
    with pytest.raises(ValueError, match="degree=1 requires"):
        nodes = _spectrum_from_columns(torch.ones(_CYCLE_N, 1))
        hodge_decompose_modes(
            nodes,
            _path_edge_index(_CYCLE_N),
            num_nodes=_CYCLE_N,
            degree=1,
        )
    nan_columns = torch.ones(_CYCLE_N, 1, dtype=torch.complex128)
    nan_columns[0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        hodge_decompose_modes(
            _spectrum_from_columns(nan_columns),
            edge_index,
            num_nodes=_CYCLE_N,
        )


def test_hodge_mode_components_require_matching_shapes() -> None:
    """Frozen report rejects mismatched gradient / curl / harmonic blocks."""
    good = torch.zeros(3, 2)
    with pytest.raises(ValueError, match="share shape"):
        HodgeModeComponents(
            gradient=good,
            curl=torch.zeros(3, 1),
            harmonic=good,
        )
