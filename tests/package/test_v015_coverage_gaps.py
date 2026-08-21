"""Release-gate coverage for 0.15 validation and error branches.

These tests exercise public guards that the scientific suites leave
unhit. They do not lower thresholds or invent forecast numbers.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

import koopman_graph.adaptation as adaptation
from koopman_graph.analysis.criticality import CriticalityReport
from koopman_graph.analysis.dispersion import graph_dispersion
from koopman_graph.baselines.hankel_dmd import delay_embed_rows
from koopman_graph.baselines.mpedmd import _hermitian_sqrt_factors
from koopman_graph.benchmark import (
    SCHEMA_VERSION,
    ComputeBudget,
    DatasetRef,
    ExperimentManifest,
    ManifestError,
    MethodSpec,
    OODShiftSpec,
    PreprocessingSpec,
    SplitSpec,
    UQSpec,
    dump_manifest,
    load_manifest,
    manifest_from_mapping,
)
from koopman_graph.cli.benchmark import handle_benchmark_run
from koopman_graph.identification import (
    IdentificationConfig,
    IdentificationReport,
    LatentPairs,
    OperatorSnapshot,
    SpectralReliabilityBlock,
)
from koopman_graph.nn.receptive_field import (
    check_encoder_operator_receptive_field,
)
from koopman_graph.nn.separable import (
    SeparableDictionaryDecoder,
    is_separable_dictionary,
)
from koopman_graph.operators import GraphKoopmanOperator
from koopman_graph.operators.polynomial_graph import (
    apply_monomial_powers,
    dense_polynomial_kronecker,
)
from koopman_graph.uq import JointCoverageSpec, energy_score, gaussian_nll
from koopman_graph.uq.scores import _reduce

_DIGEST = hashlib.sha256(b"fixture-bytes").hexdigest()


def _dataset() -> DatasetRef:
    """Return a valid dataset reference."""
    return DatasetRef(
        name="toy-path",
        version="1",
        sha256=_DIGEST,
        card="docs/data/toy.md",
    )


def _manifest(**overrides: object) -> ExperimentManifest:
    """Build a valid telemetry manifest."""
    payload: dict[str, object] = {
        "manifest_id": "smoke-telemetry",
        "schema_version": SCHEMA_VERSION,
        "track": "telemetry",
        "dataset": _dataset(),
        "split": SplitSpec(0.7, 0.1, 0.2, history_len=12),
        "preprocessing": PreprocessingSpec(zscore=True),
        "methods": (MethodSpec(name="graph_koopman", role="koopman"),),
        "seeds": (0, 1, 2),
        "horizons": (1, 3, 12),
        "metrics": ("mae", "rmse"),
        "compute_budget": ComputeBudget(max_epochs=2),
        "controls": ("pernode",),
    }
    payload.update(overrides)
    return ExperimentManifest(**payload)  # type: ignore[arg-type]


def test_adaptation_lazy_exports_and_unknown_name() -> None:
    """Joint-observer names resolve lazily; unknown names raise."""
    assert "JointObserverResult" in dir(adaptation)
    result = adaptation.JointObserverResult
    observer = adaptation.JointStateTopologyObserver
    assert result.__name__ == "JointObserverResult"
    assert observer.__name__ == "JointStateTopologyObserver"
    with pytest.raises(AttributeError, match="has no attribute"):
        _ = adaptation.NotAnExport


def test_benchmark_cli_run_reports_io_errors(tmp_path, capsys) -> None:
    """``benchmark run`` returns 1 when the manifest is missing."""
    args = argparse.Namespace(
        manifest=str(tmp_path / "missing.yaml"),
        data=str(tmp_path / "data.bin"),
        out=str(tmp_path / "out"),
    )
    assert handle_benchmark_run(args) == 1
    captured = capsys.readouterr()
    assert "error:" in captured.err


def test_mpedmd_empty_gram_and_zero_rank_raise() -> None:
    """Truncated Gram square-root refuses empty or rank-zero factors."""
    gram = torch.zeros(2, 2)
    with pytest.raises(ValueError, match="empty"):
        _hermitian_sqrt_factors(gram, rank=None)
    identity = torch.eye(2)
    with pytest.raises(ValueError, match="empty"):
        _hermitian_sqrt_factors(identity, rank=0)
    sqrt, inv_sqrt = _hermitian_sqrt_factors(identity, rank=1)
    assert sqrt.shape == (2, 2)
    assert inv_sqrt.shape == (2, 2)


def test_identification_config_select_on_guards() -> None:
    """``select_on`` must be a non-empty string sequence; gate is a bool."""
    with pytest.raises(ValueError, match="sequence of strings"):
        IdentificationConfig(select_on="rollout")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty strings"):
        IdentificationConfig(select_on=("rollout", ""))
    IdentificationConfig(select_on=["rollout"])
    with pytest.raises(ValueError, match="gate_resdmd must be a bool"):
        IdentificationConfig(gate_resdmd=1)  # type: ignore[arg-type]


def test_latent_pairs_and_operator_snapshot_guards() -> None:
    """Latent pair dtype/device and empty snapshots raise."""
    left = torch.zeros(3, 2)
    with pytest.raises(ValueError, match="dtype"):
        LatentPairs(z_t=left, z_next=left.to(dtype=torch.float64))
    if torch.cuda.is_available():
        with pytest.raises(ValueError, match="device"):
            LatentPairs(z_t=left, z_next=left.cuda())
    with pytest.raises(TypeError, match="Tensor or None"):
        OperatorSnapshot(matrix="dense")  # type: ignore[arg-type]


def test_identification_report_field_guards() -> None:
    """Report blocks reject non-bool pollution and bad rejected names."""
    with pytest.raises(ValueError, match="finite float"):
        SpectralReliabilityBlock(residual_max="high")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="polluted"):
        SpectralReliabilityBlock(polluted="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sequence of strings"):
        IdentificationReport(rejected_alternatives="rank-2")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty strings"):
        IdentificationReport(rejected_alternatives=("rank-2", ""))
    IdentificationReport(rejected_alternatives=["rank-2"])


def test_joint_coverage_spec_rejects_unknown_target_and_block() -> None:
    """Named coverage spec validates target, alpha, and block."""
    with pytest.raises(ValueError, match="coverage target"):
        JointCoverageSpec(target="joint_box")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="coverage alpha"):
        JointCoverageSpec(alpha=1.5)
    with pytest.raises(ValueError, match="coverage block"):
        JointCoverageSpec(block="spatial")  # type: ignore[arg-type]


def test_score_reductions_and_energy_shape_guards() -> None:
    """Score reductions and energy-score ndim / pairwise expand."""
    values = torch.tensor([1.0, 3.0])
    assert torch.equal(_reduce(values, "none"), values)
    assert float(_reduce(values, "sum")) == pytest.approx(4.0)
    observation = torch.zeros(2, 3)
    samples = torch.randn(4, 2, 3)
    score = energy_score(observation, samples, reduction="none")
    assert score.shape == (2,)
    with pytest.raises(ValueError, match="\\(S, ..., d\\)"):
        energy_score(torch.tensor(0.0), torch.tensor([1.0, 2.0]))
    nll = gaussian_nll(torch.zeros(3), torch.zeros(3), torch.ones(3), reduction="sum")
    assert nll.ndim == 0


def test_manifest_io_rejects_non_mapping_and_bad_suffix(tmp_path) -> None:
    """Manifest load/dump reject lists, unknown suffixes, and YAML-import gaps."""
    json_list = tmp_path / "list.json"
    json_list.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="mapping"):
        load_manifest(json_list)
    with pytest.raises(ManifestError, match="Unsupported manifest suffix"):
        dump_manifest(_manifest(), tmp_path / "manifest.txt")
    monkey_yaml = pytest.MonkeyPatch()
    try:
        monkey_yaml.delitem(__import__("sys").modules, "yaml", raising=False)
        import builtins

        real_import = builtins.__import__

        def _block_yaml(name, *args, **kwargs):
            if name == "yaml":
                raise ImportError("blocked")
            return real_import(name, *args, **kwargs)

        monkey_yaml.setattr(builtins, "__import__", _block_yaml)
        with pytest.raises(ImportError, match="PyYAML"):
            dump_manifest(_manifest(), tmp_path / "manifest.yaml")
    finally:
        monkey_yaml.undo()


def test_schema_validation_error_paths() -> None:
    """Manifest helpers reject malformed nested fields."""
    with pytest.raises(ManifestError, match="must be a mapping"):
        manifest_from_mapping([])  # type: ignore[arg-type]
    with pytest.raises(ManifestError, match="keys must be strings"):
        manifest_from_mapping({1: "x"})  # type: ignore[dict-item]
    with pytest.raises(ManifestError, match="non-empty string"):
        DatasetRef(name=" ", version="1", sha256=_DIGEST, card="card.md")
    with pytest.raises(ManifestError, match="sequence of strings"):
        PreprocessingSpec(notes="zscore")  # type: ignore[arg-type]
    with pytest.raises(ManifestError, match="non-empty strings"):
        PreprocessingSpec(notes=("",))
    with pytest.raises(ManifestError, match="sequence of ints"):
        _manifest(seeds="012")
    with pytest.raises(ManifestError, match="must contain ints"):
        _manifest(seeds=(0, 1, 2.0))
    with pytest.raises(ManifestError, match=">= 0"):
        _manifest(seeds=(-1, 0, 1))
    with pytest.raises(ManifestError, match="must be a float"):
        SplitSpec("0.7", 0.1, 0.2)  # type: ignore[arg-type]
    with pytest.raises(ManifestError, match="0 < ratio"):
        SplitSpec(0.0, 0.5, 0.5)
    with pytest.raises(ManifestError, match="sum to 1"):
        SplitSpec(0.5, 0.5, 0.5)
    with pytest.raises(ManifestError, match="history_len"):
        SplitSpec(0.7, 0.1, 0.2, history_len=0)
    with pytest.raises(ManifestError, match="method.role"):
        MethodSpec(name="graph_koopman", role="oracle")  # type: ignore[arg-type]
    with pytest.raises(ManifestError, match="ood.description"):
        OODShiftSpec(name="night", kind="time", description=1)  # type: ignore[arg-type]
    with pytest.raises(ManifestError, match="uq.coverage"):
        UQSpec(method="conformal", coverage="0.9")  # type: ignore[arg-type]
    with pytest.raises(ManifestError, match="max_epochs"):
        ComputeBudget(max_epochs=0)
    with pytest.raises(ManifestError, match="schema_version"):
        _manifest(schema_version="benchmark_manifest_v0")
    with pytest.raises(ManifestError, match="track must be"):
        _manifest(track="libcity")
    with pytest.raises(ManifestError, match="dataset must be"):
        _manifest(dataset={"name": "toy"})
    with pytest.raises(ManifestError, match="split must be"):
        _manifest(split={"train_ratio": 0.7})
    with pytest.raises(ManifestError, match="preprocessing must be"):
        _manifest(preprocessing={"zscore": True})
    with pytest.raises(ManifestError, match="compute_budget must be"):
        _manifest(compute_budget={"max_epochs": 2})
    with pytest.raises(ManifestError, match="uq must be"):
        _manifest(uq={"method": "conformal"})
    with pytest.raises(ManifestError, match="methods must be"):
        _manifest(methods="graph_koopman")
    with pytest.raises(ManifestError, match="methods must be a non-empty"):
        _manifest(methods=())
    with pytest.raises(ManifestError, match="MethodSpec"):
        _manifest(methods=({"name": "graph_koopman", "role": "koopman"},))
    with pytest.raises(ManifestError, match="horizons must be a non-empty"):
        _manifest(horizons=())
    with pytest.raises(ManifestError, match="metrics must be a non-empty"):
        _manifest(metrics=())
    with pytest.raises(ManifestError, match="subset"):
        _manifest(metrics=("mae", "madeup"))
    with pytest.raises(ManifestError, match="ood_shifts must be"):
        _manifest(ood_shifts="night")

    mapping = {
        "manifest_id": "smoke",
        "schema_version": SCHEMA_VERSION,
        "track": "telemetry",
        "dataset": {
            "name": "toy",
            "version": "1",
            "sha256": _DIGEST,
            "card": "card.md",
        },
        "split": {"train_ratio": 0.7, "val_ratio": 0.1, "test_ratio": 0.2},
        "methods": [{"name": "graph_koopman", "role": "koopman"}],
        "seeds": [0, 1, 2],
        "horizons": [1, 3],
        "metrics": ["mae"],
        "compute_budget": {"max_epochs": 2},
        "controls": ["pernode"],
    }
    incomplete = copy.deepcopy(mapping)
    del incomplete["dataset"]["card"]  # type: ignore[index]
    with pytest.raises(ManifestError, match="dataset missing"):
        manifest_from_mapping(incomplete)
    split_missing = copy.deepcopy(mapping)
    split_missing["split"] = {"train_ratio": 0.7, "val_ratio": 0.1}
    with pytest.raises(ManifestError, match="split missing"):
        manifest_from_mapping(split_missing)
    method_missing = copy.deepcopy(mapping)
    method_missing["methods"] = [{"name": "graph_koopman"}]
    with pytest.raises(ManifestError, match="method missing"):
        manifest_from_mapping(method_missing)
    ood_missing = copy.deepcopy(mapping)
    ood_missing["ood_shifts"] = [{"name": "night"}]
    with pytest.raises(ManifestError, match="ood_shift missing"):
        manifest_from_mapping(ood_missing)
    uq_missing = copy.deepcopy(mapping)
    uq_missing["uq"] = {"coverage": 0.9}
    with pytest.raises(ManifestError, match="uq missing"):
        manifest_from_mapping(uq_missing)
    budget_missing = copy.deepcopy(mapping)
    budget_missing["compute_budget"] = {"device": "cpu"}
    with pytest.raises(ManifestError, match="compute_budget missing"):
        manifest_from_mapping(budget_missing)
    keys_missing = copy.deepcopy(mapping)
    del keys_missing["seeds"]
    with pytest.raises(ManifestError, match="manifest missing"):
        manifest_from_mapping(keys_missing)
    with pytest.raises(ManifestError, match="methods must be a sequence"):
        manifest_from_mapping({**mapping, "methods": "graph_koopman"})
    with pytest.raises(ManifestError, match="ood_shifts must be a sequence"):
        manifest_from_mapping({**mapping, "ood_shifts": "night"})


def test_polynomial_graph_shape_and_min_power_guards() -> None:
    """Polynomial assembly and matvec helpers reject mismatched hops."""
    adjacency = torch.eye(2)
    hops = (torch.eye(2), torch.ones(2, 3))
    with pytest.raises(ValueError, match="hop_matrices\\[1\\]"):
        dense_polynomial_kronecker(adjacency, hops)
    states = torch.zeros(3, 2)
    with pytest.raises(ValueError, match="min_power"):
        apply_monomial_powers(states, (torch.eye(2),), lambda z: z, min_power=0)
    empty = apply_monomial_powers(states, (), lambda z: z, min_power=1)
    assert torch.equal(empty, states)
    with pytest.raises(ValueError, match="hop_matrices\\[0\\]"):
        apply_monomial_powers(states, (torch.ones(3, 3),), lambda z: z)


def test_receptive_field_unwraps_and_skips_bad_metadata() -> None:
    """Hop inference ignores bools, negatives, and raising getters."""

    class RaisingEncoder(nn.Module):
        def receptive_field_hops(self) -> int:
            raise TypeError("broken hops")

    class Diffusive(nn.Module):
        diffusion_steps = 2
        num_layers = 3

    class Wrapped(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base_encoder = Diffusive()

    class NegativeHops(nn.Module):
        def receptive_field_hops(self) -> int:
            return -1

    class RaisingOperator(nn.Module):
        def receptive_field_hops(self) -> int:
            raise ValueError("broken operator hops")

    silent = check_encoder_operator_receptive_field(RaisingEncoder(), nn.Linear(2, 2))
    assert silent.encoder_hops is None
    hops = check_encoder_operator_receptive_field(Wrapped(), RaisingOperator())
    assert hops.encoder_hops == 6
    assert hops.operator_hops is None
    skipped = check_encoder_operator_receptive_field(NegativeHops(), nn.Identity())
    assert skipped.encoder_hops is None


def test_separable_kind_flag_and_decoder_hops() -> None:
    """A duck-typed separable wrapper and decoder hops are recognized."""
    wrapper = SimpleNamespace(encoder_kind="separable", base_encoder=None)
    assert is_separable_dictionary(wrapper)
    decoder = SeparableDictionaryDecoder(3, 4, 2, num_layers=3, activation="sigmoid")
    assert decoder.receptive_field_hops() == 0


def test_criticality_report_and_dispersion_none_path() -> None:
    """Criticality validation and a mocked ineligible dispersion raise."""
    gap = torch.tensor([0.2, 0.1])
    with pytest.raises(ValueError, match="shape \\(T,\\)"):
        CriticalityReport(
            spectral_gap=gap,
            gap_closure_rate=torch.tensor([0.1]),
            max_gap_closure_rate=0.1,
            near_defective=torch.tensor([True]),
        )
    with pytest.raises(ValueError, match="boolean tensor"):
        CriticalityReport(
            spectral_gap=gap,
            gap_closure_rate=torch.tensor([0.1]),
            max_gap_closure_rate=0.1,
            near_defective=torch.tensor([1, 0]),
        )
    with pytest.raises(ValueError, match="length <= T"):
        CriticalityReport(
            spectral_gap=gap,
            gap_closure_rate=torch.tensor([0.1, 0.2, 0.3]),
            max_gap_closure_rate=0.3,
            near_defective=torch.zeros(2, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="non-negative"):
        CriticalityReport(
            spectral_gap=torch.tensor([-0.1, 0.2]),
            gap_closure_rate=torch.tensor([0.1]),
            max_gap_closure_rate=0.1,
            near_defective=torch.zeros(2, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="must not be NaN"):
        CriticalityReport(
            spectral_gap=gap,
            gap_closure_rate=torch.tensor([0.1]),
            max_gap_closure_rate=float("nan"),
            near_defective=torch.zeros(2, dtype=torch.bool),
        )

    import koopman_graph.analysis.criticality as crit_mod

    with pytest.raises(ValueError, match="at least two"):
        crit_mod._pairwise_spectral_gap(torch.tensor([1.0]))
    with pytest.raises(ValueError, match="must be finite"):
        crit_mod._pairwise_spectral_gap(torch.tensor([1.0, float("nan")]))

    import koopman_graph.analysis.dispersion as disp_mod

    operator = GraphKoopmanOperator(2, init_mode="identity")
    edges = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    with (
        pytest.raises(ValueError, match="not eligible"),
        pytest.MonkeyPatch.context() as monkey,
    ):
        monkey.setattr(disp_mod, "spectrum_k_eff_kronecker_sum", lambda **_: None)
        graph_dispersion(operator, edges, 2)

    spectrum = MagicMock()
    spectrum.eigenvalues = torch.ones(2)
    spectrum.growth_rates.real = torch.ones(8)
    with pytest.MonkeyPatch.context() as monkey:
        monkey.setattr(disp_mod, "spectrum_k_eff_kronecker_sum", lambda **_: spectrum)
        relation = graph_dispersion(operator, edges, 2)
    assert relation.growth_rates.numel() == 2


def test_hankel_delay_matrix_rejects_non_matrix_states() -> None:
    """Hankel assembly requires a 2-D state matrix."""
    with pytest.raises(ValueError, match="2-D"):
        delay_embed_rows(torch.zeros(3), n_delays=2)
