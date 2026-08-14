"""Public topology-transfer evaluation API (TASK-1900–1903).

Appendix B transfer-advantage inequality uses
:data:`~koopman_graph.analysis.TRANSFER_ADVANTAGE_EPSILON` as the single
source for :math:`\\varepsilon` — do not redefine it here.

Example-37 control-comparison oracle (TASK-1903)
------------------------------------------------
On the seeded path-diffusion fixture (``N1=4→N2=6``, 100 epochs), the
public API must reproduce the archived MSEs within ``_EXAMPLE37_MSE_ABS``
and report ``transfer_advantage is False``. Inverting that flag or moving
anchors beyond tolerance requires a **versioned fixture bump** and a
``CHANGELOG.md`` entry — do not silently green the suite by widening
tolerances.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
from koopman_graph.analysis import (
    TRANSFER_ADVANTAGE_EPSILON,
    TopologyTransferReport,
    evaluate_topology_transfer,
)
from koopman_graph.analysis.transfer import (
    TRANSFER_ADVANTAGE_EPSILON as TRANSFER_EPS_MODULE,
)
from koopman_graph.datasets.synthetic import SyntheticDynamicGraphBenchmark

_N1 = 4
_N2 = 6
_FEATURES = 2
_LATENT = 4
_STEPS = 4

# Example-37 oracle anchors (archived ~0.26 / ~0.21; live API 2026-08-01).
# abs=0.05 catches ordering inversions while allowing BLAS / float drift.
_EXAMPLE37_MSE_ABS = 0.05
_EXAMPLE37_ANCHORS = {
    "graph": {"in_dist": 0.26, "transfer": 0.26},
    "pernode": {"in_dist": 0.21, "transfer": 0.21},
}
_EXAMPLE37_EPOCHS = 100
_EXAMPLE37_LR = 5e-3
_EXAMPLE37_STEPS = 8
_EXAMPLE37_HIDDEN = 32


def _diffusion_params() -> dict:
    return {
        "in_channels": _FEATURES,
        "topology": "path",
        "diffusion_rate": 0.35,
        "decay_rate": 0.92,
        "noise_std": 0.0,
        "initial_state": "ones",
    }


def _template_model(**kwargs) -> GraphKoopmanModel:
    return GraphKoopmanModel(
        GNNEncoder(_FEATURES, 16, _LATENT, num_layers=2),
        GNNDecoder(_LATENT, 16, _FEATURES, num_layers=2),
        latent_dim=_LATENT,
        time_step=1.0,
        koopman="graph",
        **kwargs,
    )


def _tiny_sequences():
    params = _diffusion_params()
    train = SyntheticDynamicGraphBenchmark.generate(
        num_nodes=_N1, num_timesteps=16, seed=0, **params
    )
    hold_n1 = SyntheticDynamicGraphBenchmark.generate(
        num_nodes=_N1, num_timesteps=_STEPS + 2, seed=1, **params
    )
    hold_n2 = SyntheticDynamicGraphBenchmark.generate(
        num_nodes=_N2, num_timesteps=_STEPS + 2, seed=2, **params
    )
    return train, hold_n1, hold_n2


def test_transfer_advantage_epsilon_is_single_source() -> None:
    """ε lives in transfer.py and is re-exported from analysis."""
    assert TRANSFER_ADVANTAGE_EPSILON == TRANSFER_EPS_MODULE
    assert pytest.approx(1e-6) == TRANSFER_ADVANTAGE_EPSILON


def test_omit_pernode_control_raises() -> None:
    train, hold_n1, hold_n2 = _tiny_sequences()
    with pytest.raises(ValueError, match="mandatory 'pernode'"):
        evaluate_topology_transfer(
            _template_model(),
            train,
            hold_n1,
            hold_n2,
            controls=("graph",),
            steps=_STEPS,
            epochs=1,
        )


def test_pernode_only_controls_raises() -> None:
    train, hold_n1, hold_n2 = _tiny_sequences()
    with pytest.raises(ValueError, match="at least one subject kind"):
        evaluate_topology_transfer(
            _template_model(),
            train,
            hold_n1,
            hold_n2,
            controls=("pernode",),
            steps=_STEPS,
            epochs=1,
        )


def test_unknown_transfer_mode_raises() -> None:
    train, hold_n1, hold_n2 = _tiny_sequences()
    with pytest.raises(ValueError, match="transfer_mode must be"):
        evaluate_topology_transfer(
            _template_model(),
            train,
            hold_n1,
            hold_n2,
            transfer_mode="not_a_mode",  # type: ignore[arg-type]
            steps=_STEPS,
            epochs=1,
        )


def test_unsupported_control_kind_raises() -> None:
    train, hold_n1, hold_n2 = _tiny_sequences()
    with pytest.raises(ValueError, match="unsupported control"):
        evaluate_topology_transfer(
            _template_model(),
            train,
            hold_n1,
            hold_n2,
            controls=("graph", "pernode", "hypergraph"),
            steps=_STEPS,
            epochs=1,
        )


def test_short_holdout_raises() -> None:
    train, hold_n1, _ = _tiny_sequences()
    short = SyntheticDynamicGraphBenchmark.generate(
        num_nodes=_N2, num_timesteps=2, seed=3, **_diffusion_params()
    )
    with pytest.raises(ValueError, match="steps \\+ 1"):
        evaluate_topology_transfer(
            _template_model(),
            train,
            hold_n1,
            short,
            steps=_STEPS,
            epochs=1,
        )


def test_evaluate_topology_transfer_zero_shot_smoke() -> None:
    """Tiny seeded N1→N2 smoke: frozen report with Appendix B advantage bool."""
    train, hold_n1, hold_n2 = _tiny_sequences()
    report = evaluate_topology_transfer(
        _template_model(),
        train,
        hold_n1,
        hold_n2,
        steps=_STEPS,
        controls=("graph", "pernode"),
        transfer_mode="zero_shot",
        seed=0,
        epochs=3,
        lr=5e-3,
        device="cpu",
    )

    assert isinstance(report, TopologyTransferReport)
    assert dataclasses.is_dataclass(report)
    assert report.__dataclass_fields__["in_dist_mse"].name  # frozen dataclass
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.steps = 99  # type: ignore[misc]

    assert set(report.in_dist_mse) == {"graph", "pernode"}
    assert set(report.transfer_mse) == {"graph", "pernode"}
    for key in ("graph", "pernode"):
        assert report.in_dist_mse[key] >= 0.0
        assert report.transfer_mse[key] >= 0.0
        assert torch.isfinite(torch.tensor(report.in_dist_mse[key]))
        assert torch.isfinite(torch.tensor(report.transfer_mse[key]))

    expected_advantage = (
        report.transfer_mse["graph"]
        < report.transfer_mse["pernode"] - TRANSFER_ADVANTAGE_EPSILON
    )
    assert report.transfer_advantage is expected_advantage
    assert report.excluded_configs == ()
    assert report.transfer_mode == "zero_shot"
    assert report.seed == 0
    assert report.steps == _STEPS


def test_analysis_exports_transfer_api() -> None:
    import koopman_graph.analysis as analysis

    assert "evaluate_topology_transfer" in analysis.__all__
    assert "TopologyTransferReport" in analysis.__all__
    assert "TRANSFER_ADVANTAGE_EPSILON" in analysis.__all__


def test_self_adaptive_template_lists_excluded_and_continues() -> None:
    """Implicit self_adaptive on the template → list, do not abort."""
    train, hold_n1, hold_n2 = _tiny_sequences()
    report = evaluate_topology_transfer(
        _template_model(learn_topology="self_adaptive"),
        train,
        hold_n1,
        hold_n2,
        steps=_STEPS,
        epochs=2,
        seed=0,
    )
    assert "self_adaptive" in report.excluded_configs
    assert set(report.transfer_mse) == {"graph", "pernode"}


def test_orbit_partition_template_lists_excluded_and_continues() -> None:
    """Implicit orbit ties on the template → list as orbit_partition."""
    train, hold_n1, hold_n2 = _tiny_sequences()
    report = evaluate_topology_transfer(
        _template_model(koopman_orbit_partition=((0, 1), (2, 3))),
        train,
        hold_n1,
        hold_n2,
        steps=_STEPS,
        epochs=2,
        seed=0,
    )
    assert report.excluded_configs == ("orbit_partition",)
    assert set(report.transfer_mse) == {"graph", "pernode"}


def test_auto_orbits_template_lists_as_orbit_partition() -> None:
    """koopman_auto_orbits binds N; report uses the orbit_partition label."""
    train, hold_n1, hold_n2 = _tiny_sequences()
    report = evaluate_topology_transfer(
        _template_model(koopman_auto_orbits=True),
        train,
        hold_n1,
        hold_n2,
        steps=_STEPS,
        epochs=2,
        seed=0,
    )
    assert report.excluded_configs == ("orbit_partition",)


@pytest.mark.parametrize(
    "name",
    ["self_adaptive", "orbit_partition", "isotypic"],
)
def test_request_excluded_raises_for_known_incompatible(name: str) -> None:
    """Explicit request_excluded raises for known incompatible names."""
    train, hold_n1, hold_n2 = _tiny_sequences()
    with pytest.raises(ValueError, match=rf"incompatible configuration '{name}'"):
        evaluate_topology_transfer(
            _template_model(),
            train,
            hold_n1,
            hold_n2,
            steps=_STEPS,
            epochs=1,
            request_excluded=(name,),
        )


def test_isotypic_template_lists_excluded_and_continues() -> None:
    """``koopman_symmetry='isotypic'`` → excluded_configs includes isotypic."""
    train, hold_n1, hold_n2 = _tiny_sequences()
    report = evaluate_topology_transfer(
        _template_model(koopman_symmetry="isotypic"),
        train,
        hold_n1,
        hold_n2,
        steps=_STEPS,
        epochs=2,
        seed=0,
    )
    assert report.excluded_configs == ("isotypic",)
    assert set(report.transfer_mse) == {"graph", "pernode"}


def test_request_excluded_unknown_name_raises() -> None:
    train, hold_n1, hold_n2 = _tiny_sequences()
    with pytest.raises(ValueError, match="unknown name"):
        evaluate_topology_transfer(
            _template_model(),
            train,
            hold_n1,
            hold_n2,
            steps=_STEPS,
            epochs=1,
            request_excluded=("not_a_real_config",),
        )


def _finetune_sequences(*, burn_in: int = 4, steps: int = _STEPS):
    """Holdouts long enough for burn-in + scored rollout."""
    params = _diffusion_params()
    train = SyntheticDynamicGraphBenchmark.generate(
        num_nodes=_N1, num_timesteps=16, seed=0, **params
    )
    hold_n1 = SyntheticDynamicGraphBenchmark.generate(
        num_nodes=_N1, num_timesteps=steps + 2, seed=1, **params
    )
    hold_n2 = SyntheticDynamicGraphBenchmark.generate(
        num_nodes=_N2,
        num_timesteps=burn_in + steps + 2,
        seed=2,
        **params,
    )
    return train, hold_n1, hold_n2


def test_finetune_koopman_smoke() -> None:
    """finetune_koopman records mode and scores both controls on the suffix."""
    burn_in = 4
    train, hold_n1, hold_n2 = _finetune_sequences(burn_in=burn_in)
    report = evaluate_topology_transfer(
        _template_model(),
        train,
        hold_n1,
        hold_n2,
        transfer_mode="finetune_koopman",
        steps=_STEPS,
        seed=0,
        epochs=2,
        burn_in_timesteps=burn_in,
        finetune_epochs=2,
        lr=5e-3,
        device="cpu",
    )
    assert report.transfer_mode == "finetune_koopman"
    assert set(report.in_dist_mse) == {"graph", "pernode"}
    assert set(report.transfer_mse) == {"graph", "pernode"}
    assert isinstance(report.transfer_advantage, bool)


def test_finetune_freezes_encoder_decoder() -> None:
    """Encoder weights are unchanged across the operator-only fine-tune."""
    from koopman_graph.analysis.transfer import (
        _build_control_model,
        _split_burn_in_and_score,
    )
    from koopman_graph.model.online_adaptation import freeze_modules

    burn_in = 4
    train, _, hold_n2 = _finetune_sequences(burn_in=burn_in)
    burn, _scored = _split_burn_in_and_score(
        hold_n2, burn_in_timesteps=burn_in, steps=_STEPS
    )

    torch.manual_seed(0)
    model = _build_control_model(_template_model(), koopman="graph")
    model.fit(train, epochs=2, lr=5e-3, device="cpu")
    freeze_modules((model.encoder, model.decoder))
    before = [p.detach().clone() for p in model.encoder.parameters()]
    koopman_before = [p.detach().clone() for p in model.koopman.parameters()]

    assert all(not p.requires_grad for p in model.encoder.parameters())
    assert all(not p.requires_grad for p in model.decoder.parameters())
    assert any(p.requires_grad for p in model.koopman.parameters())

    model.fit(burn, epochs=3, lr=5e-3, device="cpu")
    for old, new in zip(before, model.encoder.parameters(), strict=True):
        assert torch.equal(old, new.detach())
    # Operator should move under fine-tune (seeded; not a numeric oracle).
    moved = any(
        not torch.equal(old, new.detach())
        for old, new in zip(koopman_before, model.koopman.parameters(), strict=True)
    )
    assert moved


def test_finetune_burn_in_too_short_raises() -> None:
    train, hold_n1, hold_n2 = _finetune_sequences(burn_in=4)
    with pytest.raises(ValueError, match="burn_in_timesteps must be >= 2"):
        evaluate_topology_transfer(
            _template_model(),
            train,
            hold_n1,
            hold_n2,
            transfer_mode="finetune_koopman",
            steps=_STEPS,
            epochs=1,
            burn_in_timesteps=1,
            finetune_epochs=1,
        )


def test_finetune_holdout_too_short_for_disjoint_score_raises() -> None:
    """Leakage guard: need burn_in + steps + 1 snapshots."""
    params = _diffusion_params()
    train = SyntheticDynamicGraphBenchmark.generate(
        num_nodes=_N1, num_timesteps=12, seed=0, **params
    )
    hold_n1 = SyntheticDynamicGraphBenchmark.generate(
        num_nodes=_N1, num_timesteps=_STEPS + 2, seed=1, **params
    )
    # Only enough for burn-in; no scored suffix for rollout.
    hold_n2 = SyntheticDynamicGraphBenchmark.generate(
        num_nodes=_N2, num_timesteps=6, seed=2, **params
    )
    with pytest.raises(ValueError, match="burn_in_timesteps \\+ steps \\+ 1"):
        evaluate_topology_transfer(
            _template_model(),
            train,
            hold_n1,
            hold_n2,
            transfer_mode="finetune_koopman",
            steps=_STEPS,
            epochs=1,
            burn_in_timesteps=4,
            finetune_epochs=1,
        )


def test_example37_control_comparison_oracle() -> None:
    """G1 lock: public API reproduces example-37 control comparison.

    Protocol matches ``tests/analysis/test_cross_topology.py`` / notebook 37
    (path diffusion, N=4→6, 100 epochs, seed 0). Appendix B
    ``TRANSFER_ADVANTAGE_EPSILON`` is the sole ε; MSE anchors use
    ``_EXAMPLE37_MSE_ABS``.

    If this test fails because ``transfer_advantage`` flips to ``True`` or
    MSEs leave the abs band, do **not** widen tolerances in place — bump the
    versioned fixture anchors and record the change in ``CHANGELOG.md``.
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
    template = GraphKoopmanModel(
        GNNEncoder(_FEATURES, _EXAMPLE37_HIDDEN, _LATENT, num_layers=2),
        GNNDecoder(_LATENT, _EXAMPLE37_HIDDEN, _FEATURES, num_layers=2),
        latent_dim=_LATENT,
        time_step=1.0,
        koopman="graph",
    )

    report = evaluate_topology_transfer(
        template,
        train,
        hold_n1,
        hold_n2,
        steps=_EXAMPLE37_STEPS,
        controls=("graph", "pernode"),
        transfer_mode="zero_shot",
        seed=0,
        epochs=_EXAMPLE37_EPOCHS,
        lr=_EXAMPLE37_LR,
        device="cpu",
    )

    for kind, anchors in _EXAMPLE37_ANCHORS.items():
        assert report.in_dist_mse[kind] == pytest.approx(
            anchors["in_dist"], abs=_EXAMPLE37_MSE_ABS
        ), (
            f"{kind} in_dist MSE={report.in_dist_mse[kind]:.4f} "
            f"vs anchor {anchors['in_dist']} (abs={_EXAMPLE37_MSE_ABS})"
        )
        assert report.transfer_mse[kind] == pytest.approx(
            anchors["transfer"], abs=_EXAMPLE37_MSE_ABS
        ), (
            f"{kind} transfer MSE={report.transfer_mse[kind]:.4f} "
            f"vs anchor {anchors['transfer']} (abs={_EXAMPLE37_MSE_ABS})"
        )

    # Appendix B: graph does not beat pernode by ε on this seeded fixture.
    assert report.transfer_advantage is False
    assert not (
        report.transfer_mse["graph"]
        < report.transfer_mse["pernode"] - TRANSFER_ADVANTAGE_EPSILON
    )
    # Documented ordering: per-node remains competitive with graph.
    assert report.transfer_mse["pernode"] <= report.transfer_mse["graph"] * 1.25 + 1e-6
