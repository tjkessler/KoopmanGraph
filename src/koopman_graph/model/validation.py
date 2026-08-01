"""Control and sequence validation helpers for GraphKoopmanModel.

Shallow peer of :mod:`koopman_graph.model.estimator`. Prefer calling these through
the model façade; peer imports are for power-user / package-internal use.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from torch import Tensor, nn
from torch_geometric.data import Data

from koopman_graph.data import (
    HeteroGraphSnapshotSequence,
    SnapshotSequence,
)
from koopman_graph.data.validation import require_no_hyperedges
from koopman_graph.nn import (
    HypergraphDecoder,
    HypergraphEncoder,
    RelGraphDecoder,
    RelGraphEncoder,
    SimplicialDecoder,
    SimplicialEncoder,
)
from koopman_graph.training import (
    EarlyStoppingMonitor,
    TrainingInput,
    ValidationInput,
    resolve_early_stopping_monitor,
    resolve_training_sequences,
    resolve_validation_sequences,
)


def uses_relgraph_modules(encoder: nn.Module, decoder: nn.Module) -> bool:
    """Return whether encoder and decoder are a matched RelGraph pair.

    Parameters
    ----------
    encoder : nn.Module
        Model encoder module.
    decoder : nn.Module
        Model decoder module.

    Returns
    -------
    bool
        ``True`` when both modules are RelGraph peers.

    Raises
    ------
    ValueError
        If exactly one of encoder/decoder is a RelGraph peer, or the peers
        disagree on ``num_relations`` / ``normalization``.
    """
    enc_rel = isinstance(encoder, RelGraphEncoder)
    dec_rel = isinstance(decoder, RelGraphDecoder)
    if enc_rel != dec_rel:
        msg = (
            "RelGraphEncoder and RelGraphDecoder must be used together "
            f"(got encoder={type(encoder).__name__}, "
            f"decoder={type(decoder).__name__})"
        )
        raise ValueError(msg)
    if not (enc_rel and dec_rel):
        return False
    if encoder.num_relations != decoder.num_relations:
        msg = (
            "RelGraphEncoder.num_relations "
            f"({encoder.num_relations}) must match "
            f"RelGraphDecoder.num_relations ({decoder.num_relations})"
        )
        raise ValueError(msg)
    if encoder.normalization != decoder.normalization:
        msg = (
            "RelGraphEncoder.normalization "
            f"({encoder.normalization!r}) must match "
            f"RelGraphDecoder.normalization ({decoder.normalization!r})"
        )
        raise ValueError(msg)
    return True


def uses_hypergraph_modules(encoder: nn.Module, decoder: nn.Module) -> bool:
    """Return whether encoder and decoder are a matched hypergraph pair.

    Parameters
    ----------
    encoder : nn.Module
        Model encoder module.
    decoder : nn.Module
        Model decoder module.

    Returns
    -------
    bool
        ``True`` when both modules are hypergraph peers.

    Raises
    ------
    ValueError
        If exactly one of encoder/decoder is a hypergraph peer.
    """
    enc_hyper = isinstance(encoder, HypergraphEncoder)
    dec_hyper = isinstance(decoder, HypergraphDecoder)
    if enc_hyper != dec_hyper:
        msg = (
            "HypergraphEncoder and HypergraphDecoder must be used together "
            f"(got encoder={type(encoder).__name__}, "
            f"decoder={type(decoder).__name__})"
        )
        raise ValueError(msg)
    return enc_hyper and dec_hyper


def uses_simplicial_modules(encoder: nn.Module, decoder: nn.Module) -> bool:
    """Return whether encoder and decoder are a matched simplicial-1 pair.

    Parameters
    ----------
    encoder : nn.Module
        Model encoder module.
    decoder : nn.Module
        Model decoder module.

    Returns
    -------
    bool
        ``True`` when both modules are simplicial peers.

    Raises
    ------
    ValueError
        If exactly one of encoder/decoder is a simplicial peer.
    """
    enc_sim = isinstance(encoder, SimplicialEncoder)
    dec_sim = isinstance(decoder, SimplicialDecoder)
    if enc_sim != dec_sim:
        msg = (
            "SimplicialEncoder and SimplicialDecoder must be used together "
            f"(got encoder={type(encoder).__name__}, "
            f"decoder={type(decoder).__name__})"
        )
        raise ValueError(msg)
    return enc_sim and dec_sim


def validate_sequence_hyperedges(
    sequence: SnapshotSequence,
    *,
    allow_hyperedges: bool,
) -> None:
    """Enforce hyperedge policy for a training or validation sequence.

    Parameters
    ----------
    sequence : GraphSnapshotSequence or HeteroGraphSnapshotSequence
        Candidate fit/validation trajectory.
    allow_hyperedges : bool
        When ``True``, require static hyperedge incidence; when ``False``,
        reject hyperedge-carrying sequences. Multiplex hetero sequences never
        carry homogeneous hyperedges and are accepted when
        ``allow_hyperedges=False``.

    Raises
    ------
    ValueError
        If the sequence violates the requested hyperedge policy.
    """
    if isinstance(sequence, HeteroGraphSnapshotSequence):
        if allow_hyperedges:
            msg = (
                "HypergraphEncoder / HypergraphDecoder require a homogeneous "
                "hyperedge-carrying GraphSnapshotSequence, not "
                "HeteroGraphSnapshotSequence"
            )
            raise ValueError(msg)
        return
    if allow_hyperedges:
        if not sequence.has_hyperedges:
            msg = (
                "HypergraphEncoder / HypergraphDecoder require a "
                "hyperedge-carrying GraphSnapshotSequence "
                "(set hyperedge_index on each snapshot)"
            )
            raise ValueError(msg)
        return
    require_no_hyperedges(sequence)


@dataclass(frozen=True, slots=True)
class PreparedFitInputs:
    """Validated training / validation sequences and early-stopping monitor.

    Attributes
    ----------
    train_sequences : list
        Training trajectory sequences.
    val_sequences : list | None
        Optional validation sequences.
    early_stopping_monitor : Any | None
        Optional early-stopping callback state.
    """

    train_sequences: list[SnapshotSequence]
    val_sequences: list[SnapshotSequence] | None
    early_stopping_monitor: Literal["train", "val"]


def as_data(
    x_or_data: Tensor | Data,
    edge_index: Tensor,
    edge_weight: Tensor | None,
) -> Data:
    """Build a PyG ``Data`` object from tensor or ``Data`` inputs.

    Parameters
    ----------
    x_or_data : Tensor or Data
        Node features or an existing snapshot.
    edge_index : Tensor
        Edge index with shape ``(2, num_edges)``.
    edge_weight : Tensor or None
        Optional edge weights with shape ``(num_edges,)``.

    Returns
    -------
    Data
        Snapshot suitable for physics lifting callables.
    """
    if isinstance(x_or_data, Data):
        return x_or_data
    data = Data(x=x_or_data, edge_index=edge_index)
    if edge_weight is not None:
        data.edge_weight = edge_weight
    return data


def validate_controls(
    *,
    control_dim: int,
    controls: Sequence[Tensor] | None,
    steps: int,
) -> None:
    """Validate rollout controls against model control settings.

    Parameters
    ----------
    control_dim : int
        Model control dimension.
    controls : sequence of Tensor or None
        Control inputs for each rollout step.
    steps : int
        Number of rollout steps.

    Raises
    ------
    ValueError
        If controls are missing, surplus, or provided to an uncontrolled model.
    """
    if control_dim == 0:
        if controls is not None:
            msg = "controls provided to an uncontrolled model"
            raise ValueError(msg)
        return
    if controls is None:
        msg = "controls are required when control_dim > 0"
        raise ValueError(msg)
    if len(controls) != steps:
        msg = f"expected {steps} control inputs for rollout, got {len(controls)}"
        raise ValueError(msg)


def validate_sequence_controls(
    *,
    control_dim: int,
    sequence: SnapshotSequence,
) -> None:
    """Validate sequence controls against a model control dimension.

    Parameters
    ----------
    control_dim : int
        Model control dimension.
    sequence : GraphSnapshotSequence or HeteroGraphSnapshotSequence
        Training or validation sequence.

    Raises
    ------
    ValueError
        If controls are missing or dimensions disagree.
    """
    if control_dim == 0:
        if sequence.has_controls:
            msg = "sequence contains control inputs but model control_dim is 0"
            raise ValueError(msg)
        return
    if not sequence.has_controls:
        msg = "controlled model requires sequences with control inputs"
        raise ValueError(msg)
    if sequence.control_dim != control_dim:
        msg = (
            f"sequence control_dim ({sequence.control_dim}) must match "
            f"model control_dim ({control_dim})"
        )
        raise ValueError(msg)


def prepare_fit_inputs(
    *,
    control_dim: int,
    data_sequence: TrainingInput,
    validation_sequence: ValidationInput = None,
    epochs: int = 100,
    early_stopping_patience: int | None = None,
    early_stopping_monitor: EarlyStoppingMonitor = "auto",
    allow_hyperedges: bool = False,
) -> PreparedFitInputs:
    """Validate ``fit`` inputs and resolve training / validation sequences.

    Parameters
    ----------
    control_dim : int
        Model control dimension used for sequence control checks.
    data_sequence : TrainingInput
        Training trajectory input accepted by :meth:`GraphKoopmanModel.fit`.
    validation_sequence : ValidationInput, optional
        Optional held-out validation input.
    epochs : int, optional
        Requested training epoch count. Must be ``>= 1``.
    early_stopping_patience : int or None, optional
        When set, must be ``>= 1``.
    early_stopping_monitor : {"auto", "train", "val"}, optional
        Early-stopping monitor mode. Default is ``"auto"``.
    allow_hyperedges : bool, optional
        When ``True``, require hyperedge-carrying sequences (hypergraph
        encoder/decoder). When ``False``, reject hyperedge incidence.
        Default is ``False``.

    Returns
    -------
    PreparedFitInputs
        Resolved sequences and concrete ``"train"`` / ``"val"`` monitor.

    Raises
    ------
    ValueError
        If epoch / patience bounds fail, controls disagree, hyperedge policy
        fails, or fewer than two snapshots are provided for training or
        validation.
    """
    if epochs < 1:
        msg = f"epochs must be >= 1, got {epochs}"
        raise ValueError(msg)
    if early_stopping_patience is not None and early_stopping_patience < 1:
        msg = (
            f"early_stopping_patience must be >= 1 when set, "
            f"got {early_stopping_patience}"
        )
        raise ValueError(msg)

    train_sequences = resolve_training_sequences(data_sequence)
    for sequence in train_sequences:
        validate_sequence_controls(
            control_dim=control_dim,
            sequence=sequence,
        )
        validate_sequence_hyperedges(
            sequence,
            allow_hyperedges=allow_hyperedges,
        )
        if sequence.num_timesteps < 2:
            msg = "data_sequence must contain at least 2 snapshots for training"
            raise ValueError(msg)

    val_sequences = resolve_validation_sequences(
        validation_sequence,
        num_training_sequences=len(train_sequences),
    )
    if val_sequences is not None:
        for sequence in val_sequences:
            validate_sequence_controls(
                control_dim=control_dim,
                sequence=sequence,
            )
            validate_sequence_hyperedges(
                sequence,
                allow_hyperedges=allow_hyperedges,
            )
            if sequence.num_timesteps < 2:
                msg = (
                    "validation_sequence must contain at least 2 snapshots "
                    "for validation"
                )
                raise ValueError(msg)

    monitor = resolve_early_stopping_monitor(
        early_stopping_monitor,
        has_validation=val_sequences is not None,
    )
    return PreparedFitInputs(
        train_sequences=train_sequences,
        val_sequences=val_sequences,
        early_stopping_monitor=monitor,
    )
