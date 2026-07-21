"""Hierarchical GraphKoopman wrapper: pool → coarse Koopman → unpool.

Composes :class:`~koopman_graph.model.GraphKoopmanModel` without subclassing
or forking the shared latent rollout loop. Forecasting at the coarsest level
is faster on large graphs but can lose fine-scale accuracy — this is
**coarse-level forecasting with learned unpooling**, not P-K-GCN-style
physics-augmented spatiotemporal super-resolution.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.hierarchical.pooling import (
    PoolingKind,
    PoolStep,
    ScatterUnpool,
    apply_pool_layer,
    build_pool_layer,
    pool_control,
    pool_control_sequence,
    resolve_snapshot_inputs,
    snapshot_from_features,
)
from koopman_graph.model import GraphKoopmanModel
from koopman_graph.training import FitHistory

ResolutionArg = Literal["fine", "coarse"] | int

_MANIFEST_NAME = "hierarchical_manifest.json"
_MODEL_NAME = "model.pt"
_WRAPPER_NAME = "wrapper.pt"


def _encoder_in_channels(model: GraphKoopmanModel) -> int:
    """Infer scoring / unpool feature width from the composed encoder.

    Returns
    -------
    int
        Positive encoder input width.
    """
    encoder = model.encoder
    in_channels = getattr(encoder, "in_channels", None)
    if isinstance(in_channels, int) and in_channels > 0:
        return in_channels
    msg = (
        "could not infer encoder.in_channels for hierarchical pooling; "
        "pass in_channels= explicitly"
    )
    raise ValueError(msg)


def _encoder_out_channels(model: GraphKoopmanModel) -> int:
    """Return the decoded feature width used for unpooling.

    Returns
    -------
    int
        Positive decoder output width.
    """
    decoder = model.decoder
    out_channels = getattr(decoder, "out_channels", None)
    if isinstance(out_channels, int) and out_channels > 0:
        return out_channels
    msg = "could not infer decoder.out_channels for hierarchical unpooling"
    raise ValueError(msg)


class HierarchicalGraphKoopmanModel(nn.Module):
    """Pool a fine graph, advance with a composed GraphKoopman model, unpool.

    Power-user type under :mod:`koopman_graph.hierarchical` (not on the root
    façade). Spectrum and graph-operator regularization use the **pooled**
    topology. Global controls pass through; per-node controls are indexed by
    the pooling ``perm`` chain so additive and bilinear semantics on the
    coarse latent match the composed operator.

    Parameters
    ----------
    model : GraphKoopmanModel
        Composed fine→latent→decode model used on the coarsest graph.
    pool_ratios : sequence of float, optional
        Per-level retention ratios in ``(0, 1]``. Default ``(0.5,)``.
        ``(1.0,)`` keeps all nodes (no-op size; may reorder).
    pooling : {"topk", "sag"}, optional
        PyG pooling operator. Default ``"topk"``.
    in_channels : int or None, optional
        Scoring feature width. Defaults to ``model.encoder.in_channels``.
    refine_unpool : bool, optional
        Identity-initialized linear refine after scatter-unpool. Default
        ``True``.
    """

    def __init__(
        self,
        model: GraphKoopmanModel,
        *,
        pool_ratios: Sequence[float] = (0.5,),
        pooling: PoolingKind = "topk",
        in_channels: int | None = None,
        refine_unpool: bool = True,
    ) -> None:
        """Store the composed model and build pool / unpool modules.

        Notes
        -----
        Constructor parameters are documented on the class.
        """
        super().__init__()
        if len(pool_ratios) == 0:
            msg = "pool_ratios must contain at least one ratio"
            raise ValueError(msg)
        self.model = model
        self.pool_ratios = tuple(float(r) for r in pool_ratios)
        self.pooling: PoolingKind = pooling
        channels = _encoder_in_channels(model) if in_channels is None else in_channels
        self.in_channels = channels
        out_channels = _encoder_out_channels(model)
        self.out_channels = out_channels

        self.pool_layers = nn.ModuleList(
            [
                build_pool_layer(channels, ratio, pooling=pooling)
                for ratio in self.pool_ratios
            ]
        )
        # One unpool per level (coarse→…→fine), operating on decoded features.
        self.unpool_layers = nn.ModuleList(
            [
                ScatterUnpool(out_channels, refine=refine_unpool)
                for _ in self.pool_ratios
            ]
        )

    @property
    def n_levels(self) -> int:
        """Return the number of pooling levels.

        Returns
        -------
        int
            Number of configured pooling ratios.
        """
        return len(self.pool_ratios)

    @property
    def control_dim(self) -> int:
        """Return the exogenous control dimension of the composed model.

        Returns
        -------
        int
            Control feature width.
        """
        return self.model.control_dim

    def pool_down(
        self,
        graph: Tensor | Data,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
    ) -> tuple[Data, list[PoolStep]]:
        """Apply all pooling levels fine → coarse.

        Parameters
        ----------
        graph : Tensor or Data
            Fine snapshot or node features.
        edge_index, edge_weight
            Required when ``graph`` is a tensor.

        Returns
        -------
        tuple
            Coarse ``Data`` and per-level :class:`PoolStep` metadata (fine→coarse
            order).
        """
        x, edge_index, edge_weight = resolve_snapshot_inputs(
            graph, edge_index, edge_weight
        )
        steps: list[PoolStep] = []
        for layer in self.pool_layers:
            num_fine = x.size(0)
            x, edge_index, edge_weight, perm = apply_pool_layer(
                layer, x, edge_index, edge_weight
            )
            steps.append(
                PoolStep(
                    perm=perm,
                    num_fine=num_fine,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                )
            )
        return snapshot_from_features(x, edge_index, edge_weight), steps

    def unpool_up(
        self,
        coarse_x: Tensor,
        steps: Sequence[PoolStep],
        *,
        levels: int | None = None,
    ) -> Tensor:
        """Unpool coarse features toward fine resolution.

        Parameters
        ----------
        coarse_x : Tensor
            Features at the coarsest level (or an intermediate start).
        steps : sequence of PoolStep
            Pool metadata in fine→coarse order (same as :meth:`pool_down`).
        levels : int or None, optional
            Number of unpool steps from the coarse end. ``None`` fully restores
            the finest node count.

        Returns
        -------
        Tensor
            Features after the requested unpool steps.
        """
        if len(steps) != self.n_levels:
            msg = f"expected {self.n_levels} pool steps, got {len(steps)}"
            raise ValueError(msg)
        n_unpool = self.n_levels if levels is None else levels
        if n_unpool < 0 or n_unpool > self.n_levels:
            msg = f"levels must be in [0, {self.n_levels}], got {n_unpool}"
            raise ValueError(msg)
        x = coarse_x
        # Unpool reverse: last pool step first.
        for offset in range(n_unpool):
            step = steps[-(offset + 1)]
            unpool = self.unpool_layers[-(offset + 1)]
            x = unpool(x, step.perm, step.num_fine)
        return x

    def _perms(self, steps: Sequence[PoolStep]) -> list[Tensor]:
        """Extract fine-to-coarse node permutations.

        Returns
        -------
        list of Tensor
            One permutation per pooling level.
        """
        return [step.perm for step in steps]

    def _pool_controls(
        self,
        controls: Sequence[Tensor] | None,
        steps: Sequence[PoolStep],
    ) -> list[Tensor] | None:
        """Pool global or per-node controls through the permutation chain.

        Returns
        -------
        list of Tensor or None
            Coarse controls, or ``None`` when controls are absent.
        """
        if controls is None:
            return None
        perms = self._perms(steps)
        return [pool_control(control, perms) for control in controls]

    def _resolve_resolution(self, resolution: ResolutionArg) -> int:
        """Return how many unpool steps to apply from the coarse end.

        Returns
        -------
        int
            Number of unpooling levels.
        """
        if resolution == "coarse":
            return 0
        if resolution == "fine":
            return self.n_levels
        if isinstance(resolution, int):
            if resolution < 0 or resolution > self.n_levels:
                msg = (
                    f"resolution int must be in [0, {self.n_levels}] "
                    f"(0=coarse, {self.n_levels}=fine), got {resolution}"
                )
                raise ValueError(msg)
            return resolution
        msg = f"resolution must be 'fine', 'coarse', or int, got {resolution!r}"
        raise ValueError(msg)

    def spectrum(
        self,
        reference_graph: Tensor | Data,
        *,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
        delta_t: float | None = None,
    ) -> Any:
        """Spectrum of the composed operator on the **pooled** topology.

        For ``koopman="graph"``, requires pooling a reference graph so the
        effective matrix uses the coarse ``edge_index`` / node count — never
        report ``K_self`` alone as the networked spectrum.

        Returns
        -------
        Any
            Spectrum result returned by the composed model.
        """
        coarse, _steps = self.pool_down(
            reference_graph, edge_index=edge_index, edge_weight=edge_weight
        )
        return self.model.spectrum(
            delta_t=delta_t,
            edge_index=coarse.edge_index,
            num_nodes=int(coarse.x.size(0)),
            edge_weight=getattr(coarse, "edge_weight", None),
        )

    def predict(
        self,
        initial_graph: Tensor | Data,
        steps: int,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
        controls: Sequence[Tensor] | None = None,
        future_topologies: Sequence[Data] | None = None,
        history: Sequence[Data] | None = None,
        *,
        resolution: ResolutionArg = "fine",
    ) -> list[Data]:
        """Pool once, forecast on the coarse graph, optionally unpool.

        Parameters
        ----------
        initial_graph : Tensor or Data
            Fine initial snapshot.
        steps : int
            Forecast horizon.
        edge_index, edge_weight
            Topology when ``initial_graph`` is a tensor.
        controls : sequence of Tensor or None, optional
            Fine-level controls (global or per-node). Per-node rows are pooled
            with the initial pooling ``perm`` chain.
        future_topologies : sequence of Data or None, optional
            Fine future topologies; each is pooled with the **same** pool
            layers (scores recomputed) before being forwarded to the composed
            model.
        history : sequence of Data or None, optional
            Delay history; each snapshot is pooled independently.
        resolution : {"fine", "coarse"} or int, optional
            ``"fine"`` / ``n_levels`` fully unpools; ``"coarse"`` / ``0`` returns
            coarse forecasts; intermediate ints unpool that many levels.

        Returns
        -------
        list of Data
            Forecasts at the requested resolution. Fine outputs carry the
            initial fine topology (hold-last at the fine level).
        """
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                fine_x, fine_edge, fine_weight = resolve_snapshot_inputs(
                    initial_graph, edge_index, edge_weight
                )
                fine_template = snapshot_from_features(fine_x, fine_edge, fine_weight)
                coarse, pool_steps = self.pool_down(fine_template)
                coarse_controls = self._pool_controls(controls, pool_steps)

                coarse_future: list[Data] | None = None
                if future_topologies is not None:
                    coarse_future = []
                    for topo in future_topologies:
                        pooled_topo, _ = self.pool_down(topo)
                        coarse_future.append(
                            snapshot_from_features(
                                pooled_topo.x.new_zeros(pooled_topo.x.shape),
                                pooled_topo.edge_index,
                                getattr(pooled_topo, "edge_weight", None),
                            )
                        )

                coarse_history: list[Data] | None = None
                if history is not None:
                    coarse_history = [self.pool_down(snap)[0] for snap in history]

                coarse_preds = self.model.predict(
                    coarse,
                    steps,
                    controls=coarse_controls,
                    future_topologies=coarse_future,
                    history=coarse_history,
                )

                n_unpool = self._resolve_resolution(resolution)
                if n_unpool == 0:
                    return coarse_preds

                output: list[Data] = []
                for pred in coarse_preds:
                    fine_feat = self.unpool_up(pred.x, pool_steps, levels=n_unpool)
                    if n_unpool == self.n_levels:
                        output.append(
                            snapshot_from_features(fine_feat, fine_edge, fine_weight)
                        )
                    else:
                        # Intermediate resolution: use topology after
                        # (n_levels - n_unpool) pool steps.
                        stop = self.n_levels - n_unpool
                        mid = pool_steps[stop - 1]
                        output.append(
                            snapshot_from_features(
                                fine_feat, mid.edge_index, mid.edge_weight
                            )
                        )
                return output
        finally:
            self.train(was_training)

    def _pool_sequence(
        self,
        sequence: GraphSnapshotSequence,
    ) -> tuple[GraphSnapshotSequence, list[list[PoolStep]]]:
        """Pool every snapshot and retain per-step metadata.

        Returns
        -------
        tuple
            Coarse sequence and pooling metadata for every snapshot.
        """
        coarse_snaps: list[Data] = []
        all_steps: list[list[PoolStep]] = []
        for snap in sequence:
            coarse, steps = self.pool_down(snap)
            coarse_snaps.append(coarse)
            all_steps.append(steps)

        control_inputs = None
        if sequence.has_controls:
            assert sequence.control_inputs is not None
            perms = [self._perms(steps) for steps in all_steps]
            control_inputs = pool_control_sequence(sequence.control_inputs, perms)

        kwargs: dict[str, Any] = {
            # Feature-dependent TopK/SAG perms can change coarse edges over time.
            "allow_dynamic_topology": True,
        }
        if sequence.timestamps is not None:
            kwargs["timestamps"] = sequence.timestamps
        # Observation masks are fine-node specific; drop on coarse (documented).
        return (
            GraphSnapshotSequence(
                coarse_snaps,
                control_inputs=control_inputs,
                **kwargs,
            ),
            all_steps,
        )

    def _fit_unpool(
        self,
        sequence: GraphSnapshotSequence,
        all_steps: Sequence[Sequence[PoolStep]],
        *,
        epochs: int,
        lr: float,
    ) -> None:
        """Train scatter-unpool refine layers to reconstruct fine features.

        Notes
        -----
        A non-positive epoch count leaves the refine layers unchanged.
        """
        if epochs <= 0 or len(self.unpool_layers) == 0:
            return
        params = [p for layer in self.unpool_layers for p in layer.parameters()]
        if not params:
            return
        opt = torch.optim.Adam(params, lr=lr)
        for _ in range(epochs):
            opt.zero_grad(set_to_none=True)
            loss = sequence[0].x.new_zeros(())
            for snap, steps in zip(sequence, all_steps, strict=True):
                # Teacher: pool features without refine, then unpool back.
                x = snap.x
                assert x is not None
                # Reconstruct from the last coarse features obtained by
                # indexing fine features with the perm chain (no score net).
                coarse_x = x
                for step in steps:
                    coarse_x = coarse_x[step.perm]
                recon = self.unpool_up(coarse_x, steps)
                loss = loss + torch.mean((recon - x) ** 2)
            loss = loss / len(sequence)
            loss.backward()
            opt.step()

    def fit(
        self,
        sequence: GraphSnapshotSequence,
        *,
        epochs: int = 100,
        lr: float = 1e-3,
        unpool_epochs: int | None = None,
        unpool_lr: float | None = None,
        **kwargs: Any,
    ) -> FitHistory:
        """Pool the sequence, fit the composed model, then train unpool.

        Pooling scores are held fixed (eval) during the composed ``fit`` so the
        coarse topology stays a consistent reduction for the inner training
        loop. Unpool refine layers are trained afterward to map coarse features
        back toward fine node features.

        Parameters
        ----------
        sequence : GraphSnapshotSequence
            Fine-resolution training sequence.
        epochs : int, optional
            Epochs for the composed :meth:`GraphKoopmanModel.fit`.
        lr : float, optional
            Learning rate for the composed model.
        unpool_epochs : int or None, optional
            Epochs for unpool refine training. Defaults to ``max(5, epochs // 5)``.
        unpool_lr : float or None, optional
            Unpool learning rate. Defaults to ``lr``.
        **kwargs
            Forwarded to :meth:`GraphKoopmanModel.fit`.

        Returns
        -------
        FitHistory
            History from the composed model ``fit``.
        """
        was_training = self.training
        self.pool_layers.eval()
        try:
            with torch.no_grad():
                coarse_sequence, all_steps = self._pool_sequence(sequence)
            history = self.model.fit(coarse_sequence, epochs=epochs, lr=lr, **kwargs)
        finally:
            self.train(was_training)

        u_epochs = max(5, epochs // 5) if unpool_epochs is None else unpool_epochs
        u_lr = lr if unpool_lr is None else unpool_lr
        self._fit_unpool(sequence, all_steps, epochs=u_epochs, lr=u_lr)
        return history

    def save(self, directory: str | Path) -> None:
        """Persist composed model (format-1) plus wrapper weights.

        Parameters
        ----------
        directory : str or Path
            Destination directory (created when missing).
        """
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        self.model.save(root / _MODEL_NAME)
        torch.save(
            {
                "pool_ratios": list(self.pool_ratios),
                "pooling": self.pooling,
                "in_channels": self.in_channels,
                "out_channels": self.out_channels,
                "pool_state_dict": self.pool_layers.state_dict(),
                "unpool_state_dict": self.unpool_layers.state_dict(),
            },
            root / _WRAPPER_NAME,
        )
        manifest = {
            "kind": "HierarchicalGraphKoopmanModel",
            "model_file": _MODEL_NAME,
            "wrapper_file": _WRAPPER_NAME,
            "member_format": "GraphKoopmanModel.save",
            "pool_ratios": list(self.pool_ratios),
            "pooling": self.pooling,
        }
        (root / _MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        directory: str | Path,
        *,
        map_location: str | torch.device | None = None,
    ) -> HierarchicalGraphKoopmanModel:
        """Load a hierarchical wrapper saved by :meth:`save`.

        Parameters
        ----------
        directory : str or Path
            Directory with manifest, model checkpoint, and wrapper weights.
        map_location : str, device, or None, optional
            Forwarded to model / wrapper loaders.

        Returns
        -------
        HierarchicalGraphKoopmanModel
            Reconstructed wrapper.
        """
        root = Path(directory)
        manifest_path = root / _MANIFEST_NAME
        if not manifest_path.is_file():
            msg = f"hierarchical manifest not found: {manifest_path}"
            raise FileNotFoundError(msg)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        model = GraphKoopmanModel.load(
            root / manifest.get("model_file", _MODEL_NAME),
            map_location=map_location,
        )
        wrapper_path = root / manifest.get("wrapper_file", _WRAPPER_NAME)
        payload = torch.load(
            wrapper_path,
            map_location=map_location,
            weights_only=False,
        )
        inst = cls(
            model,
            pool_ratios=payload["pool_ratios"],
            pooling=payload["pooling"],
            in_channels=payload["in_channels"],
        )
        inst.pool_layers.load_state_dict(payload["pool_state_dict"])
        inst.unpool_layers.load_state_dict(payload["unpool_state_dict"])
        return inst
