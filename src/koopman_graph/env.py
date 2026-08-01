"""Gymnasium environment wrapper for latent-space graph control.

Optional dependency convention
------------------------------
Gymnasium is an optional ``[rl]`` extra. This module soft-imports it at load
time so ``import koopman_graph.env`` and
``from koopman_graph.env import GraphKoopmanEnv``
succeed without Gymnasium installed. Construction fails at call time via
:func:`_require_gymnasium` with install guidance. Soft import is required here
because :class:`GraphKoopmanEnv` subclasses ``gymnasium.Env`` when available.

For optional dependencies that are not base classes (e.g. ``h5py`` in METR-LA
loaders), prefer a call-site ``import`` that raises ``ImportError`` with the
same install-guidance style. Never fail the core package import for an optional
extra.

Heterogeneous observation layout
--------------------------------
For ``koopman="hetero_graph"`` models the observation is the row-major
flatten of the latent state. Shared-d path: stacked block ``Z`` with shape
``(N, d)`` where ``N = Σ_τ N_τ``;
:meth:`GraphKoopmanEnv.reshape_observation` restores ``(N, d)``. Opt-in
unequal ``d_τ`` (``latent_dims``): flat vector of length ``Σ_τ N_τ·d_τ``
(same C-order type blocks as :mod:`koopman_graph.data.hetero_layout`);
reshape restores that 1-D layout.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data, HeteroData

from koopman_graph.data import GraphSnapshotSequence, HeteroGraphSnapshotSequence
from koopman_graph.data.hetero_layout import stack_typed_features, stacked_latent_numel
from koopman_graph.data.validation import require_no_hyperedges
from koopman_graph.graph_utils import (
    advance_and_decode,
    propagate_latent,
    snapshot_edge_weight,
    snapshot_to_device,
)
from koopman_graph.nn.heterogeneous import (
    resolve_multiplex_relation_inputs,
    resolve_typed_relation_inputs,
)
from koopman_graph.protocols import ModeShapeModel

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - exercised in test_env_requires_gymnasium
    gym = None  # type: ignore[assignment,misc]
    spaces = None  # type: ignore[assignment,misc]

SnapshotLike = Data | HeteroData
RewardFn = Callable[[SnapshotLike, int], float]
ReferenceSequence = GraphSnapshotSequence | HeteroGraphSnapshotSequence

_GYMNASIUM_IMPORT_ERROR = (
    "Gymnasium is required for GraphKoopmanEnv. "
    "Install with: pip install koopman-graph[rl]"
)


def _require_gymnasium() -> tuple[Any, Any]:
    """Return Gymnasium modules or raise a helpful import error.

    Returns
    -------
    tuple
        ``(gymnasium, spaces)`` modules.

    Raises
    ------
    ImportError
        If Gymnasium is not installed (``pip install koopman-graph[rl]``).
    """
    if gym is None or spaces is None:
        raise ImportError(_GYMNASIUM_IMPORT_ERROR)
    return gym, spaces


def flatten_latent(z: Tensor) -> np.ndarray:
    """Flatten node latent states for Gymnasium observations.

    Parameters
    ----------
    z : Tensor
        Shared-d latents ``(num_nodes, latent_dim)``, or rectangular flat
        latents ``(Σ N_τ·d_τ,)``.

    Returns
    -------
    ndarray
        Observation vector (``float32``) with length ``z.numel()``.
    """
    return z.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)


def unflatten_latent(
    observation: np.ndarray,
    *,
    num_nodes: int,
    latent_dim: int,
    obs_numel: int | None = None,
) -> Tensor:
    """Reshape a flattened observation back to latent features.

    Parameters
    ----------
    observation : ndarray
        Flattened latent vector.
    num_nodes : int
        Number of graph nodes (stacked ``Σ_τ N_τ`` when hetero).
    latent_dim : int
        Shared latent width ``d`` (ignored when ``obs_numel`` is set for
        rectangular layouts).
    obs_numel : int or None, optional
        When set (rectangular ``Σ N_τ·d_τ``), validate against this length
        and return a 1-D tensor of that size instead of ``(N, d)``.

    Returns
    -------
    Tensor
        Latent tensor with shape ``(num_nodes, latent_dim)`` or ``(obs_numel,)``.
    """
    if obs_numel is not None:
        if observation.size != obs_numel:
            msg = (
                f"expected flattened observation size {obs_numel}, "
                f"got {observation.size}"
            )
            raise ValueError(msg)
        return torch.from_numpy(np.asarray(observation, dtype=np.float32).reshape(-1))
    expected = num_nodes * latent_dim
    if observation.size != expected:
        msg = f"expected flattened observation size {expected}, got {observation.size}"
        raise ValueError(msg)
    return torch.from_numpy(observation.reshape(num_nodes, latent_dim))


def _reference_num_nodes(sequence: ReferenceSequence) -> int:
    """Return stacked node count for observation sizing.

    Parameters
    ----------
    sequence : GraphSnapshotSequence or HeteroGraphSnapshotSequence
        Reference trajectory.

    Returns
    -------
    int
        ``N`` (homo / multiplex) or ``Σ_τ N_τ`` (typed hetero).
    """
    if isinstance(sequence, HeteroGraphSnapshotSequence):
        return int(sequence.num_nodes_total)
    return int(sequence.num_nodes)


class GraphKoopmanEnv(gym.Env if gym is not None else object):  # type: ignore[misc]
    """Gymnasium environment for closed-loop control in Koopman latent space.

    Requires the optional ``[rl]`` extra (``pip install koopman-graph[rl]``).
    Importing this class succeeds without Gymnasium; constructing an instance
    raises ``ImportError`` until the extra is installed.

    The environment exposes flattened latent node states as observations and
    global control vectors as actions. Each ``step`` applies the learned Koopman
    operator in latent space, decodes to physical node features, and evaluates
    a user-supplied reward on the decoded graph snapshot.

    The encoder and decoder are frozen during interaction; only the action
    (control input) changes the latent transition. This mirrors latent-space MPC
    and actor-critic pipelines described in Recursive Koopman Learning
    (Zhang et al., CoRL 2025): a fixed lifting map provides a low-dimensional
    linear control interface while rewards are defined on interpretable physical
    states.

    Heterogeneous models (``koopman="hetero_graph"``) use
    :class:`~koopman_graph.data.HeteroGraphSnapshotSequence` references.
    Shared-d observations flatten ``(Σ_τ N_τ) · d`` in the operator
    ``node_types`` order; :meth:`reshape_observation` restores ``(N, d)``.
    With opt-in ``latent_dims`` (unequal ``d_τ``), the observation length is
    ``Σ_τ N_τ·d_τ`` and reshape restores the flat 1-D layout (see module
    ``Heterogeneous observation layout``). This documents the Gymnasium
    vector layout only; it does not imply type-balanced rewards or certified
    control performance. Rewards receive decoded ``HeteroData`` snapshots;
    there is no silent cast to homogeneous ``Data``. Typed operators
    currently reject ``control_dim > 0``, so controlled typed envs are
    unavailable until that operator support lands; multiplex controlled
    hetero envs work today.

    **Limitations.** Rewards see decoded states that depend on a frozen
    encoder/decoder trained offline. Global controls with shape
    ``(control_dim,)`` are supported; per-node action spaces are not. Topology
    is held fixed from the reset snapshot for the episode; sequences with
    :attr:`~koopman_graph.data.GraphSnapshotSequence.is_dynamic_topology`
    ``True`` are rejected at construction. Discrete models advance by one
    ``K``-step per ``step`` call; only continuous models honor a custom
    ``delta_t`` integration interval.

    Parameters
    ----------
    model : GraphKoopmanModel
        Trained controlled model with ``control_dim > 0``.
    reference_sequence : GraphSnapshotSequence or HeteroGraphSnapshotSequence
        Sequence supplying reset snapshots and fixed episode topology.
        Must have ``is_dynamic_topology=False``. Container type must match
        the model (homo vs hetero).
    reward_fn : callable
        ``reward_fn(decoded_snapshot, step_index) -> float`` where
        ``decoded_snapshot`` is a PyG ``Data`` (homogeneous) or
        ``HeteroData`` (hetero) object with physical node features and
        ``step_index`` counts environment steps since ``reset``.
    control_low : float or sequence of float, optional
        Lower bounds for the action space. Default is ``-1.0``.
    control_high : float or sequence of float, optional
        Upper bounds for the action space. Default is ``1.0``.
    max_episode_steps : int, optional
        Maximum steps per episode. Default is ``50``.
    start_index : int or None, optional
        Fixed reference-sequence index for every ``reset``. When set,
        ``random_start`` is ignored.
    random_start : bool, optional
        Sample a random reference snapshot on each ``reset``. Default is
        ``True``.
    delta_t : float or None, optional
        Integration interval for continuous latent advance on each ``step``.
        When ``None`` (default), uses ``model.time_step`` via
        :meth:`~koopman_graph.model.GraphKoopmanModel.resolve_delta_t`. For continuous
        models this enables closed-loop control at a horizon other than the
        training ``time_step``. For discrete models, ``delta_t`` must be
        ``None`` or equal to ``model.time_step`` (strict check); irregular
        stepping is not supported in discrete mode.
    device : torch.device or str or None, optional
        Device for model inference. Defaults to the model's current parameter
        device.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        model: ModeShapeModel,
        reference_sequence: ReferenceSequence,
        reward_fn: RewardFn,
        *,
        control_low: float | Sequence[float] = -1.0,
        control_high: float | Sequence[float] = 1.0,
        max_episode_steps: int = 50,
        start_index: int | None = None,
        random_start: bool = True,
        delta_t: float | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        """Initialize the latent-space Gymnasium environment.

        Parameters
        ----------
        model
            Value for ``model``.
        reference_sequence
            Value for ``reference_sequence``.
        reward_fn
            Value for ``reward_fn``.
        control_low
            Value for ``control_low``.
        control_high
            Value for ``control_high``.
        max_episode_steps
            Value for ``max_episode_steps``.
        start_index
            Value for ``start_index``.
        random_start
            Value for ``random_start``.
        delta_t
            Value for ``delta_t``.
        device
            Value for ``device``.
        """
        _require_gymnasium()
        # Duck-type against the ModeShapeModel / trainable façade surface so
        # ``env`` does not import ``koopman_graph.model`` (import cycle).
        required = (
            "encode",
            "decoder",
            "koopman",
            "control_dim",
            "latent_dim",
            "time_step",
            "resolve_delta_t",
        )
        if not all(hasattr(model, name) for name in required):
            msg = "model must provide GraphKoopmanModel encode/decode/koopman surface"
            raise TypeError(msg)
        uses_hetero = bool(getattr(model, "uses_hetero_koopman", False))
        rectangular_zero_control = (
            uses_hetero
            and bool(getattr(model.koopman, "is_rectangular", False))
            and int(model.control_dim) == 0
        )
        if model.control_dim < 0:
            msg = f"model.control_dim must be non-negative, got {model.control_dim}"
            raise ValueError(msg)
        if model.control_dim == 0 and not rectangular_zero_control:
            msg = (
                "GraphKoopmanEnv requires model.control_dim > 0 "
                "(control_dim=0 is allowed only for rectangular hetero "
                "observation smokes)"
            )
            raise ValueError(msg)

        seq_is_hetero = isinstance(reference_sequence, HeteroGraphSnapshotSequence)
        if uses_hetero and not seq_is_hetero:
            msg = (
                "hetero GraphKoopmanEnv requires HeteroGraphSnapshotSequence "
                "reference_sequence"
            )
            raise TypeError(msg)
        if not uses_hetero and seq_is_hetero:
            msg = (
                "homogeneous GraphKoopmanEnv cannot use "
                "HeteroGraphSnapshotSequence reference_sequence"
            )
            raise TypeError(msg)

        if reference_sequence.num_timesteps < 1:
            msg = "reference_sequence must contain at least one snapshot"
            raise ValueError(msg)
        if reference_sequence.is_dynamic_topology:
            msg = (
                "GraphKoopmanEnv requires a fixed graph topology; "
                "reference_sequence.is_dynamic_topology must be False "
                "(topology is held from the reset snapshot for the episode)"
            )
            raise ValueError(msg)
        require_no_hyperedges(reference_sequence)
        if max_episode_steps < 1:
            msg = f"max_episode_steps must be >= 1, got {max_episode_steps}"
            raise ValueError(msg)
        if start_index is not None and not (
            0 <= start_index < reference_sequence.num_timesteps
        ):
            msg = (
                f"start_index must be in [0, {reference_sequence.num_timesteps}), "
                f"got {start_index}"
            )
            raise ValueError(msg)
        if delta_t is not None and delta_t <= 0:
            msg = f"delta_t must be positive, got {delta_t}"
            raise ValueError(msg)
        if (
            not model.is_continuous
            and delta_t is not None
            and not np.isclose(delta_t, model.time_step)
        ):
            msg = (
                "discrete models require delta_t is None or equal to "
                f"model.time_step ({model.time_step}), got {delta_t}"
            )
            raise ValueError(msg)

        super().__init__()

        self.model = model
        self.reference_sequence = reference_sequence
        self.reward_fn = reward_fn
        self.max_episode_steps = max_episode_steps
        self._fixed_start_index = start_index
        self._random_start = random_start
        self._delta_t = float(delta_t) if delta_t is not None else None
        self._device = (
            torch.device(device)
            if device is not None
            else next(model.parameters()).device
        )
        self._uses_hetero = uses_hetero

        self.num_nodes = _reference_num_nodes(reference_sequence)
        self.latent_dim = model.latent_dim
        self.control_dim = model.control_dim
        self._latent_dims: dict[str, int] | None = None
        self._is_rectangular = False
        if uses_hetero:
            koopman = model.koopman
            self.node_type_names: tuple[str, ...] = tuple(koopman.node_types)
            self._edge_types: tuple[tuple[str, str, str], ...] = tuple(
                tuple(edge_type) for edge_type in koopman.edge_types
            )
            if isinstance(reference_sequence, HeteroGraphSnapshotSequence):
                self._num_nodes_dict: dict[str, int] | None = dict(
                    reference_sequence.num_nodes_dict
                )
            else:  # pragma: no cover - guarded above
                self._num_nodes_dict = None
            latent_dims = getattr(koopman, "latent_dims", None)
            is_rectangular = bool(getattr(koopman, "is_rectangular", False))
            if is_rectangular:
                if self._num_nodes_dict is None or latent_dims is None:
                    msg = (
                        "rectangular hetero GraphKoopmanEnv requires "
                        "HeteroGraphSnapshotSequence.num_nodes_dict and "
                        "operator.latent_dims"
                    )
                    raise ValueError(msg)
                self._latent_dims = dict(latent_dims)
                self._is_rectangular = True
        else:
            self.node_type_names = ()
            self._edge_types = ()
            self._num_nodes_dict = None

        control_low_arr = np.full(self.control_dim, control_low, dtype=np.float32)
        control_high_arr = np.full(self.control_dim, control_high, dtype=np.float32)
        if isinstance(control_low, Sequence) and not isinstance(control_low, str):
            control_low_arr = np.asarray(control_low, dtype=np.float32)
        if isinstance(control_high, Sequence) and not isinstance(control_high, str):
            control_high_arr = np.asarray(control_high, dtype=np.float32)
        if control_low_arr.shape != (self.control_dim,):
            msg = (
                f"control_low must have shape ({self.control_dim},), "
                f"got {control_low_arr.shape}"
            )
            raise ValueError(msg)
        if control_high_arr.shape != (self.control_dim,):
            msg = (
                f"control_high must have shape ({self.control_dim},), "
                f"got {control_high_arr.shape}"
            )
            raise ValueError(msg)

        self._control_low = control_low_arr
        self._control_high = control_high_arr

        if self._is_rectangular:
            assert self._num_nodes_dict is not None
            assert self._latent_dims is not None
            obs_size = stacked_latent_numel(
                self.node_type_names,
                self._num_nodes_dict,
                self._latent_dims,
            )
        else:
            obs_size = self.num_nodes * self.latent_dim
        self._obs_numel = int(obs_size)
        self.observation_space = spaces.Box(  # type: ignore[union-attr]
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(  # type: ignore[union-attr]
            low=self._control_low,
            high=self._control_high,
            dtype=np.float32,
        )

        self._edge_index: Tensor | None = None
        self._edge_weight: Tensor | None = None
        self._edge_indices: list[Tensor] | None = None
        self._edge_weights: list[Tensor | None] | None = None
        self._topology_template: HeteroData | None = None
        self._latent: Tensor | None = None
        self._step_count = 0
        self._start_index = 0

        self._freeze_model()

    def _freeze_model(self) -> None:
        """Freeze encoder/decoder and run the model in eval mode.

        Notes
        -----
        Internal helper with no parameters.
        """
        self.model.eval()
        for parameter in self.model.encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self.model.decoder.parameters():
            parameter.requires_grad_(False)

    def reshape_observation(self, observation: np.ndarray) -> Tensor:
        """Reshape a flattened observation to model latent layout.

        Shared-d: ``(num_nodes, latent_dim)``. Rectangular hetero: flat
        ``(Σ N_τ·d_τ,)``. See the module ``Heterogeneous observation layout``
        section.

        Parameters
        ----------
        observation : ndarray
            Flattened observation returned by :meth:`reset` or :meth:`step`.

        Returns
        -------
        Tensor
            Latent features matching model ``encode`` layout.
        """
        return unflatten_latent(
            observation,
            num_nodes=self.num_nodes,
            latent_dim=self.latent_dim,
            obs_numel=self._obs_numel if self._is_rectangular else None,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment from a reference snapshot.

        Parameters
        ----------
        seed : int, optional
            Random seed forwarded to Gymnasium's RNG.
        options : dict, optional
            Optional ``{"start_index": int}`` to override the configured start
            index for this episode only.

        Returns
        -------
        tuple
            ``(observation, info)`` where ``observation`` is a flattened latent
            vector and ``info`` contains ``step_index``, ``start_index``, and
            decoded physical features (see :meth:`_build_info`).
        """
        super().reset(seed=seed)
        options = options or {}

        if "start_index" in options:
            start_index = int(options["start_index"])
            if not (0 <= start_index < self.reference_sequence.num_timesteps):
                msg = (
                    "options['start_index'] must be in "
                    f"[0, {self.reference_sequence.num_timesteps})"
                )
                raise ValueError(msg)
        elif self._fixed_start_index is not None:
            start_index = self._fixed_start_index
        elif self._random_start:
            start_index = int(
                self.np_random.integers(0, self.reference_sequence.num_timesteps)
            )
        else:
            start_index = 0

        snapshot = self.reference_sequence[start_index]
        with torch.no_grad():
            snapshot_device = snapshot_to_device(snapshot, self._device)
            if self._uses_hetero:
                assert isinstance(snapshot_device, HeteroData)
                self._cache_hetero_topology(snapshot_device)
                self._latent = self.model.encode(snapshot_device)
            else:
                assert isinstance(snapshot_device, Data)
                self._edge_index = snapshot_device.edge_index
                self._edge_weight = snapshot_edge_weight(snapshot_device)
                if self._edge_weight is not None:
                    self._edge_weight = self._edge_weight.to(self._device)
                self._edge_indices = None
                self._edge_weights = None
                self._topology_template = None
                self._latent = self.model.encode(snapshot_device)

        self._step_count = 0
        self._start_index = start_index

        decoded = self._decode_current()
        observation = flatten_latent(self._latent)
        info = self._build_info(decoded)
        return observation, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply a control action and advance the latent state.

        Parameters
        ----------
        action : ndarray
            Control vector with shape ``(control_dim,)``. Values are clipped to
            the configured action bounds before application.

        Returns
        -------
        tuple
            ``(observation, reward, terminated, truncated, info)`` in Gymnasium
            v0.26+ format. Episodes truncate after ``max_episode_steps``.

        Notes
        -----
        Latent advance uses the environment's ``delta_t`` (or
        ``model.time_step`` when unset). Continuous models integrate the
        generator over that interval; discrete models always take one
        ``K``-step. Hetero models advance with cached relation banks.
        """
        if self._latent is None:
            msg = "reset() must be called before step()"
            raise RuntimeError(msg)
        if self._uses_hetero:
            if self._edge_indices is None:
                msg = "reset() must be called before step()"
                raise RuntimeError(msg)
        elif self._edge_index is None:
            msg = "reset() must be called before step()"
            raise RuntimeError(msg)

        clipped = np.clip(action, self._control_low, self._control_high)
        control: Tensor | None
        if self.control_dim == 0:
            control = None
        else:
            control = torch.as_tensor(
                clipped,
                dtype=self._latent.dtype,
                device=self._device,
            )

        with torch.no_grad():
            if self._uses_hetero:
                assert self._edge_indices is not None
                assert self._edge_weights is not None
                self._latent = propagate_latent(
                    self.model.koopman,
                    self._latent,
                    control=control,
                    delta_t=self.model.resolve_delta_t(self._delta_t),
                    default_delta_t=self.model.time_step,
                    edge_indices=self._edge_indices,
                    edge_weights=self._edge_weights,
                    num_nodes_dict=self._num_nodes_dict,
                )
                prediction = self._decode_hetero_latent(self._latent)
            else:
                assert self._edge_index is not None
                self._latent, prediction = advance_and_decode(
                    self.model.koopman,
                    self.model.decoder,
                    self._latent,
                    self._edge_index,
                    self._edge_weight,
                    control=control,
                    delta_t=self.model.resolve_delta_t(self._delta_t),
                    default_delta_t=self.model.time_step,
                )

        decoded = self._package_decoded(prediction)
        reward = float(self.reward_fn(decoded, self._step_count))
        self._step_count += 1

        terminated = False
        truncated = self._step_count >= self.max_episode_steps
        observation = flatten_latent(self._latent)
        info = self._build_info(decoded)
        return observation, reward, terminated, truncated, info

    def _cache_hetero_topology(self, snapshot: HeteroData) -> None:
        """Resolve and cache relation banks from a hetero reset snapshot.

        Parameters
        ----------
        snapshot : HeteroData
            Device-placed reset snapshot.
        """
        encoder = self.model.encoder
        if bool(getattr(encoder, "is_typed", False)):
            _, edge_indices, edge_weights, num_nodes_dict = (
                resolve_typed_relation_inputs(
                    snapshot,
                    None,
                    None,
                    node_types=encoder.node_types,
                    edge_types=encoder.edge_types,
                    num_relations=encoder.num_relations,
                )
            )
            self._num_nodes_dict = dict(num_nodes_dict)
        else:
            _, edge_indices, edge_weights = resolve_multiplex_relation_inputs(
                snapshot,
                None,
                None,
                num_relations=encoder.num_relations,
            )
            self._num_nodes_dict = None
        self._edge_indices = [bank.to(self._device) for bank in edge_indices]
        self._edge_weights = [
            None if weight is None else weight.to(self._device)
            for weight in edge_weights
        ]
        self._edge_index = None
        self._edge_weight = None
        # Keep type-local topology for packing rewards (not global banks).
        self._topology_template = snapshot.clone().cpu()

    def _decode_hetero_latent(
        self,
        latent: Tensor,
    ) -> Tensor | dict[str, Tensor]:
        """Decode a stacked hetero latent with the RelGraph decoder.

        Parameters
        ----------
        latent : Tensor
            Shared-d stacked block ``(N, d)`` or rectangular flat
            ``(Σ N_τ·d_τ,)``.

        Returns
        -------
        Tensor or dict of str to Tensor
            Multiplex tensor or per-type feature mapping.
        """
        assert self._edge_indices is not None
        assert self._edge_weights is not None
        decoder = self.model.decoder
        if bool(getattr(decoder, "is_typed", False)):
            return decoder(
                latent,
                self._edge_indices,
                self._edge_weights,
                num_nodes_dict=self._num_nodes_dict,
            )
        return decoder(latent, self._edge_indices, self._edge_weights)

    def _decode_current(self) -> SnapshotLike:
        """Decode the current latent state to a physical graph snapshot.

        Returns
        -------
        Data or HeteroData
            Decoded graph snapshot on CPU.
        """
        assert self._latent is not None
        with torch.no_grad():
            if self._uses_hetero:
                prediction = self._decode_hetero_latent(self._latent)
            else:
                assert self._edge_index is not None
                prediction = self.model.decoder(
                    self._latent,
                    self._edge_index,
                    self._edge_weight,
                )
        return self._package_decoded(prediction)

    def _package_decoded(
        self,
        prediction: Tensor | Mapping[str, Tensor],
    ) -> SnapshotLike:
        """Package decoded features into a CPU graph snapshot.

        Parameters
        ----------
        prediction : Tensor or mapping of str to Tensor
            Decoded physical node features (per-type when typed hetero).

        Returns
        -------
        Data or HeteroData
            Snapshot with CPU tensors and the environment topology.
        """
        if self._uses_hetero:
            assert self._topology_template is not None
            out = self._topology_template.clone()
            if isinstance(prediction, Mapping):
                for name in self.node_type_names:
                    out[name].x = prediction[name].detach().cpu()
            else:
                if len(self.node_type_names) != 1:
                    msg = (
                        "typed hetero decode must return a per-type feature "
                        "mapping; got a tensor"
                    )
                    raise TypeError(msg)
                out[self.node_type_names[0]].x = prediction.detach().cpu()
            return out

        assert self._edge_index is not None
        assert isinstance(prediction, Tensor)
        fields: dict[str, Tensor] = {
            "x": prediction.detach().cpu(),
            "edge_index": self._edge_index.detach().cpu(),
        }
        if self._edge_weight is not None:
            fields["edge_weight"] = self._edge_weight.detach().cpu()
        return Data(**fields)

    def _build_info(self, decoded: SnapshotLike) -> dict[str, Any]:
        """Build the info dictionary returned by reset/step.

        Parameters
        ----------
        decoded : Data or HeteroData
            Latest decoded physical snapshot.

        Returns
        -------
        dict
            Info payload with step index and decoded node features.
            Homogeneous paths expose ``decoded_x``. Hetero paths expose
            stacked ``decoded_x`` plus ``decoded_x_by_type``.
        """
        info: dict[str, Any] = {
            "step_index": self._step_count,
            "start_index": self._start_index,
            "num_nodes": self.num_nodes,
            "latent_dim": self.latent_dim,
        }
        if isinstance(decoded, HeteroData):
            by_type = {
                name: decoded[name].x.detach().cpu().numpy()
                for name in self.node_type_names
            }
            stacked = stack_typed_features(
                {name: torch.as_tensor(by_type[name]) for name in self.node_type_names},
                self.node_type_names,
            )
            info["decoded_x"] = stacked.numpy()
            info["decoded_x_by_type"] = by_type
            info["node_type_names"] = self.node_type_names
        else:
            info["decoded_x"] = decoded.x.numpy()
        return info
