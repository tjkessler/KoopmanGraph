"""Gymnasium environment wrapper for latent-space graph control."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data

from koopman_graph.data import GraphSnapshotSequence, _snapshot_edge_weight

if TYPE_CHECKING:
    from koopman_graph.model import GraphKoopmanModel

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - exercised via import guard test
    gym = None  # type: ignore[assignment,misc]
    spaces = None  # type: ignore[assignment,misc]

RewardFn = Callable[[Data, int], float]

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
        If Gymnasium is not installed.
    """
    if gym is None or spaces is None:
        raise ImportError(_GYMNASIUM_IMPORT_ERROR)
    return gym, spaces


def flatten_latent(z: Tensor) -> np.ndarray:
    """Flatten node latent states for Gymnasium observations.

    Parameters
    ----------
    z : Tensor
        Latent node features with shape ``(num_nodes, latent_dim)``.

    Returns
    -------
    ndarray
        Observation vector with shape ``(num_nodes * latent_dim,)`` and
        dtype ``float32``.
    """
    return z.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)


def unflatten_latent(
    observation: np.ndarray,
    *,
    num_nodes: int,
    latent_dim: int,
) -> Tensor:
    """Reshape a flattened observation back to node latent features.

    Parameters
    ----------
    observation : ndarray
        Flattened latent vector.
    num_nodes : int
        Number of graph nodes.
    latent_dim : int
        Latent feature dimension per node.

    Returns
    -------
    Tensor
        Latent tensor with shape ``(num_nodes, latent_dim)``.
    """
    expected = num_nodes * latent_dim
    if observation.size != expected:
        msg = f"expected flattened observation size {expected}, got {observation.size}"
        raise ValueError(msg)
    return torch.from_numpy(observation.reshape(num_nodes, latent_dim))


class GraphKoopmanEnv(gym.Env if gym is not None else object):  # type: ignore[misc]
    """Gymnasium environment for closed-loop control in Koopman latent space.

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

    **Limitations.** Rewards see decoded states that depend on a frozen
    encoder/decoder trained offline. Global controls with shape
    ``(control_dim,)`` are supported; per-node action spaces are not. Topology
    is held fixed from the reset snapshot for the episode. Discrete models
    advance by one ``K``-step per ``step`` call; only continuous models honor
    a custom ``delta_t`` integration interval.

    Parameters
    ----------
    model : GraphKoopmanModel
        Trained controlled model with ``control_dim > 0``.
    reference_sequence : GraphSnapshotSequence
        Sequence supplying reset snapshots and fixed episode topology.
    reward_fn : callable
        ``reward_fn(decoded_snapshot, step_index) -> float`` where
        ``decoded_snapshot`` is a PyG ``Data`` object with physical node
        features and ``step_index`` counts environment steps since ``reset``.
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
        Integration interval passed to
        :meth:`~koopman_graph.model.GraphKoopmanModel._advance_latent` on each
        ``step``. When ``None`` (default), uses ``model.time_step``. For
        continuous models this enables closed-loop control at a horizon other
        than the training ``time_step``. For discrete models, ``delta_t`` must
        be ``None`` or equal to ``model.time_step`` (strict check); irregular
        stepping is not supported in discrete mode.
    device : torch.device or str or None, optional
        Device for model inference. Defaults to the model's current parameter
        device.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        model: GraphKoopmanModel,
        reference_sequence: GraphSnapshotSequence,
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

        See the class docstring for parameter descriptions.

        Raises
        ------
        TypeError
            If ``model`` is not a :class:`~koopman_graph.model.GraphKoopmanModel`.
        ValueError
            If ``control_dim`` is zero or arguments are invalid.
        ImportError
            If Gymnasium is not installed.
        """
        _require_gymnasium()
        from koopman_graph.model import GraphKoopmanModel as _GraphKoopmanModel

        if not isinstance(model, _GraphKoopmanModel):
            msg = "model must be a GraphKoopmanModel instance"
            raise TypeError(msg)
        if model.control_dim <= 0:
            msg = "GraphKoopmanEnv requires model.control_dim > 0"
            raise ValueError(msg)
        if reference_sequence.num_timesteps < 1:
            msg = "reference_sequence must contain at least one snapshot"
            raise ValueError(msg)
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

        self.num_nodes = reference_sequence.num_nodes
        self.latent_dim = model.latent_dim
        self.control_dim = model.control_dim

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

        obs_size = self.num_nodes * self.latent_dim
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
        self._latent: Tensor | None = None
        self._step_count = 0
        self._start_index = 0

        self._freeze_model()

    def _freeze_model(self) -> None:
        """Freeze encoder/decoder and run the model in eval mode.

        Returns
        -------
        None
        """
        self.model.eval()
        for parameter in self.model.encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self.model.decoder.parameters():
            parameter.requires_grad_(False)

    def reshape_observation(self, observation: np.ndarray) -> Tensor:
        """Reshape a flattened observation to ``(num_nodes, latent_dim)``.

        Parameters
        ----------
        observation : ndarray
            Flattened observation returned by :meth:`reset` or :meth:`step`.

        Returns
        -------
        Tensor
            Node latent features.
        """
        return unflatten_latent(
            observation,
            num_nodes=self.num_nodes,
            latent_dim=self.latent_dim,
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
            ``decoded_x`` (physical node features as a NumPy array).
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
        self._edge_index = snapshot.edge_index.to(self._device)
        self._edge_weight = _snapshot_edge_weight(snapshot)
        if self._edge_weight is not None:
            self._edge_weight = self._edge_weight.to(self._device)

        with torch.no_grad():
            snapshot_device = self.model._snapshot_to_device(snapshot, self._device)
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
        ``K``-step.
        """
        if self._latent is None or self._edge_index is None:
            msg = "reset() must be called before step()"
            raise RuntimeError(msg)

        clipped = np.clip(action, self._control_low, self._control_high)
        control = torch.as_tensor(
            clipped,
            dtype=self._latent.dtype,
            device=self._device,
        )

        with torch.no_grad():
            self._latent = self.model._advance_latent(
                self._latent,
                control=control,
                delta_t=self._delta_t,
            )

        decoded = self._decode_current()
        reward = float(self.reward_fn(decoded, self._step_count))
        self._step_count += 1

        terminated = False
        truncated = self._step_count >= self.max_episode_steps
        observation = flatten_latent(self._latent)
        info = self._build_info(decoded)
        return observation, reward, terminated, truncated, info

    def _decode_current(self) -> Data:
        """Decode the current latent state to a physical graph snapshot.

        Returns
        -------
        Data
            Decoded graph snapshot on CPU.
        """
        assert self._latent is not None
        assert self._edge_index is not None
        with torch.no_grad():
            prediction = self.model.decoder(
                self._latent,
                self._edge_index,
                self._edge_weight,
            )
        fields: dict[str, Tensor] = {
            "x": prediction.detach().cpu(),
            "edge_index": self._edge_index.detach().cpu(),
        }
        if self._edge_weight is not None:
            fields["edge_weight"] = self._edge_weight.detach().cpu()
        return Data(**fields)

    def _build_info(self, decoded: Data) -> dict[str, Any]:
        """Build the info dictionary returned by reset/step.

        Parameters
        ----------
        decoded : Data
            Latest decoded physical snapshot.

        Returns
        -------
        dict
            Info payload with step index and decoded node features.
        """
        return {
            "step_index": self._step_count,
            "start_index": self._start_index,
            "decoded_x": decoded.x.numpy(),
            "num_nodes": self.num_nodes,
            "latent_dim": self.latent_dim,
        }
