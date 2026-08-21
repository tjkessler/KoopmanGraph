"""Node-wise separable dictionary encoder and decoder.

A dictionary is separable when each node's lift depends only on that
node's features (Peng, Shen & Zhu, arXiv:2606.17797, Def. 2.2;
``Peng2026KoopmanGKFA``). Neighbor-mixing GNN encoders are not
separable. These modules ignore ``edge_index`` for mixing. They are
not a factory ``encoder="separable"`` kind and are not Koopman-GKFA.

References
----------
Peng, C., Shen, X. & Zhu, Y. (2026). Koopman lifting with certified
error bounds for joint inference in nonlinear networks. *arXiv*
2606.17797. (``Peng2026KoopmanGKFA``). Provisional preprint. Cited
for the separable-dictionary homomorphism precondition only.
"""

from __future__ import annotations

from typing import Literal

from torch import Tensor, nn
from torch_geometric.data import Data

from koopman_graph.nn.gnn import ActivationName, validate_positive_dims

__all__ = [
    "SeparableDictionaryDecoder",
    "SeparableDictionaryEncoder",
    "is_separable_dictionary",
]


def _activation(name: ActivationName) -> nn.Module:
    """Return the hidden activation module.

    Parameters
    ----------
    name : {"relu", "sigmoid", "tanh"}
        Activation identifier.

    Returns
    -------
    nn.Module
        Pointwise activation.
    """
    if name == "relu":
        return nn.ReLU()
    if name == "sigmoid":
        return nn.Sigmoid()
    return nn.Tanh()


def _nodewise_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    *,
    num_layers: int,
    activation: ActivationName,
) -> nn.Sequential:
    """Build a per-node MLP that does not mix graph neighbors.

    Parameters
    ----------
    in_dim, hidden_dim, out_dim : int
        Channel widths.
    num_layers : int
        Affine layers. ``1`` is a single linear map.
    activation : {"relu", "sigmoid", "tanh"}
        Hidden activation (unused when ``num_layers == 1``).

    Returns
    -------
    nn.Sequential
        Row-wise MLP on ``(N, in_dim)``.
    """
    if num_layers == 1:
        return nn.Sequential(nn.Linear(in_dim, out_dim))
    layers: list[nn.Module] = [
        nn.Linear(in_dim, hidden_dim),
        _activation(activation),
    ]
    for _ in range(num_layers - 2):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(_activation(activation))
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


def is_separable_dictionary(module: object) -> bool:
    """Return whether ``module`` is a node-wise separable dictionary.

    Unwraps ``base_encoder`` (delay wrappers). Default GNN encoders are
    not separable.

    Parameters
    ----------
    module : object
        Encoder module or ``None``.

    Returns
    -------
    bool
        ``True`` when the lift is :class:`SeparableDictionaryEncoder`
        (or declares ``encoder_kind="separable"``).
    """
    current: object | None = module
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, SeparableDictionaryEncoder):
            return True
        kind = getattr(current, "encoder_kind", None)
        if kind == "separable":
            return True
        current = getattr(current, "base_encoder", None)
    return False


class SeparableDictionaryEncoder(nn.Module):
    """Per-node lift :math:`z_i = \\psi(x_i)` (no neighbor mixing).

    ``encoder_kind`` is ``"separable"``. This is the homomorphism
    precondition in ``Peng2026KoopmanGKFA``, not a GNN encoder and not a
    certified Koopman-GKFA implementation.

    Attributes
    ----------
    encoder_kind : str
        Always ``"separable"``.
    in_channels : int
        Input feature width per node.
    hidden_channels : int
        Hidden MLP width.
    latent_dim : int
        Output latent width per node.
    num_layers : int
        Number of affine layers.
    """

    encoder_kind: Literal["separable"] = "separable"

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        latent_dim: int,
        *,
        num_layers: int = 2,
        activation: ActivationName = "relu",
    ) -> None:
        """Initialize a node-wise dictionary encoder.

        Parameters
        ----------
        in_channels : int
            Input feature width.
        hidden_channels : int
            Hidden width when ``num_layers > 1``.
        latent_dim : int
            Latent width.
        num_layers : int, optional
            Affine layers. Default ``2``.
        activation : {"relu", "sigmoid", "tanh"}, optional
            Hidden activation. Default ``"relu"``.

        Raises
        ------
        ValueError
            If a dimension is not a positive int.
        """
        super().__init__()
        validate_positive_dims(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            latent_dim=latent_dim,
            num_layers=num_layers,
        )
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.latent_dim = int(latent_dim)
        self.num_layers = int(num_layers)
        self.activation_name = activation
        self.mlp = _nodewise_mlp(
            self.in_channels,
            self.hidden_channels,
            self.latent_dim,
            num_layers=self.num_layers,
            activation=activation,
        )

    def receptive_field_hops(self) -> int:
        """Return zero graph hops (node-wise lift).

        Returns
        -------
        int
            ``0``.
        """
        return 0

    def forward(
        self,
        x_or_data: Tensor | Data,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        """Lift each node independently.

        ``edge_index`` / ``edge_weight`` are accepted for encoder API
        symmetry and are not used.

        Parameters
        ----------
        x_or_data : Tensor or Data
            Node features ``(N, in_channels)`` or a ``Data`` snapshot.
        edge_index : Tensor or None, optional
            Ignored.
        edge_weight : Tensor or None, optional
            Ignored.

        Returns
        -------
        Tensor
            Latents ``(N, latent_dim)``.

        Raises
        ------
        ValueError
            If feature rank or width is wrong.
        """
        del edge_index, edge_weight
        x = x_or_data.x if isinstance(x_or_data, Data) else x_or_data
        if x.ndim != 2 or int(x.shape[1]) != self.in_channels:
            msg = (
                "SeparableDictionaryEncoder expects features "
                f"(N, {self.in_channels}), got {tuple(x.shape)}"
            )
            raise ValueError(msg)
        return self.mlp(x)


class SeparableDictionaryDecoder(nn.Module):
    """Per-node map :math:`\\hat x_i = \\psi^{-1}(z_i)` (no neighbor mixing).

    Matching decoder for :class:`SeparableDictionaryEncoder`. Topology
    arguments are ignored.

    Attributes
    ----------
    latent_dim : int
        Input latent width per node.
    hidden_channels : int
        Hidden MLP width.
    out_channels : int
        Reconstructed feature width per node.
    num_layers : int
        Number of affine layers.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_channels: int,
        out_channels: int,
        *,
        num_layers: int = 2,
        activation: ActivationName = "relu",
    ) -> None:
        """Initialize a node-wise dictionary decoder.

        Parameters
        ----------
        latent_dim : int
            Latent width.
        hidden_channels : int
            Hidden width when ``num_layers > 1``.
        out_channels : int
            Output feature width.
        num_layers : int, optional
            Affine layers. Default ``2``.
        activation : {"relu", "sigmoid", "tanh"}, optional
            Hidden activation. Default ``"relu"``.

        Raises
        ------
        ValueError
            If a dimension is not a positive int.
        """
        super().__init__()
        validate_positive_dims(
            latent_dim=latent_dim,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
        )
        self.latent_dim = int(latent_dim)
        self.hidden_channels = int(hidden_channels)
        self.out_channels = int(out_channels)
        self.num_layers = int(num_layers)
        self.activation_name = activation
        self.mlp = _nodewise_mlp(
            self.latent_dim,
            self.hidden_channels,
            self.out_channels,
            num_layers=self.num_layers,
            activation=activation,
        )

    def receptive_field_hops(self) -> int:
        """Return zero graph hops (node-wise map).

        Returns
        -------
        int
            ``0``.
        """
        return 0

    def forward(
        self,
        z: Tensor,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        """Decode each node independently.

        Parameters
        ----------
        z : Tensor
            Latents ``(N, latent_dim)``.
        edge_index : Tensor or None, optional
            Ignored.
        edge_weight : Tensor or None, optional
            Ignored.

        Returns
        -------
        Tensor
            Features ``(N, out_channels)``.

        Raises
        ------
        ValueError
            If latent rank or width is wrong.
        """
        del edge_index, edge_weight
        if z.ndim != 2 or int(z.shape[1]) != self.latent_dim:
            msg = (
                "SeparableDictionaryDecoder expects latents "
                f"(N, {self.latent_dim}), got {tuple(z.shape)}"
            )
            raise ValueError(msg)
        return self.mlp(z)
