"""Extended DMD baseline with polynomial, RBF, and kernel dictionaries.

Kernel dictionaries support exact center sections plus Nyström and random
Fourier feature (RFF) approximations for medium-``T`` paths. Primary-source
citations are deferred to Sphinx Phase 61 verification.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor
from torch_geometric.data import Data

from koopman_graph.baselines.base import (
    ClassicalBaseline,
    RankSpec,
    check_initial_graph,
    copy_topology,
    fit_row_operator,
    flatten_snapshots,
    require_static_topology,
    resolve_fit_rank,
)
from koopman_graph.data import (
    GraphSnapshotSequence,
    resolve_sequence,
)
from koopman_graph.spectrum_types import KoopmanSpectrum, compute_spectrum

DictionaryKind = Literal["polynomial", "rbf", "kernel"]
PolynomialDegree = Literal[1, 2]
KernelKind = Literal["gaussian", "polynomial", "linear"]
KernelApproximation = Literal["centers", "nystrom", "random_features"]


class EDMDBaseline(ClassicalBaseline):
    """Extended DMD baseline with polynomial, RBF, or kernel dictionaries.

    EDMD lifts flattened graph states into a fixed observable space, fits a
    linear Koopman operator there, and learns a least-squares
    ``reconstruction_matrix`` back to physical node features (not a GNN
    decoder). Dictionary modes:

    * ``polynomial`` — identity (``polynomial_degree=1``) or identity plus
      elementwise squares (``polynomial_degree=2``); Williams et al. (2015)
      EDMD with a monomial dictionary.
    * ``rbf`` — explicit Gaussian radial basis functions centered on a subset
      of training states (``num_centers``, ``length_scale``).
    * ``kernel`` — dictionary of kernel sections ``k(x, c_i)`` against training
      centers (Gaussian / polynomial), or the linear-kernel identity lift that
      reduces to DMD. Full-data centers are ``O(T^2)``; use
      ``kernel_approximation="nystrom"`` or ``"random_features"`` for fixed-width
      lifts on medium ``T``.

    All modes remain **topology-blind**: snapshots are flattened and graph
    edges are ignored during fit/predict (topology is only copied onto
    predicted ``Data`` objects).

    Satisfies :class:`~koopman_graph.protocols.ForecastModel` and the narrower
    :class:`~koopman_graph.protocols.UncontrolledForecastModel` peer set.
    ``predict`` accepts a PyG ``Data`` snapshot only (not raw feature tensors)
    and does not take controls or future topologies.

    Parameters
    ----------
    time_step : float, optional
        Physical duration represented by one snapshot transition. Used by
        :meth:`spectrum`. Default is ``1.0``.
    rank : int or None or {"auto"}, optional
        Optional truncated-SVD rank for the observable data matrix. ``None``
        uses the full least-squares solution. ``"auto"`` selects the
        Gavish–Donoho hard threshold. Default is ``None``.
    dictionary : {"polynomial", "rbf", "kernel"}, optional
        Observable dictionary family. Default is ``"polynomial"``.
    polynomial_degree : {1, 2}, optional
        Used when ``dictionary="polynomial"``. Default is ``2``.
    num_centers : int or None, optional
        Number of RBF / kernel centers drawn uniformly from training snapshots.
        ``None`` selects ``min(32, T)`` for ``rbf`` and all ``T`` snapshots for
        ``kernel`` center sections. Ignored for ``polynomial``.
    length_scale : float, optional
        Gaussian bandwidth ``σ`` for ``rbf`` and ``kernel="gaussian"``.
        Default is ``1.0``.
    kernel : {"gaussian", "polynomial", "linear"}, optional
        Kernel family when ``dictionary="kernel"``. ``"linear"`` uses the
        identity feature map (DMD reduction). Default is ``"gaussian"``.
    kernel_degree : int, optional
        Degree of the polynomial kernel
        ``(γ ⟨x, y⟩ + coef0)^degree``. Default is ``2``.
    kernel_coef0 : float, optional
        Additive coefficient in the polynomial kernel. Default is ``1.0``.
    kernel_gamma : float or None, optional
        Scale ``γ`` for the polynomial kernel. ``None`` uses ``1.0``.
        Default is ``None``.
    kernel_approximation : {"centers", "nystrom", "random_features"}, optional
        Kernel lift when ``dictionary="kernel"`` and ``kernel != "linear"``.
        Default ``"centers"`` preserves the exact section path. Nyström and
        RFF require a non-linear kernel dictionary.
    num_features : int or None, optional
        Feature width for Nyström / RFF (default ``min(32, T)``). Nyström also
        uses this as the landmark count when ``num_centers`` is ``None``.
    feature_seed : int or None, optional
        RNG seed for RFF weights. Default ``0`` when RFF is used.
    """

    def __init__(
        self,
        *,
        time_step: float = 1.0,
        rank: RankSpec = None,
        dictionary: DictionaryKind = "polynomial",
        polynomial_degree: PolynomialDegree = 2,
        num_centers: int | None = None,
        length_scale: float = 1.0,
        kernel: KernelKind = "gaussian",
        kernel_degree: int = 2,
        kernel_coef0: float = 1.0,
        kernel_gamma: float | None = None,
        kernel_approximation: KernelApproximation = "centers",
        num_features: int | None = None,
        feature_seed: int | None = None,
    ) -> None:
        """Initialize the EDMD baseline.

        Parameters
        ----------
        time_step : float, optional
            Physical duration represented by one snapshot transition. Default
            is ``1.0``.
        rank : int or None or {"auto"}, optional
            Optional truncated-SVD rank in observable space. ``None`` uses full
            least squares. ``"auto"`` selects the Gavish–Donoho hard threshold.
        dictionary : {"polynomial", "rbf", "kernel"}, optional
            Observable dictionary family. Default is ``"polynomial"``.
        polynomial_degree : {1, 2}, optional
            Polynomial observable degree when ``dictionary="polynomial"``.
            Default is ``2``.
        num_centers : int or None, optional
            Center count for ``rbf`` / ``kernel``. See class docstring.
        length_scale : float, optional
            Gaussian bandwidth. Default is ``1.0``.
        kernel : {"gaussian", "polynomial", "linear"}, optional
            Kernel family for ``dictionary="kernel"``. Default is
            ``"gaussian"``.
        kernel_degree : int, optional
            Polynomial kernel degree. Default is ``2``.
        kernel_coef0 : float, optional
            Polynomial kernel offset. Default is ``1.0``.
        kernel_gamma : float or None, optional
            Polynomial kernel scale. ``None`` uses ``1.0``.
        kernel_approximation : {"centers", "nystrom", "random_features"}, optional
            Kernel lift approximation. Default is ``"centers"``.
        num_features : int or None, optional
            Nyström / RFF width. Default ``min(32, T)`` at fit time.
        feature_seed : int or None, optional
            RFF RNG seed. Default ``0`` when RFF is used.

        Raises
        ------
        ValueError
            If ``time_step`` is not positive or a dictionary knob is invalid.
        """
        super().__init__(time_step=time_step, rank=rank)
        if dictionary not in ("polynomial", "rbf", "kernel"):
            msg = (
                "dictionary must be 'polynomial', 'rbf', or 'kernel', "
                f"got {dictionary!r}"
            )
            raise ValueError(msg)
        if polynomial_degree not in (1, 2):
            msg = f"polynomial_degree must be 1 or 2, got {polynomial_degree}"
            raise ValueError(msg)
        if num_centers is not None and num_centers < 1:
            msg = f"num_centers must be >= 1 when provided, got {num_centers}"
            raise ValueError(msg)
        if length_scale <= 0.0:
            msg = f"length_scale must be positive, got {length_scale}"
            raise ValueError(msg)
        if kernel not in ("gaussian", "polynomial", "linear"):
            msg = (
                f"kernel must be 'gaussian', 'polynomial', or 'linear', got {kernel!r}"
            )
            raise ValueError(msg)
        if kernel_degree < 1:
            msg = f"kernel_degree must be >= 1, got {kernel_degree}"
            raise ValueError(msg)
        if kernel_gamma is not None and kernel_gamma <= 0.0:
            msg = f"kernel_gamma must be positive when provided, got {kernel_gamma}"
            raise ValueError(msg)
        if kernel_approximation not in ("centers", "nystrom", "random_features"):
            msg = (
                "kernel_approximation must be 'centers', 'nystrom', or "
                f"'random_features', got {kernel_approximation!r}"
            )
            raise ValueError(msg)
        if num_features is not None and num_features < 1:
            msg = f"num_features must be >= 1 when provided, got {num_features}"
            raise ValueError(msg)
        if kernel_approximation != "centers":
            if dictionary != "kernel":
                msg = (
                    "kernel_approximation requires dictionary='kernel', "
                    f"got {dictionary!r}"
                )
                raise ValueError(msg)
            if kernel == "linear":
                msg = (
                    "kernel_approximation requires a non-linear kernel "
                    "(not kernel='linear')"
                )
                raise ValueError(msg)

        self.dictionary = dictionary
        self.polynomial_degree = polynomial_degree
        self.num_centers = num_centers
        self.length_scale = float(length_scale)
        self.kernel = kernel
        self.kernel_degree = int(kernel_degree)
        self.kernel_coef0 = float(kernel_coef0)
        self.kernel_gamma = None if kernel_gamma is None else float(kernel_gamma)
        self.kernel_approximation = kernel_approximation
        self.num_features = num_features
        self.feature_seed = feature_seed
        self.reconstruction_matrix: Tensor | None = None
        self.observable_dim: int | None = None
        self.centers: Tensor | None = None
        self._nystrom_whitener: Tensor | None = None
        self._rff_weight: Tensor | None = None
        self._rff_bias: Tensor | None = None

    def _is_fitted(self) -> bool:
        """Return whether the EDMD operator and reconstruction matrix are fit.

        Returns
        -------
        bool
            ``True`` when ``K`` and ``reconstruction_matrix`` are available.
        """
        return self.K is not None and self.reconstruction_matrix is not None

    def _require_reconstruction_matrix(self) -> Tensor:
        """Return the fitted reconstruction matrix after a fitted-state check.

        Returns
        -------
        Tensor
            Least-squares map from observables to flattened physical states.

        Raises
        ------
        RuntimeError
            If the baseline has not been fit.
        """
        self._check_fitted()
        if self.reconstruction_matrix is None:
            raise RuntimeError(self._unfitted_message())
        return self.reconstruction_matrix

    def fit(
        self,
        sequence: GraphSnapshotSequence | Sequence[Data],
    ) -> EDMDBaseline:
        """Fit EDMD operator and linear reconstruction matrix.

        Parameters
        ----------
        sequence : GraphSnapshotSequence or sequence of Data
            Training snapshots with shared topology.

        Returns
        -------
        EDMDBaseline
            The fitted baseline (``self``) for sklearn-style chaining. Unlike
            :meth:`~koopman_graph.model.GraphKoopmanModel.fit`, which returns
            :class:`~koopman_graph.training.FitHistory`, classical baselines
            return ``self``; see the ``ForecastModel`` call-site matrix in
            :doc:`architecture`.

        Raises
        ------
        ValueError
            If fewer than two snapshots are provided, the sequence has dynamic
            topology, rank is invalid, or center selection is impossible.
        """
        resolved = resolve_sequence(sequence)
        require_static_topology(resolved)
        if resolved.num_timesteps < 2:
            msg = "EDMDBaseline.fit requires at least two snapshots"
            raise ValueError(msg)

        states = flatten_snapshots(resolved)
        self._nystrom_whitener = None
        self._rff_weight = None
        self._rff_bias = None
        if self.kernel_approximation == "random_features":
            self.centers = None
            self._init_random_features(states)
        else:
            self.centers = self._select_centers(states)
            if self.kernel_approximation == "nystrom":
                self._init_nystrom_whitener()
        observables = self._observables(states)
        left = observables[:-1]
        self.selected_rank = resolve_fit_rank(left, self.rank)
        self.K = fit_row_operator(left, observables[1:], self.selected_rank)
        self.reconstruction_matrix = torch.linalg.lstsq(observables, states).solution.T
        self.num_nodes = resolved.num_nodes
        self.in_channels = resolved.in_channels
        self.state_dim = states.shape[1]
        self.observable_dim = observables.shape[1]
        return self

    def predict(self, initial_graph: Data, steps: int) -> list[Data]:
        """Autoregressively predict future graph snapshots (Data-only).

        Uncontrolled peer call site shared with :class:`DMDBaseline` and
        :class:`~koopman_graph.model.GraphKoopmanModel` when the latter is
        invoked as ``predict(data, steps)``. Does not accept raw feature
        tensors or a ``controls`` argument.

        Parameters
        ----------
        initial_graph : Data
            Initial graph snapshot. Its topology is copied to every prediction.
        steps : int
            Number of future snapshots to predict.

        Returns
        -------
        list of Data
            Predicted graph snapshots with the same node/feature shape as the
            fitted training data.

        Raises
        ------
        RuntimeError
            If the baseline has not been fit.
        ValueError
            If ``steps < 1`` or graph metadata does not match the fit data.
        """
        operator = self._require_operator()
        reconstruction = self._require_reconstruction_matrix()
        num_nodes, in_channels = self._require_graph_metadata()
        if steps < 1:
            msg = f"steps must be >= 1, got {steps}"
            raise ValueError(msg)
        check_initial_graph(
            initial_graph,
            num_nodes=num_nodes,
            in_channels=in_channels,
        )

        observable = self._observables(initial_graph.x.reshape(1, -1)).squeeze(0)
        topology = copy_topology(initial_graph)
        predictions: list[Data] = []
        for _ in range(steps):
            observable = observable @ operator.T
            state = observable @ reconstruction.T
            x = state.reshape(num_nodes, in_channels)
            predictions.append(Data(x=x, **topology))
        return predictions

    def spectrum(self) -> KoopmanSpectrum:
        """Return the EDMD observable-space operator spectrum.

        Takes no kwargs (unlike continuous
        :meth:`~koopman_graph.model.GraphKoopmanModel.spectrum`).

        Returns
        -------
        KoopmanSpectrum
            Eigendecomposition and continuous-time mode characteristics of the
            fitted observable-space operator.
        """
        return compute_spectrum(self._require_operator(), self.time_step)

    def _select_centers(self, states: Tensor) -> Tensor | None:
        """Choose dictionary centers from flattened training states.

        Parameters
        ----------
        states : Tensor
            Training states with shape ``(T, state_dim)``.

        Returns
        -------
        Tensor or None
            Center matrix with shape ``(num_centers, state_dim)``, or ``None``
            when the dictionary does not use centers.

        Raises
        ------
        ValueError
            If ``num_centers`` exceeds the number of training snapshots.
        """
        if self.dictionary == "polynomial":
            return None
        if self.dictionary == "kernel" and self.kernel == "linear":
            return None
        if self.kernel_approximation == "random_features":
            return None

        num_timesteps = int(states.shape[0])
        if self.dictionary == "rbf":
            requested = 32 if self.num_centers is None else self.num_centers
        elif self.kernel_approximation == "nystrom":
            if self.num_centers is not None:
                requested = self.num_centers
            elif self.num_features is not None:
                requested = self.num_features
            else:
                requested = min(32, num_timesteps)
        else:
            requested = num_timesteps if self.num_centers is None else self.num_centers
        if requested > num_timesteps:
            msg = f"num_centers={requested} exceeds training length T={num_timesteps}"
            raise ValueError(msg)
        if requested == num_timesteps:
            return states.detach().clone()
        indices = (
            torch.linspace(
                0,
                num_timesteps - 1,
                steps=requested,
                device=states.device,
            )
            .round()
            .long()
        )
        return states[indices].detach().clone()

    def _feature_width(self, num_timesteps: int) -> int:
        """Resolve Nyström / RFF feature width for training length ``T``.

        Parameters
        ----------
        num_timesteps
            Value for ``num_timesteps``.

        Returns
        -------
        object
            Function result.
        """
        if self.num_features is not None:
            return int(self.num_features)
        return min(32, num_timesteps)

    def _init_random_features(self, states: Tensor) -> None:
        """Sample seeded RFF weights for Gaussian kernel approximation.

        Parameters
        ----------
        states
            Value for ``states``.
        """
        if self.kernel != "gaussian":
            msg = "random_features currently supports kernel='gaussian' only"
            raise ValueError(msg)
        num_timesteps = int(states.shape[0])
        width = self._feature_width(num_timesteps)
        state_dim = int(states.shape[1])
        seed = 0 if self.feature_seed is None else int(self.feature_seed)
        generator = torch.Generator(device=states.device)
        generator.manual_seed(seed)
        self._rff_weight = (
            torch.randn(
                state_dim,
                width,
                dtype=states.dtype,
                device=states.device,
                generator=generator,
            )
            / self.length_scale
        )
        self._rff_bias = (
            2.0
            * math.pi
            * torch.rand(
                width,
                dtype=states.dtype,
                device=states.device,
                generator=generator,
            )
        )

    def _init_nystrom_whitener(self) -> None:
        """Build the Nyström whitener ``pinv(sqrtm(K_mm))`` from landmarks.

        Notes
        -----
        Internal helper with no parameters.
        """
        if self.centers is None:
            msg = "Nyström approximation requires landmark centers"
            raise RuntimeError(msg)
        gram = self._kernel_features(self.centers, self.centers)
        # Symmetric eigendecomposition for a principal square-root inverse.
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        positive = torch.clamp(eigenvalues, min=0.0)
        inv_sqrt = torch.zeros_like(positive)
        mask = positive > 1e-12
        inv_sqrt[mask] = positive[mask].rsqrt()
        self._nystrom_whitener = eigenvectors @ torch.diag(inv_sqrt) @ eigenvectors.T

    def _observables(self, states: Tensor) -> Tensor:
        """Lift flattened states into the configured observable dictionary.

        Parameters
        ----------
        states : Tensor
            Flattened physical states with shape ``(..., state_dim)``.

        Returns
        -------
        Tensor
            Observable matrix with shape ``(..., observable_dim)``.
        """
        if self.dictionary == "polynomial":
            if self.polynomial_degree == 1:
                return states
            return torch.cat([states, states.square()], dim=-1)

        if self.dictionary == "kernel" and self.kernel == "linear":
            # Linear kernel k(x, y) = ⟨x, y⟩ has feature map φ(x) = x on R^d,
            # so Kernel EDMD reduces to standard DMD on flattened states.
            return states

        if self.kernel_approximation == "random_features":
            if self._rff_weight is None or self._rff_bias is None:
                msg = "random features are not available; call fit() first"
                raise RuntimeError(msg)
            projection = states @ self._rff_weight + self._rff_bias
            scale = math.sqrt(2.0 / float(self._rff_weight.shape[1]))
            return scale * torch.cos(projection)

        if self.centers is None:
            msg = "dictionary centers are not available; call fit() first"
            raise RuntimeError(msg)

        if self.dictionary == "rbf":
            return self._rbf_features(states, self.centers)

        if self.kernel_approximation == "nystrom":
            if self._nystrom_whitener is None:
                msg = "Nyström whitener is not available; call fit() first"
                raise RuntimeError(msg)
            kernel_block = self._kernel_features(states, self.centers)
            return kernel_block @ self._nystrom_whitener

        return self._kernel_features(states, self.centers)

    def _rbf_features(self, states: Tensor, centers: Tensor) -> Tensor:
        """Evaluate Gaussian RBF features against stored centers.

        Parameters
        ----------
        states : Tensor
            Query states ``(..., state_dim)``.
        centers : Tensor
            Center matrix ``(C, state_dim)``.

        Returns
        -------
        Tensor
            Features ``(..., C)`` with
            ``φ_i(x) = exp(-‖x - c_i‖² / (2 σ²))``.
        """
        # (..., C, state_dim)
        delta = states.unsqueeze(-2) - centers
        squared = delta.square().sum(dim=-1)
        scale = 2.0 * self.length_scale * self.length_scale
        return torch.exp(-squared / scale)

    def _kernel_features(self, states: Tensor, centers: Tensor) -> Tensor:
        """Evaluate kernel sections ``k(x, c_i)`` against stored centers.

        Parameters
        ----------
        states : Tensor
            Query states ``(..., state_dim)``.
        centers : Tensor
            Center matrix ``(C, state_dim)``.

        Returns
        -------
        Tensor
            Kernel feature matrix ``(..., C)``.
        """
        if self.kernel == "gaussian":
            return self._rbf_features(states, centers)

        gamma = 1.0 if self.kernel_gamma is None else self.kernel_gamma
        # (..., C)
        dots = states @ centers.T
        return (gamma * dots + self.kernel_coef0).pow(self.kernel_degree)
