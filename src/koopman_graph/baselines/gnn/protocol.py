"""Teaching forecaster protocol metadata for spatiotemporal GNN baselines.

``ForecasterProtocol`` records the documented teaching split, horizon, and
metric together with an explicit **non-empty** ``deviations`` tuple versus the
primary paper / LibCity-style scripts. An empty deviation list is rejected:
claiming zero deviation is the overclaim this package forbids for teaching
ports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ForecastMetric = Literal["mae", "rmse", "mape"]

_RATIO_SUM_ATOL = 1e-6
_ALLOWED_METRICS: frozenset[str] = frozenset({"mae", "rmse", "mape"})


class EmptyProtocolDeviationsError(ValueError):
    """Raised when a teaching forecaster protocol has an empty ``deviations`` tuple.

    Teaching baselines must name at least one simplification or mismatch versus
    the paper / reference training script. An empty tuple is treated as an
    accidental leaderboard-parity claim.

    Notes
    -----
    See class definition.
    """


@dataclass(frozen=True)
class ForecasterProtocol:
    """Documented teaching protocol for a spatiotemporal GNN forecaster.

    Parameters
    ----------
    name : str
        Short baseline identifier (e.g. ``"stgcn"``).
    history_len : int
        Temporal lookback length used by the teaching port.
    horizon : int
        Evaluation forecast horizon the protocol claims (may differ from the
        in-repo next-frame ``fit`` window objective — list that gap in
        ``deviations`` when applicable).
    train_ratio, val_ratio, test_ratio : float
        Temporal split fractions; must each lie in ``(0, 1]`` and sum to ``1``
        within absolute tolerance ``1e-6``.
    metric : {"mae", "rmse", "mape"}
        Primary teaching metric name.
    deviations : tuple of str
        Non-empty list of simplifications or mismatches versus the paper /
        LibCity-style protocol. Empty tuples raise
        :class:`EmptyProtocolDeviationsError`.
    """

    name: str
    history_len: int
    horizon: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    """Validate field bounds and the non-empty deviations contract.

Notes
-----
See signature.
    """
    metric: ForecastMetric
    deviations: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate field bounds and the non-empty deviations contract.

        Notes
        -----
        See signature."""
        if not isinstance(self.name, str) or not self.name.strip():
            msg = "ForecasterProtocol.name must be a non-empty string"
            raise ValueError(msg)
        if int(self.history_len) < 1:
            msg = f"history_len must be >= 1, got {self.history_len}"
            raise ValueError(msg)
        if int(self.horizon) < 1:
            msg = f"horizon must be >= 1, got {self.horizon}"
            raise ValueError(msg)
        for label, ratio in (
            ("train_ratio", self.train_ratio),
            ("val_ratio", self.val_ratio),
            ("test_ratio", self.test_ratio),
        ):
            if not isinstance(ratio, (int, float)) or isinstance(ratio, bool):
                msg = f"{label} must be a float, got {type(ratio).__name__}"
                raise TypeError(msg)
            if not (0.0 < float(ratio) <= 1.0):
                msg = f"{label} must satisfy 0 < ratio <= 1, got {ratio}"
                raise ValueError(msg)
        ratio_sum = (
            float(self.train_ratio) + float(self.val_ratio) + float(self.test_ratio)
        )
        if abs(ratio_sum - 1.0) > _RATIO_SUM_ATOL:
            msg = (
                "train_ratio + val_ratio + test_ratio must equal 1 within "
                f"atol={_RATIO_SUM_ATOL}, got sum={ratio_sum}"
            )
            raise ValueError(msg)
        if self.metric not in _ALLOWED_METRICS:
            msg = f"metric must be one of 'mae', 'rmse', or 'mape', got {self.metric!r}"
            raise ValueError(msg)
        deviations = self.deviations
        if not isinstance(deviations, tuple):
            object.__setattr__(self, "deviations", tuple(deviations))
            deviations = self.deviations
        if len(deviations) == 0:
            msg = (
                "ForecasterProtocol.deviations must be non-empty; teaching "
                "baselines cannot claim zero deviation from a paper / LibCity "
                "protocol"
            )
            raise EmptyProtocolDeviationsError(msg)
        for index, item in enumerate(deviations):
            if not isinstance(item, str) or not item.strip():
                msg = (
                    "ForecasterProtocol.deviations entries must be non-empty "
                    f"strings; index {index} is invalid"
                )
                raise ValueError(msg)


__all__ = [
    "EmptyProtocolDeviationsError",
    "ForecastMetric",
    "ForecasterProtocol",
]
