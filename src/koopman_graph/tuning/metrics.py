"""Flatten :class:`~koopman_graph.training.FitHistory` for HPO reporters."""

from __future__ import annotations

from koopman_graph.training.history import FitHistory


def _last(series: tuple[float, ...] | None) -> float | None:
    """Return the last finite-friendly float from a series, or ``None``.

    Parameters
    ----------
    series : tuple of float or None
        Epoch series; empty or ``None`` yields ``None``.

    Returns
    -------
    float or None
        Last element as ``float``, or ``None`` when unavailable.
    """
    if series is None or len(series) == 0:
        return None
    return float(series[-1])


def fit_history_metrics(history: FitHistory) -> dict[str, float]:
    """Extract scalar metrics from a fit history for Tune / Optuna reporters.

    Returns a flat ``str → float`` mapping. Optional series that are missing
    or empty are omitted (no sentinel NaNs). Boolean flags are encoded as
    ``0.0`` / ``1.0``.

    Parameters
    ----------
    history : FitHistory
        History returned by :meth:`~koopman_graph.GraphKoopmanModel.fit` or
        :func:`~koopman_graph.training.run_fit_loop`.

    Returns
    -------
    dict of str to float
        Keys always include ``epochs`` and ``stopped_early``. When the
        corresponding series or fields are present, also includes
        ``loss`` / ``final_loss``, ``reconstruction_loss``, ``best_loss``,
        ``best_epoch``, ``val_loss`` / ``final_val_loss``, and
        ``val_reconstruction_loss``.

    Notes
    -----
    This helper does not choose an optimization objective. Callers decide
    which key (e.g. ``loss`` vs ``val_loss``) to minimize or maximize.
    """
    metrics: dict[str, float] = {
        "epochs": float(history.epochs),
        "stopped_early": 1.0 if history.stopped_early else 0.0,
    }

    final_loss = _last(history.loss)
    if final_loss is not None:
        metrics["loss"] = final_loss
        metrics["final_loss"] = final_loss

    reconstruction = _last(history.reconstruction_loss)
    if reconstruction is not None:
        metrics["reconstruction_loss"] = reconstruction

    if history.best_loss is not None:
        metrics["best_loss"] = float(history.best_loss)
    if history.best_epoch is not None:
        metrics["best_epoch"] = float(history.best_epoch)

    final_val = _last(history.val_loss)
    if final_val is not None:
        metrics["val_loss"] = final_val
        metrics["final_val_loss"] = final_val

    val_reconstruction = _last(history.val_reconstruction_loss)
    if val_reconstruction is not None:
        metrics["val_reconstruction_loss"] = val_reconstruction

    return metrics
