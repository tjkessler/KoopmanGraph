"""Coverage and error-path tests for :mod:`koopman_graph.distributed`."""

from __future__ import annotations

import pytest


def test_distributed_lazy_exports() -> None:
    """Package ``__getattr__`` resolves Ray/Lightning/Dask names."""
    import koopman_graph.distributed as distributed_pkg

    assert callable(distributed_pkg.fit_ensemble_with_ray)
    assert callable(distributed_pkg.run_ray_train_fit_loop)
    assert callable(distributed_pkg.materialize_sequences)
    assert callable(distributed_pkg.materialize_window_index_list)
    try:
        cls = distributed_pkg.KoopmanLightningModule
    except ImportError:
        cls = None
    else:
        assert cls is not None
    with pytest.raises(AttributeError, match="no attribute"):
        _ = distributed_pkg.not_a_real_symbol
