"""Coverage and error-path tests for :mod:`koopman_graph.baselines`."""

from __future__ import annotations

import pytest


def test_agcrn_mtgnn_graphcast_constructor_guards() -> None:
    """Constructor / set_num_nodes / teaching-mesh validation branches."""
    from koopman_graph.baselines.gnn import AGCRNBaseline, MTGNNBaseline
    from koopman_graph.baselines.gnn.graphcast import build_teaching_mesh_edge_index

    with pytest.raises(ValueError, match="embed_dim must be positive"):
        AGCRNBaseline(1, 4, 1, embed_dim=0)
    agcrn = AGCRNBaseline(1, 4, 1, embed_dim=2, num_nodes=3)
    with pytest.raises(ValueError, match="num_nodes must be positive"):
        agcrn.set_num_nodes(0)
    with pytest.raises(ValueError, match="node embeddings are static"):
        agcrn.set_num_nodes(4)

    with pytest.raises(ValueError, match="embed_dim must be positive"):
        MTGNNBaseline(1, 4, 1, embed_dim=0)
    with pytest.raises(ValueError, match="num_layers must be positive"):
        MTGNNBaseline(1, 4, 1, num_layers=0)
    mtgnn = MTGNNBaseline(1, 4, 1, embed_dim=2, num_layers=1, num_nodes=2)
    with pytest.raises(ValueError, match="node embeddings are static"):
        mtgnn.set_num_nodes(5)

    with pytest.raises(ValueError, match="teaching mesh requires"):
        build_teaching_mesh_edge_index(num_lat=1, num_lon=2)
