API Reference
=============

Public classes and functions are re-exported from :mod:`koopman_graph`. Detailed
documentation is organized by module below.

Package
-------

.. automodule:: koopman_graph
   :members: __version__
   :no-index:

Model
-----

.. automodule:: koopman_graph.model
   :members:
   :exclude-members: EvaluationResult, GATEncoder, GNNDecoder, GNNEncoder, GraphSnapshotSequence, KoopmanOperator
   :show-inheritance:

Encoders
--------

.. automodule:: koopman_graph.encoder
   :members:
   :show-inheritance:

Decoder
-------

.. automodule:: koopman_graph.decoder
   :members:
   :show-inheritance:

Physics-Informed Observables
----------------------------

.. automodule:: koopman_graph.observables
   :members:
   :show-inheritance:

Koopman Operator
----------------

.. automodule:: koopman_graph.operator
   :members:
   :show-inheritance:

Continuous-Time Operator
------------------------

.. automodule:: koopman_graph.continuous
   :members:
   :show-inheritance:

Spectral Analysis
-----------------

.. automodule:: koopman_graph.analysis
   :members:
   :show-inheritance:

Baselines
---------

.. automodule:: koopman_graph.baselines
   :members:
   :show-inheritance:

Data Utilities
--------------

.. automodule:: koopman_graph.data
   :members:
   :show-inheritance:

Losses
------

.. automodule:: koopman_graph.losses
   :members:
   :exclude-members: GraphSnapshotSequence
   :show-inheritance:

Training
--------

.. automodule:: koopman_graph.training
   :members:
   :exclude-members: GraphSnapshotSequence
   :show-inheritance:

Metrics
-------

.. automodule:: koopman_graph.metrics
   :members:
   :show-inheritance:

Online Adaptation
-----------------

.. automodule:: koopman_graph.adaptation
   :members:
   :show-inheritance:

RL Environment
--------------

.. automodule:: koopman_graph.env
   :members:
   :show-inheritance:

Serialization
-------------

.. automodule:: koopman_graph.serialization
   :members:
   :show-inheritance:

Datasets
--------

.. automodule:: koopman_graph.datasets
   :members:
   :show-inheritance:

.. automodule:: koopman_graph.datasets.synthetic
   :members:
   :show-inheritance:

.. automodule:: koopman_graph.datasets.grid
   :members:
   :show-inheritance:

.. automodule:: koopman_graph.datasets.ieee118
   :members:
   :show-inheritance:

.. automodule:: koopman_graph.datasets.metr_la
   :members:
   :show-inheritance:
