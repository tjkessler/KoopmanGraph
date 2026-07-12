Quickstart
==========

This example trains a :class:`~koopman_graph.model.GraphKoopmanModel` on a
synthetic spatiotemporal graph and predicts future snapshots. It follows the
core KoopmanGraph workflow: encode → linear Koopman advance → decode.

Generate data
-------------

Use the built-in synthetic benchmark (Laplacian diffusion on a path graph):

.. code-block:: python

   from koopman_graph.datasets import SyntheticDynamicGraphBenchmark

   data_sequence = SyntheticDynamicGraphBenchmark.generate(
       num_nodes=20,
       num_timesteps=30,
       in_channels=3,
       seed=42,
       noise_std=0.01,
   )
   print(f"Training sequence: {data_sequence.num_timesteps} snapshots")

Build the model
---------------

.. code-block:: python

   import torch
   from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel

   in_channels = 3
   hidden_channels = 64
   latent_dim = 64
   out_channels = 3

   encoder = GNNEncoder(in_channels, hidden_channels, latent_dim)
   decoder = GNNDecoder(latent_dim, hidden_channels, out_channels)
   model = GraphKoopmanModel(
       encoder=encoder,
       decoder=decoder,
       latent_dim=latent_dim,
       time_step=0.1,
   )

Train
-----

.. code-block:: python

   torch.manual_seed(0)
   history = model.fit(data_sequence, epochs=20, lr=1e-3)
   print(f"Final training loss: {history.loss[-1]:.6f}")

Predict
-------

Roll out from the first snapshot in the sequence:

.. code-block:: python

   initial_graph = data_sequence[0]
   future_graphs = model.predict(initial_graph, steps=5)
   print(f"Predicted {len(future_graphs)} future snapshots")
   print(f"First prediction shape: {future_graphs[0].x.shape}")

Save and reload
---------------

Persist a trained model (weights and architecture) to a ``.pt`` checkpoint and
reload it without reconstructing encoder/decoder classes manually:

.. code-block:: python

   model.save("checkpoints/synthetic_model.pt")
   loaded_model = GraphKoopmanModel.load("checkpoints/synthetic_model.pt")
   future_graphs = loaded_model.predict(initial_graph, steps=5)

During training you can optionally restore or persist the lowest-loss epoch:

.. code-block:: python

   history = model.fit(
       data_sequence,
       epochs=20,
       restore_best_weights=True,
       checkpoint_path="checkpoints/best_model.pt",
   )
   print(f"Best epoch: {history.best_epoch}, loss: {history.best_loss:.6f}")

Advanced training options
-------------------------

:meth:`~koopman_graph.model.GraphKoopmanModel.fit` also supports learning-rate
schedulers, per-term loss history, multi-origin rollout loss, and multiple
training trajectories:

.. code-block:: python

   from torch.optim.lr_scheduler import StepLR
   from koopman_graph.training import constant_loss_weights

   history = model.fit(
       [trajectory_a, trajectory_b],
       epochs=50,
       lr_scheduler=lambda optim: StepLR(optim, step_size=10, gamma=0.5),
       rollout_start_indices="all",
       loss_weights=constant_loss_weights(reconstruction=1.0, rollout=0.5),
   )
   print(history.reconstruction_loss[-1], history.rollout_loss[-1])

For longer trajectories, opt into fixed-length window mini-batches. By default,
every valid window is shuffled and used once per epoch; set
``windows_per_epoch`` to cap the work:

.. code-block:: python

   history = model.fit(
       data_sequence,
       epochs=50,
       window_length=12,
       batch_size=8,
       windows_per_epoch=64,
       window_seed=42,
   )

This performs one optimizer update per window batch. Leaving
``window_length=None`` preserves the full-sequence, one-update-per-epoch
behavior.

Complete script
---------------

Copy and run this end-to-end script after :doc:`installation`:

.. code-block:: python

   import torch
   from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
   from koopman_graph.datasets import SyntheticDynamicGraphBenchmark

   data_sequence = SyntheticDynamicGraphBenchmark.generate(
       num_nodes=20,
       num_timesteps=30,
       in_channels=3,
       seed=42,
       noise_std=0.01,
   )

   in_channels = 3
   hidden_channels = 64
   latent_dim = 64
   out_channels = 3

   encoder = GNNEncoder(in_channels, hidden_channels, latent_dim)
   decoder = GNNDecoder(latent_dim, hidden_channels, out_channels)
   model = GraphKoopmanModel(
       encoder=encoder,
       decoder=decoder,
       latent_dim=latent_dim,
       time_step=0.1,
   )

   torch.manual_seed(0)
   history = model.fit(data_sequence, epochs=20, lr=1e-3)

   initial_graph = data_sequence[0]
   future_graphs = model.predict(initial_graph, steps=5)

   print(f"Snapshots: {data_sequence.num_timesteps}, loss: {history.loss[-1]:.6f}")
   print(f"Predictions: {len(future_graphs)}, shape: {future_graphs[0].x.shape}")

Expected output (values may vary slightly by platform):

.. code-block:: text

   Snapshots: 30, loss: <float>
   Predictions: 5, shape: torch.Size([20, 3])

Learn more
----------

- :doc:`api` — full API reference
- `Synthetic graph dynamics tutorial
  <https://github.com/tjkessler/KoopmanGraph/blob/main/examples/01_synthetic_graph.ipynb>`_
  — end-to-end Jupyter notebook with plots
- `SyntheticDynamicGraphBenchmark
  <https://github.com/tjkessler/KoopmanGraph/blob/main/src/koopman_graph/datasets/synthetic.py>`_
  — benchmark parameters and dynamics
