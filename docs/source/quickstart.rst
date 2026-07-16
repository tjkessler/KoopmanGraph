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

Operator stability modes
------------------------

``GraphKoopmanModel`` accepts ``koopman_parameterization`` to control how the
Koopman matrix **K** (or continuous generator **L**) is stored. Modes fall
into two categories:

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * - Mode
     - Category
     - When to use
   * - ``"dense"``
     - Soft
     - Default. Maximum flexibility; add ``LossWeights(..., eigenvalue=...)`` for empirical stability.
   * - ``"odo"``
     - Soft
     - Cayley ODO factorization; bounds diagonal factors only — assembled **K** is not structurally stable. Pair with eigenvalue loss for long rollouts.
   * - ``"schur"``, ``"dissipative"``, ``"lyapunov"``
     - Structural
     - Eigenvalues forced inside the unit disk by construction; use for 200+ step rollouts without retuning.

Soft regularization example (loss-based stability):

.. code-block:: python

   from koopman_graph.training import LossWeights

   model = GraphKoopmanModel(
       encoder=encoder,
       decoder=decoder,
       latent_dim=latent_dim,
       time_step=0.1,
       koopman_parameterization="odo",
       koopman_max_spectral_radius=1.0,
   )
   history = model.fit(
       data_sequence,
       epochs=50,
       loss_weights=LossWeights(reconstruction=1.0, eigenvalue=0.1),
   )

Structural guarantee example (certified stable **K**):

.. code-block:: python

   model = GraphKoopmanModel(
       encoder=encoder,
       decoder=decoder,
       latent_dim=latent_dim,
       time_step=0.1,
       koopman_parameterization="lyapunov",
   )
   cert = model.koopman.stability_certificate()
   print(cert["margin"], model.koopman.spectral_radius())

For ``"odo"``, :meth:`~koopman_graph.operator.KoopmanOperator.spectral_radius`
reports a diagonal-factor bound, not ``max |λᵢ(K)|``. See
``examples/08_loss_stability.ipynb`` (soft modes) and
``examples/11_long_horizon_stability.ipynb`` (structural modes).

Continuous-time dynamics
------------------------

For irregularly sampled telemetry, set ``dynamics_mode="continuous"`` so the
model learns a generator :math:`L` with :math:`K(\Delta t) = \exp(L \Delta t)`.
Attach monotone ``timestamps`` to the sequence and forecast at arbitrary query
times with :meth:`~koopman_graph.model.GraphKoopmanModel.predict_at`:

.. code-block:: python

   import torch
   from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
   from koopman_graph.data import GraphSnapshotSequence

   # sequence must carry strictly increasing timestamps (one per snapshot)
   model = GraphKoopmanModel(
       encoder=encoder,
       decoder=decoder,
       latent_dim=latent_dim,
       time_step=0.1,
       dynamics_mode="continuous",
   )
   history = model.fit(sequence_with_timestamps, epochs=30, lr=1e-3)

   query_times = [0.15, 0.37, 0.82]
   future_graphs = model.predict_at(sequence_with_timestamps[0], query_times=query_times)

Discrete mode (the default) requires uniform :attr:`~koopman_graph.model.GraphKoopmanModel.time_step`
increments; use continuous mode when sampling intervals vary. See
``examples/12_irregular_sampling_continuous_time.ipynb``.

Partial node observations
~~~~~~~~~~~~~~~~~~~~~~~~~

When sensors report only a subset of nodes at each timestamp, attach boolean
``observation_masks`` with shape ``(num_timesteps, num_nodes)`` (``True`` =
observed). Training averages reconstruction loss over ``mask[t+1]`` and
consistency losses over ``mask[t] & mask[t+1]``. :meth:`~koopman_graph.model.GraphKoopmanModel.predict_at`
still returns full-graph predictions; use sequence masks with
:meth:`~koopman_graph.model.GraphKoopmanModel.evaluate` for masked scoring.
This path is validated for static topology only.

.. code-block:: python

   masks = torch.ones(num_timesteps, num_nodes, dtype=torch.bool)
   masks[:, ::4] = False  # example dropout pattern
   sequence = GraphSnapshotSequence(
       snapshots,
       timestamps=timestamps,
       observation_masks=masks,
   )

Physics-informed observables
----------------------------

Prepend domain features to the GNN latent with ``physics_preset`` or a custom
``physics_lifting_fn``. Custom callables are not serialized — re-supply them on
:meth:`~koopman_graph.model.GraphKoopmanModel.load`. See
``examples/14_physics_informed_diffusion.ipynb`` for Laplacian presets and a
west/north directional custom function (absolute neighbor states) with save/load
round-trip.

Online adaptation
-----------------

After offline training, freeze the GNN encoder and adapt only the dense Koopman
operator with recursive least squares (RLS) as new snapshots arrive:

.. code-block:: python

   model.fit(data_sequence, epochs=50)
   model.enable_online_adaptation(forgetting_factor=0.99)

   for snapshot_t, snapshot_tp1 in zip(data_sequence[:-1], data_sequence[1:]):
       result = model.adapt_step(snapshot_t, snapshot_tp1)
       print(result.operator_change_norm.item())

``adapt_step`` encodes each pair with the frozen encoder and updates ``K`` (or
the continuous generator) in place. Requires ``koopman_parameterization="dense"``.
See ``examples/13_online_adaptation_traffic_drift.ipynb``.

**Discrete vs. continuous RLS fidelity.** Discrete models (the default) adapt
``K`` directly and are exact for the fitted row convention. Continuous models
fit a discrete propagator ``K(Δt)`` per interval and write back a generator via
``L ≈ logm(K(Δt)) / Δt`` with control scaling ``B̃ ≈ B(Δt) / Δt``. These
approximations differ from the Van Loan integration used in
:meth:`~koopman_graph.continuous.ContinuousKoopmanOperator.advance` and degrade
for large or varying ``Δt`` and for controlled dynamics. Use discrete RLS when
sampling is uniform; treat continuous RLS as a local linearization only.

Latent-space RL environment
---------------------------

After training a controlled model (``control_dim > 0``), wrap it as a
Gymnasium environment for closed-loop control in latent space. Install the
optional RL dependencies first:

.. code-block:: bash

   pip install "koopman-graph[rl]"

.. code-block:: python

   from koopman_graph import GraphKoopmanModel

   def voltage_reward(snapshot, step_index):
       target = 1.0
       vm = snapshot.x[:, 0]
       return float(-((vm - target) ** 2).mean())

   env = model.to_latent_env(
       data_sequence,
       voltage_reward,
       max_episode_steps=20,
       random_start=False,
       start_index=0,
   )
   observation, info = env.reset(seed=0)
   action = env.action_space.sample()
   observation, reward, terminated, truncated, info = env.step(action)

Observations are flattened latent vectors with shape
``(num_nodes * latent_dim,)``. Reshape with
:meth:`~koopman_graph.env.GraphKoopmanEnv.reshape_observation` when needed.
Actions are global controls clipped to ``[-1, 1]`` by default. The encoder and
decoder stay frozen during RL; only the Koopman control input changes the latent
transition.

**Continuous-time stepping.** For ``dynamics_mode="continuous"``, pass
``delta_t`` to integrate the generator over a custom horizon per ``step``
(defaults to ``model.time_step`` when omitted)::

   env = model.to_latent_env(
       data_sequence,
       voltage_reward,
       delta_t=0.25,  # advance by 0.25 time units each step
       max_episode_steps=20,
   )

Discrete models reject a ``delta_t`` that differs from ``time_step``. See
``examples/15_closed_loop_voltage_control_rl.ipynb`` for a PPO demo on IEEE
118-bus voltage regulation.

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
