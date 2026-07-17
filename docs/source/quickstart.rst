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

The constructor factory-builds a discrete :class:`~koopman_graph.operators.KoopmanOperator`
from string-mode settings (``dynamics_mode``, ``koopman_parameterization``, …).
Pass ``koopman="graph"`` for a networked
:class:`~koopman_graph.operators.GraphKoopmanOperator` (discrete only) so
``edge_index`` enters the linear step. To inject a pre-built or custom operator
instead, pass ``koopman=...`` (module) and leave those factory kwargs at their
defaults. Custom operators are not checkpoint round-trippable; see the
architecture page.

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
training trajectories. Multi-trajectory input must use
:class:`~koopman_graph.data.MultiTrajectory` (or
:func:`~koopman_graph.data.as_multi_trajectory`):

.. code-block:: python

   from torch.optim.lr_scheduler import StepLR
   from koopman_graph import MultiTrajectory
   from koopman_graph.training import constant_loss_weights

   history = model.fit(
       MultiTrajectory((trajectory_a, trajectory_b)),
       epochs=50,
       lr_scheduler=lambda optim: StepLR(optim, step_size=10, gamma=0.5),
       rollout_start_indices="all",
       loss_weights=constant_loss_weights(reconstruction=1.0, rollout=0.5),
   )
   print(history.reconstruction_loss[-1], history.rollout_loss[-1])

A plain ``list`` of ``Data`` snapshots is always one trajectory. A bare list of
:class:`~koopman_graph.data.GraphSnapshotSequence` is rejected (wrap it in
``MultiTrajectory``). Empty lists and mixed ``GraphSnapshotSequence`` /
``Data`` lists raise ``ValueError``. See :doc:`architecture` for the full
discrimination rules.

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
     - Cayley ODO factorization; orthogonal factors imply ``ρ(K) ≤ max|dᵢ| ≤ max_spectral_radius`` via the operator 2-norm, but without a strict ε-interior certificate. Prefer structural modes for long-horizon guarantees; continuous ODO still needs eigenvalue loss on the true spectrum.
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
   print(cert.margin, model.koopman.bound_metric(), model.koopman.spectral_radius())

For ``"odo"``, :meth:`~koopman_graph.operators.KoopmanOperator.bound_metric`
(and continuous :meth:`~koopman_graph.operators.ContinuousKoopmanOperator.bound_metric`)
reports the **diagonal-factor** bound, not the true spectrum of assembled
``K`` / ``L``. Use :meth:`~koopman_graph.operators.KoopmanOperator.spectral_radius`
or :meth:`~koopman_graph.operators.ContinuousKoopmanOperator.max_real_part`
for ``\\max |\\lambda_i|`` / ``\\max \\operatorname{Re}(\\lambda_i)`` via
``eigvals``. See ``examples/08_loss_stability.ipynb`` (soft modes) and
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

When ``delta_t`` is omitted on continuous model paths (``forward``, training
pair losses without timestamps, and the Gymnasium env), the interval resolves
to :attr:`~koopman_graph.model.GraphKoopmanModel.time_step` via
:meth:`~koopman_graph.model.GraphKoopmanModel.resolve_delta_t`. Standalone
:class:`~koopman_graph.operators.ContinuousKoopmanOperator` calls soft-default
to ``1.0``; pass an explicit interval outside a model.

Partial node observations
~~~~~~~~~~~~~~~~~~~~~~~~~

When sensors report only a subset of nodes at each timestamp, attach boolean
``observation_masks`` with shape ``(num_timesteps, num_nodes)`` (``True`` =
observed). Training averages reconstruction loss over ``mask[t+1]`` and
consistency losses over ``mask[t] & mask[t+1]``. :meth:`~koopman_graph.model.GraphKoopmanModel.predict_at`
still returns full-graph predictions; use sequence masks with
:meth:`~koopman_graph.model.GraphKoopmanModel.evaluate` for masked scoring.
This path is validated for static topology only.

For **imputation** of masked sensors (not just masked loss), use
:class:`~koopman_graph.adaptation.KoopmanObserver` — a latent-space Kalman
filter / smoother. Default ``observation_model="latent_encode"`` uses a
selection of identity blocks on observed nodes (heuristic, fast);
``"decoder_jacobian"`` is an EKF-style local linearization with documented
cost and validity limits. See
``examples/25_kalman_koopman_state_estimation.ipynb``.

.. code-block:: python

   masks = torch.ones(num_timesteps, num_nodes, dtype=torch.bool)
   masks[:, ::4] = False  # example dropout pattern
   sequence = GraphSnapshotSequence(
       snapshots,
       timestamps=timestamps,
       observation_masks=masks,
   )

Delay embeddings
~~~~~~~~~~~~~~~~

For partially observed systems, stack the last ``n_delays`` snapshots into the
encoder channels (delay-coordinate / Takens-style lifting; related Hankel
constructions appear in HAVOK and Hankel-DMD). Size the base encoder yourself as
``in_channels = n_delays * feature_dim`` (composition; layers are not rebuilt),
then pass ``n_delays`` to the model. Training uses teacher-forced history;
``predict`` encodes the provided observation window once and advances in latent
space (decoded forecasts are not recycled into the delay buffer by default).
See ``examples/17_delay_embedding_partial_observability.ipynb``.

.. code-block:: python

   n_delays = 5
   feature_dim = 4
   encoder = GNNEncoder(n_delays * feature_dim, 64, 32)
   decoder = GNNDecoder(32, 64, feature_dim)
   model = GraphKoopmanModel(
       encoder=encoder,
       decoder=decoder,
       latent_dim=32,
       time_step=1.0,
       n_delays=n_delays,
   )
   history = model.fit(sequence, epochs=50)
   # Optional prior observations (oldest → newest), excluding initial_graph:
   future = model.predict(
       sequence[10],
       steps=5,
       history=list(sequence[6:10]),
   )

Physics-informed observables
----------------------------

Prepend domain features to the GNN latent with ``physics_preset`` or a custom
``physics_lifting_fn`` (see :func:`~koopman_graph.observables.graph_laplacian_features`
and :mod:`koopman_graph.observables`). Custom callables are not serialized —
re-supply them on :meth:`~koopman_graph.model.GraphKoopmanModel.load`. See
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
fit a discrete propagator ``K(Δt)`` per interval and write back a generator
aligned with
:meth:`~koopman_graph.operators.ContinuousKoopmanOperator.advance`:
``L = logm(K(Δt)) / Δt`` when uncontrolled, and a Van Loan block-matrix inverse
when controlled. Prefer discrete RLS when sampling is uniform and a discrete
operator is acceptable. Continuous write-back can still degrade for very large
``Δt`` or when ``K(Δt)`` approaches a matrix-logarithm branch cut.

Historical note: earlier releases used a first-order controlled approximation
``B̃ ≈ B(Δt) / Δt`` that disagreed with Van Loan integration; that path was
replaced in the Phase 8 fidelity update.

Bilinear / control-affine control
---------------------------------

Default Koopman-with-control is additive (``z @ K.T + u @ B``), matching DMDc-style
linear predictors. For control-affine systems, set ``control_mode="bilinear"`` to
add state–control couplings ``sum_i u_i N_i z`` (optional low-rank
``bilinear_rank``). Continuous operators integrate piecewise-constant ``u`` with
an effective generator ``L_eff = L + sum_i u_i N_i`` via Van Loan. See
``examples/19_bilinear_control_koopman.ipynb`` (synthetic recovery plus networked
SIR contact-reduction with ``EpidemicNetworkBenchmark.expose_intervention_control``).

.. code-block:: python

   model = GraphKoopmanModel(
       encoder=encoder,
       decoder=decoder,
       latent_dim=32,
       time_step=1.0,
       control_dim=1,
       control_mode="bilinear",
       bilinear_rank=4,
   )

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
