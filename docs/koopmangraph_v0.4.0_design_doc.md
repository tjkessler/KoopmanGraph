# KoopmanGraph Extension Design Document

**Scope:** Design proposals addressing all fourteen literature-gap shortcomings identified in the KoopmanGraph repository review.
**Status:** Draft for maintainer review
**Target repo:** `tjkessler/KoopmanGraph` (v0.3.0 baseline)
**Conventions:** All proposals extend the existing public API style (`GraphKoopmanModel`, `GNNEncoder`/`GATEncoder`, `KoopmanOperator`/`ContinuousKoopmanOperator`, `koopman_graph.datasets`, `koopman_graph.analysis`) rather than introducing a parallel API surface. New symbols are additive and backward compatible; existing constructors gain optional keyword arguments with safe defaults, so `model.fit(...)` code written today continues to work unchanged.

Each section below follows the same structure: **Problem**, **Design**, **API sketch**, **Highlighting example** (a runnable-style snippet suitable for a new tutorial notebook), and **Validation / tests**. A prioritization roadmap is given at the end.

---

## 1. Delay (Hankel) embeddings for partial observability

### Problem
The encoder only ever consumes a single instantaneous snapshot `x_t`. Systems that are partially observed (missing state variables, sparse sensors) rely in the literature on delay-coordinate (Hankel/Takens) embeddings to reconstruct an effectively Markovian state from a window of past observations. KoopmanGraph currently has no notion of a temporal window at the encoder boundary.

### Design
Introduce a `DelayEmbeddingEncoder` wrapper that stacks the last `n_delays` snapshots' node features (respecting the shared graph topology) into a single augmented feature tensor before handing off to an existing `GNNEncoder`/`GATEncoder`. This is a *wrapper*, not a new base encoder, so it composes with any current or future encoder:

- `GraphSnapshotSequence` gains a `windowed(n_delays, stride=1)` method that returns overlapping windows of consecutive snapshots (raising if the underlying topology changes across a window unless `dynamic_topology=True`).
- `DelayEmbeddingEncoder(base_encoder, n_delays)` concatenates per-node features across the window along the channel dimension `(N, n_delays * F)` before calling `base_encoder`. Missing history at the start of a sequence is handled via zero-padding with a boolean `history_mask` passed through to the loss (reusing the existing `observation_masks` machinery).
- `GraphKoopmanModel` gains an optional `n_delays: int = 1` constructor argument; when `> 1` it automatically wraps whatever encoder is passed in with `DelayEmbeddingEncoder`, and `predict`/`fit` internally maintain a rolling buffer of the last `n_delays` *decoded* outputs during autoregressive rollout.

### API sketch
```python
class DelayEmbeddingEncoder(nn.Module):
    """Wraps a GNNEncoder/GATEncoder with Hankel/delay-coordinate lifting."""

    def __init__(self, base_encoder: nn.Module, n_delays: int = 3) -> None: ...

    def forward(
        self,
        x_window: Tensor,        # (n_delays, num_nodes, in_channels)
        edge_index: Tensor,
        edge_weight: Tensor | None = None,
    ) -> Tensor: ...             # (num_nodes, latent_dim)


class GraphSnapshotSequence:
    def windowed(
        self, n_delays: int, *, stride: int = 1, pad: bool = True
    ) -> "GraphSnapshotSequence": ...
```

### Highlighting example
```python
from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
from koopman_graph.datasets import IEEE118DynamicBenchmark

# Only every 3rd bus reports Vm/Va at each timestep -> partial observability.
data_sequence = IEEE118DynamicBenchmark.generate(num_timesteps=200, seed=7)
sparse_sequence = data_sequence.with_observation_masks(drop_fraction=0.6, seed=7)

encoder = GNNEncoder(in_channels=4, hidden_channels=64, latent_dim=32)
decoder = GNNDecoder(latent_dim=32, hidden_channels=64, out_channels=4)

model = GraphKoopmanModel(
    encoder=encoder, decoder=decoder, latent_dim=32,
    n_delays=5,              # <- Hankel/delay embedding of the last 5 snapshots
    time_step=1.0,
)
history = model.fit(sparse_sequence, epochs=50, lr=1e-3)

# Compare against n_delays=1 (no delay embedding) on the same masked data.
baseline = GraphKoopmanModel(
    encoder=GNNEncoder(4, 64, 32), decoder=GNNDecoder(32, 64, 4),
    latent_dim=32, n_delays=1, time_step=1.0,
)
baseline.fit(sparse_sequence, epochs=50, lr=1e-3)

print("Delay-embedded RMSE:", model.evaluate(sparse_sequence).rmse[-1])
print("Single-snapshot RMSE:", baseline.evaluate(sparse_sequence).rmse[-1])
```
Proposed new notebook: `examples/17_delay_embedding_partial_observability.ipynb`, showing forecast-error degradation as `drop_fraction` increases, with and without delay embedding, on IEEE118 and METR-LA.

### Validation / tests
`tests/test_delay_embedding.py`: shape checks for windowing at sequence boundaries, gradient flow through padded windows, round-trip serialization of `n_delays` in `save`/`load`, and a synthetic Lorenz-96-on-a-ring regression test where `n_delays > 1` provably reduces RMSE relative to `n_delays = 1` under partial observability (fixed seed, asserted improvement threshold).

---

## 2. Networked (spatially-coupled) Koopman operator

### Problem
`KoopmanOperator.forward` applies `z_next = z @ K.T` identically and independently to every node's latent vector — there is no cross-node term inside the linear step itself. All spatial coupling therefore lives in the (nonlinear, single-shot-at-t0) encoder/decoder, which is a meaningful departure from the "distributed geometric Koopman operator" and "compositional Koopman operator" literature, where the operator's block structure directly encodes inter-node coupling and can respond to topology change *during* rollout, not just at decode time.

### Design
Add a `GraphKoopmanOperator` variant that operates on the *full* stacked latent state `Z ∈ R^{N×d}` rather than per-node, using a learnable, edge-index-aware linear map:

```
vec(Z_{t+1}) = (I_N ⊗ K_self + Â ⊗ K_nbr) vec(Z_t)
```

where `Â` is the (fixed per-sequence, or per-snapshot for dynamic topology) normalized adjacency, `K_self ∈ R^{d×d}` governs each node's own linear evolution, and `K_nbr ∈ R^{d×d}` governs the linear contribution propagated from neighbors — i.e. a *linear* message-passing step, sparse and implemented via `torch_geometric` scatter operations rather than the dense Kronecker form (which is written above only to state the operator precisely). This keeps the operator genuinely linear (required for spectral analysis) while letting it react to `edge_index` at every rollout step, which finally makes "dynamic topology" support affect the actual forward dynamics, not only the decode readout.

For very large graphs, expose a `sparsity: Literal["dense", "block_diagonal", "distributed"]` option that mirrors the distributed geometric Koopman formulation (block-diagonal `K` matrices per community, computed via existing graph-partitioning utilities in `graph_utils.py`).

### API sketch
```python
class GraphKoopmanOperator(nn.Module):
    """Linear operator with an explicit self-term and a neighbor-coupling term."""

    def __init__(
        self,
        latent_dim: int,
        *,
        parameterization: Parameterization = "dense",  # reuses schur/dissipative/lyapunov
        sparsity: Literal["dense", "block_diagonal", "distributed"] = "dense",
        max_spectral_radius: float = 1.0,
    ) -> None: ...

    def forward(
        self, z: Tensor, edge_index: Tensor, edge_weight: Tensor | None = None
    ) -> Tensor: ...  # (num_nodes, latent_dim), couples across nodes via edge_index

    def spectrum(self, edge_index: Tensor, num_nodes: int) -> KoopmanSpectrum:
        """Eigendecomposition of the *effective* N*d x N*d coupled operator
        for a given topology (computed via sparse eigensolvers for large N)."""
```

`GraphKoopmanModel(..., koopman="graph")` selects this operator; `koopman="pernode"` keeps today's behavior for backward compatibility. Because the operator now needs `edge_index` at every step, `predict()`/`_rollout()` are updated to consult `future_topologies` inside the linear-advance loop (not only at decode), which directly closes the dynamic-topology gap.

### Highlighting example
```python
from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
from koopman_graph.datasets import SyntheticDynamicGraphBenchmark
from torch_geometric.data import Data

# Contact network rewires halfway through the rollout (e.g. epidemic contact change).
seq = SyntheticDynamicGraphBenchmark.generate(num_nodes=30, num_timesteps=40, seed=1)
rewired_topologies = [Data(edge_index=seq.rewired_edge_index(t)) for t in range(20, 40)]

model = GraphKoopmanModel(
    encoder=GNNEncoder(3, 64, 32), decoder=GNNDecoder(32, 64, 3),
    latent_dim=32, koopman="graph", time_step=0.1,
)
model.fit(seq, epochs=30, lr=1e-3)

# Rollout that must react to a mid-sequence rewiring event during the LINEAR step,
# not only at decode.
forecasts = model.predict(seq[19], steps=20, future_topologies=rewired_topologies)
print("Effective spectral radius before/after rewiring:",
      model.koopman.spectrum(seq[0].edge_index, num_nodes=30).radius,
      model.koopman.spectrum(rewired_topologies[-1].edge_index, num_nodes=30).radius)
```
Proposed new notebook: `examples/18_networked_koopman_dynamic_topology.ipynb`, directly comparing `koopman="pernode"` vs `koopman="graph"` forecast error on a mid-sequence rewiring event.

### Validation / tests
`tests/test_graph_operator.py`: verify `K_nbr=0` recovers today's per-node operator exactly (regression safety); verify a hand-constructed two-node coupled linear system is recovered exactly by `GraphKoopmanOperator` in the noiseless case; verify rollout forecasts differ from `koopman="pernode"` under a mid-sequence rewiring (this is the behavioral fix being validated); sparse-vs-dense eigenvalue agreement test for `spectrum()`.

---

## 3. Bilinear / control-affine Koopman with control

### Problem
Control enters only additively: `z_{t+1} = K z_t + B u_t`, with constant `B`. The literature (Korda & Mezić; Proctor et al. DMDc) shows this is insufficient for control-affine nonlinear systems, which naturally produce **bilinear** lifted dynamics with state–control cross terms.

### Design
Extend `KoopmanOperator`/`ContinuousKoopmanOperator` with an optional bilinear term:

```
z_{t+1} = K z_t + B u_t + sum_i u_t[i] * N_i z_t
```

where each `N_i ∈ R^{d×d}` is a learnable per-control-channel bilinear coupling matrix. To avoid parameter blow-up for high control dimension, `N_i` is optionally low-rank factored (`N_i = P_i Q_i^T`, rank `r << d`), configurable via `bilinear_rank`.

### API sketch
```python
class KoopmanOperator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        *,
        control_dim: int = 0,
        control_mode: Literal["additive", "bilinear"] = "additive",
        bilinear_rank: int | None = None,   # None = full-rank N_i
        ...,
    ) -> None: ...

    def forward(self, z: Tensor, u: Tensor | None = None) -> Tensor:
        """z_next = z @ K.T + u @ B                       (control_mode='additive')
                   + sum_i u[..., i:i+1] * (z @ N_i.T)     (control_mode='bilinear')
        """
```

### Highlighting example
```python
import torch
from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
from koopman_graph.datasets import IEEE118DynamicBenchmark

# Reactive power injection u_t multiplicatively affects voltage dynamics
# (a textbook control-affine setting where additive-only B underfits).
seq = IEEE118DynamicBenchmark.generate(
    num_timesteps=300, expose_load_ramp_control=True, seed=3,
)

model_additive = GraphKoopmanModel(
    encoder=GNNEncoder(4, 64, 32), decoder=GNNDecoder(32, 64, 4),
    latent_dim=32, control_dim=1, control_mode="additive", time_step=1.0,
)
model_bilinear = GraphKoopmanModel(
    encoder=GNNEncoder(4, 64, 32), decoder=GNNDecoder(32, 64, 4),
    latent_dim=32, control_dim=1, control_mode="bilinear", bilinear_rank=4,
    time_step=1.0,
)
for m in (model_additive, model_bilinear):
    m.fit(seq, epochs=60, lr=1e-3)

print("Additive-control RMSE :", model_additive.evaluate(seq).rmse[-1])
print("Bilinear-control RMSE :", model_bilinear.evaluate(seq).rmse[-1])
```
Proposed new notebook: `examples/19_bilinear_control_koopman.ipynb`, extending the existing IEEE118 RL notebook (`15_closed_loop_voltage_control_rl.ipynb`) to show the bilinear operator improving closed-loop tracking of a 1.0 p.u. voltage setpoint under larger control authority than the additive model can represent.

### Validation / tests
`tests/test_control.py` additions: exact recovery of a synthetic bilinear system (`z_{t+1} = Kz_t + (u_t N) z_t`) that an additive-only model provably cannot fit below a fixed loss floor, confirming the bilinear model closes that gap; gradient/shape tests for `bilinear_rank`; serialization round-trip for the new control mode.

---

## 4. Continuous-spectrum handling (frequency-parameterized eigenvalues)

### Problem
`ContinuousKoopmanOperator` learns one fixed generator matrix. Systems with continuous spectra (e.g., nonlinear oscillators whose frequency depends on amplitude) are not specifically addressed; the literature handles this via an auxiliary network that maps the current latent state to instantaneous eigenvalues.

### Design
Add a `parameterization="auxiliary_spectral"` mode to `ContinuousKoopmanOperator`. Instead of a fixed generator `L`, an auxiliary MLP `ω_θ(z) -> (real_part, imag_freq)` per (conjugate-pair) mode predicts state-dependent eigenvalues, and the generator is assembled per-forward-pass as a block-diagonal matrix of 2x2 rotation-scaling blocks `[[a, -b], [b, a]]` from `ω_θ`. This keeps the *instantaneous* dynamics linear (satisfying local Koopman linearity) while letting the effective spectrum vary smoothly with state/amplitude, matching the auxiliary-network construction from the deep-learning-for-universal-linear-embeddings literature.

### API sketch
```python
class ContinuousKoopmanOperator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        *,
        parameterization: Parameterization = "dense",  # add "auxiliary_spectral"
        num_conjugate_pairs: int | None = None,  # required for auxiliary_spectral
        aux_hidden_dim: int = 32,
        ...,
    ) -> None: ...

    def generator_at(self, z: Tensor) -> Tensor:
        """State-dependent generator L(z) for parameterization='auxiliary_spectral';
        falls back to the fixed L for other modes."""
```

### Highlighting example
```python
import torch
from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
from koopman_graph.operators import ContinuousKoopmanOperator
from koopman_graph.datasets import SyntheticDynamicGraphBenchmark

# Amplitude-dependent oscillator network (continuous spectrum by construction).
seq = SyntheticDynamicGraphBenchmark.generate(
    num_nodes=12, num_timesteps=200, dynamics="amplitude_dependent_oscillator", seed=5,
)

koopman = ContinuousKoopmanOperator(
    latent_dim=8, parameterization="auxiliary_spectral", num_conjugate_pairs=4,
)
model = GraphKoopmanModel(
    encoder=GNNEncoder(2, 32, 8), decoder=GNNDecoder(8, 32, 2),
    latent_dim=8, koopman=koopman, dynamics_mode="continuous", time_step=0.05,
)
model.fit(seq, epochs=80, lr=1e-3)

spec = model.spectrum(z=model.encode(seq[0]))
print("Instantaneous frequency at t=0:", spec.frequencies)
spec_late = model.spectrum(z=model.encode(seq[150]))
print("Instantaneous frequency at t=150 (should differ if amplitude changed):",
      spec_late.frequencies)
```
Proposed new notebook: `examples/20_continuous_spectrum_auxiliary_network.ipynb`, replicating the classic amplitude-dependent-frequency pendulum example from the deep-Koopman literature on a small oscillator network, showing frequency vs. amplitude recovered by the auxiliary network against ground truth.

### Validation / tests
`tests/test_continuous.py` additions: on a synthetic amplitude-dependent oscillator, verify `auxiliary_spectral` achieves materially lower long-horizon rollout error than a fixed-generator continuous operator at matched latent dimension; verify recovered instantaneous frequency correlates with true instantaneous amplitude (Pearson r above a fixed threshold, seeded).

---

## 5. Uncertainty quantification / probabilistic Koopman

### Problem
Every model in KoopmanGraph is deterministic. There is no predictive interval, ensembling, or variational latent state anywhere in the codebase, despite Deep Probabilistic Koopman and Kalman-augmented Koopman approaches being established for representing forecast uncertainty.

### Design
Two complementary, independently-usable mechanisms:

1. **Variational latent state.** `VariationalKoopmanOperator` wraps any base operator and treats `z_t` as the mean of a diagonal Gaussian `q(z_t) = N(μ_t, diag(σ_t²))`, propagated in closed form under linear dynamics (`μ_{t+1} = Kμ_t`, `Σ_{t+1} = KΣ_tK^T + Q`, with learnable process-noise `Q`). Training adds a KL term against a unit-variance prior at `t=0` and a Gaussian NLL reconstruction term in place of MSE.
2. **Deep ensembles.** `EnsembleGraphKoopmanModel(models: list[GraphKoopmanModel])` trains `M` independently-seeded models and aggregates `predict()`/`evaluate()` into mean + empirical quantiles, requiring no architecture change — the simplest path to calibrated intervals and a useful baseline against option 1.

Both report `predict_interval(initial_graph, steps, alpha=0.1) -> list[(Data, Data, Data)]` (lower, mean, upper), added to the public `ForecastModel` protocol as an optional method (`hasattr` checked, not required, to keep existing models valid).

### API sketch
```python
class VariationalKoopmanOperator(nn.Module):
    def __init__(self, base_operator: nn.Module, *, learnable_process_noise: bool = True): ...
    def forward(self, mu: Tensor, sigma2: Tensor) -> tuple[Tensor, Tensor]: ...

class EnsembleGraphKoopmanModel:
    def __init__(self, models: Sequence[GraphKoopmanModel]) -> None: ...
    def fit(self, data_sequence, **kwargs) -> list[TrainingHistory]: ...
    def predict_interval(
        self, initial_graph, steps: int, alpha: float = 0.1
    ) -> list[tuple[Data, Data, Data]]: ...
```

### Highlighting example
```python
from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
from koopman_graph.uq import EnsembleGraphKoopmanModel
from koopman_graph.datasets import MetrLaTrafficBenchmark

seq = MetrLaTrafficBenchmark.load(split="train")

models = [
    GraphKoopmanModel(
        encoder=GNNEncoder(1, 64, 32), decoder=GNNDecoder(32, 64, 1),
        latent_dim=32, time_step=5.0,
    )
    for _ in range(5)
]
ensemble = EnsembleGraphKoopmanModel(models)
ensemble.fit(seq, epochs=40, lr=1e-3, seeds=[0, 1, 2, 3, 4])

lower, mean, upper = zip(*ensemble.predict_interval(seq[0], steps=12, alpha=0.1))
coverage = ensemble.empirical_coverage(seq, alpha=0.1)  # fraction of true values in [lower, upper]
print(f"90% interval empirical coverage on held-out sensors: {coverage:.2%}")
```
Proposed new notebook: `examples/21_uncertainty_quantification.ipynb`, showing calibration curves (coverage vs. nominal `1-alpha`) on METR-LA and comparing ensemble vs. variational interval widths.

### Validation / tests
`tests/test_uq.py`: closed-form propagation check for `VariationalKoopmanOperator` on a scalar linear system against an analytic Gaussian solution; ensemble coverage test asserting empirical coverage is within a tolerance band of the nominal level on a synthetic system with known noise; serialization of ensemble members.

---

## 6. Richer physics-informed observables

### Problem
`koopman_graph.observables` ships exactly one preset (`graph_laplacian`). There is no PDE-residual loss, no Lie-derivative/generator-consistency constraint, and no symplectic/Hamiltonian structure preservation — all established mechanisms in the physics-informed Koopman literature.

### Design
Three additions, independently toggleable:

1. **More presets** in `PHYSICS_PRESETS`: `graph_gradient` (edge-difference features), `graph_curvature` (Laplacian-of-Laplacian), and a generic `polynomial(degree)` preset matching classical EDMD dictionaries at the observable level (complementary to item 7's baseline-side dictionary).
2. **Lie-derivative consistency loss.** `LieConsistencyLoss` penalizes `||∇_z φ(z) · f(x) − K φ(z)||` when a known (even approximate/partially-known) vector field `f` is supplied by the user via `physics_dynamics_fn`, directly implementing the Physics-Informed Koopman Network constraint.
3. **PDE-residual loss.** `PDEResidualLoss(pde_fn)` for grid-graph benchmarks accepts a callable that computes a finite-difference PDE residual on the decoded node field (e.g., diffusion/advection residual) and adds it as an auxiliary training term, implementing a SPIKE-style residual penalty without requiring full autograd-based PINN machinery.

### API sketch
```python
def polynomial_observables(degree: int) -> PhysicsLiftingFn: ...

class LieConsistencyLoss(nn.Module):
    def __init__(self, physics_dynamics_fn: Callable[[Data], Tensor]) -> None: ...
    def forward(self, z: Tensor, encoder_jacobian: Tensor, koopman_matrix: Tensor,
                dynamics_at_x: Tensor) -> Tensor: ...

class PDEResidualLoss(nn.Module):
    def __init__(self, pde_fn: Callable[[Tensor, Tensor], Tensor]) -> None: ...
    def forward(self, predicted_x: Tensor, edge_index: Tensor, dt: float) -> Tensor: ...
```

### Highlighting example
```python
from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel, LossWeights
from koopman_graph.datasets import GridDynamicGraphBenchmark
from koopman_graph.observables import graph_laplacian_features
from koopman_graph.losses import PDEResidualLoss

def diffusion_residual(x_pred, edge_index, dt, alpha=0.2):
    # residual of d x/dt - alpha * Laplacian(x) = 0, finite-differenced
    ...

seq = GridDynamicGraphBenchmark.generate(rows=8, cols=8, num_timesteps=60, seed=2)

model = GraphKoopmanModel(
    encoder=GNNEncoder(1, 32, 16), decoder=GNNDecoder(16, 32, 1),
    latent_dim=16, time_step=0.1,
    physics_preset="graph_laplacian",
    physics_dim=1,
)
history = model.fit(
    seq, epochs=60, lr=1e-3,
    extra_losses={"pde_residual": PDEResidualLoss(diffusion_residual)},
    loss_weights=LossWeights(reconstruction=1.0, forward=1.0, backward=0.5, pde_residual=0.3),
)
print("Final PDE residual loss term:", history.loss_terms["pde_residual"][-1])
```
Proposed new notebook: extends `examples/14_physics_informed_diffusion.ipynb` with a "PDE-residual" section, comparing rollout stability with vs. without the residual penalty at long horizons.

### Validation / tests
`tests/test_observables.py` additions: `LieConsistencyLoss` reduces to near-zero on a system where `physics_dynamics_fn` is exact; `PDEResidualLoss` correctly flags a deliberately wrong `pde_fn` (residual stays high); new presets pass the existing `validate_physics_output` shape contract.

---

## 7. Richer classical-baseline dictionaries (kernel / RBF EDMD)

### Problem
`EDMDBaseline` supports only degree-1/2 polynomial observables on flattened, topology-ignoring features — a small fraction of the dictionary flexibility in the classical EDMD/kernel-Koopman literature.

### Design
Add `dictionary: Literal["polynomial", "rbf", "kernel"]` to `EDMDBaseline`:
- `"rbf"`: fixed random-center or k-means-center radial basis functions, count configurable via `num_centers`.
- `"kernel"`: full kernel EDMD (Gaussian/polynomial kernel Gram-matrix regression), matching the operator-valued-RKHS line of work, provided as an O(T²) method suitable for the same small/medium benchmark scale the baselines already target.

### API sketch
```python
class EDMDBaseline(ClassicalBaseline):
    def __init__(
        self,
        *,
        dictionary: Literal["polynomial", "rbf", "kernel"] = "polynomial",
        polynomial_degree: Literal[1, 2] = 1,
        num_centers: int | None = None,       # for "rbf"
        kernel: Literal["gaussian", "polynomial"] | None = None,  # for "kernel"
        kernel_bandwidth: float | None = None,
    ) -> None: ...
```

### Highlighting example
```python
from koopman_graph.baselines import EDMDBaseline
from koopman_graph.datasets import MetrLaTrafficBenchmark

seq = MetrLaTrafficBenchmark.load(split="train")

poly = EDMDBaseline(dictionary="polynomial", polynomial_degree=2).fit(seq)
rbf = EDMDBaseline(dictionary="rbf", num_centers=200).fit(seq)
kernel = EDMDBaseline(dictionary="kernel", kernel="gaussian", kernel_bandwidth=1.0).fit(seq)

for name, m in [("poly-2", poly), ("rbf-200", rbf), ("gaussian-kernel", kernel)]:
    print(name, m.evaluate(seq).rmse)
```
Proposed update: `examples/03_traffic_network.ipynb` gains an extra baseline row comparing `GraphKoopmanModel` against all three EDMD dictionary variants, not just the current degree-1/2 polynomial default.

### Validation / tests
`tests/test_baselines.py` additions: `"rbf"`/`"kernel"` fit-then-evaluate smoke tests on the synthetic benchmark; numerical agreement check that `dictionary="kernel"` with a linear kernel reduces to ordinary DMD on a linear synthetic system.

---

## 8. Comparison against modern spatiotemporal GNN forecasters

### Problem
`paper.md`'s stated motivation contrasts KoopmanGraph with plain GNN forecasters that "learn nonlinear recurrent or autoregressive maps," but the only benchmarked baselines are vector DMD/EDMD. No STGCN/DCRNN/Graph WaveNet-style comparison exists, so that motivating claim is untested in-repo.

### Design
Add a `koopman_graph.baselines.gnn` submodule with lightweight, dependency-free reference implementations of three standard spatiotemporal GNN forecasters, sized appropriately for the existing benchmark scale (not intended to replace dedicated libraries, just to give an apples-to-apples in-repo comparator with the same `ForecastModel`/`fit`/`predict`/`evaluate` protocol):

- `STGCNBaseline` — spatial GCN + temporal gated convolution blocks.
- `DCRNNBaseline` — diffusion convolution + GRU recurrent core (directly relevant given the existing `AnisotropicAdvectionGridBenchmark`).
- `GraphWaveNetBaseline` — dilated causal convolutions + adaptive adjacency.

All three satisfy the existing `ForecastModel` protocol so they slot directly into `evaluate_forecast` and existing plotting utilities without any notebook-side special-casing.

### API sketch
```python
class STGCNBaseline(nn.Module, ForecastModel):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 *, num_st_blocks: int = 2, kernel_size: int = 3) -> None: ...

class DCRNNBaseline(nn.Module, ForecastModel):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 *, diffusion_steps: int = 2) -> None: ...

class GraphWaveNetBaseline(nn.Module, ForecastModel):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 *, num_layers: int = 4, adaptive_adj: bool = True) -> None: ...
```

### Highlighting example
```python
from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
from koopman_graph.baselines.gnn import DCRNNBaseline, GraphWaveNetBaseline, STGCNBaseline
from koopman_graph.datasets import MetrLaTrafficBenchmark
from koopman_graph import evaluate_forecast

seq_train = MetrLaTrafficBenchmark.load(split="train")
seq_test = MetrLaTrafficBenchmark.load(split="test")

koopman_model = GraphKoopmanModel(
    encoder=GNNEncoder(1, 64, 32), decoder=GNNDecoder(32, 64, 1),
    latent_dim=32, time_step=5.0,
).fit(seq_train, epochs=40, lr=1e-3)

candidates = {
    "GraphKoopmanModel": koopman_model,
    "STGCN": STGCNBaseline(1, 64, 1).fit(seq_train, epochs=40, lr=1e-3),
    "DCRNN": DCRNNBaseline(1, 64, 1).fit(seq_train, epochs=40, lr=1e-3),
    "GraphWaveNet": GraphWaveNetBaseline(1, 64, 1).fit(seq_train, epochs=40, lr=1e-3),
}
for name, model in candidates.items():
    result = evaluate_forecast(model, seq_test, horizons=[1, 3, 6, 12])
    print(name, result.rmse)
```
Proposed new notebook: `examples/22_gnn_forecaster_comparison.ipynb`, directly answering the "why Koopman-structured latent dynamics over a plain spatiotemporal GNN" question the JOSS paper poses but does not currently test, on METR-LA and IEEE118.

### Validation / tests
`tests/test_gnn_baselines.py`: shape/protocol conformance tests for all three baselines against `ForecastModel`; overfitting sanity check (near-zero training loss achievable on a small synthetic sequence) as a correctness smoke test, mirroring the existing `DMDBaseline`/`EDMDBaseline` test style.

---

## 9. Graph pooling / multi-resolution modeling

### Problem
No coarsening, pooling, or hierarchical latent structure exists anywhere in the source. Large graphs and multiscale spatiotemporal phenomena (as in P-K-GCN's coarsen-then-Koopman super-resolution architecture) cannot be handled except at full node resolution.

### Design
Add a `HierarchicalGraphKoopmanModel` that pairs each encoder stage with a pooling operator (initially wrapping PyG's existing `TopKPooling`/`SAGPooling`, avoiding a new pooling algorithm implementation), producing a stack of progressively coarsened latent graphs `G_0 ⊃ G_1 ⊃ ... ⊃ G_L`. A separate `KoopmanOperator` (or `GraphKoopmanOperator` from item 2) advances the *coarsest* level; a learned un-pooling/interpolation path (mirroring the encoder's pooling assignment matrices) reconstructs finer levels, giving both a compute-efficient forecasting path for very large graphs and a super-resolution mode (train coarse, evaluate at fine resolution).

### API sketch
```python
class HierarchicalGraphKoopmanModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        *,
        latent_dim: int,
        pool_ratios: Sequence[float] = (0.5, 0.5),   # per level
        pooling: Literal["topk", "sag"] = "topk",
        time_step: float = 1.0,
    ) -> None: ...

    def fit(self, data_sequence, **kwargs) -> TrainingHistory: ...
    def predict(self, initial_graph, steps: int, *, resolution: int = -1) -> list[Data]:
        """resolution=-1 decodes at full (finest) resolution regardless of the
        level the Koopman operator advances at; resolution=0..L decodes at a
        specific coarsened level for inspection/efficiency."""
```

### Highlighting example
```python
from koopman_graph.hierarchical import HierarchicalGraphKoopmanModel
from koopman_graph import GNNDecoder, GNNEncoder
from koopman_graph.datasets import GridDynamicGraphBenchmark

# 64x64 grid (4096 nodes) - too large to advance a dense per-node Koopman
# operator efficiently without coarsening.
seq = GridDynamicGraphBenchmark.generate(rows=64, cols=64, num_timesteps=40, seed=9)

model = HierarchicalGraphKoopmanModel(
    encoder=GNNEncoder(1, 32, 16),
    decoder=GNNDecoder(16, 32, 1),
    latent_dim=16,
    pool_ratios=(0.25, 0.25),   # 4096 -> 1024 -> 256 nodes at the coarsest level
    time_step=0.1,
)
model.fit(seq, epochs=30, lr=1e-3)

# Forecast cheaply at the coarsest resolution, then super-resolve to full grid.
coarse_forecast = model.predict(seq[0], steps=10, resolution=-2)
full_res_forecast = model.predict(seq[0], steps=10, resolution=-1)
print("Coarse nodes:", coarse_forecast[0].x.shape[0],
      "Full-res nodes:", full_res_forecast[0].x.shape[0])
```
Proposed new notebook: `examples/23_hierarchical_multiresolution.ipynb`, benchmarking wall-clock rollout time and RMSE at multiple `pool_ratios` settings on a large synthetic grid, directly analogous to P-K-GCN's super-resolution evaluation.

### Validation / tests
`tests/test_hierarchical.py`: pooling/un-pooling shape round-trips at each level; a no-op sanity check (`pool_ratios=(1.0,)` reduces to the existing flat `GraphKoopmanModel` numerically); wall-clock scaling test confirming sub-quadratic runtime growth vs. node count relative to the flat model at fixed accuracy tolerance.

---

## 10. Additional topology-aware encoders

### Problem
Encoders are limited to GCN and GAT. There is no GraphSAGE, no directional/diffusion convolution (despite the existing `AnisotropicAdvectionGridBenchmark` specifically containing asymmetric edge weights that a diffusion-convolution layer is designed to exploit), and no graph-transformer or mesh-aware encoder for unstructured-mesh CFD-style data.

### Design
Add three encoder/decoder pairs following the exact `BaseGNNModule` pattern already used by `GNNEncoder`/`GATEncoder` (so they're drop-in replacements requiring no other API change):

- `SAGEEncoder`/`SAGEDecoder` — `torch_geometric.nn.SAGEConv` stack, useful as an inductive baseline for generalizing to unseen node subsets.
- `DiffConvEncoder`/`DiffConvDecoder` — bidirectional (in/out) random-walk diffusion convolution, matching DCRNN-style directional propagation; the natural encoder to pair with `AnisotropicAdvectionGridBenchmark`.
- `GraphTransformerEncoder`/`GraphTransformerDecoder` — `torch_geometric.nn.TransformerConv` stack with optional edge-feature conditioning, for long-range spatial dependencies beyond a GAT's local attention.

### API sketch
```python
class DiffConvEncoder(BaseGNNModule):
    def __init__(
        self, in_channels: int, hidden_channels: int, latent_dim: int,
        *, num_layers: int = 2, max_diffusion_steps: int = 2,
        activation: ActivationName = "relu",
    ) -> None: ...
    # forward(x, edge_index, edge_weight=None) propagates separately along
    # edge_index and its reverse, matching directional traffic/flow physics.

class GraphTransformerEncoder(BaseGNNModule):
    def __init__(
        self, in_channels: int, hidden_channels: int, latent_dim: int,
        *, num_layers: int = 2, heads: int = 4, edge_dim: int | None = None,
    ) -> None: ...
```

### Highlighting example
```python
from koopman_graph import GraphKoopmanModel
from koopman_graph.nn import DiffConvDecoder, DiffConvEncoder
from koopman_graph.datasets import AnisotropicAdvectionGridBenchmark

# Directional advection: a plain (undirected-message) GCN encoder cannot
# represent the asymmetric in/out flow without extra parameters; DiffConv can.
seq = AnisotropicAdvectionGridBenchmark.generate(
    rows=16, cols=16, drift=(0.6, 0.1), num_timesteps=50, seed=4,
)

model = GraphKoopmanModel(
    encoder=DiffConvEncoder(1, 32, 16, max_diffusion_steps=3),
    decoder=DiffConvDecoder(16, 32, 1, max_diffusion_steps=3),
    latent_dim=16, time_step=0.1,
)
model.fit(seq, epochs=40, lr=1e-3)
print("DiffConv RMSE:", model.evaluate(seq).rmse)
```
Proposed update: `examples/09_topology_ablation.ipynb` gains `SAGE`/`DiffConv`/`GraphTransformer` rows alongside the existing `GCN`/`GAT` ablation, with `AnisotropicAdvectionGridBenchmark` specifically highlighting DiffConv's advantage on directional dynamics.

### Validation / tests
`tests/test_encoder.py` additions mirroring the existing `GNNEncoder`/`GATEncoder` test suite (shape contracts, `validate_positive_dims`, activation options) for each new encoder; a directional-advection regression test asserting `DiffConvEncoder` achieves lower RMSE than `GNNEncoder` on `AnisotropicAdvectionGridBenchmark` at matched parameter count (fixed seed, asserted margin).

---

## 11. Genuinely nonlinear / chaotic benchmark datasets

### Problem
`SyntheticDynamicGraphBenchmark`, `GridDynamicGraphBenchmark`, and `IEEE118DynamicBenchmark` all reduce to the same linear Laplacian-diffusion step plus exogenous sinusoidal forcing and Gaussian noise. The only genuinely nonlinear dynamics (SIR epidemic) exists only ad hoc inside a notebook, not as a tested library class. None of the classical chaotic-PDE stress tests from the wider Koopman literature (Kuramoto–Sivashinsky, Burgers, Navier–Stokes/cylinder wake, Lorenz-96) are present.

### Design
Promote the notebook's SIR simulation to a first-class, tested dataset, and add three new nonlinear benchmark classes following the existing `*.generate(...)` classmethod convention:

- `EpidemicNetworkBenchmark` — networked SIR/SEIR with configurable `beta`, `gamma`, and contact-graph topology (ring, small-world, or user-supplied `edge_index`), promoting `examples/06_epidemic_ring.ipynb`'s inline simulation.
- `Lorenz96GraphBenchmark` — the classic Lorenz-96 chaotic system mapped onto a ring graph (each node is one Lorenz-96 variable, coupling matches the graph's cyclic structure), a standard chaotic-dynamics stress test.
- `KuramotoSivashinskyBenchmark` — 1D KS PDE discretized onto a path/ring graph, the exact system used in the original EDMD dictionary-learning paper, enabling direct literature comparison.
- `CylinderWakeBenchmark` — 2D incompressible vortex-shedding flow (precomputed/cached snapshots, following the same caching pattern as `MetrLaTrafficBenchmark`) on an unstructured mesh graph, the most common fluid-dynamics Koopman benchmark in the literature.

### API sketch
```python
class EpidemicNetworkBenchmark:
    @classmethod
    def generate(cls, *, num_nodes: int, topology: Literal["ring", "small_world", "custom"],
                 beta: float = 0.45, gamma: float = 0.12, num_timesteps: int = 60,
                 edge_index: Tensor | None = None, seed: int | None = None
                 ) -> GraphSnapshotSequence: ...

class Lorenz96GraphBenchmark:
    @classmethod
    def generate(cls, *, num_nodes: int = 40, forcing: float = 8.0,
                 num_timesteps: int = 500, dt: float = 0.01,
                 seed: int | None = None) -> GraphSnapshotSequence: ...

class KuramotoSivashinskyBenchmark:
    @classmethod
    def generate(cls, *, num_nodes: int = 64, domain_length: float = 22.0,
                 num_timesteps: int = 400, dt: float = 0.25,
                 seed: int | None = None) -> GraphSnapshotSequence: ...
```

### Highlighting example
```python
from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
from koopman_graph.baselines import DMDBaseline
from koopman_graph.datasets import KuramotoSivashinskyBenchmark

# Reproduce the original EDMD-dictionary-learning paper's KS stress test,
# now on a graph-native path topology.
seq = KuramotoSivashinskyBenchmark.generate(num_nodes=64, num_timesteps=400, seed=0)
train, test = seq.split(train_fraction=0.7)

koopman_model = GraphKoopmanModel(
    encoder=GNNEncoder(1, 64, 24), decoder=GNNDecoder(24, 64, 1),
    latent_dim=24, time_step=0.25,
).fit(train, epochs=100, lr=1e-3)

dmd_model = DMDBaseline().fit(train)

print("GraphKoopmanModel RMSE (chaotic KS):", koopman_model.evaluate(test).rmse)
print("Linear DMD RMSE (chaotic KS):       ", dmd_model.evaluate(test).rmse)
```
Proposed new notebook: `examples/24_chaotic_pde_benchmarks.ipynb`, running all four new nonlinear benchmarks and explicitly reporting the GraphKoopmanModel-vs-linear-DMD RMSE gap, which is currently untestable in-repo since every packaged benchmark is linear-plus-forcing.

### Validation / tests
`tests/test_datasets.py` additions: determinism under fixed seed for each new benchmark; energy/conservation sanity checks specific to each system (Lorenz-96 known Lyapunov-exponent range, KS known dominant-wavenumber band); a regression test asserting `GraphKoopmanModel` beats `DMDBaseline` by a fixed margin on `KuramotoSivashinskyBenchmark` (this is the assertion that currently has no benchmark to run against).

---

## 12. State estimation / imputation from observation masks

### Problem
`observation_masks` currently only gate the loss and evaluation metrics — unobserved nodes are excluded from those computations, but there is no imputation or observer mechanism, and no fusion with the delay-embedding machinery (item 1) for reconstructing missing sensor values before encoding.

### Design
Add a `KoopmanObserver` that fuses the existing `RecursiveKoopmanAdapter` (already shipped for RLS operator updates) with a standard Kalman filter/smoother operating in latent space: since latent dynamics are linear (`z_{t+1} = Kz_t`), the Kalman equations apply exactly (no EKF/UKF approximation needed) once node-level observations are mapped through the encoder's *local* Jacobian for masked-node imputation. Concretely:

1. `impute_missing(data_with_mask) -> Data`: fills unobserved node features via a graph-diffusion-based initial guess (reusing `graph_laplacian_features` machinery), then
2. `KoopmanObserver.filter(sequence)`: runs a latent-space Kalman filter across the sequence, producing filtered/smoothed latent states `z_t` and their covariances even when some nodes are unobserved at some timesteps, by zeroing rows of the observation matrix `H` (constructed from the decoder's local linearization) for masked nodes at each step.

### API sketch
```python
class KoopmanObserver:
    def __init__(self, model: GraphKoopmanModel, *, process_noise: float = 1e-3,
                 observation_noise: float = 1e-2) -> None: ...

    def filter(self, sequence: GraphSnapshotSequence) -> "FilterResult":
        """Returns per-timestep filtered latent states + covariances, using
        sequence.observation_masks to zero rows of H for unobserved nodes."""

    def smooth(self, sequence: GraphSnapshotSequence) -> "FilterResult":
        """RTS smoother pass over the full sequence."""

    def impute(self, sequence: GraphSnapshotSequence) -> GraphSnapshotSequence:
        """Decodes smoothed latent states back to node space, filling in
        physically-plausible values for every masked (node, timestep) pair."""
```

### Highlighting example
```python
from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
from koopman_graph.adaptation import KoopmanObserver
from koopman_graph.datasets import MetrLaTrafficBenchmark

# 40% of sensors drop out at random times (realistic METR-LA sensor outages).
seq = MetrLaTrafficBenchmark.load(split="test").with_observation_masks(
    drop_fraction=0.4, seed=11,
)

model = GraphKoopmanModel(
    encoder=GNNEncoder(1, 64, 32), decoder=GNNDecoder(32, 64, 1),
    latent_dim=32, time_step=5.0,
)
model.fit(seq, epochs=40, lr=1e-3)

observer = KoopmanObserver(model, process_noise=1e-3, observation_noise=1e-2)
imputed_seq = observer.impute(seq)

print("Masked-node imputation RMSE vs. ground truth:",
      imputed_seq.rmse_against(MetrLaTrafficBenchmark.load(split="test")))
```
Proposed new notebook: `examples/25_kalman_koopman_state_estimation.ipynb`, showing imputation accuracy vs. `drop_fraction` on METR-LA, and comparing filtered vs. smoothed latent trajectories.

### Validation / tests
`tests/test_adaptation.py` additions: on a fully-observed synthetic linear system, `KoopmanObserver.filter` output must match a hand-rolled reference Kalman filter to numerical tolerance; imputation-RMSE-vs-`drop_fraction` monotonicity check; smoother must not perform worse than filter on average (standard Kalman-smoother property, checked empirically).

---

## 13. Bibliography corrections in `paper.bib`

### Problem
The forward/backward "consistency" loss the README foregrounds as a headline feature is functionally the core contribution of the Consistent Koopman Autoencoder, yet that paper is absent from `paper.bib`. The EDMD dictionary-learning paper, Korda & Mezić's bilinear/DMDc Koopman-MPC formulation, and the IEEE 68-bus GNN-vs-Koopman comparative study (the closest prior work to the packaged IEEE-118 example) are also missing.

### Design
This is a documentation-only change, but is included here because it should land alongside the items above (items 1–3 in particular directly implement ideas from the papers being added) so the citations accurately track the code that motivated them.

### Proposed `paper.bib` additions
```bibtex
@inproceedings{Azencot2020,
  author    = {Azencot, Omri and Erichson, N. Benjamin and Lin, Vanessa and Mahoney, Michael W.},
  title     = {Forecasting Sequential Data Using Consistent {Koopman} Autoencoders},
  booktitle = {Proceedings of the 37th International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {119},
  pages     = {475--485},
  year      = {2020},
  publisher = {PMLR}
}

@article{Li2017EDMD,
  author  = {Li, Qianxiao and Dietrich, Felix and Bollt, Erik M. and Kevrekidis, Ioannis G.},
  title   = {Extended Dynamic Mode Decomposition with Dictionary Learning: A
             Data-Driven Adaptive Spectral Decomposition of the {Koopman} Operator},
  journal = {Chaos: An Interdisciplinary Journal of Nonlinear Science},
  volume  = {27},
  number  = {10},
  pages   = {103111},
  year    = {2017},
  doi     = {10.1063/1.4993854}
}

@article{Korda2018,
  author  = {Korda, Milan and Mezi{\'c}, Igor},
  title   = {Linear Predictors for Nonlinear Dynamical Systems: {Koopman} Operator
             Meets Model Predictive Control},
  journal = {Automatica},
  volume  = {93},
  pages   = {149--160},
  year    = {2018},
  doi     = {10.1016/j.automatica.2018.03.046}
}

@article{Nandanoori2022,
  author  = {Nandanoori, Sai Pushpak and Guan, Sheng and Kundu, Soumya and Pal,
             Seemita and Agarwal, Khushbu and Wu, Yinghui and Choudhury, Sutanay},
  title   = {Graph Neural Network and {Koopman} Models for Learning Networked
             Dynamics: A Comparative Study on Power Grid Transients Prediction},
  journal = {IEEE Access},
  year    = {2022},
  doi     = {10.1109/ACCESS.2022.3162205}
}
```
Update `paper.md`'s "State of the field" and "Software design" sections to cite `Azencot2020` where the forward/backward consistency loss is introduced, `Li2017EDMD` where `EDMDBaseline` is described, `Korda2018` in the new bilinear-control section (item 3), and `Nandanoori2022` alongside the existing IEEE-118 example description.

### Highlighting example
No code example applies; the "highlight" here is a documentation diff. Suggested verification: a `tests/test_docs.py` check (already exists for other doc invariants) extended to assert every class/loss introduced by name in `README.md`'s feature table has a corresponding `paper.bib` entry when a direct literature precedent exists in this design document's mapping table (see Appendix).

---

## 14. Sparsity-promoting and worst-case (L∞) loss terms

### Problem
Beyond the existing eigenvalue-radius regularizer, there is no SINDy-style sparsity penalty on `K` for interpretability, and no L∞/worst-case reconstruction loss to bound generalization error, both of which appear in the broader deep-Koopman literature.

### Design
Two new loss terms, added to `koopman_graph.losses` and wired into `LossWeights` alongside the existing `reconstruction`/`forward`/`backward`/`eigenvalue` terms:

- `KoopmanSparsityLoss(p: float = 1.0)`: an `L1` (or general `Lp`, `p<1` via a smoothed surrogate) penalty on the entries of `K` (or, for structural parameterizations, on the underlying factor matrices), encouraging a sparse operator whose nonzero pattern is directly interpretable — mirroring SINDy-style sparse regression applied to the *learned* Koopman matrix rather than to a hand-built dictionary.
- `WorstCaseReconstructionLoss()`: replaces/augments the mean-squared reconstruction term with a per-batch max (`L∞`) term, bounding worst-case single-node/single-timestep error rather than only the average.

### API sketch
```python
class KoopmanSparsityLoss(nn.Module):
    def __init__(self, p: float = 1.0, threshold: float | None = None) -> None: ...
    def forward(self, koopman_matrix: Tensor) -> Tensor: ...

class WorstCaseReconstructionLoss(nn.Module):
    def forward(self, predicted: Tensor, target: Tensor) -> Tensor:
        return (predicted - target).abs().amax()

@dataclass
class LossWeights:
    reconstruction: float = 1.0
    forward: float = 1.0
    backward: float = 1.0
    eigenvalue: float = 0.0
    sparsity: float = 0.0          # new
    worst_case: float = 0.0        # new
```

### Highlighting example
```python
from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel, LossWeights
from koopman_graph.datasets import IEEE118DynamicBenchmark

seq = IEEE118DynamicBenchmark.generate(num_timesteps=200, seed=6)

model = GraphKoopmanModel(
    encoder=GNNEncoder(4, 64, 16), decoder=GNNDecoder(16, 64, 4),
    latent_dim=16, time_step=1.0,
)
history = model.fit(
    seq, epochs=60, lr=1e-3,
    loss_weights=LossWeights(
        reconstruction=1.0, forward=1.0, backward=0.5,
        sparsity=0.05, worst_case=0.1,
    ),
)

K = model.koopman.matrix()
sparsity_fraction = (K.abs() < 1e-3).float().mean()
print(f"Fraction of near-zero K entries: {sparsity_fraction:.1%}")
print("Worst-case node error (final epoch):", history.loss_terms["worst_case"][-1])
```
Proposed new notebook: `examples/26_sparse_interpretable_operator.ipynb`, visualizing the learned `K` matrix's sparsity pattern against the IEEE-118 admittance/adjacency structure (do sparse `K` entries align with electrically-coupled buses?), and comparing worst-case node error with and without the `WorstCaseReconstructionLoss` term.

### Validation / tests
`tests/test_losses.py` additions: `KoopmanSparsityLoss` gradient check (subgradient sanity at zero); sparsity-weight sweep test confirming monotonically increasing sparsity fraction as the `sparsity` weight increases, on a fixed seed; `WorstCaseReconstructionLoss` correctly dominates MSE-driven training on an adversarial single-outlier-node synthetic case where MSE alone would under-penalize that node.

---

## Appendix A: Mapping of design items to literature precedent

| # | Feature | Primary literature precedent |
|---|---|---|
| 1 | Delay/Hankel embeddings | Brunton et al. delay coordinates; Arbabi, Korda & Mezić sparse-measurement Koopman-MPC |
| 2 | Networked/coupled Koopman operator | Mukherjee et al., distributed geometric Koopman; Li et al., Compositional Koopman Operators |
| 3 | Bilinear/control-affine Koopman | Korda & Mezić, linear predictors for nonlinear systems; Proctor et al., DMDc |
| 4 | Continuous-spectrum auxiliary network | Lusch, Kutz & Brunton, deep learning for universal linear embeddings; Goyal, Yıldız & Benner, structure-preserving Koopman embeddings |
| 5 | Uncertainty quantification | Mallen et al., Deep Probabilistic Koopman; Wu et al., K²VAE |
| 6 | Physics-informed observables | Physics-Informed Koopman Networks; SPIKE; Goyal et al. symplectic embeddings |
| 7 | Kernel/RBF EDMD dictionaries | Williams et al., original EDMD; operator-valued RKHS Koopman methods |
| 8 | Modern GNN forecaster baselines | Yu, Yin & Zhu, STGCN; Li et al., DCRNN; Wu et al., Graph WaveNet |
| 9 | Graph pooling / multi-resolution | P-K-GCN, physics-augmented Koopman-enhanced GCN for spatiotemporal super-resolution |
| 10 | Additional encoders (SAGE/DiffConv/Transformer) | Hamilton et al., GraphSAGE; Li et al., DCRNN (diffusion convolution) |
| 11 | Nonlinear/chaotic benchmarks | Li et al., EDMD dictionary learning (Kuramoto–Sivashinsky); classical Lorenz-96 and cylinder-wake Koopman literature |
| 12 | Kalman-Koopman state estimation | Arbabi, Korda & Mezić sparse-measurement Koopman-MPC; K²VAE (Kalman-augmented Koopman) |
| 13 | Bibliography corrections | Azencot et al., Consistent Koopman Autoencoders; Nandanoori et al., GNN-vs-Koopman power grid study |
| 14 | Sparsity / worst-case losses | SINDy-style sparse system identification; L∞ bounds in autoencoder-Koopman generalization discussions |

## Appendix B: Suggested prioritization

**Phase 1 (highest literature-alignment value, moderate effort):** Items 2 (networked operator — closes the biggest conceptual gap), 8 (GNN baselines — needed to substantiate the paper's own motivating claim), 13 (bibliography — nearly free).

**Phase 2 (high value, moderate-to-high effort):** Items 1 (delay embeddings), 3 (bilinear control), 11 (nonlinear benchmarks — unblocks meaningful evaluation of everything else), 14 (sparsity/worst-case losses).

**Phase 3 (valuable, higher effort / narrower audience):** Items 4 (continuous spectrum), 5 (UQ), 6 (physics-informed losses), 12 (Kalman-Koopman observer).

**Phase 4 (nice-to-have, largest scope):** Items 7 (kernel EDMD), 9 (hierarchical/pooling), 10 (additional encoders) — each is a clean, additive extension but touches the largest amount of new surface area (new baseline algorithms, new architectural class, new encoder family respectively) for the narrowest immediate payoff relative to Phases 1–2.

## Appendix C: New/updated example notebooks introduced by this document

| Notebook | Item(s) covered |
|---|---|
| `17_delay_embedding_partial_observability.ipynb` | 1 |
| `18_networked_koopman_dynamic_topology.ipynb` | 2 |
| `19_bilinear_control_koopman.ipynb` | 3 |
| `20_continuous_spectrum_auxiliary_network.ipynb` | 4 |
| `21_uncertainty_quantification.ipynb` | 5 |
| `14_physics_informed_diffusion.ipynb` (extended) | 6 |
| `03_traffic_network.ipynb` (extended) | 7 |
| `22_gnn_forecaster_comparison.ipynb` | 8 |
| `23_hierarchical_multiresolution.ipynb` | 9 |
| `09_topology_ablation.ipynb` (extended) | 10 |
| `24_chaotic_pde_benchmarks.ipynb` | 11 |
| `25_kalman_koopman_state_estimation.ipynb` | 12 |
| — (documentation only) | 13 |
| `26_sparse_interpretable_operator.ipynb` | 14 |
