# GbSB + QUBO Solver

A production-ready Python implementation of the **Generalized Ballistic Simulated Bifurcation (GbSB)** algorithm for solving combinatorial optimization problems. This codebase implements and extends the method described in:

> Goto et al., *"Edge-of-chaos-enhanced quantum-inspired algorithm for combinatorial optimization"*  
> Physical Review Applied 25, 044011, 2026 · [arXiv:2508.17655](https://arxiv.org/abs/2508.17655)

GbSB is a **quantum-inspired heuristic** that finds high-quality solutions to NP-hard Ising and QUBO problems orders of magnitude faster than classical methods, with optional GPU acceleration via PyTorch.

---

## Table of Contents

- [What is this for?](#what-is-this-for)
- [How it works](#how-it-works)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Quick start](#quick-start)
- [API reference](#api-reference)
- [Extending the solver](#extending-the-solver)
- [Potential improvements](#potential-improvements)
- [Performance notes](#performance-notes)
- [Background: Ising & QUBO problems](#background-ising--qubo-problems)

---

## What is this for?

This solver targets **binary combinatorial optimization** — problems where you need to find the best assignment of 0/1 (or ±1) variables that minimises some objective. These problems appear constantly in industry and research:

| Problem | Formulation |
|---|---|
| Portfolio optimization | QUBO |
| Max-Cut / graph partitioning | Ising |
| Vehicle routing | Constrained QUBO |
| Protein folding | Ising |
| Feature selection | QUBO |
| Boolean satisfiability (SAT) | Ising |
| Scheduling & bin packing | Constrained QUBO |

All of these reduce to either the **Ising model** or **QUBO** (Quadratic Unconstrained Binary Optimization) form, both of which this solver handles directly.

---

## How it works

GbSB is a continuous dynamical system that naturally bifurcates (snaps) its variables toward ±1 solutions. Each variable $x_i$ evolves under **symplectic Euler integration** of:

$$\dot{y}_i = -(p_i x_i - c \cdot (\sum_j J_{ij} x_j + h_i))$$
$$\dot{x}_i = y_i$$

where $p_i$ is a per-variable **bifurcation parameter** that decays from 1 towards 0 over $M$ steps. As $p_i$ crosses zero, $x_i$ bifurcates: the stable fixed points at $x_i = 0$ become unstable and $x_i = \pm 1$ become attractors. The coupling matrix $J$ (or its QUBO equivalent) biases *which* way each variable bifurcates — driving the system toward low-energy configurations.

The **"edge of chaos"** refers to the nonlinear control parameter $A$ which tunes how aggressively the per-variable dynamics are controlled. The name comes from the observation that the best solutions are found near the boundary between ordered (all decided) and chaotic (oscillating) behaviour.

### Key enhancements in this implementation

1. **Adaptive A calibration** — Automatically finds the best $A$ for each problem instance via a short sweep, rather than using a fixed default.
2. **Instance-adaptive time stepping** — Uses a larger $dt$ early in the run (exploration) and shrinks it as variables become decided (exploitation).
3. **Early stopping** — Exits a trial when >95% of variables are decided and energy has stagnated, saving significant computation.
4. **Smooth p decay** — Replaces the numerically unstable $1/(M-m)$ decay formula with a smooth exponential-logistic variant.
5. **Per-variable p clamping** — Prevents divergence in the chaotic regime by enforcing $p_i \in [p_{\min}, p_{\max}]$.
6. **Sobol low-discrepancy initialisation** — Better coverage of the initial condition space versus uniform random.
7. **Auto-scaling** — Normalises $\max|J| = 1$ for numerical stability across diverse problem scales.
8. **Input validation** — Symmetry enforcement, diagonal zeroing, shape assertions, and range warnings.
9. **k-flip local search** — Post-processing hill-climbing (1-flip and 2-flip) refines the best solution found.
10. **Diversity penalty** — Hamming-distance term discourages the solver from revisiting previously found optima.
11. **GPU batched solver** — All trials run as a single `(num_trials, N)` tensor pass via PyTorch (CUDA/MPS/CPU).
12. **Rich diagnostics** — `GbSBResult` container reports per-trial energies, success rate, A calibration map, and step counts.
13. **TTS benchmarking** — Standard time-to-solution metric for comparing solver configurations.
14. **Sparse matrix support** — `scipy.sparse` passthrough for memory-efficient large QUBOs.
15. **QUBO penalty utilities** — Auto penalty weight and equality constraint builder for constrained problems.

---

## Repository structure

```
gbsb/
├── gbsb_solver.py          # Core solver — Ising + QUBO interface (NumPy)
├── gbsb_torch.py           # GPU-accelerated batched solver (PyTorch)
├── gbsb_local_search.py    # Post-processing: local search & noise injection
├── gbsb_utils.py           # Input handling, QUBO utilities, benchmarking
├── gbsb_demo.py            # End-to-end demo and edge-of-chaos sweep
└── requirements.txt        # Dependencies
```

### Module responsibilities

**`gbsb_solver.py`** — The heart of the solver. Contains:
- `GbSBResult` — dataclass holding all outputs and diagnostics
- `gbsb_optimize(J, h, ...)` — main Ising solver (NumPy, multi-trial)
- `gbsb_qubo_optimize(Q, ...)` — QUBO interface (auto-converts to Ising)
- `qubo_to_ising(Q)` — standalone QUBO→Ising conversion
- Internal helpers: `_run_trial`, `_calibrate_A`, `_sobol_uniform`, `_validate_inputs`, `_auto_scale`

**`gbsb_torch.py`** — Drop-in GPU replacement:
- `gbsb_optimize_torch(J, h, ...)` — batched PyTorch solver, auto-detects CUDA/MPS/CPU
- `gbsb_qubo_optimize_torch(Q, ...)` — QUBO interface for the GPU solver
- Gracefully falls back to the NumPy solver if PyTorch is unavailable

**`gbsb_local_search.py`** — Post-processing and diversity tools:
- `k_flip_hillclimb(J, h, spins, ...)` — steepest-descent 1-flip and 2-flip hill climbing
- `inject_noise(spins, p, ...)` — perturbation for hard instances (near-bifurcation variables)
- `diversity_penalty(J, h, spins, previous_best, ...)` — Hamming-distance augmentation

**`gbsb_utils.py`** — Benchmarking and problem construction:
- `random_dense_ising`, `random_sparse_ising` — standard Ising benchmarks
- `random_qubo`, `maxcut_from_adjacency` — QUBO and graph problem generators
- `auto_penalty_weight`, `add_equality_penalty` — constrained QUBO helpers
- `time_to_solution(solver_fn, ...)` — TTS metric calculation
- `print_result(result)` — pretty-printer for `GbSBResult`

---

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd gbsb

# Core dependencies (required)
pip install numpy>=1.24

# Strongly recommended
pip install scipy>=1.10 tqdm>=4.65

# GPU acceleration (optional)
pip install torch>=2.0   # CUDA or MPS builds as appropriate for your hardware

# Development
pip install pytest>=7.0
```

Or install all at once:

```bash
pip install -r requirements.txt
```

---

## Quick start

### Minimal Ising solve

```python
import numpy as np
from gbsb_solver import gbsb_optimize

N = 100
rng = np.random.default_rng(0)
J = rng.normal(0, 1/N**0.5, (N, N))
J = (J + J.T) / 2
np.fill_diagonal(J, 0)

result = gbsb_optimize(J, M=2000, num_trials=30, auto_tune_A=True)
print(result)
# GbSBResult(best_energy=-47.321, success_rate=83.33%, A_used=2.0, wall_time=1.24s)

print(result.best_spins)   # ±1 solution vector
print(result.A_calibration) # {A: energy} map from the calibration sweep
```

### QUBO solve

```python
from gbsb_solver import gbsb_qubo_optimize
import numpy as np

Q = np.array([[0, 1, -2],
              [1, 0,  3],
              [-2, 3, 0]], dtype=float)

x, energy, result = gbsb_qubo_optimize(Q, M=5000, num_trials=50, auto_tune_A=True)
print("Solution:", x)       # binary {0, 1} vector
print("QUBO energy:", energy)
```

### GPU-accelerated solve

```python
from gbsb_torch import gbsb_optimize_torch
import numpy as np

N = 1000
rng = np.random.default_rng(1)
J = rng.normal(0, 1/N**0.5, (N, N))
J = (J + J.T) / 2
np.fill_diagonal(J, 0)

result = gbsb_optimize_torch(J, M=3000, num_trials=256, device='cuda')
print(result)
```

### Max-Cut on a graph

```python
import numpy as np
from gbsb_solver import gbsb_optimize
from gbsb_utils import maxcut_from_adjacency

# Build a random weighted adjacency matrix
N = 50
rng = np.random.default_rng(42)
W = rng.integers(0, 2, (N, N)).astype(float)
W = (W + W.T) / 2
np.fill_diagonal(W, 0)

J = maxcut_from_adjacency(W)
result = gbsb_optimize(J, M=2000, num_trials=30, auto_tune_A=True)

cut_value = -result.best_energy   # Max-Cut = -Ising energy for J = W
print(f"Max-Cut value: {cut_value:.2f}")
# Partition: nodes where best_spins == +1 are in set A, -1 in set B
```

### Constrained QUBO (equality constraint)

```python
from gbsb_solver import gbsb_qubo_optimize
from gbsb_utils import add_equality_penalty, auto_penalty_weight
import numpy as np

N = 20
rng = np.random.default_rng(0)
Q_obj = rng.normal(0, 1, (N, N))
Q_obj = (Q_obj + Q_obj.T) / 2

# Add penalty: sum(x[0:5]) == 2
penalty = auto_penalty_weight(Q_obj, np.ones(5), 2.0)
Q_pen = add_equality_penalty(Q_obj, list(range(5)), target_sum=2, penalty=penalty)

x, energy, result = gbsb_qubo_optimize(Q_pen, M=3000, num_trials=50, auto_tune_A=True)
print(f"Constraint satisfied: {x[:5].sum() == 2}")
```

### Run the full demo

```bash
python gbsb_demo.py
```

This runs six demonstrations: dense Ising, QUBO, edge-of-chaos A-sweep, GPU batched, Max-Cut, and TTS benchmarking.

---

## API reference

### `gbsb_optimize`

```python
gbsb_optimize(
    J,                       # (N, N) symmetric coupling matrix, zero diagonal
    h=None,                  # (N,) linear field; defaults to zeros
    *,
    M=2000,                  # integration steps per trial
    dt=1.25,                 # base time step (paper default)
    A=None,                  # nonlinear control; None → auto-calibrate
    c=1.0,                   # coupling scale factor
    num_trials=20,           # independent random restarts
    seed=42,
    auto_tune_A=True,        # run calibration sweep to choose A
    A_search_grid=(1.0, 1.5, 2.0, 2.5, 3.0, 3.5),
    calib_trials=5,          # restarts per A during calibration
    calib_M_fraction=0.25,   # fraction of M for calibration
    adaptive_dt=True,        # scale dt based on fraction of decided variables
    early_stop_frac=0.95,    # early-stop when this fraction of |x_i| > 0.95
    early_stop_window=15,    # stagnation steps before early stop triggers
    p_min=0.0,               # lower clamp for bifurcation parameter
    p_max=2.0,               # upper clamp for bifurcation parameter
    auto_scale=True,         # normalise max|J| = 1
    local_search=True,       # run k-flip hill climbing on best solution
    local_search_flips=2,    # max flip size (1 or 2)
    verbose=True,            # tqdm progress bar
) -> GbSBResult
```

### `gbsb_qubo_optimize`

```python
gbsb_qubo_optimize(
    Q,          # (N, N) QUBO matrix; Q need not be symmetric
    **kwargs,   # all arguments forwarded to gbsb_optimize
) -> (best_x, qubo_energy, GbSBResult)
```

Returns the binary {0,1} solution vector, true QUBO energy, and the full `GbSBResult` for diagnostics.

### `GbSBResult` fields

| Field | Type | Description |
|---|---|---|
| `best_spins` | `ndarray (N,)` | ±1 Ising solution |
| `best_energy` | `float` | Ising energy of best solution |
| `energies` | `ndarray (num_trials,)` | Best energy per trial |
| `success_rate` | `float` | Fraction of trials matching the best energy |
| `wall_time` | `float` | Total seconds elapsed |
| `A_used` | `float` | A value used for the main run |
| `A_calibration` | `dict` or `None` | `{A: energy}` from the calibration sweep |
| `steps_used` | `ndarray (num_trials,)` | Actual steps per trial (early-stop aware) |
| `final_p_mean` | `ndarray (N,)` | Mean final bifurcation parameter across trials |

---

## Extending the solver

### Adding a new benchmark problem

Add a generator function to `gbsb_utils.py`:

```python
def tsp_qubo(distances: np.ndarray, penalty: float = 10.0) -> np.ndarray:
    """
    Construct a QUBO for the Travelling Salesman Problem.
    Variables x_{i,t} = 1 if city i is visited at time t.
    """
    N = len(distances)
    n_vars = N * N
    Q = np.zeros((n_vars, n_vars))
    # ... add distance objective and permutation constraints ...
    return Q
```

### Adding a new post-processing step

Extend `gbsb_local_search.py` with a new refinement function:

```python
def tabu_search(J, h, spins, tabu_tenure=10, max_iter=1000):
    """
    Tabu search on top of the GbSB solution — prevents re-visiting recent flips.
    """
    tabu_list = []
    best_spins = spins.copy()
    best_energy = ising_energy(J, h, spins)
    # ... tabu search implementation ...
    return best_spins, best_energy
```

Then activate it in `gbsb_optimize` by adding a `tabu_search` parameter and calling it after `k_flip_hillclimb`.

### Adding a new backend

To add a JAX or CuPy backend, follow the pattern in `gbsb_torch.py`:

1. Import your array library with a graceful fallback.
2. Implement `_best_device()` for that library.
3. Port the inner loop — symplectic Euler, inelastic walls, early stopping — using the library's tensor ops.
4. Return a `GbSBResult` (the NumPy result container is reused).

### Changing the bifurcation dynamics

The core physics is in `_run_trial` in `gbsb_solver.py`. To experiment with different decay schedules or integration schemes, modify that function. For example, to try a **cosine p decay**:

```python
# Replace the current decay line:
decay_rate = factor / max(remaining, 1)
p = p - decay_rate * p

# With a cosine schedule:
progress = m / M
p = p_max * 0.5 * (1 + np.cos(np.pi * progress)) * np.ones(N)
```

---

## Potential improvements

### Algorithmic

1. **Parallel tempering** — Run multiple replica chains at different temperatures simultaneously and exchange configurations periodically. This is highly effective for instances with many local optima (spin glasses, constrained problems).

2. **Population-based diversity** — Maintain a pool of solutions across restarts. Crossover operators (e.g. combining the ±1 assignments of two good solutions) can seed new trials in unexplored regions.

3. **Simulated annealing hybrid** — After GbSB bifurcates to a solution, apply a brief SA pass with a temperature schedule calibrated to the local energy landscape, rather than the greedy hill-climbing used now.

4. **Adaptive Sobol restart** — Instead of fixed Sobol initialisation, use the `final_p_mean` from previous trials to warm-start subsequent ones, biasing initial conditions toward regions that recently bifurcated well.

5. **Continuous relaxation rounding** — Instead of `sign(x)` at the end of each trial, use SDP-style randomised rounding: project the final continuous state onto a random hyperplane several times and keep the best binary assignment.

6. **Better 2-flip local search** — The current `k_flip_hillclimb` with `max_flips=2` has $O(N^2)$ cost per pass. For large $N$, replace with a priority-queue-based **Lin-Kernighan-style** move that scores candidate 2-flips in $O(N \log N)$.

7. **Constraint-aware bifurcation** — For constrained QUBOs, incorporate penalty gradient information directly into the $p_i$ decay schedule, so variables that violate constraints get extra pressure to decide earlier.

### Software quality

8. **Comprehensive test suite** — Add `pytest` tests covering: known-optimum small instances (N ≤ 10, brute-force verifiable), QUBO↔Ising round-trip, energy monotonicity of local search, reproducibility under fixed seeds, and shape/dtype edge cases.

9. **Type annotations and mypy** — The codebase uses `Optional` and basic annotations but `gbsb_utils.py` is missing several. A full `mypy --strict` pass would catch latent type issues.

10. **Profiling and `numba` JIT** — The inner `_run_trial` loop in `gbsb_solver.py` is the performance bottleneck for CPU runs. Decorating `_run_trial` with `@numba.njit` (with `parallel=True` for the `J @ x` matmul) could give a 5–20× CPU speedup without PyTorch.

11. **Logging vs print** — `gbsb_demo.py` mixes `print` statements with the `logger` used in `gbsb_solver.py`. Unifying on the `logging` module makes the library easier to embed in larger applications.

12. **Config object** — The `gbsb_optimize` function has 20+ keyword arguments. Introducing a `GbSBConfig` dataclass that bundles all hyperparameters would improve readability, enable easy serialisation, and facilitate hyperparameter search.

### Scalability

13. **Sparse J in the main loop** — The batched matmul `J @ x` dominates runtime for large $N$. `gbsb_torch.py` uses dense tensors. For sparse graphs (e.g. Max-Cut on a planar graph) switching to `torch.sparse_csr_tensor` could give 10–100× memory and compute savings for $N > 10{,}000$.

14. **Distributed trials** — Each trial is independent, making the outer trial loop trivially parallel. Wrapping `_run_trial` with `concurrent.futures.ProcessPoolExecutor` (CPU) or splitting `num_trials` across multiple GPUs (PyTorch `DataParallel`) is a straightforward scaling path.

15. **Mixed-precision GPU** — `gbsb_torch.py` uses `float32` throughout. For very large $N$ on modern GPUs, `bfloat16` (with a `float32` accumulator for `J @ x`) would halve memory bandwidth and potentially double throughput.

16. **Streaming for huge instances** — For $N > 50{,}000$ where $J$ doesn't fit in GPU VRAM, implement a **tiled matmul** that loads $J$ in blocks from CPU memory, computes partial `J @ x` products on GPU, and accumulates the result.

### Usability

17. **CLI interface** — A `python -m gbsb solve --input problem.npz --trials 100 --gpu` command-line tool would make the solver accessible without writing Python.

18. **Problem file I/O** — Support reading/writing standard formats: DIMACS (Max-Cut), QUBO `.bqp` files (used in D-Wave toolchains), and `.npz` archives for Ising instances.

19. **Callback hooks** — Allow users to pass a `callback(trial, result)` function that is called after each trial, enabling live monitoring, early termination based on external criteria, or logging to experiment tracking tools (MLflow, Weights & Biases).

20. **Warm-starting from partial assignments** — Accept an optional `x_init` argument to `gbsb_optimize` that fixes certain spin values before running, useful for decomposition-based approaches on structured large instances.

---

## Performance notes

| Instance size | Recommended backend | Typical time |
|---|---|---|
| N ≤ 200 | `gbsb_solver.py` (NumPy, CPU) | < 5s for 50 trials |
| 200 < N ≤ 2000 | `gbsb_torch.py` (CPU batched) | 10–60s for 128 trials |
| N > 2000 | `gbsb_torch.py` (CUDA) | 5–30s for 256 trials |

**Tuning tips:**

- Start with `auto_tune_A=True` and `local_search=True` — these are the two highest-impact improvements on most instances.
- Increase `num_trials` before increasing `M`. The solver benefits more from diversity (more restarts) than from longer integration on most instances.
- For very hard instances (low success rate), increase `M` and widen `A_search_grid` to include values outside the default [1.0, 3.5] range.
- The 2-flip local search (`local_search_flips=2`) is $O(N^2)$ per pass and can be slow for $N > 5{,}000$. Set `local_search_flips=1` in that regime.

---

## Background: Ising & QUBO problems

The **Ising model** asks: given a symmetric coupling matrix $J$ and linear field $h$, find $s^* \in \{-1, +1\}^N$ minimising:

$$E(s) = -\frac{1}{2} s^\top J s - h^\top s$$

**QUBO** asks: given a matrix $Q$, find $x^* \in \{0, 1\}^N$ minimising:

$$E(x) = x^\top Q x$$

These are equivalent under the substitution $s_i = 2x_i - 1$. `qubo_to_ising` implements this conversion exactly. Because both problems are NP-hard in general, GbSB is a heuristic — it finds high-quality solutions quickly but does not guarantee global optimality. For small instances (N ≤ 20) where exact solutions are needed, compare against brute force or branch-and-bound.
