# RLatScale

**Benchmarking GPU-accelerated simulation for Reinforcement Learning at Scale**

A controlled empirical study comparing CPU and JAX-native GPU simulation backends
using a consistent PPO implementation (Ion), targeting publication as an empirical
research contribution.

---

## Research Questions

1. What wall-clock and throughput speedup does JAX-native simulation (Gymnax, Brax, MJX) provide over CPU equivalents (Gymnasium, MuJoCo) for PPO training?
2. How does throughput scale with the number of parallel environments for each backend, and where does each backend saturate or degrade?
3. Is sample efficiency (IQM of normalised return vs environment steps) equivalent across backends, or does large-scale GPU parallelism affect policy quality?
4. Does physics simulation fidelity (Brax vs MJX vs CPU MuJoCo) measurably affect learned policy quality on contact-rich tasks?

> **Status**: RQ1–4 answered for the Ion implementation. Linen and NNX comparisons (originally RQ3) are supported by the codebase but not yet benchmarked.

---

## Results Summary

All experiments were run on an **RTX 5090 Laptop GPU** (24 GB GDDR7). Results are
stored under `results/` with one timestamped subdirectory per evaluation run.

### Simple Control (CartPole-v1, Pendulum-v1)

| Backend | Environment | IQM | Steps/s | Time to Threshold (s) |
|---|---|---|---|---|
| CPU (Gymnasium) | CartPole-v1 | 1.029 | 3,088 | 13.2 |
| GPU (Gymnax) | CartPole-v1 | 0.995 | 1,112,262 | 9.3 |
| CPU (Gymnasium) | Pendulum-v1 | 0.766 | 2,616 | 201.5 |
| GPU (Gymnax) | Pendulum-v1 | 0.961 | 1,173,198 | 20.2 |

GPU throughput is **360× faster** on CartPole and **448× faster** on Pendulum. The
wall-clock advantage is 1.4× on CartPole and 10× on Pendulum — smaller than the
throughput ratio because the CPU requires far fewer steps to converge (33k vs 5.2M
for CartPole). IQM is not directly comparable due to different total step budgets
(1M CPU vs 50M GPU). Both backends used 1 seed; results are qualitative only.

### Simple Control Throughput Scaling (CartPole-v1)

GPU and CPU throughput are equivalent below ~512 parallel environments. Above that
the GPU scales near-linearly, reaching **11.3M steps/s at 524,288 environments**
while the CPU saturates at ~228k steps/s above ~131k environments. At the default
eval configuration (2,048 GPU environments) raw throughput is 312k steps/s; the
difference from the eval figure (1.11M steps/s) is JIT amortisation over ~190
rollouts vs 5 in the scaling probe.

### Continuous Locomotion (HalfCheetah-v4, Ant-v4)

| Backend | Environment | Seeds | IQM | IQM 95% CI | Steps/s | Time to Threshold (s) |
|---|---|---|---|---|---|---|
| CPU (MuJoCo) | HalfCheetah-v4 | 5 | 1.077 | [1.035, 1.141] | 7,840 | 468.8 |
| Brax | HalfCheetah-v4 | 10 | **1.872** | [1.805, 1.926] | 70,311 | **187.8** |
| MJX | HalfCheetah-v4 | 10 | 1.271 | [1.243, 1.337] | 58,476 | 221.1 |
| CPU (MuJoCo) | Ant-v4 | 5 | 0.714 | [0.552, 0.891] | 4,111 | 1,670.6 |
| Brax | Ant-v4 | 10 | 0.861 | [0.841, 0.875] | 225,647 | **114.8** |
| MJX | Ant-v4 | 10 | **0.895** | [0.854, 0.935] | 16,999 | 1,602.1 |

CPU runs: 16 environments, 10M total steps, 5 seeds. Brax/MJX: 50M total steps, 10 seeds.

Key findings:
- **Brax achieves IQM 1.872 on HalfCheetah** — substantially above CPU (1.077) and MJX (1.271) — suggesting that 4,096-way parallelism improves policy quality beyond mere speed.
- **MJX beats Brax on Ant** (0.895 vs 0.861, barely overlapping CIs) despite 13× lower throughput. Ant's contact-rich morphology is sensitive to physics fidelity; Brax's simplified contact model biases the learned policy.
- **MJX Ant wall-clock nearly matches CPU** (1,602s vs 1,671s) because MJX's throughput advantage (4×) is consumed by the harder learning problem under faithful contact dynamics.
- **Brax is 54.9× faster than CPU on Ant** (225,647 vs 4,111 steps/s) but only 9× on HalfCheetah (70,311 vs 7,840 steps/s), reflecting Brax's insensitivity to contact geometry.

### Continuous Locomotion Throughput Scaling (HalfCheetah)

| Num Envs | CPU | Brax | MJX |
|---|---|---|---|
| 1,024 | 14,240 | — | 4,800 |
| 4,096 | 24,144 | 28,052 | 17,749 |
| 8,192 | 27,513 | 29,999 | 32,093 |
| 32,768 | — | 40,090 | **62,188** ← peak |
| 65,536 | — | 42,528 | 29,498 ← collapse |

- **MJX peaks at 32,768 environments then drops 52%** at 65,536 — a hard VRAM ceiling on 24 GB; the eval config (2,048 envs) is safely below it.
- **GPU backends do not surpass CPU until ~8,192 environments** for MuJoCo-class physics; below that the CPU baseline is faster.
- **Brax saturates early**: only 1.5× gain across a 16× increase in environment count, suggesting a GPU kernel occupancy ceiling.

---

## Environments

| CPU Baseline | JAX-native GPU | Action Space | Obs / Act dims |
|---|---|---|---|
| Gymnasium `CartPole-v1` | Gymnax `CartPole-v1` | Discrete | 4 / 1 |
| Gymnasium `Pendulum-v1` | Gymnax `Pendulum-v1` | Continuous | 3 / 1 |
| MuJoCo `HalfCheetah-v4` | Brax `halfcheetah` · MJX `halfcheetah` | Continuous | 17 / 6 |
| MuJoCo `Ant-v4` | Brax `ant` · MJX `ant` | Continuous | 27 / 8 |

> MJX loads the same XML model files as Gymnasium MuJoCo, making physics directly comparable. Brax uses a position-based contact model that diverges from the reference MuJoCo dynamics; the IQM gap on Ant (0.895 MJX vs 0.861 Brax) is attributable to this difference.

---

## PPO Implementation

All benchmarks use the **Ion** implementation. Linen and NNX implementations exist in the codebase and can be selected via the `impls` config field but have not been benchmarked in the current evaluation runs.

| Implementation | Framework | Notes |
|---|---|---|
| **Ion** | Ion (`ion-nn`) | Pure-pytree networks; dict-based optimizer for separate actor/critic treatment. Used for all current benchmarks. |
| **Linen** | Flax Linen | Full training loop via `jax.lax.scan`. Implemented, not yet benchmarked. |
| **NNX** | Flax NNX | Python-level update loop. Implemented, not yet benchmarked. |

### Ion optimizer — separate actor/critic treatment

```python
ion.Optimizer(
    {
        ("actor", "std_raw"): optax.chain(   # continuous: actor + log-std
            optax.clip_by_global_norm(0.5),
            optax.adam(lr_actor, eps=1e-5),
        ),
        "critic": optax.chain(
            optax.clip_by_global_norm(10.0), # critic tolerates larger gradient steps
            optax.adam(lr_critic, eps=1e-5),
        ),
    },
    network,
)
```

For discrete environments the actor key is `"actor"` (no `std_raw`).

---

## Network Architecture

**Continuous** (`ActorCriticContinuous` — MuJoCo / Brax / MJX / Pendulum):
- Actor and critic are independent 2-layer MLPs, orthogonal init, tanh activations
- Actor head: mean + `softplus(std_raw) + 1e-6` (state-independent std)
- Critic head: scalar value

**Discrete** (`ActorCritic` — CartPole):
- 2-layer MLP, tanh activations
- Actor head: logits (categorical policy)
- Critic head: scalar value

---

## Configuration

Current defaults in `algo/config.py`. **Note**: the MuJoCo evaluation runs stored in
`results/` used `num_envs=16` and `total_timesteps=10_000_000`; the current
`MuJoCoConfig` values differ and will not replicate those results directly.

| Config | Environments | `hidden_dim` | `num_envs` | `num_steps` | `total_timesteps` | `num_seeds` |
|---|---|---|---|---|---|---|
| `Config` | CartPole, Pendulum (CPU) | 64 | 4 | 128 | 1 M | 5 |
| `GPUConfig` | CartPole, Pendulum (Gymnax) | 64 | 2,048 | 128 | 50 M | 10 |
| `MuJoCoConfig` | HalfCheetah, Ant (CPU) | 256 | 10 | 2,048 | 2 M | 10 |
| `BraxConfig` | halfcheetah, ant (Brax) | 256 | 4,096 | 64 | 50 M | 10 |
| `MjxConfig` | halfcheetah, ant (MJX) | 256 | 2,048 | 128 | 50 M | 10 |

**Shared hyperparameters** (MuJoCo tier): `lr_actor=3e-4`, `lr_critic=1e-3`,
`gamma=0.99`, `gae_lambda=0.95`, `clip_eps=0.2`, `entropy_beta=0.0`,
`max_grad_norm_actor=0.5`, `max_grad_norm_critic=10.0`, 10 epochs, 32 minibatches.

**Simple control** uses `lr_actor=2.5e-4`, `entropy_beta=0.01`, 4 epochs, 4 minibatches.

---

## Project Structure

```
RLatScale/
  algo/
    config.py           # Config dataclasses (Config, GPUConfig, MuJoCoConfig, BraxConfig, MjxConfig)
    ppo_ion.py          # Ion actor-critic networks (discrete + continuous)
    ppo_linen.py        # Flax Linen PPO (implemented, not yet benchmarked)
    ppo_nnx.py          # Flax NNX PPO (implemented, not yet benchmarked)
    ppo_continous.py    # Reference continuous PPO (used to validate Ion optimizer)
    distributions.py    # Normal distribution helpers

  gym_test/
    cpu_test.py         # Gymnasium CartPole + Pendulum — Linen, NNX, Ion
    gpu_test.py         # Gymnax CartPole + Pendulum — Ion (JAX scan)

  mujoco_test/
    cpu_test.py         # MuJoCo HalfCheetah + Ant — Ion (Python loop)
    brax_test.py        # Brax HalfCheetah + Ant — Ion
    mjx_test.py         # MJX HalfCheetah + Ant — Ion

  utils/
    gym_eval.py         # Full CPU vs GPU evaluation: learning curves, throughput, summary table
    gym_scaling.py      # Throughput sweep: CPU vs GPU, 1→524k envs, CartPole
    mujoco_eval.py      # Full CPU vs Brax vs MJX evaluation: learning curves, throughput, summary
    mujoco_scaling.py   # Throughput sweep: CPU vs Brax vs MJX, 1→65k envs, HalfCheetah

results/
  gym_eval_YYYYMMDD_HHMMSS/
    summary.csv             # Per-run metrics
    summary_table.png
    {env}_curves.png        # IQM learning curves
    throughput.png          # Steps/s bar chart
    runs/{hw}_{impl}_{env}_{ts}/
      config.json           # Exact config used (authoritative record)
      metrics.json          # Aggregated rliable stats
      raw_seeds.json        # Per-seed return curves and timing
      curves.json           # Full normalised matrix for re-plotting
      learning_curve.png

  mujoco_eval_YYYYMMDD_HHMMSS/
    (same structure)

  gym_scaling_YYYYMMDD_HHMMSS/
    scaling.csv
    scaling.png

  mujoco_scaling_YYYYMMDD_HHMMSS/
    scaling.csv
    scaling.png
```

---

## Running Experiments

```bash
# Install (CPU only)
uv sync

# Install (NVIDIA GPU — adds jax[cuda12])
uv sync --extra gpu

# --- Evaluation runs (recommended entry points) ---

# Gymnasium: CPU vs GPU learning curves + throughput (CartPole + Pendulum)
python -m RLatScale.utils.gym_eval

# MuJoCo: CPU vs Brax vs MJX learning curves + throughput (HalfCheetah + Ant)
python -m RLatScale.utils.mujoco_eval

# --- Throughput scaling sweeps ---

# Gymnasium: steps/s vs num_envs, CartPole (CPU and GPU)
python -m RLatScale.utils.gym_scaling

# MuJoCo: steps/s vs num_envs, HalfCheetah (CPU, Brax, MJX)
# Each probe point runs in a fresh subprocess to avoid Brax/MJX C-library conflicts
python -m RLatScale.utils.mujoco_scaling

# --- Individual backend tests (for development/debugging) ---
python -m RLatScale.gym_test.cpu_test
python -m RLatScale.gym_test.gpu_test
python -m RLatScale.mujoco_test.cpu_test
python -m RLatScale.mujoco_test.brax_test
python -m RLatScale.mujoco_test.mjx_test
```

Each evaluation run writes a timestamped directory under `results/`. The
`config.json` inside each run subdirectory is the authoritative record of the
exact configuration used; the current `config.py` defaults may differ.

---

## Evaluation Metrics

| Metric | Detail |
|---|---|
| **IQM** | Interquartile Mean of per-seed AUC, 95% CI via stratified bootstrap (50k resamples), using `rliable` |
| **AUC** | Area under normalised learning curve; returns scaled to [0, 1] using per-task random and threshold baselines |
| **Steps/s** | Steady-state environment steps per second (wall-clock, includes all training overhead) |
| **Time to threshold** | Wall-clock seconds until mean episode return first exceeds the performance threshold |
| **Resource utilisation** | CPU %, peak RAM (psutil), GPU % and peak VRAM (NVML), sampled at 500ms intervals |

---

## Hardware

| Role | Machine | GPU | JAX backend |
|---|---|---|---|
| All benchmark runs | RTX 5090 Laptop | NVIDIA RTX 5090 Laptop (24 GB GDDR7) | CUDA |
| Development | MacBook | Apple M-series | CPU / Metal |

Hardware is auto-detected at runtime via NVML and recorded in each run directory.
The `hardware_tag` field in `config.py` is used only as a fallback label when
auto-detection is ambiguous.

---

## Dependencies

Python 3.11+. Key packages: `jax`, `flax`, `ion-nn`, `gymnax`, `brax`,
`mujoco` (includes MJX), `gymnasium[mujoco]`, `optax`, `rliable`, `psutil`.
