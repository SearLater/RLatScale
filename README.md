# RLatScale

**Investigation of GPU-based simulation for Reinforcement Learning at Scale**

A benchmarking study comparing CPU and JAX-native GPU simulation backends using a consistent PPO implementation, targeting publication as an empirical research contribution.

---

## Research Questions

1. What wall-clock speedup does JAX-native simulation (Gymnax, Brax, MJX) provide over CPU equivalents (Gymnasium, MuJoCo) for PPO training?
2. Does the speedup profile differ between Apple Silicon (M3, Metal) and NVIDIA (4090, CUDA)?
3. Within the JAX ecosystem, do different neural-network frameworks (Linen, NNX, Ion) carry measurable throughput differences?
4. Is sample efficiency (return vs environment steps) equivalent across backends, confirming algorithmic consistency?

---

## Environments

| CPU Baseline | JAX-native GPU | Action Space | Obs / Act |
|---|---|---|---|
| Gymnasium `CartPole-v1` | Gymnax `CartPole-v1` | Discrete | 4 / 2 |
| Gymnasium `Pendulum-v1` | Gymnax `Pendulum-v1` | Continuous | 3 / 1 |
| MuJoCo `HalfCheetah-v4` | Brax `halfcheetah` · MJX `halfcheetah` | Continuous | 17 / 6 |
| MuJoCo `Ant-v4` | Brax `ant` · MJX `ant` | Continuous | 27 / 8 |

> Brax uses a different contact model from MuJoCo/MJX. Environments are treated as functionally analogous; comparisons use shared performance thresholds rather than raw reward parity. MJX loads the same XML files as Gymnasium, so observations and rewards are identical to the CPU baseline.

---

## PPO Implementations

Three JAX implementations of PPO, all sharing identical GAE computation and minibatch update logic:

| Implementation | Framework | JIT scope | Notes |
|---|---|---|---|
| **Linen** | Flax Linen | Full training loop via `jax.lax.scan` | Functional, XLA-transparent; primary comparison against PureJAXRL |
| **NNX** | Flax NNX | Rollout only; Python-level update loop | Imperative/PyTorch-like; benchmarks the ergonomics tradeoff |
| **Ion** | Ion (`ion-nn`) | Full training loop via `jax.lax.scan` | Pure-pytree networks; dict-based optimizer for separate actor/critic treatment |

### Ion optimizer design

Ion exposes pure pytree networks compatible with `jax.lax.scan`. The optimizer uses dict-based param groups so the actor and critic receive independent gradient clipping and learning-rate schedules:

```python
ion.Optimizer(
    {
        ("actor", "std_raw"): optax.chain(          # continuous: actor + log-std
            optax.clip_by_global_norm(0.5),
            optax.adam(lr_actor, eps=1e-5),
        ),
        "critic": optax.chain(
            optax.clip_by_global_norm(10.0),        # critic needs larger gradient steps early
            optax.adam(lr_critic, eps=1e-5),
        ),
    },
    network,
)
```

For discrete environments the actor key is `"actor"` (no `std_raw`). Combined actor+critic losses are computed in a single forward pass; gradients flow only to the relevant param group for each term.

---

## Network Architecture

`ActorCriticContinuous` (MuJoCo / Brax / MJX / Pendulum):
- Shared encoder: none — actor and critic are independent MLPs
- 2 hidden layers of `hidden_dim` units, orthogonal init, tanh activations
- Actor head: mean + `softplus(std_raw) + 1e-6` (state-dependent std)
- Critic head: scalar value

`ActorCritic` (CartPole):
- 2 hidden layers of `hidden_dim` units, tanh activations
- Actor head: logits (categorical policy)
- Critic head: scalar value

---

## Configuration

```
algo/config.py
```

| Config | Environments | `hidden_dim` | `num_envs` | `num_steps` | `total_timesteps` |
|---|---|---|---|---|---|
| `Config` | CartPole, Pendulum (CPU) | 64 | 4 | 128 | 1 M |
| `GPUConfig` | CartPole, Pendulum (Gymnax) | 64 | 2 048 | 128 | 50 M |
| `MuJoCoConfig` | HalfCheetah, Ant (CPU) | 256 | 1 | 2 048 | 2 M |
| `BraxConfig` | halfcheetah, ant (Brax) | 256 | 4 096 | 64 | 50 M |
| `MjxConfig` | halfcheetah, ant (MJX) | 256 | 2 048 | 128 | 50 M |

Key shared hyperparameters: `lr_actor=3e-4`, `lr_critic=1e-3`, `max_grad_norm_actor=0.5`, `max_grad_norm_critic=10.0`, `clip_eps=0.2`, `gae_lambda=0.95`, `gamma=0.99`.

---

## Project Structure

```
algo/
  config.py          # Config dataclasses (Config, GPUConfig, MuJoCoConfig, BraxConfig, MjxConfig)
  ppo_linen.py       # Flax Linen PPO (discrete + continuous)
  ppo_nnx.py         # Flax NNX PPO (discrete + continuous)
  ppo_ion.py         # Ion PPO networks (ActorCritic, ActorCriticContinuous)
  ppo_continous.py   # Reference continuous PPO (used to validate Ion optimizer design)
  distributions.py   # Normal distribution helpers

gym_test/
  cpu_test.py        # Gymnasium CartPole + Pendulum — Linen, NNX, Ion
  gpu_test.py        # Gymnax CartPole + Pendulum — Linen, NNX, Ion (JAX scan)

mujoco_test/
  cpu_test.py        # MuJoCo HalfCheetah + Ant — Linen, NNX, Ion (Python loop)
  brax_test.py       # Brax HalfCheetah + Ant — Linen, NNX, Ion (JAX scan)
  mjx_test.py        # MJX HalfCheetah + Ant — Linen, NNX, Ion (JAX scan)

utils/
  gym_eval.py        # Evaluation helpers for Gymnasium experiments
  gym_scaling.py     # Scaling / throughput analysis for Gymnasium
  mujoco_eval.py     # Evaluation helpers for MuJoCo experiments
  mujoco_scaling.py  # Scaling / throughput analysis for MuJoCo

results/             # JSON result files written by each test module
results/brax/
results/mjx/
```

---

## Evaluation Metrics

| Metric | Measurement |
|---|---|
| Sample efficiency | Episode return vs cumulative env steps; AUC normalised to a performance threshold |
| Wall-clock training time | Time-to-threshold (s); steady-state steps/second |
| Resource utilisation | GPU util %, peak VRAM, CPU util %, memory bandwidth |
| Algorithm stability | IQM and 95% CI across seeds via `rliable` |

---

## Running Experiments

```bash
# Install (CPU)
uv sync

# Install (GPU — adds jax[cuda12])
uv sync --extra gpu

# Gymnasium CPU baseline (CartPole + Pendulum, Linen / NNX / Ion)
python -m RLatScale.gym_test.cpu_test

# Gymnax GPU (CartPole + Pendulum, Linen / NNX / Ion, full scan)
python -m RLatScale.gym_test.gpu_test

# MuJoCo CPU baseline (HalfCheetah + Ant)
python -m RLatScale.mujoco_test.cpu_test

# Brax GPU (HalfCheetah + Ant, JAX-native physics)
python -m RLatScale.mujoco_test.brax_test

# MJX GPU (HalfCheetah + Ant, identical XML to MuJoCo baseline)
python -m RLatScale.mujoco_test.mjx_test
```

Results are written as JSON files under `results/`.

---

## Hardware

| System | CPU | GPU | Backend |
|---|---|---|---|
| MacBook Air M3 | Apple M3 | Apple M3 (Metal) | JAX Metal |
| Linux workstation | — | NVIDIA RTX 4090 | JAX CUDA 12 |

---

## Dependencies

Python 3.11. Key packages: `jax`, `flax`, `ion-nn`, `gymnax`, `brax`, `mujoco` (MJX included), `gymnasium`, `optax`, `rliable`.
