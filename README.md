# RLatScale

**Investigation of GPU-based simulation for Reinforcement Learning at Scale**

A benchmarking study comparing CPU and JAX-native GPU simulation backends using a consistent PPO implementation, targeting publication as an empirical research contribution.

---

## Research Questions

1. What wall-clock speedup does JAX-native simulation (Gymnax, Brax) provide over CPU equivalents (Gymnasium, MuJoCo) for PPO training?
2. Does the speedup profile differ between Apple Silicon (M3, Metal) and NVIDIA (4090, CUDA)?
3. Within the JAX ecosystem, does full end-to-end JIT (Flax Linen) outperform a more imperative style (Flax NNX) for training throughput?
4. Is sample efficiency (return vs environment steps) equivalent across backends, confirming algorithmic consistency?

---

## Environments

| CPU Baseline | JAX-native | Action Space |
|---|---|---|
| Gymnasium `CartPole-v1` | Gymnax `CartPole-v1` | Discrete |
| Gymnasium `Pendulum-v1` | Gymnax `Pendulum-v1` | Continuous |
| MuJoCo `HalfCheetah-v4` | Brax `halfcheetah` | Continuous |
| MuJoCo `Ant-v4` | Brax `ant` | Continuous |

Note: Brax and MuJoCo use different contact models. Environments are treated as functionally analogous rather than identical; comparisons use shared performance thresholds rather than raw reward values.

---

## PPO Implementations

Two JAX implementations of PPO supporting both discrete and continuous action spaces:

- **Flax Linen** — functional, JIT-transparent. Enables full end-to-end training loop compilation via `jax.lax.scan`. Primary comparison against PureJAXRL.
- **Flax NNX** — imperative, PyTorch-like. Mutable state model limits full-loop JIT; Python-level update loop. Tests whether the ergonomics tradeoff carries a throughput cost.

Both implementations share identical hyperparameters, GAE computation, and minibatch update logic. The only variable is the neural network framework.

---

## Evaluation Metrics

| Metric | Measurement |
|---|---|
| Sample efficiency | Return vs millions of env steps; AUC to a performance threshold |
| Wall-clock training time | Time-to-threshold (seconds); steady-state steps/second |
| Resource utilisation | GPU util %, peak VRAM, CPU util %, memory bandwidth |
| Algorithm stability | IQM and 95% CI across 10 seeds via `rliable` |

---

## Hardware

| System | CPU | GPU | Backend |
|---|---|---|---|
| MacBook Air M3 | Apple M3 | Apple M3 (Metal) | JAX Metal, MPS |
| Linux workstation | — | NVIDIA RTX 4090 | JAX CUDA 12 |

---

## Project Structure

```
algo/          # PPO implementations (Flax Linen, Flax NNX)
gym_test/      # Gymnasium and Gymnax environment experiments
mujoco_test/   # MuJoCo and Brax environment experiments
```
