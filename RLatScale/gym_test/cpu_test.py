"""
CPU baseline: PPO on Gymnasium environments.

Trains Flax Linen and Flax NNX PPO on:
  - CartPole-v1  (discrete action space)
  - Pendulum-v1  (continuous action space)

NOTE: ppo_linen.make_train and ppo_nnx.train both require a Gymnax-style
JAX-vmappable environment API and cannot be called directly with Gymnasium.
This module imports the network classes and shared helpers from algo and
wraps them in a Python-loop training function that uses gymnasium.vector.
The update logic is identical to the algo implementations.

Metrics collected
-----------------
- Sample efficiency  : episode return vs cumulative env steps; AUC to threshold
- Wall-clock time    : time-to-threshold (s); steady-state steps/second
- Resource usage     : CPU %, RAM MB, GPU % (NVML) / VRAM MB
- Stability          : IQM + 95 % CI across NUM_SEEDS seeds via rliable

Usage
-----
    python -m RLatScale.gym_test.cpu_test
"""

from __future__ import annotations

import dataclasses
import json
import platform
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — no display required
import matplotlib.pyplot as plt

import gymnasium
import jax
import jax.numpy as jnp
import numpy as np
import optax
import psutil
import flax.linen as nn
from flax import nnx
from flax.training.train_state import TrainState
from rliable import library as rly
from rliable import metrics as rl_metrics
from tqdm import tqdm

from RLatScale.algo.config import Config
from RLatScale.algo.ppo_linen import Actor as LinenDiscreteActor
from RLatScale.algo.ppo_linen import Critic as LinenCritic
from RLatScale.algo.ppo_nnx import Actor as NNXDiscreteActor
from RLatScale.algo.ppo_nnx import Critic as NNXCritic

# ---------------------------------------------------------------------------
# Optional GPU monitoring (NVIDIA only; graceful fallback for Metal / CPU)
# ---------------------------------------------------------------------------
try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    _NVML = True
except Exception:
    _NVML = False

# ---------------------------------------------------------------------------
# Environment metadata
# ---------------------------------------------------------------------------
# random_score / threshold are used for rliable normalisation and AUC cutoff.
_ENV_META: dict[str, dict] = {
    "CartPole-v1": {
        "action_type": "discrete",
        "threshold": 475.0,
        "random_score": 10.0,
    },
    "Pendulum-v1": {
        "action_type": "continuous",
        "threshold": -200.0,
        "random_score": -1200.0,
    },
}

# ---------------------------------------------------------------------------
# Continuous actor networks (not in algo — algo currently discrete-only)
# ---------------------------------------------------------------------------

class ContinuousActorLinen(nn.Module):
    """Gaussian policy head. Returns (mean, log_std) with learnable log_std."""

    action_dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.tanh(x)
        mean = nn.Dense(self.action_dim)(x)
        log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        return mean, jnp.clip(log_std, -20.0, 2.0)


class ContinuousActorNNX(nnx.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, rngs: nnx.Rngs):
        self.l1 = nnx.Linear(obs_dim, hidden_dim, rngs=rngs)
        self.l2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.mean_head = nnx.Linear(hidden_dim, action_dim, rngs=rngs)
        self.log_std = nnx.Param(jnp.zeros(action_dim))

    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        x = jax.nn.tanh(self.l1(x))
        x = jax.nn.tanh(self.l2(x))
        mean = self.mean_head(x)
        log_std = jnp.clip(self.log_std.value, -20.0, 2.0)
        return mean, jnp.broadcast_to(log_std, mean.shape)


# ---------------------------------------------------------------------------
# Distribution helpers
# ---------------------------------------------------------------------------

def _cat_log_prob(logits: jax.Array, actions: jax.Array) -> jax.Array:
    return jax.nn.log_softmax(logits, axis=-1)[jnp.arange(logits.shape[0]), actions.astype(jnp.int32)]


def _cat_entropy(logits: jax.Array) -> jax.Array:
    p = jax.nn.softmax(logits, axis=-1)
    return -jnp.sum(p * jax.nn.log_softmax(logits, axis=-1), axis=-1)


def _normal_log_prob(x: jax.Array, mean: jax.Array, log_std: jax.Array) -> jax.Array:
    std = jnp.exp(log_std)
    return (-0.5 * ((x - mean) / std) ** 2 - log_std - 0.5 * jnp.log(2 * jnp.pi)).sum(-1)


def _normal_entropy(log_std: jax.Array) -> jax.Array:
    return (0.5 + 0.5 * jnp.log(2 * jnp.pi) + log_std).sum(-1)


# ---------------------------------------------------------------------------
# GAE — numpy implementation for Python-loop rollouts
# ---------------------------------------------------------------------------

def _gae_numpy(
    rewards: np.ndarray,     # (T, num_envs)
    values: np.ndarray,      # (T, num_envs)
    dones: np.ndarray,       # (T, num_envs)
    last_values: np.ndarray, # (num_envs,)
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    T = rewards.shape[0]
    advantages = np.zeros_like(rewards)
    gae = np.zeros(rewards.shape[1], dtype=np.float32)
    for t in reversed(range(T)):
        nv = last_values if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * nv * (1.0 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1.0 - dones[t]) * gae
        advantages[t] = gae
    return advantages, advantages + values


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

def _detect_hardware() -> str:
    """Return a short tag identifying the current compute backend.

    Used to prefix result directories so CPU (M3), Metal (M3 + jax-metal),
    and CUDA GPU runs are kept separate automatically.

    Examples
    --------
    "cpu_m3"      — Apple Silicon Mac, JAX CPU backend
    "metal_arm"   — Apple Silicon Mac, JAX Metal backend (jax-metal installed)
    "rtx_4090"    — NVIDIA RTX 4090, JAX CUDA backend
    "gpu"         — unrecognised NVIDIA GPU (NVML unavailable)
    "cpu"         — non-Apple CPU-only machine
    """
    backend = jax.default_backend()

    if backend == "metal":
        chip = platform.processor() or "apple_silicon"
        return f"metal_{chip.lower().replace(' ', '_')}"

    if backend == "gpu":
        if _NVML:
            try:
                raw = pynvml.nvmlDeviceGetName(_NVML_HANDLE)
                name = raw.decode() if isinstance(raw, bytes) else raw
                # "NVIDIA GeForce RTX 4090" → "rtx_4090"
                tag = (
                    name.lower()
                    .replace("nvidia geforce ", "")
                    .replace("nvidia ", "")
                    .replace(" ", "_")
                )
                return tag
            except Exception:
                pass
        return "gpu"

    # CPU backend — distinguish Apple Silicon from x86
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "cpu_m3"
    return "cpu"


# ---------------------------------------------------------------------------
# Resource monitor
# ---------------------------------------------------------------------------

@dataclass
class ResourceMonitor:
    """Samples CPU/RAM and GPU/VRAM (if NVML available) in a background thread."""

    interval: float = 0.5
    cpu_pct:  list[float] = field(default_factory=list)
    ram_mb:   list[float] = field(default_factory=list)
    gpu_pct:  list[float] = field(default_factory=list)
    vram_mb:  list[float] = field(default_factory=list)
    _stop:    threading.Event = field(default_factory=threading.Event)
    _thread:  threading.Thread | None = field(default=None)

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self.cpu_pct.append(psutil.cpu_percent())
            self.ram_mb.append(psutil.virtual_memory().used / 1024 ** 2)
            if _NVML:
                r = pynvml.nvmlDeviceGetUtilizationRates(_NVML_HANDLE)
                m = pynvml.nvmlDeviceGetMemoryInfo(_NVML_HANDLE)
                self.gpu_pct.append(float(r.gpu))
                self.vram_mb.append(m.used / 1024 ** 2)

    @property
    def summary(self) -> dict[str, float]:
        def _s(lst: list[float]) -> tuple[float, float]:
            return (float(np.mean(lst)), float(np.max(lst))) if lst else (float("nan"), float("nan"))

        mean_cpu, peak_cpu = _s(self.cpu_pct)
        _, peak_ram        = _s(self.ram_mb)
        _, peak_gpu        = _s(self.gpu_pct) if _NVML else (float("nan"), float("nan"))
        _, peak_vram       = _s(self.vram_mb) if _NVML else (float("nan"), float("nan"))
        return {
            "mean_cpu_pct": mean_cpu,
            "peak_cpu_pct": peak_cpu,
            "peak_ram_mb":  peak_ram,
            "peak_gpu_pct": peak_gpu,
            "peak_vram_mb": peak_vram,
        }


# ---------------------------------------------------------------------------
# Linen — Gymnasium training
# ---------------------------------------------------------------------------
# The scan-based make_train from ppo_linen requires a Gymnax JAX-vmappable
# environment.  Here we use a Python rollout loop and jax.jit the per-minibatch
# update, which is equivalent in algorithm but not in compilation strategy.

def train_linen_gymnasium(
    config: Config,
    env_id: str,
    seed: int,
) -> dict:
    """Run one PPO (Flax Linen) training trial and return per-rollout metrics."""

    meta = _ENV_META[env_id]
    is_cont  = meta["action_type"] == "continuous"
    threshold = meta["threshold"]

    envs = gymnasium.vector.SyncVectorEnv(
        [lambda: gymnasium.make(env_id) for _ in range(config.num_envs)]
    )
    obs_dim    = envs.single_observation_space.shape[0]
    action_dim = (
        envs.single_action_space.shape[0] if is_cont
        else int(envs.single_action_space.n)
    )
    act_low  = envs.single_action_space.low  if is_cont else None
    act_high = envs.single_action_space.high if is_cont else None

    rng = jax.random.key(seed)
    rng, rng_a, rng_c = jax.random.split(rng, 3)
    dummy = jnp.zeros((1, obs_dim))

    actor_net  = (
        ContinuousActorLinen(action_dim=action_dim, hidden_dim=config.hidden_dim)
        if is_cont else
        LinenDiscreteActor(action_dim=action_dim, hidden_dim=config.hidden_dim)
    )
    critic_net = LinenCritic(hidden_dim=config.hidden_dim)

    n_updates = config.num_rollouts * config.num_epochs * config.num_minibatches

    def _opt(lr: float, clip: float) -> optax.GradientTransformation:
        sched = optax.linear_schedule(lr, 0.0, n_updates) if config.anneal_lr else lr
        return optax.chain(optax.clip_by_global_norm(clip), optax.adam(sched, eps=1e-5))

    actor_st = TrainState.create(
        apply_fn=actor_net.apply,
        params=actor_net.init(rng_a, dummy),
        tx=_opt(config.lr_actor, config.max_grad_norm_actor),
    )
    critic_st = TrainState.create(
        apply_fn=critic_net.apply,
        params=critic_net.init(rng_c, dummy),
        tx=_opt(config.lr_critic, config.max_grad_norm_critic),
    )

    @jax.jit
    def _update_mb(a_st, c_st, obs_mb, act_mb, lp_old, adv, tgt):
        if config.advantage_norm:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        def actor_loss(params):
            if is_cont:
                mean, log_std = a_st.apply_fn(params, obs_mb)
                lp  = _normal_log_prob(act_mb, mean, log_std)
                ent = _normal_entropy(log_std)
            else:
                logits = a_st.apply_fn(params, obs_mb)
                lp  = _cat_log_prob(logits, act_mb)
                ent = _cat_entropy(logits)
            ratio = jnp.exp(lp - lp_old)
            pg = jnp.maximum(
                -adv * ratio,
                -adv * jnp.clip(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps),
            ).mean()
            return pg - config.entropy_beta * ent.mean()

        def critic_loss(params):
            return 0.5 * jnp.mean((c_st.apply_fn(params, obs_mb) - tgt) ** 2)

        return (
            a_st.apply_gradients(grads=jax.grad(actor_loss)(a_st.params)),
            c_st.apply_gradients(grads=jax.grad(critic_loss)(c_st.params)),
        )

    obs_np, _ = envs.reset(seed=seed)
    ep_buf    = np.zeros(config.num_envs, dtype=np.float32)
    completed: list[tuple[int, float]] = []

    returns_by_rollout: list[float] = []
    steps_by_rollout:   list[int]   = []
    time_by_rollout:    list[float] = []
    time_to_threshold  = -1.0
    steps_to_threshold = -1
    total_steps        = 0
    t0 = time.perf_counter()
    rng_np = np.random.default_rng(seed)

    for _ in tqdm(range(config.num_rollouts), desc=f"linen/{env_id}/s{seed}", leave=False):
        T, N = config.num_steps, config.num_envs
        O  = np.zeros((T, N, obs_dim),  np.float32)
        A  = np.zeros((T, N, action_dim) if is_cont else (T, N),
                      np.float32 if is_cont else np.int32)
        R  = np.zeros((T, N),            np.float32)
        D  = np.zeros((T, N),            np.float32)
        LP = np.zeros((T, N),            np.float32)
        V  = np.zeros((T, N),            np.float32)

        for t in range(T):
            obs_j = jnp.array(obs_np)
            rng, key = jax.random.split(rng)

            if is_cont:
                mean, log_std = actor_st.apply_fn(actor_st.params, obs_j)
                eps   = jax.random.normal(key, mean.shape)
                act_j = jnp.clip(mean + jnp.exp(log_std) * eps, act_low, act_high)
                lp    = _normal_log_prob(act_j, mean, log_std)
                act_np = np.array(act_j)
            else:
                logits = actor_st.apply_fn(actor_st.params, obs_j)
                act_j  = jax.random.categorical(key, logits)
                lp     = _cat_log_prob(logits, act_j)
                act_np = np.array(act_j)

            val = critic_st.apply_fn(critic_st.params, obs_j)
            obs_next, rew, term, trunc, _ = envs.step(act_np)
            done = (term | trunc).astype(np.float32)

            O[t] = obs_np; A[t] = act_np; R[t] = rew
            D[t] = done;   LP[t] = np.array(lp); V[t] = np.array(val)

            ep_buf += rew
            for i in range(N):
                if done[i]:
                    completed.append((total_steps + t * N, float(ep_buf[i])))
                    ep_buf[i] = 0.0
            obs_np = obs_next

        total_steps += config.batch_size
        last_val = np.array(critic_st.apply_fn(critic_st.params, jnp.array(obs_np)))
        adv_all, tgt_all = _gae_numpy(R, V, D, last_val, config.gamma, config.gae_lambda)

        B    = config.batch_size
        O_f  = O.reshape(B, obs_dim)
        A_f  = A.reshape(B, action_dim) if is_cont else A.reshape(B)
        LP_f = LP.reshape(B)
        adv_f = adv_all.reshape(B)
        tgt_f = tgt_all.reshape(B)

        for _ in range(config.num_epochs):
            perm = rng_np.permutation(B)
            for mb in range(config.num_minibatches):
                idx = perm[mb * config.minibatch_size : (mb + 1) * config.minibatch_size]
                actor_st, critic_st = _update_mb(
                    actor_st, critic_st,
                    jnp.array(O_f[idx]),  jnp.array(A_f[idx]),
                    jnp.array(LP_f[idx]), jnp.array(adv_f[idx]), jnp.array(tgt_f[idx]),
                )

        elapsed = time.perf_counter() - t0
        win = [r for s, r in completed if s > total_steps - config.batch_size * 5]
        mean_ret = float(np.mean(win)) if win else float("nan")
        returns_by_rollout.append(mean_ret)
        steps_by_rollout.append(total_steps)
        time_by_rollout.append(elapsed)

        if time_to_threshold < 0 and not np.isnan(mean_ret) and mean_ret >= threshold:
            time_to_threshold  = elapsed
            steps_to_threshold = total_steps

    envs.close()
    total_time = time.perf_counter() - t0
    return {
        "impl":                "linen",
        "env":                 env_id,
        "seed":                seed,
        "returns_by_rollout":  returns_by_rollout,
        "steps_by_rollout":    steps_by_rollout,
        "time_by_rollout":     time_by_rollout,
        "time_to_threshold":   time_to_threshold,
        "steps_to_threshold":  steps_to_threshold,
        "steps_per_second":    total_steps / total_time,
        "total_time":          total_time,
    }


# ---------------------------------------------------------------------------
# NNX — Gymnasium training
# ---------------------------------------------------------------------------

def train_nnx_gymnasium(
    config: Config,
    env_id: str,
    seed: int,
) -> dict:
    """Run one PPO (Flax NNX) training trial and return per-rollout metrics."""

    meta = _ENV_META[env_id]
    is_cont  = meta["action_type"] == "continuous"
    threshold = meta["threshold"]

    envs = gymnasium.vector.SyncVectorEnv(
        [lambda: gymnasium.make(env_id) for _ in range(config.num_envs)]
    )
    obs_dim    = envs.single_observation_space.shape[0]
    action_dim = (
        envs.single_action_space.shape[0] if is_cont
        else int(envs.single_action_space.n)
    )
    act_low  = envs.single_action_space.low  if is_cont else None
    act_high = envs.single_action_space.high if is_cont else None

    rng = jax.random.key(seed)
    rng, rng_a, rng_c = jax.random.split(rng, 3)
    seed_a = int(jax.random.randint(rng_a, (), 0, 2**16))
    seed_c = int(jax.random.randint(rng_c, (), 0, 2**16))

    actor = (
        ContinuousActorNNX(obs_dim, action_dim, config.hidden_dim, nnx.Rngs(seed_a))
        if is_cont else
        NNXDiscreteActor(obs_dim, action_dim, config.hidden_dim, nnx.Rngs(seed_a))
    )
    critic = NNXCritic(obs_dim, config.hidden_dim, nnx.Rngs(seed_c))

    n_updates = config.num_rollouts * config.num_epochs * config.num_minibatches

    def _opt(lr: float, clip: float) -> optax.GradientTransformation:
        sched = optax.linear_schedule(lr, 0.0, n_updates) if config.anneal_lr else lr
        return optax.chain(optax.clip_by_global_norm(clip), optax.adam(sched, eps=1e-5))

    actor_opt  = nnx.Optimizer(actor,  _opt(config.lr_actor,  config.max_grad_norm_actor))
    critic_opt = nnx.Optimizer(critic, _opt(config.lr_critic, config.max_grad_norm_critic))

    @nnx.jit
    def _forward(actor, critic, obs_j):
        return actor(obs_j), critic(obs_j)

    @nnx.jit
    def _update_mb(actor, critic, actor_opt, critic_opt, obs_mb, act_mb, lp_old, adv, tgt):
        if config.advantage_norm:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        def actor_loss(actor):
            if is_cont:
                mean, log_std = actor(obs_mb)
                lp  = _normal_log_prob(act_mb, mean, log_std)
                ent = _normal_entropy(log_std)
            else:
                logits = actor(obs_mb)
                lp  = _cat_log_prob(logits, act_mb)
                ent = _cat_entropy(logits)
            ratio = jnp.exp(lp - lp_old)
            pg = jnp.maximum(
                -adv * ratio,
                -adv * jnp.clip(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps),
            ).mean()
            return pg - config.entropy_beta * ent.mean()

        def critic_loss(critic):
            return 0.5 * jnp.mean((critic(obs_mb) - tgt) ** 2)

        actor_opt.update(nnx.grad(actor_loss)(actor))
        critic_opt.update(nnx.grad(critic_loss)(critic))

    obs_np, _ = envs.reset(seed=seed)
    ep_buf    = np.zeros(config.num_envs, dtype=np.float32)
    completed: list[tuple[int, float]] = []

    returns_by_rollout: list[float] = []
    steps_by_rollout:   list[int]   = []
    time_by_rollout:    list[float] = []
    time_to_threshold  = -1.0
    steps_to_threshold = -1
    total_steps        = 0
    t0 = time.perf_counter()
    rng_np = np.random.default_rng(seed)

    for _ in tqdm(range(config.num_rollouts), desc=f"nnx/{env_id}/s{seed}", leave=False):
        T, N = config.num_steps, config.num_envs
        O  = np.zeros((T, N, obs_dim),  np.float32)
        A  = np.zeros((T, N, action_dim) if is_cont else (T, N),
                      np.float32 if is_cont else np.int32)
        R  = np.zeros((T, N),            np.float32)
        D  = np.zeros((T, N),            np.float32)
        LP = np.zeros((T, N),            np.float32)
        V  = np.zeros((T, N),            np.float32)

        for t in range(T):
            obs_j = jnp.array(obs_np)
            rng, key = jax.random.split(rng)

            actor_out, val = _forward(actor, critic, obs_j)

            if is_cont:
                mean, log_std = actor_out
                eps   = jax.random.normal(key, mean.shape)
                act_j = jnp.clip(mean + jnp.exp(log_std) * eps, act_low, act_high)
                lp    = _normal_log_prob(act_j, mean, log_std)
                act_np = np.array(act_j)
            else:
                logits = actor_out
                act_j  = jax.random.categorical(key, logits)
                lp     = _cat_log_prob(logits, act_j)
                act_np = np.array(act_j)

            obs_next, rew, term, trunc, _ = envs.step(act_np)
            done = (term | trunc).astype(np.float32)

            O[t] = obs_np; A[t] = act_np; R[t] = rew
            D[t] = done;   LP[t] = np.array(lp); V[t] = np.array(val)

            ep_buf += rew
            for i in range(N):
                if done[i]:
                    completed.append((total_steps + t * N, float(ep_buf[i])))
                    ep_buf[i] = 0.0
            obs_np = obs_next

        total_steps += config.batch_size
        _, last_val = _forward(actor, critic, jnp.array(obs_np))
        adv_all, tgt_all = _gae_numpy(R, V, D, np.array(last_val), config.gamma, config.gae_lambda)

        B    = config.batch_size
        O_f  = O.reshape(B, obs_dim)
        A_f  = A.reshape(B, action_dim) if is_cont else A.reshape(B)
        LP_f = LP.reshape(B)
        adv_f = adv_all.reshape(B)
        tgt_f = tgt_all.reshape(B)

        for _ in range(config.num_epochs):
            perm = rng_np.permutation(B)
            for mb in range(config.num_minibatches):
                idx = perm[mb * config.minibatch_size : (mb + 1) * config.minibatch_size]
                _update_mb(
                    actor, critic, actor_opt, critic_opt,
                    jnp.array(O_f[idx]),  jnp.array(A_f[idx]),
                    jnp.array(LP_f[idx]), jnp.array(adv_f[idx]), jnp.array(tgt_f[idx]),
                )

        elapsed = time.perf_counter() - t0
        win = [r for r in [r for s, r in completed if s > total_steps - config.batch_size * 5]]
        mean_ret = float(np.mean(win)) if win else float("nan")
        returns_by_rollout.append(mean_ret)
        steps_by_rollout.append(total_steps)
        time_by_rollout.append(elapsed)

        if time_to_threshold < 0 and not np.isnan(mean_ret) and mean_ret >= threshold:
            time_to_threshold  = elapsed
            steps_to_threshold = total_steps

    envs.close()
    total_time = time.perf_counter() - t0
    return {
        "impl":                "nnx",
        "env":                 env_id,
        "seed":                seed,
        "returns_by_rollout":  returns_by_rollout,
        "steps_by_rollout":    steps_by_rollout,
        "time_by_rollout":     time_by_rollout,
        "time_to_threshold":   time_to_threshold,
        "steps_to_threshold":  steps_to_threshold,
        "steps_per_second":    total_steps / total_time,
        "total_time":          total_time,
    }


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------

def _normalise(returns: np.ndarray, env_id: str) -> np.ndarray:
    """Scale returns to [0, 1] using random and threshold baselines."""
    meta = _ENV_META[env_id]
    lo, hi = meta["random_score"], meta["threshold"]
    return np.clip((returns - lo) / (hi - lo + 1e-8), 0.0, None)


def aggregate_metrics(seed_results: list[dict], env_id: str) -> dict:
    """
    Compute rliable IQM + 95 % CI, AUC, and wall-clock statistics
    across multiple seeds.

    Parameters
    ----------
    seed_results : list of per-seed result dicts from train_*_gymnasium
    env_id       : environment id (used for normalisation constants)
    """
    n_seeds    = len(seed_results)
    n_rollouts = len(seed_results[0]["returns_by_rollout"])

    # Shape (num_seeds, num_rollouts) — mean episode return per rollout window
    raw_matrix = np.array(
        [r["returns_by_rollout"] for r in seed_results], dtype=np.float32
    )
    norm_matrix = _normalise(raw_matrix, env_id)  # (num_seeds, num_rollouts)

    # ── Sample efficiency: AUC of normalised learning curve ──────────────────
    auc_per_seed = np.trapz(norm_matrix, axis=1) / n_rollouts  # (num_seeds,)

    # ── rliable IQM + 95 % CI on per-seed AUC ────────────────────────────────
    # rliable expects shape (num_runs, num_tasks); single task → (num_seeds, 1)
    score_dict = {seed_results[0]["impl"]: auc_per_seed[:, None]}
    iqm_fn = lambda x: np.array([rl_metrics.aggregate_iqm(x)])  # noqa: E731
    iqm_scores, iqm_cis = rly.get_interval_estimates(score_dict, iqm_fn, reps=50_000)
    impl_key = seed_results[0]["impl"]

    # ── rliable IQM learning curve (IQM at each rollout checkpoint) ──────────
    iqm_curve = np.array([
        rl_metrics.aggregate_iqm(norm_matrix[:, t : t + 1])
        for t in range(n_rollouts)
    ])

    # ── Wall-clock ────────────────────────────────────────────────────────────
    t2t  = [r["time_to_threshold"]  for r in seed_results if r["time_to_threshold"]  > 0]
    s2t  = [r["steps_to_threshold"] for r in seed_results if r["steps_to_threshold"] > 0]
    sps  = [r["steps_per_second"]   for r in seed_results]

    return {
        "env":            env_id,
        "impl":           impl_key,
        "n_seeds":        n_seeds,
        # Sample efficiency
        "auc_mean":       float(np.mean(auc_per_seed)),
        "auc_std":        float(np.std(auc_per_seed)),
        "iqm":            float(iqm_scores[impl_key][0]),
        "iqm_ci_lo":      float(iqm_cis[impl_key][0][0]),
        "iqm_ci_hi":      float(iqm_cis[impl_key][1][0]),
        "iqm_curve":      iqm_curve.tolist(),
        "p25_curve":      np.percentile(norm_matrix, 25, axis=0).tolist(),
        "p75_curve":      np.percentile(norm_matrix, 75, axis=0).tolist(),
        # Wall-clock
        "mean_time_to_threshold":   float(np.mean(t2t))  if t2t else float("nan"),
        "median_time_to_threshold": float(np.median(t2t)) if t2t else float("nan"),
        "pct_reached_threshold":    len(t2t) / n_seeds,
        "mean_steps_to_threshold":  float(np.mean(s2t))  if s2t else float("nan"),
        "mean_steps_per_second":    float(np.mean(sps)),
        # Raw curves for plotting
        "steps_axis":     seed_results[0]["steps_by_rollout"],
        "norm_matrix":    norm_matrix.tolist(),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_learning_curve(metrics: dict, out_path: Path) -> None:
    """Save an IQM learning curve with P25–P75 band and per-seed traces."""
    steps = np.array(metrics["steps_axis"]) / 1e6
    norm  = np.array(metrics["norm_matrix"])   # (num_seeds, num_rollouts)
    iqm   = np.array(metrics["iqm_curve"])
    p25   = np.array(metrics["p25_curve"])
    p75   = np.array(metrics["p75_curve"])

    color = "#2563EB" if metrics["impl"] == "linen" else "#EA580C"

    fig, ax = plt.subplots(figsize=(7, 4))

    for seed_curve in norm:
        ax.plot(steps, seed_curve, color="grey", alpha=0.15, linewidth=0.7)

    ax.fill_between(steps, p25, p75, color=color, alpha=0.2, label="P25–P75")
    ax.plot(steps, iqm, color=color, linewidth=2.0,
            label=f"IQM ({metrics['impl'].upper()})")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, label="Threshold")

    ax.set_xlabel("Environment steps (M)")
    ax.set_ylabel("Normalised return")
    ax.set_title(
        f"{metrics['impl'].upper()} · {metrics['env']} · {metrics['n_seeds']} seeds"
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _json_safe(obj):
    """Recursively replace non-finite floats with None for valid JSON output."""
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def save_run(
    metrics: dict,
    seed_results: list[dict],
    config: Config,
    base_dir: str | Path = "results",
) -> Path:
    """
    Persist one (impl, env) run to a timestamped subdirectory.

    Directory layout
    ----------------
    results/cpu_{impl}_{env}_{timestamp}/
        config.json         — Config snapshot
        raw_seeds.json      — Per-seed return curves and timing
        metrics.json        — Aggregated rliable stats (no large arrays)
        curves.json         — Full normalised matrix + IQM/P25/P75 curves
        learning_curve.png  — IQM ± P25–P75 learning curve plot
    """
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    env_tag = metrics["env"].replace("-", "_")
    hw      = config.hardware_tag if config.hardware_tag else _detect_hardware()
    run_dir = Path(base_dir) / f"{hw}_{metrics['impl']}_{env_tag}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics["hardware"] = hw

    # Config snapshot
    (run_dir / "config.json").write_text(
        json.dumps(dataclasses.asdict(config), indent=2)
    )

    # Raw per-seed results (curves + timing only — omit impl/env duplication)
    _SEED_KEYS = (
        "seed", "returns_by_rollout", "steps_by_rollout", "time_by_rollout",
        "time_to_threshold", "steps_to_threshold", "steps_per_second", "total_time",
    )
    raw = [{k: r[k] for k in _SEED_KEYS if k in r} for r in seed_results]
    (run_dir / "raw_seeds.json").write_text(json.dumps(_json_safe(raw), indent=2))

    # Summary metrics (human-readable, no large arrays)
    _CURVE_KEYS = {"norm_matrix", "iqm_curve", "p25_curve", "p75_curve", "steps_axis"}
    summary = {k: v for k, v in metrics.items() if k not in _CURVE_KEYS}
    (run_dir / "metrics.json").write_text(json.dumps(_json_safe(summary), indent=2))

    # Full curve data for re-analysis and re-plotting without retraining
    curves = {k: metrics[k] for k in _CURVE_KEYS}
    (run_dir / "curves.json").write_text(json.dumps(curves, indent=2))

    # Learning curve plot
    _plot_learning_curve(metrics, run_dir / "learning_curve.png")

    return run_dir


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    config: Config,
    env_id: str,
    impl: str,
    num_seeds: int,
    monitor: ResourceMonitor,
) -> tuple[dict, list[dict]]:
    """Run `num_seeds` trials; return (aggregated metrics, raw seed results)."""

    train_fn = train_linen_gymnasium if impl == "linen" else train_nnx_gymnasium

    monitor.start()
    seed_results = []
    for seed in range(num_seeds):
        result = train_fn(config, env_id, seed)
        seed_results.append(result)
    monitor.stop()

    metrics = aggregate_metrics(seed_results, env_id)
    metrics["resource"] = monitor.summary
    return metrics, seed_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_summary(metrics: dict) -> None:
    impl = metrics["impl"]
    env  = metrics["env"]
    print(f"\n{'─'*60}")
    print(f"  {impl.upper()} · {env}  ({metrics['n_seeds']} seeds)")
    print(f"{'─'*60}")
    print(f"  Sample efficiency (AUC, normalised)")
    print(f"    mean ± std : {metrics['auc_mean']:.3f} ± {metrics['auc_std']:.3f}")
    print(f"    IQM        : {metrics['iqm']:.3f}  "
          f"95%CI [{metrics['iqm_ci_lo']:.3f}, {metrics['iqm_ci_hi']:.3f}]")
    print(f"  Wall-clock")
    print(f"    steps/s    : {metrics['mean_steps_per_second']:.0f}")
    print(f"    time→thr   : {metrics['mean_time_to_threshold']:.1f} s  "
          f"({metrics['pct_reached_threshold']*100:.0f}% seeds reached threshold)")
    r = metrics["resource"]
    print(f"  Resource utilisation")
    print(f"    CPU        : mean {r['mean_cpu_pct']:.1f}%  peak {r['peak_cpu_pct']:.1f}%")
    print(f"    RAM        : peak {r['peak_ram_mb']:.0f} MB")
    if not np.isnan(r["peak_gpu_pct"]):
        print(f"    GPU        : peak {r['peak_gpu_pct']:.1f}%  VRAM {r['peak_vram_mb']:.0f} MB")
    else:
        print(f"    GPU        : not available (NVML not found)")


def main() -> None:
    config = Config()

    for env_id in config.envs:
        for impl in config.impls:
            print(f"\nRunning {impl}/{env_id} × {config.num_seeds} seeds …")
            monitor = ResourceMonitor()
            metrics, seed_results = run_experiment(config, env_id, impl, config.num_seeds, monitor)
            _print_summary(metrics)
            run_dir = save_run(metrics, seed_results, config, base_dir=config.results_dir)
            print(f"  → {run_dir}")


if __name__ == "__main__":
    main()
