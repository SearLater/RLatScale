"""
CPU baseline: PPO on Gymnasium MuJoCo environments.

HalfCheetah-v4 and Ant-v4 are both continuous-control tasks; no discrete
branching is needed.  Structure mirrors gym_test/cpu_test.py; shared network
classes, distribution helpers, monitoring, and persistence utilities are
imported from there.

Usage
-----
    python -m RLatScale.mujoco_test.cpu_test
"""

from __future__ import annotations

import time

import gymnasium
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from flax.training.train_state import TrainState
from rliable import library as rly
from rliable import metrics as rl_metrics
from tqdm import tqdm

from RLatScale.algo.config import MuJoCoConfig
from RLatScale.algo.ppo_linen import Critic as LinenCritic
from RLatScale.algo.ppo_nnx import Critic as NNXCritic
from RLatScale.gym_test.cpu_test import (
    ContinuousActorLinen,
    ContinuousActorNNX,
    ResourceMonitor,
    _detect_hardware,
    _gae_numpy,
    _normal_entropy,
    _normal_log_prob,
    _print_summary,
    save_run,
)

# ---------------------------------------------------------------------------
# Environment metadata
# ---------------------------------------------------------------------------

_ENV_META: dict[str, dict] = {
    "HalfCheetah-v4": {
        "action_type": "continuous",
        "threshold":    3000.0,
        "random_score": -280.0,
    },
    "Ant-v4": {
        "action_type": "continuous",
        "threshold":    3000.0,
        "random_score": -150.0,
    },
}

# ---------------------------------------------------------------------------
# Metric aggregation (local — uses mujoco _ENV_META)
# ---------------------------------------------------------------------------

def _normalise(returns: np.ndarray, env_id: str) -> np.ndarray:
    meta = _ENV_META[env_id]
    lo, hi = meta["random_score"], meta["threshold"]
    return np.clip((returns - lo) / (hi - lo + 1e-8), 0.0, None)


def aggregate_metrics(seed_results: list[dict], env_id: str) -> dict:
    n_seeds    = len(seed_results)
    n_rollouts = len(seed_results[0]["returns_by_rollout"])

    raw_matrix  = np.array(
        [r["returns_by_rollout"] for r in seed_results], dtype=np.float32
    )
    norm_matrix = _normalise(raw_matrix, env_id)

    # Forward-fill NaN values (rollouts where no episode completed).
    # Leading NaN → 0 (no data yet = worst-case performance).
    norm_filled = norm_matrix.copy()
    for s in range(n_seeds):
        last = 0.0
        for t in range(n_rollouts):
            if np.isnan(norm_filled[s, t]):
                norm_filled[s, t] = last
            else:
                last = float(norm_filled[s, t])

    auc_per_seed = np.trapezoid(norm_filled, axis=1) / n_rollouts

    impl_key   = seed_results[0]["impl"]
    score_dict = {impl_key: auc_per_seed[:, None]}
    iqm_fn     = lambda x: np.array([rl_metrics.aggregate_iqm(x)])  # noqa: E731
    iqm_scores, iqm_cis = rly.get_interval_estimates(score_dict, iqm_fn, reps=50_000)

    iqm_curve = np.array([
        rl_metrics.aggregate_iqm(norm_filled[:, t : t + 1])
        for t in range(n_rollouts)
    ])

    t2t = [r["time_to_threshold"]  for r in seed_results if r["time_to_threshold"]  > 0]
    s2t = [r["steps_to_threshold"] for r in seed_results if r["steps_to_threshold"] > 0]
    sps = [r["steps_per_second"]   for r in seed_results]

    return {
        "env":     env_id,
        "impl":    impl_key,
        "n_seeds": n_seeds,
        "auc_mean":  float(np.mean(auc_per_seed)),
        "auc_std":   float(np.std(auc_per_seed)),
        "iqm":       float(iqm_scores[impl_key][0]),
        "iqm_ci_lo": float(iqm_cis[impl_key][0][0]),
        "iqm_ci_hi": float(iqm_cis[impl_key][1][0]),
        "iqm_curve": iqm_curve.tolist(),
        "p25_curve": np.percentile(norm_filled, 25, axis=0).tolist(),
        "p75_curve": np.percentile(norm_filled, 75, axis=0).tolist(),
        "mean_time_to_threshold":   float(np.mean(t2t))   if t2t else float("nan"),
        "median_time_to_threshold": float(np.median(t2t)) if t2t else float("nan"),
        "pct_reached_threshold":    len(t2t) / n_seeds,
        "mean_steps_to_threshold":  float(np.mean(s2t))   if s2t else float("nan"),
        "mean_steps_per_second":    float(np.mean(sps)),
        "steps_axis":  seed_results[0]["steps_by_rollout"],
        "norm_matrix": norm_matrix.tolist(),
    }


# ---------------------------------------------------------------------------
# Linen — Gymnasium MuJoCo training
# ---------------------------------------------------------------------------

def train_linen_mujoco(config: MuJoCoConfig, env_id: str, seed: int) -> dict:
    threshold = _ENV_META[env_id]["threshold"]

    envs = gymnasium.vector.SyncVectorEnv(
        [lambda: gymnasium.make(env_id) for _ in range(config.num_envs)]
    )
    obs_dim    = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.shape[0]
    act_low    = envs.single_action_space.low
    act_high   = envs.single_action_space.high

    rng = jax.random.key(seed)
    rng, rng_a, rng_c = jax.random.split(rng, 3)
    dummy = jnp.zeros((1, obs_dim))

    actor_net  = ContinuousActorLinen(action_dim=action_dim, hidden_dim=config.hidden_dim)
    critic_net = LinenCritic(hidden_dim=config.hidden_dim)

    n_updates = config.num_rollouts * config.num_epochs * config.num_minibatches

    def _opt(lr, clip):
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
            mean, log_std = a_st.apply_fn(params, obs_mb)
            lp  = _normal_log_prob(act_mb, mean, log_std)
            ent = _normal_entropy(log_std)
            ratio = jnp.exp(lp - lp_old)
            pg = jnp.maximum(
                -adv * ratio,
                -adv * jnp.clip(ratio, 1 - config.clip_eps, 1 + config.clip_eps),
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
    t0      = time.perf_counter()
    rng_np  = np.random.default_rng(seed)

    for _ in tqdm(range(config.num_rollouts), desc=f"linen/{env_id}/s{seed}", leave=False):
        T, N = config.num_steps, config.num_envs
        O  = np.zeros((T, N, obs_dim),  np.float32)
        A  = np.zeros((T, N, action_dim), np.float32)
        R  = np.zeros((T, N),             np.float32)
        D  = np.zeros((T, N),             np.float32)
        LP = np.zeros((T, N),             np.float32)
        V  = np.zeros((T, N),             np.float32)

        for t in range(T):
            obs_j = jnp.array(obs_np)
            rng, key = jax.random.split(rng)
            mean, log_std = actor_st.apply_fn(actor_st.params, obs_j)
            eps    = jax.random.normal(key, mean.shape)
            act_j  = jnp.clip(mean + jnp.exp(log_std) * eps, act_low, act_high)
            lp     = _normal_log_prob(act_j, mean, log_std)
            val    = critic_st.apply_fn(critic_st.params, obs_j)
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
        last_val = np.array(critic_st.apply_fn(critic_st.params, jnp.array(obs_np)))
        adv_all, tgt_all = _gae_numpy(R, V, D, last_val, config.gamma, config.gae_lambda)

        B     = config.batch_size
        O_f   = O.reshape(B, obs_dim)
        A_f   = A.reshape(B, action_dim)
        LP_f  = LP.reshape(B)
        adv_f = adv_all.reshape(B)
        tgt_f = tgt_all.reshape(B)

        for _ in range(config.num_epochs):
            perm = rng_np.permutation(B)
            for mb in range(config.num_minibatches):
                idx = perm[mb * config.minibatch_size : (mb + 1) * config.minibatch_size]
                actor_st, critic_st = _update_mb(
                    actor_st, critic_st,
                    jnp.array(O_f[idx]), jnp.array(A_f[idx]),
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
        "impl": "linen", "env": env_id, "seed": seed,
        "returns_by_rollout":  returns_by_rollout,
        "steps_by_rollout":    steps_by_rollout,
        "time_by_rollout":     time_by_rollout,
        "time_to_threshold":   time_to_threshold,
        "steps_to_threshold":  steps_to_threshold,
        "steps_per_second":    total_steps / total_time,
        "total_time":          total_time,
    }


# ---------------------------------------------------------------------------
# NNX — Gymnasium MuJoCo training
# ---------------------------------------------------------------------------

def train_nnx_mujoco(config: MuJoCoConfig, env_id: str, seed: int) -> dict:
    threshold = _ENV_META[env_id]["threshold"]

    envs = gymnasium.vector.SyncVectorEnv(
        [lambda: gymnasium.make(env_id) for _ in range(config.num_envs)]
    )
    obs_dim    = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.shape[0]
    act_low    = envs.single_action_space.low
    act_high   = envs.single_action_space.high

    rng = jax.random.key(seed)
    rng, rng_a, rng_c = jax.random.split(rng, 3)
    seed_a = int(jax.random.randint(rng_a, (), 0, 2**16))
    seed_c = int(jax.random.randint(rng_c, (), 0, 2**16))

    actor  = ContinuousActorNNX(obs_dim, action_dim, config.hidden_dim, nnx.Rngs(seed_a))
    critic = NNXCritic(obs_dim, config.hidden_dim, nnx.Rngs(seed_c))

    n_updates = config.num_rollouts * config.num_epochs * config.num_minibatches

    def _opt(lr, clip):
        sched = optax.linear_schedule(lr, 0.0, n_updates) if config.anneal_lr else lr
        return optax.chain(optax.clip_by_global_norm(clip), optax.adam(sched, eps=1e-5))

    actor_opt  = nnx.Optimizer(actor,  _opt(config.lr_actor,  config.max_grad_norm_actor), wrt=nnx.Param)
    critic_opt = nnx.Optimizer(critic, _opt(config.lr_critic, config.max_grad_norm_critic), wrt=nnx.Param)

    @nnx.jit
    def _forward(actor, critic, obs_j):
        return actor(obs_j), critic(obs_j)

    @nnx.jit
    def _update_mb(actor, critic, actor_opt, critic_opt, obs_mb, act_mb, lp_old, adv, tgt):
        if config.advantage_norm:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        def actor_loss(actor):
            mean, log_std = actor(obs_mb)
            lp  = _normal_log_prob(act_mb, mean, log_std)
            ent = _normal_entropy(log_std)
            ratio = jnp.exp(lp - lp_old)
            pg = jnp.maximum(
                -adv * ratio,
                -adv * jnp.clip(ratio, 1 - config.clip_eps, 1 + config.clip_eps),
            ).mean()
            return pg - config.entropy_beta * ent.mean()

        def critic_loss(critic):
            return 0.5 * jnp.mean((critic(obs_mb) - tgt) ** 2)

        actor_opt.update(actor, nnx.grad(actor_loss)(actor))
        critic_opt.update(critic, nnx.grad(critic_loss)(critic))

    obs_np, _ = envs.reset(seed=seed)
    ep_buf    = np.zeros(config.num_envs, dtype=np.float32)
    completed: list[tuple[int, float]] = []

    returns_by_rollout: list[float] = []
    steps_by_rollout:   list[int]   = []
    time_by_rollout:    list[float] = []
    time_to_threshold  = -1.0
    steps_to_threshold = -1
    total_steps        = 0
    t0     = time.perf_counter()
    rng_np = np.random.default_rng(seed)

    for _ in tqdm(range(config.num_rollouts), desc=f"nnx/{env_id}/s{seed}", leave=False):
        T, N = config.num_steps, config.num_envs
        O  = np.zeros((T, N, obs_dim),  np.float32)
        A  = np.zeros((T, N, action_dim), np.float32)
        R  = np.zeros((T, N),             np.float32)
        D  = np.zeros((T, N),             np.float32)
        LP = np.zeros((T, N),             np.float32)
        V  = np.zeros((T, N),             np.float32)

        for t in range(T):
            obs_j = jnp.array(obs_np)
            rng, key = jax.random.split(rng)
            (mean, log_std), val = _forward(actor, critic, obs_j)
            eps    = jax.random.normal(key, mean.shape)
            act_j  = jnp.clip(mean + jnp.exp(log_std) * eps, act_low, act_high)
            lp     = _normal_log_prob(act_j, mean, log_std)
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

        B     = config.batch_size
        O_f   = O.reshape(B, obs_dim)
        A_f   = A.reshape(B, action_dim)
        LP_f  = LP.reshape(B)
        adv_f = adv_all.reshape(B)
        tgt_f = tgt_all.reshape(B)

        for _ in range(config.num_epochs):
            perm = rng_np.permutation(B)
            for mb in range(config.num_minibatches):
                idx = perm[mb * config.minibatch_size : (mb + 1) * config.minibatch_size]
                _update_mb(
                    actor, critic, actor_opt, critic_opt,
                    jnp.array(O_f[idx]), jnp.array(A_f[idx]),
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
        "impl": "nnx", "env": env_id, "seed": seed,
        "returns_by_rollout":  returns_by_rollout,
        "steps_by_rollout":    steps_by_rollout,
        "time_by_rollout":     time_by_rollout,
        "time_to_threshold":   time_to_threshold,
        "steps_to_threshold":  steps_to_threshold,
        "steps_per_second":    total_steps / total_time,
        "total_time":          total_time,
    }


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    config: MuJoCoConfig,
    env_id: str,
    impl: str,
    num_seeds: int,
    monitor: ResourceMonitor,
) -> tuple[dict, list[dict]]:
    train_fn = train_linen_mujoco if impl == "linen" else train_nnx_mujoco
    monitor.start()
    seed_results = [train_fn(config, env_id, s) for s in range(num_seeds)]
    monitor.stop()
    metrics = aggregate_metrics(seed_results, env_id)
    metrics["resource"] = monitor.summary
    return metrics, seed_results


def main() -> None:
    config = MuJoCoConfig()
    for env_id in config.envs:
        for impl in config.impls:
            print(f"\nRunning {impl}/{env_id} × {config.num_seeds} seeds …")
            monitor = ResourceMonitor()
            metrics, seed_results = run_experiment(
                config, env_id, impl, config.num_seeds, monitor
            )
            _print_summary(metrics)
            run_dir = save_run(metrics, seed_results, config, base_dir=config.results_dir)
            print(f"  → {run_dir}")


if __name__ == "__main__":
    main()
