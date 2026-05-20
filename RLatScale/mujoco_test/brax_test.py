"""
GPU benchmark: PPO on Brax (JAX-native) environments.

HalfCheetah and Ant are both continuous-control tasks.
Full jax.lax.scan Linen training and Python-loop NNX training, both vmapping
over Brax environments on GPU/Metal.

AutoResetWrapper handles mid-rollout episode resets so the scan body stays
clean; episode returns are tracked via an ep_buf accumulator that resets on
each done flag.

Usage
-----
    python -m RLatScale.mujoco_test.brax_test
"""

from __future__ import annotations

import time
from typing import NamedTuple

import brax.envs
import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax.envs.wrappers.training import AutoResetWrapper
from flax import nnx
from flax.training.train_state import TrainState
from tqdm import tqdm

from RLatScale.algo.config import BraxConfig
from RLatScale.algo.ppo_linen import AgentState, Critic as LinenCritic
from RLatScale.algo.ppo_nnx import Critic as NNXCritic
from RLatScale.gym_test.cpu_test import (
    ContinuousActorLinen,
    ContinuousActorNNX,
    ResourceMonitor,
    _normal_entropy,
    _normal_log_prob,
    _print_summary,
    save_run,
)
from RLatScale.mujoco_test.cpu_test import _ENV_META, aggregate_metrics

# ---------------------------------------------------------------------------
# Brax env name → metadata (thresholds shared with CPU baseline)
# ---------------------------------------------------------------------------

_BRAX_TO_META: dict[str, str] = {
    "halfcheetah": "HalfCheetah-v4",
    "ant":         "Ant-v4",
}

_EPISODE_LENGTH: dict[str, int] = {
    "halfcheetah": 1000,
    "ant":         1000,
}

# ---------------------------------------------------------------------------
# Trajectory type
# ---------------------------------------------------------------------------

class BraxTransition(NamedTuple):
    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    done: jax.Array       # float32
    log_prob: jax.Array
    value: jax.Array
    ep_return: jax.Array  # episode sum at terminal steps; jnp.nan otherwise


# ---------------------------------------------------------------------------
# Linen — Python-loop training on Brax (per-rollout JIT, same as NNX)
#
# A full outer jax.lax.scan over num_rollouts hangs at XLA compile time for
# complex physics envs (Brax halfcheetah/ant).  Only the inner num_steps scan
# is JIT-compiled; the outer rollout loop runs in Python like the NNX version.
# ---------------------------------------------------------------------------

def train_linen_brax(config: BraxConfig, env_id: str, seed: int) -> dict:
    """Run PPO (Flax Linen, Python outer loop) on a Brax environment."""
    meta_key  = _BRAX_TO_META[env_id]
    threshold = _ENV_META[meta_key]["threshold"]

    episode_length = _EPISODE_LENGTH[env_id]
    base_env = brax.envs.get_environment(env_id)
    env      = AutoResetWrapper(base_env)
    obs_dim    = env.observation_size
    action_dim = env.action_size

    rng = jax.random.key(seed)
    rng, rng_a, rng_c = jax.random.split(rng, 3)

    total_updates = config.num_rollouts * config.num_epochs * config.num_minibatches
    act_bound = 1.0

    def _make_opt(lr, clip):
        sched = optax.linear_schedule(lr, 0.0, total_updates) if config.anneal_lr else lr
        return optax.chain(optax.clip_by_global_norm(clip), optax.adam(sched, eps=1e-5))

    dummy = jnp.zeros((1, obs_dim))
    actor_net  = ContinuousActorLinen(action_dim=action_dim, hidden_dim=config.hidden_dim)
    critic_net = LinenCritic(hidden_dim=config.hidden_dim)
    actor_state = TrainState.create(
        apply_fn=actor_net.apply,
        params=actor_net.init(rng_a, dummy),
        tx=_make_opt(config.lr_actor, config.max_grad_norm_actor),
    )
    critic_state = TrainState.create(
        apply_fn=critic_net.apply,
        params=critic_net.init(rng_c, dummy),
        tx=_make_opt(config.lr_critic, config.max_grad_norm_critic),
    )
    agent_state = AgentState(actor_state, critic_state)

    jit_reset = jax.jit(jax.vmap(env.reset))
    jit_step  = jax.jit(jax.vmap(env.step))

    rng, rng_reset, rng_sc = jax.random.split(rng, 3)
    brax_state = jit_reset(jax.random.split(rng_reset, config.num_envs))
    ep_buf = jnp.zeros(config.num_envs)
    # Stagger initial step counts so episode completions are spread across rollouts.
    step_counts = jax.random.randint(
        rng_sc, (config.num_envs,), 0, episode_length, dtype=jnp.int32
    )

    # JIT over the inner num_steps scan only — avoids XLA compile hang
    @jax.jit
    def _collect_rollout(agent_state, brax_state, ep_buf, step_counts, rng):
        def _env_step(carry, _):
            brax_state, ep_buf, step_counts, rng = carry
            obs = brax_state.obs
            rng, rng_act = jax.random.split(rng)
            mean, log_std = agent_state.actor_state.apply_fn(
                agent_state.actor_state.params, obs
            )
            noise  = jax.random.normal(rng_act, mean.shape)
            action = jnp.clip(mean + jnp.exp(log_std) * noise, -act_bound, act_bound)
            log_prob = _normal_log_prob(action, mean, log_std)
            value    = agent_state.critic_state.apply_fn(
                agent_state.critic_state.params, obs
            )
            step_counts = step_counts + 1
            brax_state  = jit_step(brax_state, action)
            reward = brax_state.reward
            done   = (brax_state.done | (step_counts >= episode_length)).astype(jnp.float32)
            ep_return  = jnp.where(done, ep_buf + reward, jnp.nan)
            ep_buf_new = jnp.where(done, 0.0, ep_buf + reward)
            step_counts = jnp.where(done.astype(jnp.bool_), 0, step_counts)
            t = BraxTransition(obs, action, reward, done, log_prob, value, ep_return)
            return (brax_state, ep_buf_new, step_counts, rng), t

        (brax_state, ep_buf, step_counts, rng), transitions = jax.lax.scan(
            _env_step, (brax_state, ep_buf, step_counts, rng), None, length=config.num_steps
        )
        return brax_state, ep_buf, step_counts, rng, transitions

    @jax.jit
    def _update_mb(agent_state, obs_mb, act_mb, lp_old_mb, adv_mb, tgt_mb):
        def actor_loss_fn(params):
            mean, log_std = agent_state.actor_state.apply_fn(params, obs_mb)
            lp  = _normal_log_prob(act_mb, mean, log_std)
            ent = _normal_entropy(log_std)
            ratio = jnp.exp(lp - lp_old_mb)
            pg = jnp.maximum(
                -adv_mb * ratio,
                -adv_mb * jnp.clip(ratio, 1 - config.clip_eps, 1 + config.clip_eps),
            ).mean()
            return pg - config.entropy_beta * ent.mean()

        def critic_loss_fn(params):
            return 0.5 * jnp.mean(
                (agent_state.critic_state.apply_fn(params, obs_mb) - tgt_mb) ** 2
            )

        return AgentState(
            agent_state.actor_state.apply_gradients(
                grads=jax.grad(actor_loss_fn)(agent_state.actor_state.params)
            ),
            agent_state.critic_state.apply_gradients(
                grads=jax.grad(critic_loss_fn)(agent_state.critic_state.params)
            ),
        )

    def _gae(transitions: BraxTransition, last_value: jax.Array):
        def _step(carry, t: BraxTransition):
            gae, nv = carry
            delta = t.reward + config.gamma * nv * (1 - t.done) - t.value
            gae   = delta + config.gamma * config.gae_lambda * (1 - t.done) * gae
            return (gae, t.value), (gae, gae + t.value)

        _, (adv, tgt) = jax.lax.scan(
            _step, (jnp.zeros_like(last_value), last_value), transitions, reverse=True
        )
        return adv, tgt

    returns_by_rollout: list[float] = []
    steps_by_rollout:   list[int]   = []
    time_by_rollout:    list[float] = []
    time_to_threshold  = -1.0
    steps_to_threshold = -1
    total_steps = 0
    t0       = time.perf_counter()
    rng_perm = jax.random.key(seed + 99_999)

    for _ in tqdm(range(config.num_rollouts), desc=f"linen_brax/{env_id}/s{seed}", leave=False):
        brax_state, ep_buf, step_counts, rng, transitions = _collect_rollout(
            agent_state, brax_state, ep_buf, step_counts, rng
        )
        last_val = jax.jit(agent_state.critic_state.apply_fn)(
            agent_state.critic_state.params, brax_state.obs
        )
        advantages, targets = _gae(transitions, last_val)
        if config.advantage_norm:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        B = config.batch_size
        for _ in range(config.num_epochs):
            rng_perm, rng_sub = jax.random.split(rng_perm)
            perm = jax.random.permutation(rng_sub, B)

            def _reshape(x):
                flat = x.reshape(B, *x.shape[2:])[perm]
                return flat.reshape(config.num_minibatches, config.minibatch_size, *x.shape[2:])

            for mb in range(config.num_minibatches):
                agent_state = _update_mb(
                    agent_state,
                    _reshape(transitions.obs)[mb],
                    _reshape(transitions.action)[mb],
                    _reshape(transitions.log_prob)[mb],
                    _reshape(advantages)[mb],
                    _reshape(targets)[mb],
                )

        total_steps += config.batch_size
        ep_r     = np.array(transitions.ep_return)
        mean_ret = float(np.nanmean(ep_r)) if not np.all(np.isnan(ep_r)) else float("nan")
        elapsed  = time.perf_counter() - t0

        returns_by_rollout.append(mean_ret)
        steps_by_rollout.append(total_steps)
        time_by_rollout.append(elapsed)

        if time_to_threshold < 0 and not np.isnan(mean_ret) and mean_ret >= threshold:
            time_to_threshold  = elapsed
            steps_to_threshold = total_steps

    jax.block_until_ready(brax_state.obs)
    total_time = time.perf_counter() - t0

    return {
        "impl": "linen", "env": meta_key, "seed": seed,
        "returns_by_rollout":  returns_by_rollout,
        "steps_by_rollout":    steps_by_rollout,
        "time_by_rollout":     time_by_rollout,
        "time_to_threshold":   time_to_threshold,
        "steps_to_threshold":  steps_to_threshold,
        "steps_per_second":    total_steps / total_time,
        "total_time":          total_time,
    }


# ---------------------------------------------------------------------------
# NNX — Python-loop training on Brax
# ---------------------------------------------------------------------------

def train_nnx_brax(config: BraxConfig, env_id: str, seed: int) -> dict:
    meta_key  = _BRAX_TO_META[env_id]
    threshold = _ENV_META[meta_key]["threshold"]

    episode_length = _EPISODE_LENGTH[env_id]
    base_env = brax.envs.get_environment(env_id)
    env      = AutoResetWrapper(base_env)
    obs_dim    = env.observation_size
    action_dim = env.action_size

    rng = jax.random.key(seed)
    rng, rng_a, rng_c = jax.random.split(rng, 3)
    seed_a = int(jax.random.randint(rng_a, (), 0, 2**16))
    seed_c = int(jax.random.randint(rng_c, (), 0, 2**16))

    actor  = ContinuousActorNNX(obs_dim, action_dim, config.hidden_dim, nnx.Rngs(seed_a))
    critic = NNXCritic(obs_dim, config.hidden_dim, nnx.Rngs(seed_c))

    total_updates = config.num_rollouts * config.num_epochs * config.num_minibatches

    def _opt(lr, clip):
        sched = optax.linear_schedule(lr, 0.0, total_updates) if config.anneal_lr else lr
        return optax.chain(optax.clip_by_global_norm(clip), optax.adam(sched, eps=1e-5))

    actor_opt  = nnx.Optimizer(actor,  _opt(config.lr_actor,  config.max_grad_norm_actor), wrt=nnx.Param)
    critic_opt = nnx.Optimizer(critic, _opt(config.lr_critic, config.max_grad_norm_critic), wrt=nnx.Param)

    jit_reset = jax.jit(jax.vmap(env.reset))
    jit_step  = jax.jit(jax.vmap(env.step))

    rng, rng_reset, rng_sc = jax.random.split(rng, 3)
    brax_state = jit_reset(jax.random.split(rng_reset, config.num_envs))
    ep_buf = jnp.zeros(config.num_envs)
    step_counts = jax.random.randint(
        rng_sc, (config.num_envs,), 0, episode_length, dtype=jnp.int32
    )

    def _collect_rollout(actor, critic, brax_state, ep_buf, step_counts, rng):
        graphdef_a, state_a = nnx.split(actor)
        graphdef_c, state_c = nnx.split(critic)

        def _env_step(carry, _):
            state_a, state_c, brax_state, ep_buf, step_counts, rng = carry
            obs = brax_state.obs
            rng, rng_act = jax.random.split(rng)

            mean, log_std = nnx.merge(graphdef_a, state_a)(obs)
            noise  = jax.random.normal(rng_act, mean.shape)
            action = jnp.clip(mean + jnp.exp(log_std) * noise, -1.0, 1.0)
            log_prob = _normal_log_prob(action, mean, log_std)
            value    = nnx.merge(graphdef_c, state_c)(obs)

            step_counts = step_counts + 1
            brax_state  = jit_step(brax_state, action)
            reward = brax_state.reward
            done   = (brax_state.done | (step_counts >= episode_length)).astype(jnp.float32)

            ep_return  = jnp.where(done, ep_buf + reward, jnp.nan)
            ep_buf_new = jnp.where(done, 0.0, ep_buf + reward)
            step_counts = jnp.where(done.astype(jnp.bool_), 0, step_counts)

            t = BraxTransition(obs, action, reward, done, log_prob, value, ep_return)
            return (state_a, state_c, brax_state, ep_buf_new, step_counts, rng), t

        (_, _, brax_state, ep_buf, step_counts, rng), transitions = jax.lax.scan(
            _env_step,
            (state_a, state_c, brax_state, ep_buf, step_counts, rng),
            None,
            length=config.num_steps,
        )
        return brax_state, ep_buf, step_counts, rng, transitions

    _collect_jit = nnx.jit(_collect_rollout)

    @nnx.jit
    def _update_mb(actor, critic, actor_opt, critic_opt, obs_mb, act_mb, lp_old, adv, tgt):
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

    def _gae(transitions: BraxTransition, last_value: jax.Array):
        def _step(carry, t: BraxTransition):
            gae, nv = carry
            delta = t.reward + config.gamma * nv * (1 - t.done) - t.value
            gae   = delta + config.gamma * config.gae_lambda * (1 - t.done) * gae
            return (gae, t.value), (gae, gae + t.value)

        _, (adv, tgt) = jax.lax.scan(
            _step, (jnp.zeros_like(last_value), last_value), transitions, reverse=True
        )
        return adv, tgt

    returns_by_rollout: list[float] = []
    steps_by_rollout:   list[int]   = []
    time_by_rollout:    list[float] = []
    time_to_threshold  = -1.0
    steps_to_threshold = -1
    total_steps = 0
    t0       = time.perf_counter()
    rng_perm = jax.random.key(seed + 99_999)

    for _ in tqdm(range(config.num_rollouts), desc=f"nnx_brax/{env_id}/s{seed}", leave=False):
        brax_state, ep_buf, step_counts, rng, transitions = _collect_jit(
            actor, critic, brax_state, ep_buf, step_counts, rng
        )
        last_val   = nnx.jit(critic)(brax_state.obs)
        advantages, targets = _gae(transitions, last_val)
        if config.advantage_norm:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        B = config.batch_size
        for _ in range(config.num_epochs):
            rng_perm, rng_sub = jax.random.split(rng_perm)
            perm = jax.random.permutation(rng_sub, B)

            def _reshape(x):
                flat = x.reshape(B, *x.shape[2:])[perm]
                return flat.reshape(config.num_minibatches, config.minibatch_size, *x.shape[2:])

            for mb in range(config.num_minibatches):
                _update_mb(
                    actor, critic, actor_opt, critic_opt,
                    _reshape(transitions.obs)[mb],
                    _reshape(transitions.action)[mb],
                    _reshape(transitions.log_prob)[mb],
                    _reshape(advantages)[mb],
                    _reshape(targets)[mb],
                )

        total_steps += config.batch_size
        ep_r     = np.array(transitions.ep_return)
        mean_ret = float(np.nanmean(ep_r)) if not np.all(np.isnan(ep_r)) else float("nan")
        elapsed  = time.perf_counter() - t0

        returns_by_rollout.append(mean_ret)
        steps_by_rollout.append(total_steps)
        time_by_rollout.append(elapsed)

        if time_to_threshold < 0 and not np.isnan(mean_ret) and mean_ret >= threshold:
            time_to_threshold  = elapsed
            steps_to_threshold = total_steps

    jax.block_until_ready(brax_state.obs)
    total_time = time.perf_counter() - t0

    return {
        "impl": "nnx", "env": meta_key, "seed": seed,
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
    config: BraxConfig,
    env_id: str,
    impl: str,
    num_seeds: int,
    monitor: ResourceMonitor,
) -> tuple[dict, list[dict]]:
    train_fn = train_linen_brax if impl == "linen" else train_nnx_brax
    monitor.start()
    seed_results = [train_fn(config, env_id, s) for s in range(num_seeds)]
    monitor.stop()
    meta_key = _BRAX_TO_META[env_id]
    metrics  = aggregate_metrics(seed_results, meta_key)
    metrics["resource"] = monitor.summary
    return metrics, seed_results


def main() -> None:
    config = BraxConfig()
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
