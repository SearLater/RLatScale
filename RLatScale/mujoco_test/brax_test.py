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
from brax.envs.wrappers.training import AutoResetWrapper, EpisodeWrapper
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
# Linen — full scan training on Brax
# ---------------------------------------------------------------------------

def make_train_brax_linen(config: BraxConfig, env, obs_dim: int, action_dim: int):
    """Return a pure ``train(rng) -> (AgentState, ep_return_by_rollout)``."""

    total_updates = config.num_rollouts * config.num_epochs * config.num_minibatches
    act_bound = 1.0  # Brax normalises actions to [-1, 1]

    def _make_opt(lr, clip):
        sched = optax.linear_schedule(lr, 0.0, total_updates) if config.anneal_lr else lr
        return optax.chain(optax.clip_by_global_norm(clip), optax.adam(sched, eps=1e-5))

    vmapped_reset = jax.vmap(env.reset)
    vmapped_step  = jax.vmap(env.step)

    def train(rng: jax.Array):
        rng, rng_a, rng_c = jax.random.split(rng, 3)
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

        rng, rng_reset = jax.random.split(rng)
        brax_state = vmapped_reset(jax.random.split(rng_reset, config.num_envs))
        ep_buf = jnp.zeros(config.num_envs)

        def _env_step(carry, _):
            agent_state, brax_state, ep_buf, rng = carry
            obs = brax_state.obs   # (num_envs, obs_dim)
            rng, rng_act = jax.random.split(rng)

            mean, log_std = agent_state.actor_state.apply_fn(
                agent_state.actor_state.params, obs
            )
            noise  = jax.random.normal(rng_act, mean.shape)
            action = jnp.clip(mean + jnp.exp(log_std) * noise, -act_bound, act_bound)
            log_prob = _normal_log_prob(action, mean, log_std)
            value = agent_state.critic_state.apply_fn(
                agent_state.critic_state.params, obs
            )

            brax_state = vmapped_step(brax_state, action)
            reward = brax_state.reward                          # (num_envs,)
            done   = brax_state.done.astype(jnp.float32)       # (num_envs,)

            ep_return  = jnp.where(done, ep_buf + reward, jnp.nan)
            ep_buf_new = jnp.where(done, 0.0, ep_buf + reward)

            t = BraxTransition(obs, action, reward, done, log_prob, value, ep_return)
            return (agent_state, brax_state, ep_buf_new, rng), t

        def _update_minibatch(agent_state: AgentState, mb) -> AgentState:
            obs_mb, act_mb, lp_old_mb, adv_mb, tgt_mb = mb

            adv_mb = jax.lax.cond(
                config.advantage_norm,
                lambda a: (a - a.mean()) / (a.std() + 1e-8),
                lambda a: a,
                adv_mb,
            )

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

        def _update_epoch(carry, _):
            agent_state, transitions, advantages, targets, rng = carry
            rng, rng_perm = jax.random.split(rng)
            B = config.batch_size
            perm = jax.random.permutation(rng_perm, B)

            def _reshape(x):
                flat = x.reshape(B, *x.shape[2:])[perm]
                return flat.reshape(config.num_minibatches, config.minibatch_size, *x.shape[2:])

            minibatches = (
                _reshape(transitions.obs),
                _reshape(transitions.action),
                _reshape(transitions.log_prob),
                _reshape(advantages),
                _reshape(targets),
            )
            agent_state, _ = jax.lax.scan(
                lambda s, mb: (_update_minibatch(s, mb), None), agent_state, minibatches
            )
            return (agent_state, transitions, advantages, targets, rng), None

        def _gae(transitions: BraxTransition, last_value: jax.Array):
            def _step(carry, t: BraxTransition):
                gae, nv = carry
                delta = t.reward + config.gamma * nv * (1 - t.done) - t.value
                gae = delta + config.gamma * config.gae_lambda * (1 - t.done) * gae
                return (gae, t.value), (gae, gae + t.value)

            _, (adv, tgt) = jax.lax.scan(
                _step, (jnp.zeros_like(last_value), last_value), transitions, reverse=True
            )
            return adv, tgt

        def _iteration(carry, _):
            agent_state, brax_state, ep_buf, rng = carry

            (agent_state, brax_state, ep_buf, rng), transitions = jax.lax.scan(
                _env_step,
                (agent_state, brax_state, ep_buf, rng),
                None,
                length=config.num_steps,
            )

            last_val = agent_state.critic_state.apply_fn(
                agent_state.critic_state.params, brax_state.obs
            )
            advantages, targets = _gae(transitions, last_val)

            (agent_state, _, _, _, rng), _ = jax.lax.scan(
                _update_epoch,
                (agent_state, transitions, advantages, targets, rng),
                None,
                length=config.num_epochs,
            )

            ep_r    = transitions.ep_return
            valid   = ~jnp.isnan(ep_r)
            n       = valid.sum()
            sum_v   = jnp.where(valid, ep_r, 0.0).sum()
            mean_ep = jnp.where(n > 0, sum_v / jnp.maximum(n, 1), jnp.nan)

            return (agent_state, brax_state, ep_buf, rng), mean_ep

        (agent_state, _, _, _), ep_return_by_rollout = jax.lax.scan(
            _iteration,
            (agent_state, brax_state, ep_buf, rng),
            None,
            length=config.num_rollouts,
        )
        return agent_state, ep_return_by_rollout

    return train


def train_linen_brax(config: BraxConfig, env_id: str, seed: int) -> dict:
    """Compile and time full-scan Linen PPO on a Brax environment."""
    meta_key  = _BRAX_TO_META[env_id]
    threshold = _ENV_META[meta_key]["threshold"]

    base_env = brax.envs.get_environment(env_id)
    env      = AutoResetWrapper(EpisodeWrapper(base_env, episode_length=1000, action_repeat=1))
    obs_dim    = env.observation_size
    action_dim = env.action_size

    train_fn = jax.jit(make_train_brax_linen(config, env, obs_dim, action_dim))

    rng = jax.random.key(seed)
    t0 = time.perf_counter()
    _, ep_return_by_rollout = train_fn(rng)
    jax.block_until_ready(ep_return_by_rollout)
    total_time = time.perf_counter() - t0

    n  = config.num_rollouts
    ep = np.array(ep_return_by_rollout)
    steps_by_rollout = [config.batch_size * (i + 1) for i in range(n)]
    time_by_rollout  = [total_time * (i + 1) / n      for i in range(n)]

    time_to_threshold  = -1.0
    steps_to_threshold = -1
    for i, (ret, steps) in enumerate(zip(ep, steps_by_rollout)):
        if not np.isnan(ret) and ret >= threshold:
            time_to_threshold  = time_by_rollout[i]
            steps_to_threshold = steps
            break

    # Surface results under the Gymnasium env name for consistent metric keys
    return {
        "impl": "linen", "env": meta_key, "seed": seed,
        "returns_by_rollout":  ep.tolist(),
        "steps_by_rollout":    steps_by_rollout,
        "time_by_rollout":     time_by_rollout,
        "time_to_threshold":   time_to_threshold,
        "steps_to_threshold":  steps_to_threshold,
        "steps_per_second":    config.total_timesteps / total_time,
        "total_time":          total_time,
    }


# ---------------------------------------------------------------------------
# NNX — Python-loop training on Brax
# ---------------------------------------------------------------------------

def train_nnx_brax(config: BraxConfig, env_id: str, seed: int) -> dict:
    meta_key  = _BRAX_TO_META[env_id]
    threshold = _ENV_META[meta_key]["threshold"]

    base_env = brax.envs.get_environment(env_id)
    env      = AutoResetWrapper(EpisodeWrapper(base_env, episode_length=1000, action_repeat=1))
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

    vmapped_reset = jax.vmap(env.reset)
    vmapped_step  = jax.vmap(env.step)

    rng, rng_reset = jax.random.split(rng)
    brax_state = vmapped_reset(jax.random.split(rng_reset, config.num_envs))
    ep_buf = jnp.zeros(config.num_envs)

    def _collect_rollout(actor, critic, brax_state, ep_buf, rng):
        graphdef_a, state_a = nnx.split(actor)
        graphdef_c, state_c = nnx.split(critic)

        def _env_step(carry, _):
            state_a, state_c, brax_state, ep_buf, rng = carry
            obs = brax_state.obs
            rng, rng_act = jax.random.split(rng)

            mean, log_std = nnx.merge(graphdef_a, state_a)(obs)
            noise  = jax.random.normal(rng_act, mean.shape)
            action = jnp.clip(mean + jnp.exp(log_std) * noise, -1.0, 1.0)
            log_prob = _normal_log_prob(action, mean, log_std)
            value    = nnx.merge(graphdef_c, state_c)(obs)

            brax_state = vmapped_step(brax_state, action)
            reward = brax_state.reward
            done   = brax_state.done.astype(jnp.float32)

            ep_return  = jnp.where(done, ep_buf + reward, jnp.nan)
            ep_buf_new = jnp.where(done, 0.0, ep_buf + reward)

            t = BraxTransition(obs, action, reward, done, log_prob, value, ep_return)
            return (state_a, state_c, brax_state, ep_buf_new, rng), t

        (_, _, brax_state, ep_buf, rng), transitions = jax.lax.scan(
            _env_step,
            (state_a, state_c, brax_state, ep_buf, rng),
            None,
            length=config.num_steps,
        )
        return brax_state, ep_buf, rng, transitions

    _collect_jit = nnx.jit(_collect_rollout)

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
        brax_state, ep_buf, rng, transitions = _collect_jit(
            actor, critic, brax_state, ep_buf, rng
        )
        last_val   = nnx.jit(critic)(brax_state.obs)
        advantages, targets = _gae(transitions, last_val)

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
