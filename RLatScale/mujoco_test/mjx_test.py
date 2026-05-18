"""
GPU benchmark: PPO on MJX (mujoco.mjx JAX-native) environments.

MJX runs the MuJoCo physics solver in JAX/XLA rather than the C library.
This file builds a thin env wrapper around the raw mjx API, loading the
same XML files as Gymnasium so that observations and rewards are identical
to the CPU baseline, making wall-clock comparisons directly meaningful.

Supported environments:
  halfcheetah  →  HalfCheetah-v4  (obs: 17, act: 6, horizon: 1000)
  ant          →  Ant-v4          (obs: 27, act: 8, horizon: 1000, no contact forces)

Usage
-----
    python -m RLatScale.mujoco_test.mjx_test
"""

from __future__ import annotations

import time
from typing import NamedTuple

import gymnasium
import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
import numpy as np
import optax
from flax import nnx
from flax.training.train_state import TrainState
from tqdm import tqdm

from RLatScale.algo.config import MjxConfig
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
# MJX environment wrapper
# ---------------------------------------------------------------------------

class MjxState(NamedTuple):
    mjx_data: mjx.Data
    obs:      jax.Array   # (obs_dim,)
    reward:   jax.Array   # scalar
    done:     jax.Array   # bool


class MjxEnv:
    """Thin JAX-compatible wrapper around mujoco.mjx for a single environment.

    Provides reset() and step() matching the Gymnax/Brax interface so the
    same vmap + scan training pattern works without modification.
    """

    _CONFIGS = {
        "halfcheetah": {
            "gym_id":       "HalfCheetah-v4",
            "obs_dim":      17,
            "act_dim":      6,
            "episode_len":  1000,
            "forward_weight": 1.0,
            "ctrl_weight":    0.1,
            "healthy_reward": 0.0,
            "terminate": False,
        },
        "ant": {
            "gym_id":       "Ant-v4",
            "obs_dim":      27,
            "act_dim":      8,
            "episode_len":  1000,
            "forward_weight": 1.0,
            "ctrl_weight":    0.5,
            "healthy_reward": 1.0,
            "terminate": True,
            "healthy_z_min": 0.2,
            "healthy_z_max": 1.0,
        },
    }

    def __init__(self, env_name: str):
        cfg = self._CONFIGS[env_name]
        self._cfg      = cfg
        self._env_name = env_name

        # Load XML via Gymnasium's bundled assets — same model as CPU baseline
        gym_env  = gymnasium.make(cfg["gym_id"])
        xml_path = gym_env.unwrapped.fullpath
        gym_env.close()

        self._mj_model  = mujoco.MjModel.from_xml_path(xml_path)
        self._mjx_model = mjx.put_model(self._mj_model)

        self.observation_size = cfg["obs_dim"]
        self.action_size      = cfg["act_dim"]
        self.episode_length   = cfg["episode_len"]

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    def _obs(self, data: mjx.Data) -> jax.Array:
        """Extract observation matching Gymnasium's implementation."""
        if self._env_name == "halfcheetah":
            # qpos[1:] (skip root x), qvel clipped to [-10, 10]
            return jnp.concatenate([
                data.qpos[1:],
                jnp.clip(data.qvel, -10.0, 10.0),
            ])
        else:  # ant
            # qpos[2:] (skip root x, y), qvel clipped
            return jnp.concatenate([
                data.qpos[2:],
                jnp.clip(data.qvel, -10.0, 10.0),
            ])

    def _reward(self, data_before: mjx.Data, data_after: mjx.Data, action: jax.Array) -> jax.Array:
        cfg = self._cfg
        dt  = self._mj_model.opt.timestep * self._mj_model.opt.integrator_dt \
              if hasattr(self._mj_model.opt, 'integrator_dt') else self._mj_model.opt.timestep

        if self._env_name == "halfcheetah":
            x_vel     = (data_after.qpos[0] - data_before.qpos[0]) / dt
            forward_r = cfg["forward_weight"] * x_vel
            ctrl_cost = cfg["ctrl_weight"] * jnp.sum(action ** 2)
            return forward_r - ctrl_cost

        else:  # ant
            x_vel     = (data_after.qpos[0] - data_before.qpos[0]) / dt
            forward_r = cfg["forward_weight"] * x_vel
            ctrl_cost = cfg["ctrl_weight"] * jnp.sum(action ** 2)
            return forward_r - ctrl_cost + cfg["healthy_reward"]

    def _done(self, data: mjx.Data, step_count: jax.Array) -> jax.Array:
        truncated = step_count >= self.episode_length
        if not self._cfg["terminate"]:
            return truncated
        z_pos    = data.qpos[2]
        unhealthy = (z_pos < self._cfg["healthy_z_min"]) | (z_pos > self._cfg["healthy_z_max"])
        return truncated | unhealthy

    def reset(self, rng: jax.Array) -> MjxState:
        """Reset to a slightly randomised initial state (matches Gymnasium reset)."""
        rng_q, rng_v = jax.random.split(rng)
        data = mjx.make_data(self._mjx_model)

        # Gymnasium-style init noise: qpos ± 0.1, qvel ± 0.1
        qpos = data.qpos + jax.random.uniform(
            rng_q, data.qpos.shape, minval=-0.1, maxval=0.1
        )
        qvel = data.qvel + jax.random.uniform(
            rng_v, data.qvel.shape, minval=-0.1, maxval=0.1
        )
        data = data.replace(qpos=qpos, qvel=qvel)
        data = mjx.forward(self._mjx_model, data)

        obs = self._obs(data)
        return MjxState(
            mjx_data=data,
            obs=obs,
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
        )

    def step(self, state: MjxState, action: jax.Array, step_count: jax.Array) -> MjxState:
        data_before = state.mjx_data
        data        = data_before.replace(ctrl=action)
        data        = mjx.step(self._mjx_model, data)

        reward = self._reward(data_before, data, action)
        obs    = self._obs(data)
        done   = self._done(data, step_count)

        # Auto-reset: if done, blend with a fresh state
        return MjxState(mjx_data=data, obs=obs, reward=reward, done=done)


# ---------------------------------------------------------------------------
# Trajectory type
# ---------------------------------------------------------------------------

class MjxTransition(NamedTuple):
    obs:       jax.Array
    action:    jax.Array
    reward:    jax.Array
    done:      jax.Array   # float32
    log_prob:  jax.Array
    value:     jax.Array
    ep_return: jax.Array   # episode sum at terminal steps; jnp.nan otherwise


# ---------------------------------------------------------------------------
# Linen — full scan training on MJX
# ---------------------------------------------------------------------------

def make_train_mjx_linen(config: MjxConfig, env: MjxEnv):
    """Return a pure ``train(rng) -> (AgentState, ep_return_by_rollout)``."""

    obs_dim    = env.observation_size
    action_dim = env.action_size
    total_updates = config.num_rollouts * config.num_epochs * config.num_minibatches

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
        mjx_state = vmapped_reset(jax.random.split(rng_reset, config.num_envs))
        ep_buf      = jnp.zeros(config.num_envs)
        step_counts = jnp.zeros(config.num_envs, dtype=jnp.int32)

        def _env_step(carry, _):
            agent_state, mjx_state, ep_buf, step_counts, rng = carry
            obs = mjx_state.obs   # (num_envs, obs_dim)
            rng, rng_act = jax.random.split(rng)

            mean, log_std = agent_state.actor_state.apply_fn(
                agent_state.actor_state.params, obs
            )
            noise  = jax.random.normal(rng_act, mean.shape)
            action = jnp.clip(mean + jnp.exp(log_std) * noise, -1.0, 1.0)
            log_prob = _normal_log_prob(action, mean, log_std)
            value    = agent_state.critic_state.apply_fn(
                agent_state.critic_state.params, obs
            )

            step_counts = step_counts + 1
            mjx_state   = vmapped_step(mjx_state, action, step_counts)
            reward = mjx_state.reward
            done   = mjx_state.done.astype(jnp.float32)

            ep_return  = jnp.where(done, ep_buf + reward, jnp.nan)
            ep_buf_new = jnp.where(done, 0.0, ep_buf + reward)
            # Reset step counter for done envs
            step_counts = jnp.where(done.astype(jnp.bool_), 0, step_counts)

            t = MjxTransition(obs, action, reward, done, log_prob, value, ep_return)
            return (agent_state, mjx_state, ep_buf_new, step_counts, rng), t

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

        def _gae(transitions: MjxTransition, last_value: jax.Array):
            def _step(carry, t: MjxTransition):
                gae, nv = carry
                delta = t.reward + config.gamma * nv * (1 - t.done) - t.value
                gae   = delta + config.gamma * config.gae_lambda * (1 - t.done) * gae
                return (gae, t.value), (gae, gae + t.value)

            _, (adv, tgt) = jax.lax.scan(
                _step, (jnp.zeros_like(last_value), last_value), transitions, reverse=True
            )
            return adv, tgt

        def _iteration(carry, _):
            agent_state, mjx_state, ep_buf, step_counts, rng = carry

            (agent_state, mjx_state, ep_buf, step_counts, rng), transitions = jax.lax.scan(
                _env_step,
                (agent_state, mjx_state, ep_buf, step_counts, rng),
                None,
                length=config.num_steps,
            )

            last_val = agent_state.critic_state.apply_fn(
                agent_state.critic_state.params, mjx_state.obs
            )
            advantages, targets = _gae(transitions, last_val)

            (agent_state, _, _, _, rng), _ = jax.lax.scan(
                _update_epoch,
                (agent_state, transitions, advantages, targets, rng),
                None,
                length=config.num_epochs,
            )

            ep_r  = transitions.ep_return
            valid = ~jnp.isnan(ep_r)
            n     = valid.sum()
            mean_ep = jnp.where(n > 0, jnp.where(valid, ep_r, 0.0).sum() / jnp.maximum(n, 1), jnp.nan)

            return (agent_state, mjx_state, ep_buf, step_counts, rng), mean_ep

        (agent_state, _, _, _, _), ep_return_by_rollout = jax.lax.scan(
            _iteration,
            (agent_state, mjx_state, ep_buf, step_counts, rng),
            None,
            length=config.num_rollouts,
        )
        return agent_state, ep_return_by_rollout

    return train


def train_linen_mjx(config: MjxConfig, env_id: str, seed: int) -> dict:
    gym_id    = MjxEnv._CONFIGS[env_id]["gym_id"]
    threshold = _ENV_META[gym_id]["threshold"]

    env      = MjxEnv(env_id)
    train_fn = jax.jit(make_train_mjx_linen(config, env))

    rng = jax.random.key(seed)
    t0  = time.perf_counter()
    _, ep_return_by_rollout = train_fn(rng)
    jax.block_until_ready(ep_return_by_rollout)
    total_time = time.perf_counter() - t0

    n  = config.num_rollouts
    ep = np.array(ep_return_by_rollout)
    steps_by_rollout = [config.batch_size * (i + 1) for i in range(n)]
    time_by_rollout  = [total_time * (i + 1) / n      for i in range(n)]

    time_to_threshold = -1.0; steps_to_threshold = -1
    for i, (ret, steps) in enumerate(zip(ep, steps_by_rollout)):
        if not np.isnan(ret) and ret >= threshold:
            time_to_threshold = time_by_rollout[i]; steps_to_threshold = steps; break

    return {
        "impl": "linen", "env": gym_id, "seed": seed,
        "returns_by_rollout":  ep.tolist(),
        "steps_by_rollout":    steps_by_rollout,
        "time_by_rollout":     time_by_rollout,
        "time_to_threshold":   time_to_threshold,
        "steps_to_threshold":  steps_to_threshold,
        "steps_per_second":    config.total_timesteps / total_time,
        "total_time":          total_time,
    }


# ---------------------------------------------------------------------------
# NNX — Python-loop training on MJX
# ---------------------------------------------------------------------------

def train_nnx_mjx(config: MjxConfig, env_id: str, seed: int) -> dict:
    gym_id    = MjxEnv._CONFIGS[env_id]["gym_id"]
    threshold = _ENV_META[gym_id]["threshold"]

    env        = MjxEnv(env_id)
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

    actor_opt  = nnx.Optimizer(actor,  _opt(config.lr_actor,  config.max_grad_norm_actor))
    critic_opt = nnx.Optimizer(critic, _opt(config.lr_critic, config.max_grad_norm_critic))

    vmapped_reset = jax.vmap(env.reset)
    vmapped_step  = jax.vmap(env.step)

    rng, rng_reset = jax.random.split(rng)
    mjx_state   = vmapped_reset(jax.random.split(rng_reset, config.num_envs))
    ep_buf      = jnp.zeros(config.num_envs)
    step_counts = jnp.zeros(config.num_envs, dtype=jnp.int32)

    def _collect_rollout(actor, critic, mjx_state, ep_buf, step_counts, rng):
        graphdef_a, state_a = nnx.split(actor)
        graphdef_c, state_c = nnx.split(critic)

        def _env_step(carry, _):
            state_a, state_c, mjx_state, ep_buf, step_counts, rng = carry
            obs = mjx_state.obs
            rng, rng_act = jax.random.split(rng)

            mean, log_std = nnx.merge(graphdef_a, state_a)(obs)
            noise  = jax.random.normal(rng_act, mean.shape)
            action = jnp.clip(mean + jnp.exp(log_std) * noise, -1.0, 1.0)
            log_prob = _normal_log_prob(action, mean, log_std)
            value    = nnx.merge(graphdef_c, state_c)(obs)

            step_counts = step_counts + 1
            mjx_state   = vmapped_step(mjx_state, action, step_counts)
            reward = mjx_state.reward
            done   = mjx_state.done.astype(jnp.float32)

            ep_return   = jnp.where(done, ep_buf + reward, jnp.nan)
            ep_buf_new  = jnp.where(done, 0.0, ep_buf + reward)
            step_counts = jnp.where(done.astype(jnp.bool_), 0, step_counts)

            t = MjxTransition(obs, action, reward, done, log_prob, value, ep_return)
            return (state_a, state_c, mjx_state, ep_buf_new, step_counts, rng), t

        (_, _, mjx_state, ep_buf, step_counts, rng), transitions = jax.lax.scan(
            _env_step,
            (state_a, state_c, mjx_state, ep_buf, step_counts, rng),
            None,
            length=config.num_steps,
        )
        return mjx_state, ep_buf, step_counts, rng, transitions

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

        actor_opt.update(nnx.grad(actor_loss)(actor))
        critic_opt.update(nnx.grad(critic_loss)(critic))

    def _gae(transitions: MjxTransition, last_value: jax.Array):
        def _step(carry, t: MjxTransition):
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

    for _ in tqdm(range(config.num_rollouts), desc=f"nnx_mjx/{env_id}/s{seed}", leave=False):
        mjx_state, ep_buf, step_counts, rng, transitions = _collect_jit(
            actor, critic, mjx_state, ep_buf, step_counts, rng
        )
        last_val   = nnx.jit(critic)(mjx_state.obs)
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

    jax.block_until_ready(mjx_state.obs)
    total_time = time.perf_counter() - t0

    return {
        "impl": "nnx", "env": gym_id, "seed": seed,
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
    config: MjxConfig,
    env_id: str,
    impl: str,
    num_seeds: int,
    monitor: ResourceMonitor,
) -> tuple[dict, list[dict]]:
    train_fn = train_linen_mjx if impl == "linen" else train_nnx_mjx
    monitor.start()
    seed_results = [train_fn(config, env_id, s) for s in range(num_seeds)]
    monitor.stop()
    gym_id  = MjxEnv._CONFIGS[env_id]["gym_id"]
    metrics = aggregate_metrics(seed_results, gym_id)
    metrics["resource"] = monitor.summary
    return metrics, seed_results


def main() -> None:
    config = MjxConfig()
    for env_id in config.envs:
        for impl in config.impls:
            print(f"\nRunning {impl}/{env_id} (MJX) × {config.num_seeds} seeds …")
            monitor = ResourceMonitor()
            metrics, seed_results = run_experiment(
                config, env_id, impl, config.num_seeds, monitor
            )
            _print_summary(metrics)
            run_dir = save_run(metrics, seed_results, config, base_dir=config.results_dir)
            print(f"  → {run_dir}")


if __name__ == "__main__":
    main()
