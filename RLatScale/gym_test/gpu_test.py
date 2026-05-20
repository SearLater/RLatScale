"""
GPU benchmark: PPO on Gymnax (JAX-native) environments.

Trains Flax Linen (full jax.lax.scan) and Flax NNX (Python loop) PPO on:
  - CartPole-v1  (discrete action space)
  - Pendulum-v1  (continuous action space)

Key differences from cpu_test.py
---------------------------------
- Gymnax environments: vmappable, scannable, no Python overhead per step.
- Linen: entire training compiled as a single XLA program via nested scan.
- NNX: rollout collected via scan; updates still use Python loop (benchmark).
- jax.block_until_ready() called before timing to account for async dispatch.
- Uses GPUConfig (num_envs=2048) instead of Config (num_envs=4).

Usage
-----
    python -m RLatScale.gym_test.gpu_test
"""

from __future__ import annotations

import sys
import time
from typing import NamedTuple

import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from flax.training.train_state import TrainState
from tqdm import tqdm

import ion

from RLatScale.algo.config import Config, GPUConfig
from RLatScale.algo.ppo_ion import ActorCritic as IonDiscreteActorCritic
from RLatScale.algo.ppo_ion import ActorCriticContinuous as IonContinuousActorCritic
from RLatScale.algo.ppo_linen import Actor as LinenDiscreteActor
from RLatScale.algo.ppo_linen import AgentState, Critic as LinenCritic
from RLatScale.algo.ppo_nnx import Actor as NNXDiscreteActor
from RLatScale.algo.ppo_nnx import Critic as NNXCritic
from RLatScale.gym_test.cpu_test import (
    ContinuousActorLinen,
    ContinuousActorNNX,
    ResourceMonitor,
    _ENV_META,
    _cat_entropy,
    _cat_log_prob,
    _detect_hardware,
    _normal_entropy,
    _normal_log_prob,
    _print_summary,
    aggregate_metrics,
    save_run,
)

# ---------------------------------------------------------------------------
# Trajectory type — extends cpu_test Transition with episode-return tracking
# ---------------------------------------------------------------------------

class GpuTransition(NamedTuple):
    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    done: jax.Array       # float32
    log_prob: jax.Array
    value: jax.Array
    ep_return: jax.Array  # terminal episode sum; jnp.nan for non-terminal steps


# ---------------------------------------------------------------------------
# Linen — full scan training on Gymnax
# ---------------------------------------------------------------------------

def make_train_gymnax_linen(
    config: Config,
    env,
    env_params,
    is_cont: bool,
    action_dim: int,
    obs_dim: int,
    act_low=None,
    act_high=None,
):
    """Return a pure ``train(rng) -> (AgentState, ep_return_by_rollout)``."""

    total_updates = config.num_rollouts * config.num_epochs * config.num_minibatches

    def _make_opt(lr, clip):
        sched = optax.linear_schedule(lr, 0.0, total_updates) if config.anneal_lr else lr
        return optax.chain(optax.clip_by_global_norm(clip), optax.adam(sched, eps=1e-5))

    def train(rng: jax.Array):
        rng, rng_a, rng_c = jax.random.split(rng, 3)
        dummy = jnp.zeros((1, obs_dim))

        if is_cont:
            actor_net = ContinuousActorLinen(action_dim=action_dim, hidden_dim=config.hidden_dim)
        else:
            actor_net = LinenDiscreteActor(action_dim=action_dim, hidden_dim=config.hidden_dim)
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
        obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(
            jax.random.split(rng_reset, config.num_envs), env_params
        )
        ep_buf = jnp.zeros(config.num_envs)

        # --- Single environment step ---
        def _env_step(carry, _):
            agent_state, obs, env_state, ep_buf, rng = carry
            rng, rng_act, rng_step = jax.random.split(rng, 3)

            if is_cont:
                mean, log_std = agent_state.actor_state.apply_fn(
                    agent_state.actor_state.params, obs
                )
                noise = jax.random.normal(rng_act, mean.shape)
                action = jnp.clip(mean + jnp.exp(log_std) * noise, act_low, act_high)
                log_prob = _normal_log_prob(action, mean, log_std)
            else:
                logits = agent_state.actor_state.apply_fn(
                    agent_state.actor_state.params, obs
                )
                action = jax.random.categorical(rng_act, logits)
                log_prob = _cat_log_prob(logits, action)

            value = agent_state.critic_state.apply_fn(
                agent_state.critic_state.params, obs
            )

            obs_next, env_state, reward, done, _ = jax.vmap(
                env.step, in_axes=(0, 0, 0, None)
            )(jax.random.split(rng_step, config.num_envs), env_state, action, env_params)

            done_f = done.astype(jnp.float32)
            ep_return = jnp.where(done, ep_buf + reward, jnp.nan)
            ep_buf_new = jnp.where(done, 0.0, ep_buf + reward)

            t = GpuTransition(obs, action, reward, done_f, log_prob, value, ep_return)
            return (agent_state, obs_next, env_state, ep_buf_new, rng), t

        # --- Single minibatch update ---
        def _update_minibatch(agent_state: AgentState, mb) -> AgentState:
            obs_mb, act_mb, lp_old_mb, adv_mb, tgt_mb = mb

            def actor_loss_fn(params):
                if is_cont:
                    mean, log_std = agent_state.actor_state.apply_fn(params, obs_mb)
                    lp = _normal_log_prob(act_mb, mean, log_std)
                    ent = _normal_entropy(log_std)
                else:
                    logits = agent_state.actor_state.apply_fn(params, obs_mb)
                    lp = _cat_log_prob(logits, act_mb)
                    ent = _cat_entropy(logits)
                ratio = jnp.exp(lp - lp_old_mb)
                pg = jnp.maximum(
                    -adv_mb * ratio,
                    -adv_mb * jnp.clip(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps),
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

        # --- Single epoch: shuffle and scan over minibatches ---
        def _update_epoch(carry, _):
            agent_state, transitions, advantages, targets, rng = carry
            rng, rng_perm = jax.random.split(rng)
            if config.advantage_norm:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            B = config.batch_size
            perm = jax.random.permutation(rng_perm, B)

            def _reshape(x):
                flat = x.reshape(B, *x.shape[2:])[perm]
                return flat.reshape(
                    config.num_minibatches, config.minibatch_size, *x.shape[2:]
                )

            minibatches = (
                _reshape(transitions.obs),
                _reshape(transitions.action),
                _reshape(transitions.log_prob),
                _reshape(advantages),
                _reshape(targets),
            )

            agent_state, _ = jax.lax.scan(
                lambda s, mb: (_update_minibatch(s, mb), None),
                agent_state,
                minibatches,
            )
            return (agent_state, transitions, advantages, targets, rng), None

        # --- GAE ---
        def _gae(transitions: GpuTransition, last_value: jax.Array):
            def _step(carry, t: GpuTransition):
                gae, nv = carry
                delta = t.reward + config.gamma * nv * (1.0 - t.done) - t.value
                gae = delta + config.gamma * config.gae_lambda * (1.0 - t.done) * gae
                return (gae, t.value), (gae, gae + t.value)

            _, (adv, tgt) = jax.lax.scan(
                _step,
                (jnp.zeros_like(last_value), last_value),
                transitions,
                reverse=True,
            )
            return adv, tgt

        # --- One rollout + update iteration (outer scan body) ---
        def _iteration(carry, _):
            agent_state, obs, env_state, ep_buf, rng = carry

            (agent_state, obs, env_state, ep_buf, rng), transitions = jax.lax.scan(
                _env_step,
                (agent_state, obs, env_state, ep_buf, rng),
                None,
                length=config.num_steps,
            )

            last_val = agent_state.critic_state.apply_fn(
                agent_state.critic_state.params, obs
            )
            advantages, targets = _gae(transitions, last_val)

            (agent_state, _, _, _, rng), _ = jax.lax.scan(
                _update_epoch,
                (agent_state, transitions, advantages, targets, rng),
                None,
                length=config.num_epochs,
            )

            # Mean episode return for this rollout (nan where no episode ended)
            ep_r = transitions.ep_return  # (num_steps, num_envs)
            valid = ~jnp.isnan(ep_r)
            n_valid = valid.sum()
            sum_valid = jnp.where(valid, ep_r, 0.0).sum()
            mean_ep_return = jnp.where(n_valid > 0, sum_valid / jnp.maximum(n_valid, 1), jnp.nan)

            return (agent_state, obs, env_state, ep_buf, rng), mean_ep_return

        # --- Outer scan over all rollout iterations ---
        (agent_state, _, _, _, _), ep_return_by_rollout = jax.lax.scan(
            _iteration,
            (agent_state, obs, env_state, ep_buf, rng),
            None,
            length=config.num_rollouts,
        )

        return agent_state, ep_return_by_rollout

    return train


def train_linen_gymnax(config: Config, env_id: str, seed: int) -> dict:
    """Compile and time full-scan Linen PPO on a Gymnax environment."""
    meta = _ENV_META[env_id]
    is_cont   = meta["action_type"] == "continuous"
    threshold = meta["threshold"]

    env, env_params = gymnax.make(env_id)
    obs_dim = env.observation_space(env_params).shape[0]

    if is_cont:
        action_dim = env.action_space(env_params).shape[0]
        act_low    = env.action_space(env_params).low
        act_high   = env.action_space(env_params).high
    else:
        action_dim = int(env.action_space(env_params).n)
        act_low = act_high = None

    train_fn = jax.jit(
        make_train_gymnax_linen(
            config, env, env_params, is_cont, action_dim, obs_dim, act_low, act_high
        )
    )

    rng = jax.random.key(seed)
    t0 = time.perf_counter()
    _, ep_return_by_rollout = train_fn(rng)
    jax.block_until_ready(ep_return_by_rollout)
    total_time = time.perf_counter() - t0

    n = config.num_rollouts
    ep_returns      = np.array(ep_return_by_rollout)
    steps_by_rollout = [config.batch_size * (i + 1) for i in range(n)]
    time_by_rollout  = [total_time * (i + 1) / n      for i in range(n)]

    time_to_threshold  = -1.0
    steps_to_threshold = -1
    for i, (ret, steps) in enumerate(zip(ep_returns, steps_by_rollout)):
        if not np.isnan(ret) and ret >= threshold:
            time_to_threshold  = time_by_rollout[i]
            steps_to_threshold = steps
            break

    return {
        "impl":                "linen",
        "env":                 env_id,
        "seed":                seed,
        "returns_by_rollout":  ep_returns.tolist(),
        "steps_by_rollout":    steps_by_rollout,
        "time_by_rollout":     time_by_rollout,
        "time_to_threshold":   time_to_threshold,
        "steps_to_threshold":  steps_to_threshold,
        "steps_per_second":    config.total_timesteps / total_time,
        "total_time":          total_time,
    }


# ---------------------------------------------------------------------------
# NNX — Python-loop training on Gymnax
# ---------------------------------------------------------------------------

def train_nnx_gymnax(config: Config, env_id: str, seed: int) -> dict:
    """Run PPO (Flax NNX, Python loop) on a Gymnax environment."""
    meta = _ENV_META[env_id]
    is_cont   = meta["action_type"] == "continuous"
    threshold = meta["threshold"]

    env, env_params = gymnax.make(env_id)
    obs_dim = env.observation_space(env_params).shape[0]

    if is_cont:
        action_dim = env.action_space(env_params).shape[0]
        act_low    = jnp.array(env.action_space(env_params).low)
        act_high   = jnp.array(env.action_space(env_params).high)
    else:
        action_dim = int(env.action_space(env_params).n)
        act_low = act_high = None

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

    total_updates = config.num_rollouts * config.num_epochs * config.num_minibatches

    def _opt(lr, clip):
        sched = optax.linear_schedule(lr, 0.0, total_updates) if config.anneal_lr else lr
        return optax.chain(optax.clip_by_global_norm(clip), optax.adam(sched, eps=1e-5))

    actor_opt  = nnx.Optimizer(actor,  _opt(config.lr_actor,  config.max_grad_norm_actor), wrt=nnx.Param)
    critic_opt = nnx.Optimizer(critic, _opt(config.lr_critic, config.max_grad_norm_critic), wrt=nnx.Param)

    rng, rng_reset = jax.random.split(rng)
    obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(rng_reset, config.num_envs), env_params
    )
    ep_buf = jnp.zeros(config.num_envs)

    # --- Scan-based rollout collection ---
    def _collect_rollout(actor, critic, obs, env_state, ep_buf, rng):
        graphdef_a, state_a = nnx.split(actor)
        graphdef_c, state_c = nnx.split(critic)

        def _env_step(carry, _):
            state_a, state_c, obs, env_state, ep_buf, rng = carry
            rng, rng_act, rng_step = jax.random.split(rng, 3)

            if is_cont:
                mean, log_std = nnx.merge(graphdef_a, state_a)(obs)
                noise = jax.random.normal(rng_act, mean.shape)
                action = jnp.clip(mean + jnp.exp(log_std) * noise, act_low, act_high)
                log_prob = _normal_log_prob(action, mean, log_std)
            else:
                logits = nnx.merge(graphdef_a, state_a)(obs)
                action = jax.random.categorical(rng_act, logits)
                log_prob = _cat_log_prob(logits, action)

            value = nnx.merge(graphdef_c, state_c)(obs)

            obs_next, env_state, reward, done, _ = jax.vmap(
                env.step, in_axes=(0, 0, 0, None)
            )(jax.random.split(rng_step, config.num_envs), env_state, action, env_params)

            done_f = done.astype(jnp.float32)
            ep_return = jnp.where(done, ep_buf + reward, jnp.nan)
            ep_buf_new = jnp.where(done, 0.0, ep_buf + reward)

            t = GpuTransition(obs, action, reward, done_f, log_prob, value, ep_return)
            return (state_a, state_c, obs_next, env_state, ep_buf_new, rng), t

        (_, _, obs, env_state, ep_buf, rng), transitions = jax.lax.scan(
            _env_step,
            (state_a, state_c, obs, env_state, ep_buf, rng),
            None,
            length=config.num_steps,
        )
        return obs, env_state, ep_buf, rng, transitions

    _collect_jit = nnx.jit(_collect_rollout)

    @nnx.jit
    def _update_mb(actor, critic, actor_opt, critic_opt, obs_mb, act_mb, lp_old, adv, tgt):
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

        actor_opt.update(actor, nnx.grad(actor_loss)(actor))
        critic_opt.update(critic, nnx.grad(critic_loss)(critic))

    def _gae(transitions: GpuTransition, last_value: jax.Array):
        def _step(carry, t: GpuTransition):
            gae, nv = carry
            delta = t.reward + config.gamma * nv * (1.0 - t.done) - t.value
            gae = delta + config.gamma * config.gae_lambda * (1.0 - t.done) * gae
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
    t0 = time.perf_counter()
    rng_perm = jax.random.key(seed + 99_999)

    for _ in tqdm(range(config.num_rollouts), desc=f"nnx_gpu/{env_id}/s{seed}", leave=False, file=sys.stderr):
        obs, env_state, ep_buf, rng, transitions = _collect_jit(
            actor, critic, obs, env_state, ep_buf, rng
        )
        last_val = nnx.jit(critic)(obs)
        advantages, targets = _gae(transitions, last_val)
        if config.advantage_norm:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        B = config.batch_size
        for _ in range(config.num_epochs):
            rng_perm, rng_sub = jax.random.split(rng_perm)
            perm = jax.random.permutation(rng_sub, B)

            def _reshape(x):
                flat = x.reshape(B, *x.shape[2:])[perm]
                return flat.reshape(
                    config.num_minibatches, config.minibatch_size, *x.shape[2:]
                )

            obs_s = _reshape(transitions.obs)
            act_s = _reshape(transitions.action)
            lp_s  = _reshape(transitions.log_prob)
            adv_s = _reshape(advantages)
            tgt_s = _reshape(targets)

            for mb in range(config.num_minibatches):
                _update_mb(
                    actor, critic, actor_opt, critic_opt,
                    obs_s[mb], act_s[mb], lp_s[mb], adv_s[mb], tgt_s[mb],
                )

        total_steps += config.batch_size
        ep_r = np.array(transitions.ep_return)
        mean_ret = (
            float(np.nanmean(ep_r))
            if not np.all(np.isnan(ep_r))
            else float("nan")
        )
        elapsed = time.perf_counter() - t0
        returns_by_rollout.append(mean_ret)
        steps_by_rollout.append(total_steps)
        time_by_rollout.append(elapsed)

        if time_to_threshold < 0 and not np.isnan(mean_ret) and mean_ret >= threshold:
            time_to_threshold  = elapsed
            steps_to_threshold = total_steps

    jax.block_until_ready(obs)
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
# Ion — Python-loop training on Gymnax
# ---------------------------------------------------------------------------

def train_ion_gymnax(config: Config, env_id: str, seed: int) -> dict:
    """Run PPO (ion, Python loop) on a Gymnax environment."""
    meta = _ENV_META[env_id]
    is_cont   = meta["action_type"] == "continuous"
    threshold = meta["threshold"]

    env, env_params = gymnax.make(env_id)
    obs_dim = env.observation_space(env_params).shape[0]

    if is_cont:
        action_dim = env.action_space(env_params).shape[0]
        act_low    = jnp.array(env.action_space(env_params).low)
        act_high   = jnp.array(env.action_space(env_params).high)
    else:
        action_dim = int(env.action_space(env_params).n)
        act_low = act_high = None

    rng = jax.random.key(seed)
    rng, key_net = jax.random.split(rng)

    network = (
        IonContinuousActorCritic(obs_dim, action_dim, config.hidden_dim, 2, key=key_net)
        if is_cont else
        IonDiscreteActorCritic(obs_dim, action_dim, key=key_net)
    )

    total_updates = config.num_rollouts * config.num_epochs * config.num_minibatches
    sched = optax.linear_schedule(config.lr_actor, 0.0, total_updates) if config.anneal_lr else config.lr_actor
    tx = optax.chain(optax.clip_by_global_norm(config.max_grad_norm_actor), optax.adam(sched, eps=1e-5))
    optimizer = ion.Optimizer(tx, network)

    rng, rng_reset = jax.random.split(rng)
    obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(rng_reset, config.num_envs), env_params
    )
    ep_buf = jnp.zeros(config.num_envs)

    # network NOT in the scan carry — passed as outer JIT arg so XLA treats it
    # as a constant within _env_step, avoiding unnecessary carry traffic.
    def _collect_rollout(network, obs, env_state, ep_buf, rng):
        def _env_step(carry, _):
            obs, env_state, ep_buf, rng = carry
            rng, rng_act, rng_step = jax.random.split(rng, 3)

            action, log_prob, value = network.get_action_and_value(obs, key=rng_act)
            # step env with clipped; store unclipped so lp_old stays consistent
            action_env = jnp.clip(action, act_low, act_high) if is_cont else action

            obs_next, env_state, reward, done, _ = jax.vmap(
                env.step, in_axes=(0, 0, 0, None)
            )(jax.random.split(rng_step, config.num_envs), env_state, action_env, env_params)

            done_f = done.astype(jnp.float32)
            ep_return = jnp.where(done, ep_buf + reward, jnp.nan)
            ep_buf_new = jnp.where(done, 0.0, ep_buf + reward)

            t = GpuTransition(obs, action, reward, done_f, log_prob, value, ep_return)
            return (obs_next, env_state, ep_buf_new, rng), t

        (obs, env_state, ep_buf, rng), transitions = jax.lax.scan(
            _env_step, (obs, env_state, ep_buf, rng), None, length=config.num_steps
        )
        return obs, env_state, ep_buf, rng, transitions

    _collect_jit = jax.jit(_collect_rollout)

    @jax.jit
    def _update_mb(network, optimizer, obs_mb, act_mb, lp_old, adv, tgt):
        def loss_fn(network):
            lp, ent, val = network.get_log_prob_entropy_value(obs_mb, act_mb)
            ratio = jnp.exp(lp - lp_old)
            pg = jnp.maximum(
                -adv * ratio,
                -adv * jnp.clip(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps),
            ).mean()
            vf = 0.5 * ((val - tgt) ** 2).mean()
            return pg + vf - config.entropy_beta * ent.mean()

        grads = jax.grad(loss_fn)(network)
        network, optimizer = optimizer.update(network, grads)
        return network, optimizer

    def _gae(transitions: GpuTransition, last_value: jax.Array):
        def _step(carry, t: GpuTransition):
            gae, nv = carry
            delta = t.reward + config.gamma * nv * (1.0 - t.done) - t.value
            gae = delta + config.gamma * config.gae_lambda * (1.0 - t.done) * gae
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
    t0 = time.perf_counter()
    rng_perm = jax.random.key(seed + 99_999)

    for _ in tqdm(range(config.num_rollouts), desc=f"ion_gpu/{env_id}/s{seed}", leave=False, file=sys.stderr):
        obs, env_state, ep_buf, rng, transitions = _collect_jit(
            network, obs, env_state, ep_buf, rng
        )
        last_val = jax.jit(lambda n, o: n.get_value(o))(network, obs)
        advantages, targets = _gae(transitions, last_val)
        if config.advantage_norm:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        B = config.batch_size
        for _ in range(config.num_epochs):
            rng_perm, rng_sub = jax.random.split(rng_perm)
            perm = jax.random.permutation(rng_sub, B)

            def _reshape(x):
                flat = x.reshape(B, *x.shape[2:])[perm]
                return flat.reshape(
                    config.num_minibatches, config.minibatch_size, *x.shape[2:]
                )

            obs_s = _reshape(transitions.obs)
            act_s = _reshape(transitions.action)
            lp_s  = _reshape(transitions.log_prob)
            adv_s = _reshape(advantages)
            tgt_s = _reshape(targets)

            for mb in range(config.num_minibatches):
                network, optimizer = _update_mb(
                    network, optimizer,
                    obs_s[mb], act_s[mb], lp_s[mb], adv_s[mb], tgt_s[mb],
                )

        total_steps += config.batch_size
        ep_r = np.array(transitions.ep_return)
        mean_ret = (
            float(np.nanmean(ep_r))
            if not np.all(np.isnan(ep_r))
            else float("nan")
        )
        elapsed = time.perf_counter() - t0
        returns_by_rollout.append(mean_ret)
        steps_by_rollout.append(total_steps)
        time_by_rollout.append(elapsed)

        if time_to_threshold < 0 and not np.isnan(mean_ret) and mean_ret >= threshold:
            time_to_threshold  = elapsed
            steps_to_threshold = total_steps

    jax.block_until_ready(obs)
    total_time = time.perf_counter() - t0

    return {
        "impl":                "ion",
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
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    config: Config,
    env_id: str,
    impl: str,
    num_seeds: int,
    monitor: ResourceMonitor,
) -> tuple[dict, list[dict]]:
    if impl == "linen":
        train_fn = train_linen_gymnax
    elif impl == "nnx":
        train_fn = train_nnx_gymnax
    else:
        train_fn = train_ion_gymnax

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
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    config = GPUConfig()

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
