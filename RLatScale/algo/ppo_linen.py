"""PPO with Flax Linen — discrete action spaces.

The entire training loop (rollout collection + parameter updates) is compiled
as a single XLA program via nested jax.lax.scan calls.  Call make_train to get
a pure function suitable for jax.jit.

Usage:
    train = jax.jit(make_train(config, env, env_params))
    agent_state, metrics = train(jax.random.key(0))
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training.train_state import TrainState

from RLatScale.algo.config import Config


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class Actor(nn.Module):
    action_dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.tanh(x)
        return nn.Dense(self.action_dim)(x)  # logits


class Critic(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.tanh(x)
        return nn.Dense(1)(x).squeeze(-1)  # scalar value estimate


# ---------------------------------------------------------------------------
# Carry and trajectory types
# ---------------------------------------------------------------------------

class AgentState(NamedTuple):
    actor_state: TrainState
    critic_state: TrainState


class Transition(NamedTuple):
    obs: jax.Array       # (num_envs, obs_dim)
    action: jax.Array    # (num_envs,)
    reward: jax.Array    # (num_envs,)
    done: jax.Array      # (num_envs,)  float32
    log_prob: jax.Array  # (num_envs,)
    value: jax.Array     # (num_envs,)


# ---------------------------------------------------------------------------
# Distribution helpers (inline; no distrax dependency)
# ---------------------------------------------------------------------------

def _log_prob(logits: jax.Array, actions: jax.Array) -> jax.Array:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return log_probs[jnp.arange(logits.shape[0]), actions]


def _entropy(logits: jax.Array) -> jax.Array:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    probs = jax.nn.softmax(logits, axis=-1)
    return -jnp.sum(probs * log_probs, axis=-1)


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------

def _compute_gae(
    transitions: Transition,
    last_value: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    """Reverse scan over a trajectory to compute advantages and value targets."""

    def _step(carry, t: Transition):
        gae, next_val = carry
        delta = t.reward + gamma * next_val * (1.0 - t.done) - t.value
        gae = delta + gamma * gae_lambda * (1.0 - t.done) * gae
        return (gae, t.value), (gae, gae + t.value)

    _, (advantages, targets) = jax.lax.scan(
        _step,
        (jnp.zeros_like(last_value), last_value),
        transitions,
        reverse=True,
    )
    return advantages, targets


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def make_train(config: Config, env, env_params):
    """Return a pure function ``train(rng) -> (AgentState, metrics)``."""

    obs_dim = env.observation_space(env_params).shape[0]
    action_dim = env.action_space(env_params).n

    total_updates = config.num_rollouts * config.num_epochs * config.num_minibatches

    def _make_opt(lr: float, clip: float) -> optax.GradientTransformation:
        schedule = optax.linear_schedule(lr, 0.0, total_updates) if config.anneal_lr else lr
        return optax.chain(optax.clip_by_global_norm(clip), optax.adam(schedule, eps=1e-5))

    def train(rng: jax.Array):
        # --- Initialise networks and optimisers ---
        rng, rng_a, rng_c = jax.random.split(rng, 3)
        dummy = jnp.zeros((1, obs_dim))

        actor = Actor(action_dim=action_dim, hidden_dim=config.hidden_dim)
        critic = Critic(hidden_dim=config.hidden_dim)

        actor_state = TrainState.create(
            apply_fn=actor.apply,
            params=actor.init(rng_a, dummy),
            tx=_make_opt(config.lr_actor, config.max_grad_norm_actor),
        )
        critic_state = TrainState.create(
            apply_fn=critic.apply,
            params=critic.init(rng_c, dummy),
            tx=_make_opt(config.lr_critic, config.max_grad_norm_critic),
        )
        agent_state = AgentState(actor_state, critic_state)

        # --- Reset environments ---
        rng, rng_reset = jax.random.split(rng)
        obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(
            jax.random.split(rng_reset, config.num_envs), env_params
        )

        # --- Single environment step (inner scan body) ---
        def _env_step(carry, _):
            agent_state, obs, env_state, rng = carry
            rng, rng_act, rng_step = jax.random.split(rng, 3)

            logits = agent_state.actor_state.apply_fn(agent_state.actor_state.params, obs)
            action = jax.random.categorical(rng_act, logits)
            log_prob = _log_prob(logits, action)
            value = agent_state.critic_state.apply_fn(agent_state.critic_state.params, obs)

            obs_next, env_state, reward, done, _ = jax.vmap(
                env.step, in_axes=(0, 0, 0, None)
            )(jax.random.split(rng_step, config.num_envs), env_state, action, env_params)

            t = Transition(obs, action, reward, done.astype(jnp.float32), log_prob, value)
            return (agent_state, obs_next, env_state, rng), t

        # --- Single minibatch gradient update ---
        def _update_minibatch(agent_state: AgentState, mb) -> AgentState:
            obs_mb, act_mb, lp_old_mb, adv_mb, tgt_mb = mb

            def actor_loss_fn(params):
                logits = agent_state.actor_state.apply_fn(params, obs_mb)
                lp = _log_prob(logits, act_mb)
                ent = _entropy(logits)
                ratio = jnp.exp(lp - lp_old_mb)
                pg_loss = jnp.maximum(
                    -adv_mb * ratio,
                    -adv_mb * jnp.clip(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps),
                ).mean()
                return pg_loss - config.entropy_beta * ent.mean()

            def critic_loss_fn(params):
                v = agent_state.critic_state.apply_fn(params, obs_mb)
                return 0.5 * jnp.mean((v - tgt_mb) ** 2)

            a_grads = jax.grad(actor_loss_fn)(agent_state.actor_state.params)
            c_grads = jax.grad(critic_loss_fn)(agent_state.critic_state.params)

            return AgentState(
                agent_state.actor_state.apply_gradients(grads=a_grads),
                agent_state.critic_state.apply_gradients(grads=c_grads),
            )

        # --- Single epoch: shuffle batch and iterate over minibatches ---
        def _update_epoch(carry, _):
            agent_state, transitions, advantages, targets, rng = carry
            rng, rng_perm = jax.random.split(rng)
            if config.advantage_norm:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            B = config.batch_size
            perm = jax.random.permutation(rng_perm, B)

            def _reshape(x: jax.Array) -> jax.Array:
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
                lambda s, mb: (_update_minibatch(s, mb), None),
                agent_state,
                minibatches,
            )
            return (agent_state, transitions, advantages, targets, rng), None

        # --- One rollout + update iteration ---
        def _iteration(carry, _):
            agent_state, obs, env_state, rng = carry

            (agent_state, obs, env_state, rng), transitions = jax.lax.scan(
                _env_step,
                (agent_state, obs, env_state, rng),
                None,
                length=config.num_steps,
            )

            last_val = agent_state.critic_state.apply_fn(
                agent_state.critic_state.params, obs
            )
            advantages, targets = _compute_gae(
                transitions, last_val, config.gamma, config.gae_lambda
            )

            (agent_state, _, _, rng), _ = jax.lax.scan(
                _update_epoch,
                (agent_state, transitions, advantages, targets, rng),
                None,
                length=config.num_epochs,
            )

            metrics = {
                "mean_reward": transitions.reward.mean(),
                "mean_value": transitions.value.mean(),
                "mean_advantage": advantages.mean(),
            }
            return (agent_state, obs, env_state, rng), metrics

        # --- Outer scan over all rollout iterations ---
        (agent_state, _, _, _), metrics = jax.lax.scan(
            _iteration,
            (agent_state, obs, env_state, rng),
            None,
            length=config.num_rollouts,
        )

        return agent_state, metrics

    return train
