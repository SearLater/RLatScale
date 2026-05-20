"""PPO with Flax NNX — discrete action spaces.

Uses a Python-level training loop with nnx.jit-compiled step and update
functions.  The rollout is collected inside a jax.lax.scan using extracted
pure parameter state; gradient updates run in a Python loop so that NNX's
mutable module API is used idiomatically.

This structural difference from ppo_linen.py — Python loop vs full scan — is
the primary variable being benchmarked.

Usage:
    actor, critic, metrics = train(config, env, env_params, jax.random.key(0))
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from RLatScale.algo.config import Config


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class Actor(nnx.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, rngs: nnx.Rngs):
        self.l1 = nnx.Linear(obs_dim, hidden_dim, rngs=rngs)
        self.l2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.out = nnx.Linear(hidden_dim, action_dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = jax.nn.tanh(self.l1(x))
        x = jax.nn.tanh(self.l2(x))
        return self.out(x)  # logits


class Critic(nnx.Module):
    def __init__(self, obs_dim: int, hidden_dim: int, rngs: nnx.Rngs):
        self.l1 = nnx.Linear(obs_dim, hidden_dim, rngs=rngs)
        self.l2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.out = nnx.Linear(hidden_dim, 1, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = jax.nn.tanh(self.l1(x))
        x = jax.nn.tanh(self.l2(x))
        return self.out(x).squeeze(-1)  # scalar value estimate


# ---------------------------------------------------------------------------
# Trajectory type
# ---------------------------------------------------------------------------

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

def train(
    config: Config,
    env,
    env_params,
    rng: jax.Array,
) -> tuple[Actor, Critic, list[dict]]:
    """Run PPO training and return ``(actor, critic, metrics_per_rollout)``."""

    obs_dim = env.observation_space(env_params).shape[0]
    action_dim = env.action_space(env_params).n

    total_updates = config.num_rollouts * config.num_epochs * config.num_minibatches

    def _make_tx(lr: float, clip: float) -> optax.GradientTransformation:
        schedule = optax.linear_schedule(lr, 0.0, total_updates) if config.anneal_lr else lr
        return optax.chain(optax.clip_by_global_norm(clip), optax.adam(schedule, eps=1e-5))

    # --- Initialise networks and optimisers ---
    rng, rng_a, rng_c = jax.random.split(rng, 3)
    seed_a = int(jax.random.randint(rng_a, (), 0, 2**16))
    seed_c = int(jax.random.randint(rng_c, (), 0, 2**16))

    actor = Actor(obs_dim, action_dim, config.hidden_dim, nnx.Rngs(seed_a))
    critic = Critic(obs_dim, config.hidden_dim, nnx.Rngs(seed_c))

    actor_opt = nnx.Optimizer(actor, _make_tx(config.lr_actor, config.max_grad_norm_actor), wrt=nnx.Param)
    critic_opt = nnx.Optimizer(critic, _make_tx(config.lr_critic, config.max_grad_norm_critic), wrt=nnx.Param)

    # --- Reset environments ---
    rng, rng_reset = jax.random.split(rng)
    obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(rng_reset, config.num_envs), env_params
    )

    # --- Rollout collection via lax.scan over extracted pure state ---
    # NNX modules are split into a static graphdef and a pytree of arrays so
    # that the inner scan body remains a pure JAX function.
    def _collect_rollout(actor: Actor, critic: Critic, obs, env_state, rng):
        graphdef_a, state_a = nnx.split(actor)
        graphdef_c, state_c = nnx.split(critic)

        def _env_step(carry, _):
            state_a, state_c, obs, env_state, rng = carry
            rng, rng_act, rng_step = jax.random.split(rng, 3)

            logits = nnx.merge(graphdef_a, state_a)(obs)
            action = jax.random.categorical(rng_act, logits)
            log_prob = _log_prob(logits, action)
            value = nnx.merge(graphdef_c, state_c)(obs)

            obs_next, env_state, reward, done, _ = jax.vmap(
                env.step, in_axes=(0, 0, 0, None)
            )(jax.random.split(rng_step, config.num_envs), env_state, action, env_params)

            t = Transition(obs, action, reward, done.astype(jnp.float32), log_prob, value)
            return (state_a, state_c, obs_next, env_state, rng), t

        (_, _, obs, env_state, rng), transitions = jax.lax.scan(
            _env_step,
            (state_a, state_c, obs, env_state, rng),
            None,
            length=config.num_steps,
        )
        return obs, env_state, rng, transitions

    # JIT-compile the rollout; the graphdefs are captured as static closures.
    _collect_rollout_jit = nnx.jit(_collect_rollout)

    # --- Minibatch update (NNX mutable API) ---
    @nnx.jit
    def _update_minibatch(
        actor: Actor,
        critic: Critic,
        actor_opt: nnx.Optimizer,
        critic_opt: nnx.Optimizer,
        obs_mb: jax.Array,
        act_mb: jax.Array,
        lp_old_mb: jax.Array,
        adv_mb: jax.Array,
        tgt_mb: jax.Array,
    ) -> None:
        def actor_loss_fn(actor: Actor) -> jax.Array:
            logits = actor(obs_mb)
            lp = _log_prob(logits, act_mb)
            ent = _entropy(logits)
            ratio = jnp.exp(lp - lp_old_mb)
            pg_loss = jnp.maximum(
                -adv_mb * ratio,
                -adv_mb * jnp.clip(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps),
            ).mean()
            return pg_loss - config.entropy_beta * ent.mean()

        def critic_loss_fn(critic: Critic) -> jax.Array:
            v = critic(obs_mb)
            return 0.5 * jnp.mean((v - tgt_mb) ** 2)

        actor_opt.update(actor, nnx.grad(actor_loss_fn)(actor))
        critic_opt.update(critic, nnx.grad(critic_loss_fn)(critic))

    # --- Outer Python loop over rollout iterations ---
    metrics: list[dict] = []

    for _ in range(config.num_rollouts):
        obs, env_state, rng, transitions = _collect_rollout_jit(
            actor, critic, obs, env_state, rng
        )

        last_val = nnx.jit(critic)(obs)
        advantages, targets = _compute_gae(
            transitions, last_val, config.gamma, config.gae_lambda
        )
        if config.advantage_norm:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Epoch and minibatch loops (Python-level, nnx.jit per minibatch)
        for _ in range(config.num_epochs):
            rng, rng_perm = jax.random.split(rng)
            B = config.batch_size
            perm = jax.random.permutation(rng_perm, B)

            def _reshape(x: jax.Array) -> jax.Array:
                flat = x.reshape(B, *x.shape[2:])[perm]
                return flat.reshape(config.num_minibatches, config.minibatch_size, *x.shape[2:])

            obs_shuf = _reshape(transitions.obs)
            act_shuf = _reshape(transitions.action)
            lp_shuf = _reshape(transitions.log_prob)
            adv_shuf = _reshape(advantages)
            tgt_shuf = _reshape(targets)

            for mb in range(config.num_minibatches):
                _update_minibatch(
                    actor, critic, actor_opt, critic_opt,
                    obs_shuf[mb], act_shuf[mb], lp_shuf[mb], adv_shuf[mb], tgt_shuf[mb],
                )

        metrics.append({
            "mean_reward": float(transitions.reward.mean()),
            "mean_value": float(transitions.value.mean()),
            "mean_advantage": float(advantages.mean()),
        })

    return actor, critic, metrics
