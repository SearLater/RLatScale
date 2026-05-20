"""Continuous PPO clip."""

import dataclasses
import time
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import NamedTuple
from collections import deque

import ion
import jax
import jax.numpy as jnp
import numpy as np
import optax
import parallax
import wandb
from ion import nn
from jax.nn.initializers import orthogonal
from jaxtyping import Array, Bool, Float, PRNGKeyArray
from tqdm import tqdm

class Tracker:
    """Tracks episode returns and lengths across vectorized environments.

    Parameters
    ----------
    num_envs : int
        Number of parallel environments.
    window_size : int, optional
        Maximum number of episodes to keep for rolling statistics, by default 100.
    """

    def __init__(self, num_envs: int, window_size: int = 100) -> None:
        self.num_envs = num_envs
        self.window_size = window_size

        self.current_returns = np.zeros(num_envs)
        self.current_lengths = np.zeros(num_envs, dtype=np.int32)

        self.episode_returns: deque[float] = deque(maxlen=window_size)
        self.episode_lengths: deque[int] = deque(maxlen=window_size)

        self.total_steps = 0
        self.total_episodes = 0
        self.best_return = -np.inf

        self.tick_start: float | None = None
        self.sps = 0.0

    @property
    def mean_return(self) -> float:
        return float(np.mean(self.episode_returns)) if self.episode_returns else 0.0

    @property
    def mean_length(self) -> float:
        return float(np.mean(self.episode_lengths)) if self.episode_lengths else 0.0

    def tick(self, rewards: NDArray, dones: NDArray) -> None:
        """Update tracker with a batch of rollout data.

        Parameters
        ----------
        rewards : NDArray
            Rewards array, shape ``(num_steps, num_envs)``.
        dones : NDArray
            Boolean done flags, shape ``(num_steps, num_envs)``.
        """
        now = time.time()
        batch_steps = rewards.shape[0] * rewards.shape[1]

        if self.tick_start is not None:
            elapsed = now - self.tick_start
            if elapsed > 0:
                self.sps = batch_steps / elapsed
        self.tick_start = now

        for step_r, step_d in zip(rewards, dones):
            self.current_returns += step_r
            self.current_lengths += 1
            self.total_steps += self.num_envs

            if step_d.any():
                for ret, length in zip(
                    self.current_returns[step_d],
                    self.current_lengths[step_d],
                ):
                    ret = float(ret)
                    self.episode_returns.append(ret)
                    self.episode_lengths.append(int(length))
                    self.total_episodes += 1

                    if ret > self.best_return:
                        self.best_return = ret

                self.current_returns[step_d] = 0.0
                self.current_lengths[step_d] = 0

    @property
    def metrics(self) -> dict:
        """Snapshot of current tracking statistics, prefixed for logging.

        Returns
        -------
        dict
            Dictionary containing rollout metrics ready for wandb logging.
        """
        return {
            "episode/mean_return": self.mean_return,
            "episode/mean_length": self.mean_length,
            "episode/best_return": self.best_return,
            "episode/total_episodes": self.total_episodes,
            "episode/sps": self.sps,
        }

def clipped_surrogate_loss(
    new_log_probs: Float[Array, " b"],
    old_log_probs: Float[Array, " b"],
    advantages: Float[Array, " b"],
    clip_eps: float,
) -> Float[Array, ""]:
    """Compute PPO clipped surrogate policy loss."""
    ratio = jnp.exp(new_log_probs - old_log_probs)
    loss_unclipped = -advantages * ratio
    loss_clipped = -advantages * jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
    return jnp.maximum(loss_unclipped, loss_clipped).mean()


def value_loss(
    values: Float[Array, " b"],
    returns: Float[Array, " b"],
) -> Float[Array, ""]:
    """Compute mean squared error value loss."""
    return 0.5 * ((returns - values) ** 2).mean()


def entropy_loss(
    entropies: Float[Array, " b"],
) -> Float[Array, ""]:
    """Compute entropy bonus loss (negative to encourage exploration)."""
    return -entropies.mean()

def calculate_gae(
    rewards: Float[Array, "t e"],
    values: Float[Array, "t e"],
    next_values: Float[Array, "t e"],
    terminations: Bool[Array, "t e"],
    truncations: Bool[Array, "t e"],
    gamma: float,
    gae_lambda: float,
) -> Float[Array, "t e"]:
    """Compute GAE advantages via reversed scan over timesteps.

    Parameters
    ----------
    rewards : Float[Array, "t e"]
        Per-step rewards.
    values : Float[Array, "t e"]
        Value estimates at each timestep.
    next_values : Float[Array, "t e"]
        Value estimates at the following timestep.
    terminations : Bool[Array, "t e"]
        Whether the episode terminated at each step.
    truncations : Bool[Array, "t e"]
        Whether the episode was truncated at each step.
    gamma : float
        Discount factor.
    gae_lambda : float
        GAE exponential weighting factor.

    Returns
    -------
    Float[Array, "t e"]
        Computed advantages for each timestep and environment.
    """

    def gae_step(advantage, transition):
        reward, value, next_value, termination, truncation = transition
        non_termination = 1.0 - termination
        non_truncation = 1.0 - truncation
        delta = reward + gamma * next_value * non_termination - value
        advantage = delta + gamma * gae_lambda * non_termination * non_truncation * advantage
        return advantage, advantage

    _, advantages = jax.lax.scan(
        f=gae_step,
        init=jnp.zeros(rewards.shape[1]),
        xs=(rewards, values, next_values, terminations, truncations),
        reverse=True,
    )
    return advantages

@jax.tree_util.register_static
@dataclass
class Config:
    """Continuous PPO training hyperparameters."""

    total_timesteps: int = 1_000_000
    num_envs: int = 64
    num_steps: int = 32
    num_epochs: int = 8
    num_minibatches: int = 4
    hidden_dim: int = 64
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_beta: float = 0.0
    max_grad_norm_actor: float = 0.5
    max_grad_norm_critic: float = 10.0
    advantage_norm: bool = True
    anneal_lr: bool = True
    seed: int = 42
    track: bool = False
    wandb_project: str = "xenon"
    run_name: str | None = None
    checkpoint: bool = False

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches

    @property
    def num_rollouts(self) -> int:
        return self.total_timesteps // self.batch_size


class ActorCritic(nn.Module):
    """Actor-critic network for continuous action spaces."""

    actor: nn.Sequential
    critic: nn.Sequential
    std_raw: nn.Param

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        *,
        key: PRNGKeyArray,
    ) -> None:
        keys = jax.random.split(key, 6)
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim, w_init=orthogonal(scale=2**0.5), key=keys[0]),
            jax.nn.tanh,
            nn.Linear(hidden_dim, hidden_dim, w_init=orthogonal(scale=2**0.5), key=keys[1]),
            jax.nn.tanh,
            nn.Linear(hidden_dim, action_dim, w_init=orthogonal(scale=0.01), key=keys[2]),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim, w_init=orthogonal(scale=2**0.5), key=keys[3]),
            jax.nn.tanh,
            nn.Linear(hidden_dim, hidden_dim, w_init=orthogonal(scale=2**0.5), key=keys[4]),
            jax.nn.tanh,
            nn.Linear(hidden_dim, 1, w_init=orthogonal(scale=1.0), key=keys[5]),
        )
        self.std_raw = nn.Param(jnp.zeros(action_dim))

    def get_action(
        self,
        observations: Float[Array, "... d"],
        *,
        key: PRNGKeyArray,
    ) -> Float[Array, "... a"]:
        """Sample actions from the policy."""
        dist = Normal(mean=self.actor(observations), std=jax.nn.softplus(self.std_raw) + 1e-6)
        return dist.sample(key=key)

    def get_value(self, observations: Float[Array, "... d"]) -> Float[Array, "..."]:
        """Estimate state value."""
        return self.critic(observations).squeeze(-1)

    def get_action_log_prob_value(
        self,
        observations: Float[Array, "... d"],
        *,
        key: PRNGKeyArray,
    ) -> tuple[Float[Array, "... a"], Float[Array, "..."], Float[Array, "..."]]:
        """Sample action, compute its log-prob and value estimate."""
        dist = Normal(mean=self.actor(observations), std=jax.nn.softplus(self.std_raw) + 1e-6)
        action = dist.sample(key=key)
        log_prob = dist.log_prob(action)
        value = self.critic(observations).squeeze(-1)
        return action, log_prob, value

    def get_log_prob_entropy_value(
        self,
        observations: Float[Array, "... d"],
        action: Float[Array, "... a"],
    ) -> tuple[Float[Array, "..."], Float[Array, "..."], Float[Array, "..."]]:
        """Compute action log-prob, entropy, and value estimate."""
        dist = Normal(mean=self.actor(observations), std=jax.nn.softplus(self.std_raw) + 1e-6)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy
        value = self.critic(observations).squeeze(-1)
        return log_prob, entropy, value


class Transition(NamedTuple):
    observations: Float[Array, "..."]
    next_observations: Float[Array, "..."]
    rewards: Float[Array, "..."]
    terminations: Bool[Array, "..."]
    truncations: Bool[Array, "..."]
    actions: Float[Array, "..."]
    log_probs: Float[Array, "..."]
    values: Float[Array, "..."]


@partial(jax.jit, static_argnums=(2,))
def rollout(
    network: ActorCritic,
    state: parallax.State,
    env: parallax.VectorEnv,
    config: Config,
    *,
    key: PRNGKeyArray,
) -> tuple[parallax.State, Transition]:
    """Collect transitions from vectorized environments via `lax.scan`."""

    def rollout_step(state, key):
        key_action, key_reset = jax.random.split(key)
        observations = state.observation

        actions, log_probs, values = network.get_action_log_prob_value(observations, key=key_action)

        next_state = env.step(state, actions)

        transition = Transition(
            observations=observations,
            next_observations=next_state.observation,
            rewards=next_state.reward,
            terminations=next_state.termination,
            truncations=next_state.truncation,
            actions=actions,
            log_probs=log_probs,
            values=values,
        )

        # Auto-reset done environments
        next_state = env.reset(key=key_reset, state=next_state, done=next_state.done)

        return next_state, transition

    state, transitions = jax.lax.scan(
        f=rollout_step, init=state, xs=jax.random.split(key, config.num_steps)
    )
    return state, transitions


def loss_fn(
    network: ActorCritic,
    observations: Float[Array, "b ..."],
    actions: Float[Array, "b ..."],
    advantages: Float[Array, " b"],
    returns: Float[Array, " b"],
    old_log_probs: Float[Array, " b"],
    config: Config,
) -> tuple[Float[Array, ""], dict]:
    """PPO clipped surrogate loss with value MSE and entropy bonus."""

    new_log_probs, entropies, values = network.get_log_prob_entropy_value(observations, actions)

    loss_policy = clipped_surrogate_loss(new_log_probs, old_log_probs, advantages, config.clip_eps)
    loss_value = value_loss(values, returns)
    loss_entropy = entropy_loss(entropies)
    total = loss_policy + loss_value + config.entropy_beta * loss_entropy

    # Calculate metrics
    ratio = jnp.exp(new_log_probs - old_log_probs)
    metrics = {
        "loss/policy": loss_policy,
        "loss/value": loss_value,
        "loss/entropy": loss_entropy,
        "policy/clip_fraction": jnp.mean(jnp.abs(ratio - 1.0) > config.clip_eps),
        "policy/approx_kl": jnp.mean((ratio - 1.0) - jnp.log(ratio)),
    }
    return total, metrics


@jax.jit
def learn(
    network: ActorCritic,
    optimizer: ion.Optimizer,
    batch: Transition,
    config: Config,
    *,
    key: PRNGKeyArray,
) -> tuple[ActorCritic, ion.Optimizer, dict]:
    """Compute GAE advantages then scan over minibatch PPO updates."""

    advantages = calculate_gae(
        rewards=batch.rewards,
        values=batch.values,
        next_values=network.get_value(batch.next_observations),
        terminations=batch.terminations,
        truncations=batch.truncations,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
    )
    returns = advantages + batch.values

    advantage_mean, advantage_std = advantages.mean(), advantages.std()
    if config.advantage_norm:
        advantages = (advantages - advantage_mean) / (advantage_std + 1e-6)

    # Flatten batch data
    batch = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), batch)
    advantages, returns = advantages.flatten(), returns.flatten()

    # Generate minibatch indices
    keys = jax.random.split(key, config.num_epochs)
    mb_indices = jax.vmap(jax.random.permutation, in_axes=(0, None))(keys, config.batch_size)
    mb_indices = mb_indices.reshape(-1, config.minibatch_size)

    def update_step(carry, indices):
        network, optimizer = carry

        grads, metrics = jax.grad(loss_fn, has_aux=True)(
            network,
            observations=batch.observations[indices],
            actions=batch.actions[indices],
            advantages=advantages[indices],
            returns=returns[indices],
            old_log_probs=batch.log_probs[indices],
            config=config,
        )
        network, optimizer = optimizer.update(network, grads)

        metrics["norm/grad_actor"] = optax.tree.norm(grads.actor)
        metrics["norm/grad_critic"] = optax.tree.norm(grads.critic)

        return (network, optimizer), metrics

    (network, optimizer), metrics = jax.lax.scan(
        f=update_step, init=(network, optimizer), xs=mb_indices
    )

    # Calculate metrics
    metrics = jax.tree.map(lambda x: x.mean(), metrics)
    explained_var = 1 - jnp.var(returns - batch.values.flatten()) / (jnp.var(returns) + 1e-8)
    metrics["value/explained_variance"] = explained_var
    metrics["value/advantage_mean"] = advantage_mean
    metrics["value/advantage_std"] = advantage_std
    metrics["norm/weight_actor"] = optax.tree.norm(network.actor.params)
    metrics["norm/weight_critic"] = optax.tree.norm(network.critic.params)

    return network, optimizer, metrics


def train(
    env: parallax.Env,
    config: Config,
    network: ActorCritic | None = None,
) -> ActorCritic:
    """Train a continuous PPO agent."""

    split = jax.jit(jax.random.split, static_argnums=(1,))
    rng = jax.random.key(config.seed)
    rng, key_network, key_reset = split(rng, 3)

    # Vectorize environment
    env = parallax.VmapWrapper(env, num_envs=config.num_envs)

    # Actor critic network
    if network is None:
        obs_dim = env.observation_space.shape[0]  # type: ignore
        action_dim = env.action_space.shape[0]  # type: ignore
        network = ActorCritic(obs_dim, action_dim, config.hidden_dim, key=key_network)

    # Optimizer setup with different LR and max grad norms for actor and critic
    transition_steps = config.num_rollouts * config.num_epochs * config.num_minibatches
    lr_schedule_actor = optax.linear_schedule(
        init_value=config.lr_actor,
        end_value=0 if config.anneal_lr else config.lr_actor,
        transition_steps=transition_steps,
    )
    lr_schedule_critic = optax.linear_schedule(
        init_value=config.lr_critic,
        end_value=0 if config.anneal_lr else config.lr_critic,
        transition_steps=transition_steps,
    )
    optimizer = ion.Optimizer(
        {
            ("actor", "std_raw"): optax.chain(
                optax.clip_by_global_norm(config.max_grad_norm_actor),
                optax.adam(learning_rate=lr_schedule_actor, eps=1e-5),
            ),
            "critic": optax.chain(
                optax.clip_by_global_norm(config.max_grad_norm_critic),
                optax.adam(learning_rate=lr_schedule_critic, eps=1e-5),
            ),
        },
        network,
    )

    # Episode tracking
    tracker = Tracker(config.num_envs)

    if config.track:
        wandb.init(
            project=config.wandb_project,
            name=config.run_name,
            config={**dataclasses.asdict(config), "device": jax.devices()[0].device_kind},
        )

    # Reset vectorized environments
    state = env.reset(key=key_reset)

    bar = tqdm(total=config.total_timesteps, desc="PPO", unit=" step", unit_scale=True)
    for i in range(config.num_rollouts):
        rng, key_rollout, key_learn = split(rng, 3)

        # Perform rollout to gather transitions
        state, transitions = rollout(network, state, env, config, key=key_rollout)

        # Learn from gathered transitions
        network, optimizer, metrics = learn(network, optimizer, transitions, config, key=key_learn)

        # Record progress
        tracker.tick(
            rewards=np.asarray(transitions.rewards),
            dones=np.asarray(transitions.terminations) | np.asarray(transitions.truncations),
        )

        bar.update(config.batch_size)
        if tracker.total_episodes > 0:
            bar.set_postfix_str(
                f"reward={tracker.mean_return:.1f}, length={tracker.mean_length:.0f}"
            )

        # Wandb logging
        if config.track:
            updates_done = (i + 1) * config.num_epochs * config.num_minibatches
            action_means = np.asarray(transitions.actions).mean(axis=(0, 1))
            action_stds = np.log1p(np.exp(np.asarray(network.std_raw)))  # type: ignore
            wandb.log(
                {
                    **{k: v.item() for k, v in metrics.items()},
                    **tracker.metrics,
                    "hyper/lr_actor": lr_schedule_actor(updates_done),
                    "hyper/lr_critic": lr_schedule_critic(updates_done),
                    **{f"action/mean/{i}": m.item() for i, m in enumerate(action_means)},
                    **{f"action/std/{i}": s.item() for i, s in enumerate(action_stds)},
                },
                step=tracker.total_steps,
            )

    bar.close()

    if config.checkpoint:
        name = f"{config.run_name or 'experiment'}_{datetime.now():%Y-%m-%d_%H-%M-%S}"
        Path("checkpoints", name).mkdir(parents=True, exist_ok=True)
        ion.save(str(Path("checkpoints", name, "network")), network)

    if config.track:
        wandb.finish()

    return network  # type: ignore