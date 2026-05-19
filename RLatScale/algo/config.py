import jax
import dataclasses

from dataclasses import dataclass

@jax.tree_util.register_static
@dataclass
class Config:
    """PPO hyperparameters and experiment settings."""

    # --- Algorithm ---
    total_timesteps: int = 1_000_000
    num_envs: int = 4          # CPU: SyncVectorEnv is sequential; 4 is standard (CleanRL)
    num_steps: int = 128       # CartPole episodes up to 500 steps; 32 cuts too many mid-episode
    num_epochs: int = 4        # standard; 8 risks overfitting each collected batch
    num_minibatches: int = 4   # batch=512, minibatch=128
    hidden_dim: int = 64
    lr_actor: float = 2.5e-4   # CleanRL standard for simple envs
    lr_critic: float = 2.5e-4  # unified with actor; no benefit to gap for CartPole/Pendulum
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_beta: float = 0.01
    max_grad_norm_actor: float = 0.5
    max_grad_norm_critic: float = 0.5  # was 10.0; standard value matches actor
    advantage_norm: bool = True
    anneal_lr: bool = True
    seed: int = 42
    track: bool = False
    run_name: str | None = None
    checkpoint: bool = False

    # --- Experiment ---
    # tuple used (not list) so Config remains hashable for jax.tree_util.register_static
    num_seeds: int = 1
    envs: tuple[str, ...] = ("CartPole-v1", "Pendulum-v1")
    impls: tuple[str, ...] = ("linen", "nnx")
    results_dir: str = "results"
    # Leave empty to auto-detect from JAX backend + system info.
    # Set explicitly to label runs (e.g. "m3_8gb", "4090_pcie") when
    # auto-detection is ambiguous or you want finer-grained separation.
    hardware_tag: str = "Macbook"

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches

    @property
    def num_rollouts(self) -> int:
        return self.total_timesteps // self.batch_size


@jax.tree_util.register_static
@dataclass
class GPUConfig(Config):
    """JAX-native GPU config: large parallel env count for vectorised simulation.

    batch_size   = 2048 * 128 = 262_144
    minibatch    = 262_144 // 8 = 32_768
    num_rollouts = 50_000_000 // 262_144 ≈ 190
    """

    total_timesteps: int = 50_000_000
    num_envs: int = 2048
    num_steps: int = 128
    num_minibatches: int = 8
    num_seeds: int = 3


@jax.tree_util.register_static
@dataclass
class MuJoCoConfig(Config):
    """Gymnasium MuJoCo CPU config (CleanRL continuous-control defaults).

    batch_size   = 1 * 2048 = 2_048
    minibatch    = 2_048 // 32 = 64
    num_rollouts = 2_000_000 // 2_048 ≈ 976
    """

    total_timesteps: int = 2_000_000
    num_envs: int = 1
    num_steps: int = 2048
    num_epochs: int = 10
    num_minibatches: int = 32
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    entropy_beta: float = 0.0
    num_seeds: int = 5
    envs: tuple[str, ...] = ("HalfCheetah-v4", "Ant-v4")


@jax.tree_util.register_static
@dataclass
class BraxConfig(GPUConfig):
    """Brax JAX-native GPU config.

    batch_size   = 2048 * 10 = 20_480
    minibatch    = 20_480 // 8 = 2_560
    num_rollouts = 2_000_000 // 20_480 ≈ 97
    """

    total_timesteps: int = 2_000_000
    num_steps: int = 10
    num_seeds: int = 5
    envs: tuple[str, ...] = ("halfcheetah", "ant")
    results_dir: str = "results/brax"


@jax.tree_util.register_static
@dataclass
class MjxConfig(GPUConfig):
    """MJX (mujoco.mjx) JAX-native GPU config.

    Same batch geometry as BraxConfig; separate class so results are
    stored under a distinct hardware/impl tag.
    """

    total_timesteps: int = 2_000_000
    num_steps: int = 10
    num_seeds: int = 5
    envs: tuple[str, ...] = ("halfcheetah", "ant")
    results_dir: str = "results/mjx"