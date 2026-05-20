"""Distribution helper objects."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray


@jax.tree_util.register_dataclass
@dataclass
class Categorical:
    """Categorical distribution over discrete categories."""

    logits: Float[Array, "... a"]
    """Unnormalised log probabilities for each category."""

    axis: int = -1
    """The array dimension to reduce over."""

    @property
    def _log_probs(self) -> Float[Array, "... a"]:
        """Compute numerically safe normalised log probabilities."""
        return jax.nn.log_softmax(self.logits, axis=self.axis)

    def sample(self, *, key: PRNGKeyArray) -> Int[Array, "..."]:
        """Draw category indices from the distribution."""
        return jax.random.categorical(key, self.logits, axis=self.axis)

    def log_prob(self, indices: Int[Array, "..."]) -> Float[Array, "..."]:
        """Evaluate the log probability of the specified category indices."""
        gathered = jnp.take_along_axis(
            self._log_probs, jnp.expand_dims(indices, self.axis), axis=self.axis
        )
        return jnp.squeeze(gathered, axis=self.axis)

    @property
    def entropy(self) -> Float[Array, "..."]:
        """Compute the Shannon entropy of the distribution."""
        return -jnp.sum(jnp.exp(self._log_probs) * self._log_probs, axis=self.axis)


@jax.tree_util.register_dataclass
@dataclass
class Normal:
    """Diagonal Gaussian distribution."""

    mean: Float[Array, "... a"]
    """Mean of the distribution."""

    std: Float[Array, "... a"]
    """Standard deviation of the distribution."""

    axis: int = -1
    """The array dimension to reduce over."""

    def sample(self, *, key: PRNGKeyArray) -> Float[Array, "... a"]:
        """Draw samples via the reparameterisation trick."""
        return self.mean + self.std * jax.random.normal(key, self.mean.shape)

    def log_prob(self, x: Float[Array, "... a"]) -> Float[Array, "..."]:
        """Log-probability under the diagonal Gaussian, summed over components."""
        var = self.std**2
        log_probs = -0.5 * ((x - self.mean) ** 2 / var + jnp.log(var) + jnp.log(2 * jnp.pi))
        return jnp.sum(log_probs, axis=self.axis)

    @property
    def entropy(self) -> Float[Array, "..."]:
        """Entropy of the diagonal Gaussian, summed over components."""
        return jnp.sum(0.5 + 0.5 * jnp.log(2 * jnp.pi) + jnp.log(self.std), axis=self.axis)