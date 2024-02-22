"""Implementation of the Marchand toad model.

This model simulates the movement of Fowler's toad species.
"""

import jax.numpy as jnp
import numpyro.distributions as dist  # type: ignore
from jax import random
from jax._src.prng import PRNGKeyArray  # for typing
import numpy as np


def levy_stable(key: PRNGKeyArray,
                alpha: jnp.ndarray,
                gamma: jnp.ndarray,
                size=None):
    """Sample from Levy-stable distribution.

    Args:
        key (PRNGKeyArray): a PRNGKeyArray for reproducibility.
        alpha (jnp.ndarray): Stability parameter
        gamma (jnp.ndarray): Scale parameter
        size (tuple): Shape of output

    Returns:
        jnp.ndarray in shape `size`
    """
    if size is None:
        size = jnp.shape(alpha)

    key1, key2, key3 = random.split(key, num=3)

    # General case
    u = random.uniform(key1,
                       minval=-0.5*jnp.pi + 1e-5,  # stop floating point error
                       maxval=0.5*jnp.pi - 1e-5,  # stop floating point error
                       shape=size)
    v = random.exponential(key2, shape=size)
    t = jnp.sin(alpha * u) / (jnp.cos(u) ** (1 / alpha))
    s = (jnp.cos((1 - alpha) * u) / v) ** ((1 - alpha) / alpha)
    output = gamma * t * s

    # Handle alpha == 1
    cauchy_sample = random.cauchy(key3, shape=size)
    output = jnp.where(alpha == 1, cauchy_sample, output)

    # # Handle alpha == 2
    normal_sample = random.normal(key3, shape=size) * jnp.sqrt(2) * gamma
    output = jnp.where(alpha == 2, normal_sample, output)

    return output


def dgp(key: PRNGKeyArray,
        alpha: jnp.ndarray,
        gamma: jnp.ndarray,
        p0: jnp.ndarray,
        model: int = 1,
        n_toads: int = 66,
        n_days: int = 63,
        batch_size: int = 1
        ) -> jnp.ndarray:
    """Sample the movement of Fowler's toad species.

    Returns:
        jnp.ndarray in shape (n_days x n_toads x batch_size)
    """
    X = jnp.zeros((n_days, n_toads, batch_size))

    # Generate step length from levy_stable distribution
    delta_x = levy_stable(key, alpha, gamma, size=(n_days, n_toads, batch_size))

    for i in range(1, n_days):
        # Generate random uniform samples for returns
        key, subkey = random.split(key)
        ret = random.uniform(subkey, shape=(n_toads, batch_size)) < jnp.squeeze(p0)

        # Calculate new positions for all toads
        new_positions = X[i-1, :] + delta_x[i, :]

        # Handle returning toads
        key, subkey = random.split(key)
        if model == 1:
            ind_refuge = random.choice(subkey, jnp.arange(i), shape=(n_toads, batch_size))
        if model == 2:
            # xn - curr
            if i > 1:
                ind_refuge = jnp.argmin(jnp.abs(new_positions - X[:i, :]), axis=0)
            else:
                ind_refuge = jnp.zeros((n_toads, batch_size), dtype=int)

        # Extract previous positions for updating
        update_values = jnp.zeros((n_toads, batch_size))
        for j in range(batch_size):
            update_values = update_values.at[:, j].set(X[ind_refuge[:, j], np.arange(n_toads), j].flatten())

        # Combine new_positions and update_values for final_positions
        final_positions = jnp.where(ret, update_values, new_positions)

        X = X.at[i, :, :].set(final_positions)

    return X


def calculate_summary_statistics(X, real_data=False, nan_idx=None, lags=[1, 2, 4, 8]):
    """Calculate summary statistics for Marchand toad example."""
    ssx = jnp.concatenate([
        calculate_summary_statistics_lag(X, lag, real_data=real_data, nan_idx=nan_idx)
        for lag in lags
    ], axis=1)
    ssx = jnp.clip(ssx, -1e+6, 1e+6)  # NOTE: fix for some extreme results
    return ssx.flatten()


def calculate_summary_statistics_lag(X, lag, p=jnp.linspace(0, 1, 11), thd=10,
                                     real_data=False, nan_idx=None):
    """Calculate summary statistics for Marchand toad example.

    Args:
        X: Output from dgp function.
        lag, p, thd: See original function.

    Returns:
        A tensor of shape (batch_size, len(p) + 1).
    """
    if nan_idx is not None:
        X = X.at[nan_idx].set(jnp.nan)

    disp = X[lag:, :] - X[:-lag, :]

    abs_disp = jnp.abs(disp)
    abs_disp = abs_disp.flatten()

    ret = abs_disp < thd
    num_ret = jnp.sum(ret, axis=0)

    abs_disp = jnp.where(ret, jnp.nan, abs_disp)

    abs_noret_median = jnp.nanmedian(abs_disp, axis=0)
    abs_noret_quantiles = jnp.nanquantile(abs_disp, p, axis=0)
    diff = jnp.diff(abs_noret_quantiles, axis=0)
    logdiff = jnp.log(jnp.maximum(diff, jnp.exp(-20)))

    ssx = jnp.vstack((num_ret, abs_noret_median, logdiff.reshape(-1, 1)))
    ssx = jnp.nan_to_num(ssx, nan=jnp.inf)

    return ssx.T


def get_prior():
    """Return prior for Marchand toad example."""
    prior = dist.Uniform(low=jnp.array([1.0, 20.0, 0.4]),
                         high=jnp.array([2.0, 70.0, 0.9]))
    return prior
