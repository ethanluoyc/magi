from typing import Callable, Optional, Tuple

import chex
import jax
from jax import lax
import jax.numpy as jnp
import jax.experimental.host_callback as hcb


def minimize_cem(fn: Callable,
                 x0: jnp.ndarray,
                 key: jnp.ndarray,
                 args: Optional[Tuple] = (),
                 *,
                 bounds: Tuple[jnp.ndarray, jnp.ndarray],
                 n_iterations: int,
                 population_size: int,
                 elite_fraction: float,
                 alpha: float,
                 fn_use_key: bool = False,
                 return_mean_elites: bool = False) -> jnp.ndarray:
  """Minimize a function `fn` using the cross-entropy method."""
  num_elites = int(population_size * elite_fraction)
  lower_bound, upper_bound = bounds
  action_shape = x0.shape

  population_shape = (population_size,) + action_shape

  # epsilon = 0.001

  def cond_fn(state):
    t, best_cost, best_solution, mu, var, rng, args = state
    return t < n_iterations
    # return jnp.logical_and(t < n_iterations, jnp.max(var) > epsilon)

  def loop(state):
    t, best_cost, best_solution, mu, var, rng, args = state
    rng, key, key2 = jax.random.split(rng, 3)
    lb_dist = mu - lower_bound
    ub_dist = upper_bound - mu
    constrained_var = jnp.minimum(
        jnp.minimum(jnp.square(lb_dist / 2), jnp.square(ub_dist / 2)), var)

    population = jax.random.truncated_normal(key,
                                             lower=-2.,
                                             upper=2.0,
                                             shape=population_shape)
    population = population * jnp.sqrt(constrained_var) + mu
    if fn_use_key:
      costs = fn(population, key2, *args)
    else:
      costs = fn(population, *args)
    costs: jnp.ndarray = jnp.where(jnp.isnan(costs), 1e8, costs)
    chex.assert_shape(costs, (population_size,))
    _, elite_idx = lax.top_k(-costs, num_elites)
    elites = population[elite_idx]
    best_costs = costs[elite_idx]

    new_mu = jnp.mean(elites, axis=0)
    new_var = jnp.var(elites, axis=0)
    new_best_cost = jnp.where(best_costs[0] < best_cost, best_costs[0], best_cost)
    new_best_solution = jnp.where(best_costs[0] < best_cost, elites[0], best_solution)
    mu = alpha * mu + (1 - alpha) * new_mu
    var = alpha * var + (1 - alpha) * new_var

    return (t + 1, new_best_cost, new_best_solution, mu, var, rng, args)

  initial_var = jnp.square(upper_bound - lower_bound) / 16.
  initial_mu = x0
  assert initial_var.shape == initial_mu.shape
  best_cost = jnp.inf
  best_solution = jnp.empty_like(initial_mu)
  state = (0, best_cost, best_solution, initial_mu, initial_var, key, args)
  # TODO: expose intermediate results
  t, new_best_cost, best, mu, _, _, _ = jax.lax.while_loop(cond_fn, loop, state)
  return mu if return_mean_elites else best, new_best_cost


def minimize_random(fn: Callable,
                    x0: jnp.ndarray,
                    key: jnp.ndarray,
                    args: Tuple = (),
                    *,
                    bounds: Tuple[jnp.ndarray, jnp.ndarray],
                    population_size: int,
                    fn_use_key: bool = False):
  """Minimize a function using the ramdom samples."""
  key, subkey = jax.random.split(key)
  population_shape = (population_size,) + x0.shape
  lb, ub = bounds
  population = jax.random.uniform(
      key,
      shape=population_shape,
      # TODO(ethan): Fix when https://github.com/google/jax/issues/4033 is in release
      minval=jnp.broadcast_to(lb, population_shape),
      maxval=jnp.broadcast_to(ub, population_shape),
  )
  if fn_use_key:
    costs = fn(population, subkey, *args)
  else:
    costs = fn(population, *args)
  return population[jnp.argmin(costs)], ()
