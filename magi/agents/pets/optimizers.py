from typing import Callable, Optional, Tuple

import chex
import jax
from jax import lax
import jax.numpy as jnp


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
  initial_var = jnp.square(upper_bound - lower_bound) / 16.
  action_shape = x0.shape

  mu = x0
  var = initial_var

  population = jnp.zeros((population_size,) + action_shape)

  best_cost = jnp.inf
  best_solution = jnp.empty_like(mu)
  state = (best_cost, best_solution, mu, var, population, key)

  def loop(i, state):
    del i
    best_cost, best, mu, var, population, rng = state
    rng, key, key2 = jax.random.split(rng, 3)
    lb_dist = mu - lower_bound
    ub_dist = upper_bound - mu
    mv = jnp.minimum(jnp.square(lb_dist / 2), jnp.square(ub_dist / 2))
    constrained_var = jnp.minimum(mv, var)

    population = jax.random.truncated_normal(key,
                                             lower=-2.,
                                             upper=2.0,
                                             shape=population.shape)
    population = population * jnp.sqrt(constrained_var) + mu
    if fn_use_key:
      costs = fn(population, key2, *args)
    else:
      costs = fn(population, *args)
    chex.assert_shape(costs, (population_size,))
    _, elite_idx = lax.top_k(-costs, num_elites)
    elite = jnp.array(population[elite_idx])
    best_costs = jnp.array(costs[elite_idx])
    new_best = elite[0]

    new_mu = jnp.mean(elite, axis=0)
    new_var = jnp.var(elite, axis=0, ddof=1)
    new_best_cost = jnp.where(best_costs[0] < best_cost, best_costs[0], best_cost)
    new_best = jnp.where(best_costs[0] < best_cost, new_best, best)
    mu = alpha * mu + (1 - alpha) * new_mu
    var = alpha * var + (1 - alpha) * new_var

    return (new_best_cost, new_best, mu, var, population, rng)

  # TODO: expose intermediate results
  _, best, mu, _, _, _ = lax.fori_loop(0, n_iterations, loop, state)
  return mu if return_mean_elites else best


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
  return population[jnp.argmin(costs)]
