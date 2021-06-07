import jax
import jax.numpy as jnp
from jax import lax
import chex


def cem_controller(cost_fn,
                   action_spec,
                   params,
                   xinit,
                   initial_actions,
                   key,
                   n_iterations,
                   pop_size,
                   elite_frac,
                   time_horizon,
                   alpha,
                   return_mean_elites=False):
  n_elite = int(pop_size * elite_frac)
  action_shape = (time_horizon,) + action_spec.shape
  lower_bound = jnp.broadcast_to(action_spec.minimum, action_shape)
  upper_bound = jnp.broadcast_to(action_spec.maximum, action_shape)
  initial_var = jnp.square(upper_bound - lower_bound) / 16.

  mu = initial_actions
  var = initial_var

  population = jnp.zeros((pop_size,) + action_shape)

  best_cost = jnp.inf
  best_solution = jnp.empty_like(mu)
  state = (best_cost, best_solution, mu, var, population, key)

  def loop(i, state):
    del i
    best_cost, best_actions, mu, var, population, rng = state
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
    # import pdb; pdb.set_trace()
    costs = cost_fn(params, key2, xinit, population)
    # chex.assert_axis_dimension(costs, 0, population)
    chex.assert_shape(costs, (pop_size,))
    _, elite_idx = lax.top_k(-costs, n_elite)
    # _, elite_idx = jnp.argsort(-costs)[]
    # elite_idx = jnp.argsort(costs, axis=0)[:n_elite]
    elite = jnp.array(population[elite_idx])
    best_costs = jnp.array(costs[elite_idx])
    new_best_actions = elite[0]

    new_mu = jnp.mean(elite, axis=0)
    new_var = jnp.var(elite, axis=0, ddof=1)
    new_best_cost = jnp.where(best_costs[0] < best_cost, best_costs[0], best_cost)
    new_best_actions = jnp.where(best_costs[0] < best_cost, new_best_actions,
                                 best_actions)
    mu = alpha * mu + (1 - alpha) * new_mu
    var = alpha * var + (1 - alpha) * new_var

    return (new_best_cost, new_best_actions, mu, var, population, rng)

  # TODO: expose intermediate results
  _, best_action, mu, _, _, _ = lax.fori_loop(0, n_iterations, loop, state)
  return mu if return_mean_elites else best_action


def random_controller(cost_fn, action_spec, params, xinit, key, num_samples,
                      time_steps):
  action_shape = (
      num_samples,
      time_steps,
  ) + action_spec.shape
  key, subkey = jax.random.split(key)
  actions = jax.random.uniform(
      key,
      shape=action_shape,
      # TODO(ethan): Fix when https://github.com/google/jax/issues/4033 is in release
      minval=jnp.broadcast_to(action_spec.minimum, action_shape),
      maxval=jnp.broadcast_to(action_spec.maximum, action_shape),
  )
  costs = cost_fn(params, subkey, xinit, actions)
  return actions[jnp.argmin(costs)]


def _clip(a, minval, maxval):
  return jnp.maximum(jnp.minimum(a, maxval), minval)
