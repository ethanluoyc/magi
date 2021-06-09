from absl.testing import absltest
import chex
import jax
import jax.numpy as jnp

from magi.agents.pets import optimizers


class OptimizerTest(chex.TestCase):
  # pylint: disable=no-value-for-parameter

  @chex.variants(with_jit=True, without_jit=True)
  def test_minimize_cem(self):

    def objective(x):
      return jnp.sum(jnp.square(x), axis=-1)

    @self.variant
    def var_fn(x):
      return optimizers.minimize_cem(objective,
                                     x,
                                     jax.random.PRNGKey(0),
                                     bounds=(jnp.ones(2) * -10., jnp.ones(2) * 10.),
                                     n_iterations=100,
                                     population_size=400,
                                     elite_fraction=0.1,
                                     alpha=0.1,
                                     return_mean_elites=False)

    x0 = jnp.ones(2) * 2
    result, _ = var_fn(x0)
    chex.assert_tree_all_close(result, jnp.zeros(2), rtol=1e-4, atol=1e-4)


if __name__ == '__main__':
  absltest.main()
