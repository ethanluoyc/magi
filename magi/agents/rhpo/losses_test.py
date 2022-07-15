from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

from magi.agents.rhpo import losses

tfd = tfp.distributions


class RHPONetworksTestCase(parameterized.TestCase):
  """Tests for RHPO policy loss"""

  @parameterized.parameters({'per_dim_constraining': True},
                            {'per_dim_constraining': True})
  def test_rhpo_policy_loss(self, per_dim_constraining):
    num_components = 2
    act_size = 5
    num_sampled_actions = 3
    batch_size = 7
    online_action_dist = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(
            logits=jnp.zeros((
                batch_size,
                num_components,
            ))),
        components_distribution=tfd.MultivariateNormalDiag(
            loc=jnp.zeros((batch_size, num_components, act_size)),
            scale_diag=jnp.ones((batch_size, num_components, act_size))),
    )
    target_action_dist = online_action_dist
    policy_loss_fn = losses.RHPO(
        epsilon=1e-1,
        epsilon_penalty=1e-3,
        epsilon_mean=1e-3,
        per_dim_constraining=per_dim_constraining,
        epsilon_stddev=1e-6,
        epsilon_categorical=1e-6,
        action_penalization=False,
        init_log_temperature=1.0,
        init_log_alpha_mean=1.0,
        init_log_alpha_stddev=10.0,
        init_log_alpha_categorical=1.0,
    )
    dual_params = policy_loss_fn.init_params(num_components, act_size)
    actions = jnp.zeros((num_sampled_actions, batch_size, act_size))
    q_values = jnp.ones((
        num_sampled_actions,
        batch_size,
    ))
    policy_loss, extras = policy_loss_fn(dual_params, online_action_dist,
                                         target_action_dist, actions, q_values)
    self.assertEqual(policy_loss.shape, (1,))


if __name__ == '__main__':
  absltest.main()
