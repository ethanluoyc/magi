"""Tests for MPO agent."""
from absl.testing import absltest
import acme
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.testing import fakes
import chex
import haiku as hk
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions


class RHPONetworksTestCase(absltest.TestCase):
  """Tests for RHPO agent."""

  def test_rhpo_policy_network(self):
    obs_size = 3
    act_size = 2
    environment = fakes.ContinuousEnvironment(
        observation_dim=obs_size, action_dim=act_size)

    policy_torso_sizes = (400, 200)
    critic_torso_sizes = (400, 400)
    policy_controller_head_size = 100
    policy_component_head_size = 100
    critic_head_size = 100

    environment_spec = acme.make_environment_spec(environment)
    num_tasks = 5
    num_components = 4

    def _policy(obs, task_ids):
      torso = networks_lib.LayerNormMLP(policy_torso_sizes, activate_final=True)
      controller_heads = []
      component_heads = []
      for task_id_ in range(num_tasks):
        controller_heads.append(
            hk.nets.MLP([policy_controller_head_size, num_components],
                        name=f'policy_controller_head_{task_id_}'))
      for _ in range(num_components):
        component_heads.append(
            hk.Sequential([
                hk.nets.MLP([policy_component_head_size], activate_final=True),
                networks_lib.MultivariateNormalDiagHead(num_dimensions=act_size)
            ]))

      embedding = torso(obs)
      components_distribution = jax.tree_util.tree_map(
          lambda *xs: jnp.stack(xs, 1),
          *[head(embedding) for head in component_heads])
      # (T, B, L)
      categoricals = jnp.stack([head(embedding) for head in controller_heads])
      # (B, T)
      task_ids = jax.nn.one_hot(task_ids, num_classes=num_tasks)
      # (T, B, 1)
      task_ids = jnp.expand_dims(jnp.swapaxes(task_ids, 0, 1), axis=-1)
      # (B, L)
      task_mixture_logits = jnp.sum(task_ids * categoricals, axis=0)
      return tfd.MixtureSameFamily(
          mixture_distribution=tfd.Categorical(logits=task_mixture_logits),
          components_distribution=components_distribution,
      )

    def _critic(obs, action):
      torso = networks_lib.LayerNormMLP(critic_torso_sizes, activate_final=True)
      critic_heads = []
      for task_id in range(num_tasks):
        critic_heads.append(
            hk.nets.MLP([critic_head_size, 1], name=f'critic_head_{task_id}'))
      inputs = utils.batch_concat([obs, action])
      embedding = torso(inputs)
      critic_values = jnp.concatenate(
          [head(embedding) for head in critic_heads], axis=-1)
      return critic_values

    policy_network = hk.without_apply_rng(hk.transform(_policy))
    critic_network = hk.without_apply_rng(hk.transform(_critic))
    key = jax.random.PRNGKey(0)
    dummy_obs = utils.add_batch_dim(
        utils.zeros_like(environment_spec.observations))
    dummy_act = utils.add_batch_dim(utils.zeros_like(environment_spec.actions))
    dummy_task_id = jnp.ones(1, dtype=jnp.int32)

    policy_params = policy_network.init(key, dummy_obs, dummy_task_id)
    critic_params = critic_network.init(key, dummy_obs, dummy_act)
    dist = policy_network.apply(policy_params, dummy_obs, dummy_task_id)
    self.assertEqual(dist.batch_shape, (1,))
    self.assertEqual(dist.event_shape, (act_size,))
    self.assertEqual(dist.log_prob(dummy_act).shape, (1,))
    # chex.assert_shape(categoricals, (1, num_components))
    qs = critic_network.apply(critic_params, dummy_obs, dummy_act)
    chex.assert_shape(qs, (1, num_tasks))


if __name__ == '__main__':
  absltest.main()
