"""Tests for MPO agent."""
from absl.testing import absltest
import acme
from acme import types
from acme.jax import networks as networks_lib
from acme.jax import types as jax_types
from acme.jax import utils
from acme.testing import fakes
import chex
import jax
import jax.numpy as jnp
import rlax
import tensorflow_probability.substrates.jax as tfp

from magi.agents.rhpo import networks

tfd = tfp.distributions


class RHPONetworksTestCase(absltest.TestCase):
  """Tests for RHPO agent."""

  def test_rhpo_policy_network(self):
    obs_size = 3
    act_size = 2
    num_tasks = 2
    num_components = 3

    environment = fakes.ContinuousEnvironment(
        observation_dim=obs_size, action_dim=act_size)

    environment_spec = acme.make_environment_spec(environment)
    agent_networks = networks.make_networks(
        environment_spec, num_tasks=num_tasks, num_components=num_components)
    policy_network = agent_networks['policy']
    critic_network = agent_networks['critic']
    key = jax.random.PRNGKey(0)

    policy_params = policy_network.init(key)
    critic_params = critic_network.init(key)
    target_policy_params = policy_params
    target_critic_params = critic_params

    dummy_obs = utils.add_batch_dim(
        utils.zeros_like(environment_spec.observations))
    dummy_act = utils.add_batch_dim(utils.zeros_like(environment_spec.actions))
    dummy_task_id = jnp.ones(1, dtype=jnp.int32)
    dist = policy_network.apply(policy_params, dummy_obs, dummy_task_id)
    self.assertEqual(dist.batch_shape, (1,))
    self.assertEqual(dist.event_shape, (act_size,))
    self.assertEqual(dist.log_prob(dummy_act).shape, (1,))
    # chex.assert_shape(categoricals, (1, num_components))
    qs = critic_network.apply(critic_params, dummy_obs, dummy_act)
    chex.assert_shape(qs, (1, num_tasks))

    discount = 0.99
    num_samples = 20
    transition = types.Transition(
        observation=dummy_obs,
        next_observation=dummy_obs,
        action=dummy_act,
        reward=jnp.ones((1, num_tasks)),
        discount=jnp.ones((1,)),
    )

    def critic_loss_fn(
        critic_params: networks_lib.Params,
        target_policy_params: networks_lib.Params,
        target_critic_params: networks_lib.Params,
        transitions: types.Transition,
        key: jax_types.PRNGKey,
    ):
      o_tm1 = transitions.observation
      o_t = transitions.next_observation
      batch_size = transitions.discount.shape[0]
      tiled_o_t = utils.tile_nested(o_t, num_samples)  # [N, B, ...]
      loss = 0.
      for task_i in range(num_tasks):
        task_ids = jnp.full((batch_size,), task_i, dtype=jnp.int32)

        # Get action distributions from policy networks.
        # online_action_distribution = policy_network.apply(
        #     policy_params, o_t, task_ids)
        target_action_distribution = policy_network.apply(
            target_policy_params, o_t, task_ids)

        # Get sampled actions to evaluate policy; of size [N, B, ...].
        policy_key, key = jax.random.split(key)
        sampled_actions = target_action_distribution.sample(
            num_samples, seed=policy_key)

        # Compute the target critic's Q-value of the sampled actions in state o_t.
        sampled_q_t = jax.vmap(critic_network.apply, (None, 0, 0))(
            target_critic_params,
            tiled_o_t,
            sampled_actions,
        )

        q_t = jnp.mean(sampled_q_t, axis=0)[:, task_i]  # [B]

        # Compute online critic value of a_tm1 in state o_tm1.
        q_tm1 = critic_network.apply(critic_params, o_tm1,
                                     transitions.action)[:, task_i]  # [B]

        # Critic loss.
        batch_td_learning = jax.vmap(rlax.td_learning)
        td_error = batch_td_learning(q_tm1, transitions.reward[:, task_i],
                                     discount * transitions.discount, q_t)
        loss += jnp.mean(jnp.square(td_error))
      return loss

    critic_loss = critic_loss_fn(critic_params, target_policy_params,
                                 target_critic_params, transition, key)
    chex.assert_shape(critic_loss, ())


if __name__ == '__main__':
  absltest.main()
