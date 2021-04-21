"""Tests for soft actor critic."""
from typing import Iterator, Tuple, List
import haiku as hk
import jax
from absl.testing import absltest
import jax.numpy as jnp
import dm_env
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import reverb
import tensorflow_probability
from acme import core, datasets, specs
from acme.utils import loggers, counting
from acme.adders import reverb as adders
from acme.jax.networks import distributional
import tree

tfp = tensorflow_probability.experimental.substrates.jax
tfd = tfp.distributions
tfb = tfp.bijectors


class Encoder(hk.Module):

  def __call__(self, observation):
    torso = hk.Sequential([
        hk.Conv2D(16, kernel_shape=3, stride=2),
        jax.nn.relu,
        hk.Conv2D(16, kernel_shape=3, stride=2),
        jax.nn.relu,
    ])
    feature = torso(observation)
    # print(feature.shape)
    return hk.Flatten()(feature)


class Decoder(hk.Module):

  def __call__(self, feature):
    return hk.Sequential([
        lambda x: jnp.reshape(x, (-1, 8, 8, 16)),
        hk.Conv2DTranspose(16, kernel_shape=3, stride=2),
        jax.nn.relu,
        hk.Conv2DTranspose(3, kernel_shape=3, stride=2),
        jax.nn.relu,
    ])(feature)


class Policy(hk.Module):

  def __init__(self, action_dim: int, name=None):
    super().__init__(name=name)
    self._action_dim = action_dim

  def __call__(self, feature):
    o = hk.Linear(self._action_dim)(feature)
    return distributional.NormalTanhDistribution(self._action_dim)(o)


class Critic(hk.Module):

  def __call__(self, feature, action):
    q1 = hk.Linear(1)
    q2 = hk.Linear(1)
    input_ = jnp.concatenate([feature, action], axis=-1)
    return q1(input_).squeeze(-1), q2(input_).squeeze(-1)


class NetworkTest(absltest.TestCase):

  @hk.testing.transform_and_run
  def test_network(self):
    # Create a fake environment to test with.
    encoder = Encoder()
    batch_size = 1
    action_dim = 2
    dummy_obs = jnp.zeros((batch_size, 32, 32, 3))
    dummy_action = jnp.zeros((batch_size, action_dim))
    features = encoder(dummy_obs)
    decoder = Decoder()
    self.assertEqual(decoder(features).shape, dummy_obs.shape)
    policy = Policy(action_dim)
    critic = Critic()
    action = policy(features).sample(seed=jax.random.PRNGKey(0))
    q1, q2 = critic(features, dummy_action)
    self.assertEqual(action.shape, (batch_size, action_dim))
    self.assertEqual(q1.shape, (batch_size,))
    self.assertEqual(q2.shape, (batch_size,))

  def test_losses(self):
    batch_size = 1
    action_dim = 2
    dummy_obs = jnp.zeros((batch_size, 32, 32, 3))
    dummy_action = jnp.zeros((batch_size, action_dim))
    dummy_rewards = jnp.zeros((batch_size,))
    dummy_discount = jnp.zeros((batch_size,))
    o_t, a_t, r_t, d_t, o_tp1 = (dummy_obs, dummy_action, dummy_rewards, dummy_discount,
                                 dummy_obs)

    key = jax.random.PRNGKey(0)

    # Set up encoder
    encoder = hk.without_apply_rng(hk.transform(lambda o: Encoder()(o)))
    encoder_params = encoder.init(key, dummy_obs)
    encoder_opt = optax.adam(1e-3)
    encoder_opt_state = encoder_opt.init(encoder_params)

    dummy_features = encoder.apply(encoder_params, dummy_obs)

    # Set up decoder
    decoder = hk.without_apply_rng(hk.transform(lambda f: Decoder()(f)))
    decoder_params = decoder.init(key, dummy_features)
    decoder_opt = optax.adam(1e-3)
    decoder_opt_state = decoder_opt.init(decoder_params)

    # Set up policy
    policy = hk.without_apply_rng(hk.transform(lambda f: Policy(action_dim)(f)))
    policy_params = policy.init(key, dummy_features)
    policy_opt = optax.adam(1e-3)
    policy_opt_state = policy_opt.init(policy_params)

    # Set up critic
    critic = hk.without_apply_rng(hk.transform(lambda o, a: Critic()(o, a)))
    critic_params = critic.init(key, dummy_features, dummy_action)
    critic_opt = optax.adam(1e-3)
    critic_opt_state = critic_opt.init(critic_params)

    # Setup log alpha
    log_alpha = jnp.zeros(())
    log_alpha_opt = optax.adam(1e-3)
    log_alpha_opt_state = log_alpha_opt.init(log_alpha)

    target_entropy = -action_dim

    # Setup target params
    critic_target_params = tree.map_structure(lambda x: x.copy(), critic_params)
    critic_encoder_target_params = tree.map_structure(lambda x: x.copy(),
                                                      encoder_params)
    decoder_latent_lambda = 0.1  # TODO customize this
    tau = 0.05

    def _calculate_target(
        params_critic_target: hk.Params,
        log_alpha: jnp.ndarray,
        reward: np.ndarray,
        discount: np.ndarray,
        next_state: np.ndarray,
        next_action: jnp.ndarray,
        next_log_pi: jnp.ndarray,
    ) -> jnp.ndarray:
      next_qs = jnp.stack(critic.apply(params_critic_target, next_state,
                                       next_action)).min(axis=0)
      next_q = next_qs - jnp.exp(log_alpha) * next_log_pi
      assert len(next_q.shape) == 1
      assert len(reward.shape) == 1
      return jax.lax.stop_gradient(reward + discount * next_q)

    def _loss_critic(params_critic: hk.Params, params_critic_target: hk.Params,
                     params_actor: hk.Params, params_encoder: hk.Params,
                     log_alpha: jnp.ndarray, state: np.ndarray, action: np.ndarray,
                     reward: np.ndarray, discount: np.ndarray, next_state: np.ndarray,
                     weight: np.ndarray or List[jnp.ndarray],
                     key) -> Tuple[jnp.ndarray, jnp.ndarray]:
      encoded = encoder.apply(params_encoder, state)
      next_encoded = encoder.apply(params_encoder, next_state)
      next_action_dist = policy.apply(params_actor, next_encoded)
      next_action = next_action_dist.sample(seed=key)
      next_log_pi = next_action_dist.log_prob(next_action)
      target = _calculate_target(params_critic_target, log_alpha, reward, discount,
                                 next_encoded, next_action, next_log_pi)
      q_list = critic.apply(params_critic, encoded, action)
      abs_td = jnp.abs(target - q_list[0])
      loss = (jnp.square(abs_td) * weight).mean()
      for value in q_list[1:]:
        loss += (jnp.square(target - value) * weight).mean()
      return loss, jax.lax.stop_gradient(abs_td)

    def _loss_actor(params_actor: hk.Params, params_encoder, params_critic: hk.Params,
                    log_alpha: jnp.ndarray, state: np.ndarray,
                    key) -> Tuple[jnp.ndarray, jnp.ndarray]:
      h = encoder.apply(params_encoder, state)
      action_dist = policy.apply(params_actor, h)
      action = action_dist.sample(seed=key)
      log_pi = action_dist.log_prob(action)

      mean_q = jnp.stack(critic.apply(params_critic, h, action)).min(axis=0).mean()
      mean_log_pi = log_pi.mean()
      return jax.lax.stop_gradient(
          jnp.exp(log_alpha)) * mean_log_pi - mean_q, jax.lax.stop_gradient(mean_log_pi)

    def _loss_alpha(log_alpha: jnp.ndarray, mean_log_pi: jnp.ndarray,
                    target_entropy) -> jnp.ndarray:
      # TODO(yl): Investigate if it should be log_alpha or exp(log_alpha)
      return -jnp.exp(log_alpha) * (target_entropy + mean_log_pi), None

    def _update_actor(params_actor, opt_state, key, params_encoder, params_critic,
                      log_alpha, state):
      (loss, aux), grad = jax.value_and_grad(_loss_actor,
                                             has_aux=True)(params_actor, params_encoder,
                                                           params_critic, log_alpha,
                                                           state, key)
      update, opt_state = policy_opt.update(grad, opt_state)
      params_actor = optax.apply_updates(params_actor, update)
      return params_actor, opt_state, loss, aux

    def _update_critic(
        params_critic,
        opt_state,
        key,
        params_critic_target,
        params_actor,
        params_encoder,
        log_alpha,
        state,
        action,
        reward,
        discount,
        next_state,
        weight,
    ):
      (loss, aux), grad = jax.value_and_grad(_loss_critic, has_aux=True)(
          params_critic, params_critic_target, params_actor, params_encoder, log_alpha,
          state, action, reward, discount, next_state, weight, key)
      update, opt_state = critic_opt.update(grad, opt_state)
      params_critic = optax.apply_updates(params_critic, update)
      return params_critic, opt_state, loss, aux

    def _update_alpha(log_alpha, opt_state, mean_log_pi, target_entropy):
      (loss, aux), grad = jax.value_and_grad(_loss_alpha,
                                             has_aux=True)(log_alpha, mean_log_pi,
                                                           target_entropy)
      update, opt_state = log_alpha_opt.update(grad, opt_state)
      log_alpha = optax.apply_updates(log_alpha, update)
      return log_alpha, opt_state, loss, aux

    def update_autoencoder(encoder_params, decoder_params, encoder_opt_state,
                           decoder_opt_state, obs):

      def _loss_fn(params, obs):
        encoder_params, decoder_params = params
        h = encoder.apply(encoder_params, obs)
        rec_obs = decoder.apply(decoder_params, h)
        rec_loss = jnp.mean(jnp.square(obs - rec_obs))
        # add L2 penalty on latent representation
        # see https://arxiv.org/pdf/1903.12436.pdf
        latent_loss = 0.5 * jnp.mean(jnp.sum(jnp.square(h), axis=-1))
        loss = rec_loss + decoder_latent_lambda * latent_loss
        return loss, None

      (loss, aux), (grad_encoder,
                    grad_decoder) = jax.value_and_grad(_loss_fn, has_aux=True)(
                        (encoder_params, decoder_params), obs)

      encoder_update, encoder_opt_state = encoder_opt.update(grad_encoder,
                                                             encoder_opt_state)
      encoder_params = optax.apply_updates(encoder_params, encoder_update)

      decoder_update, decoder_opt_state = decoder_opt.update(grad_decoder,
                                                             decoder_opt_state)
      decoder_params = optax.apply_updates(decoder_params, decoder_update)
      return encoder_params, decoder_params, encoder_opt_state, decoder_opt_state, loss, aux

    def update_target(new_params, target_params):
      return optax.incremental_update(new_params, target_params, step_size=tau)

    critic_params, critic_opt_state, _, _ = _update_critic(
        critic_params, critic_opt_state, key, critic_target_params, policy_params,
        encoder_params, log_alpha, o_t, a_t, r_t, d_t, o_tp1, jnp.zeros_like(r_t))

    policy_params, policy_opt_state, _, mean_log_pi = _update_actor(
        policy_params, policy_opt_state, key, encoder_params, critic_params, log_alpha,
        o_t)

    log_alpha, log_alpha_opt_state, _, _, = _update_alpha(log_alpha,
                                                          log_alpha_opt_state,
                                                          mean_log_pi, target_entropy)

    encoder_params, decoder_params, encoder_opt_state, decoder_opt_state, loss, aux = update_autoencoder(
        encoder_params, decoder_params, encoder_opt_state, decoder_opt_state, o_t)
    critic_target_params, critic_encoder_target_params = update_target(
        (critic_params, encoder_params),
        (critic_target_params, critic_encoder_target_params),
    )


if __name__ == '__main__':
  absltest.main()
