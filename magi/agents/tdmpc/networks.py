"""Networks used in TD-MPC."""
import dataclasses
from typing import NamedTuple

from acme import specs
from acme.jax import utils
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


class TDMPCParams(NamedTuple):
  encoder_params: hk.Params
  reward_params: hk.Params
  dynamics_params: hk.Params
  critic_params: hk.Params
  twin_critic_params: hk.Params
  policy_params: hk.Params


def _truncated_normal_sample(key, mean, std, noise_clip=0.3):
  noise = jax.random.normal(key=key, shape=mean.shape) * std
  noise = jnp.clip(noise, -noise_clip, noise_clip)
  sample = mean + noise
  clipped_sample = jnp.clip(sample, -1, 1)
  return (sample - jax.lax.stop_gradient(sample) +
          jax.lax.stop_gradient(clipped_sample))


@dataclasses.dataclass
class TDMPCNetworks:
  encoder_network: hk.Transformed
  reward_network: hk.Transformed
  dynamics_network: hk.Transformed
  critic_network: hk.Transformed
  policy_network: hk.Transformed

  def h(self, params, obs):
    return self.encoder_network.apply(params.encoder_params, obs)

  def next(self, params, z, act):
    next_z = self.dynamics_network.apply(params.dynamics_params, z, act)
    reward = self.reward_network.apply(params.reward_params, z, act)
    return next_z, reward

  def pi(self, params, z, std, key):
    a = jnp.tanh(self.policy_network.apply(params.policy_params, z))
    return _truncated_normal_sample(key, a, std, noise_clip=0.3)

  def q(self, params, z, action):
    q1 = self.critic_network.apply(params.critic_params, z, action)
    q2 = self.critic_network.apply(params.twin_critic_params, z, action)
    return q1, q2


def init_params(
    networks: TDMPCNetworks,
    spec: specs.EnvironmentSpec,
    key: jax.random.PRNGKeyArray,
):
  keys = jax.random.split(key, 6)
  dummy_obs = utils.add_batch_dim(utils.zeros_like(spec.observations))
  dummy_act = utils.add_batch_dim(utils.zeros_like(spec.actions))

  encoder_params = networks.encoder_network.init(keys[0], dummy_obs)
  embedding = networks.encoder_network.apply(encoder_params, dummy_obs)

  critic_params = networks.critic_network.init(keys[1], embedding, dummy_act)
  twin_critic_params = networks.critic_network.init(keys[2], embedding,
                                                    dummy_act)
  dynamics_params = networks.dynamics_network.init(keys[3], embedding,
                                                   dummy_act)
  reward_params = networks.reward_network.init(keys[4], embedding, dummy_act)
  policy_params = networks.policy_network.init(keys[5], embedding)

  return TDMPCParams(
      encoder_params=encoder_params,
      critic_params=critic_params,
      dynamics_params=dynamics_params,
      twin_critic_params=twin_critic_params,
      reward_params=reward_params,
      policy_params=policy_params,
  )


def make_networks(
    spec: specs.EnvironmentSpec,
    latent_size,
    encoder_hidden_size,
    mlp_hidden_size,
    zero_init: bool = True,
):
  action_size = np.prod(spec.actions.shape, dtype=int)

  w_init = hk.initializers.Orthogonal(1.0)

  def _encoder_fn(obs):
    return hk.nets.MLP(
        [encoder_hidden_size, latent_size],
        activation=jax.nn.elu,
        w_init=w_init,
        activate_final=False,
    )(
        obs)

  def _dynamics_fn(obs_embedding, act):
    inputs = jnp.concatenate([obs_embedding, act], axis=-1)
    return hk.Sequential([
        hk.nets.MLP(
            [mlp_hidden_size, mlp_hidden_size, obs_embedding.shape[-1]],
            activation=jax.nn.elu,
            w_init=w_init,
            activate_final=False,
        ),
    ])(
        inputs)

  def _actor_fn(obs_embedding):
    return hk.Sequential([
        hk.nets.MLP(
            [mlp_hidden_size, mlp_hidden_size, action_size],
            activation=jax.nn.elu,
            w_init=w_init,
            activate_final=False,
        ),
    ])(
        obs_embedding)

  def _critic_fn(obs_embedding, act):
    inputs = jnp.concatenate([obs_embedding, act], axis=-1)
    return hk.Sequential([
        hk.Linear(mlp_hidden_size, w_init=w_init),
        hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
        jax.lax.tanh,
        hk.Linear(mlp_hidden_size, w_init=w_init),
        jax.nn.elu,
        hk.Linear(
            1,
            w_init=w_init if not zero_init else jnp.zeros,
        ),
    ])(
        inputs)

  def _reward_fn(obs_embedding, act):
    inputs = jnp.concatenate([obs_embedding, act], axis=-1)
    return hk.Sequential([
        hk.nets.MLP(
            [mlp_hidden_size, mlp_hidden_size],
            activation=jax.nn.elu,
            w_init=w_init,
            activate_final=True,
        ),
        hk.Linear(1, w_init=w_init if not zero_init else jnp.zeros),
    ])(
        inputs)

  encoder_network = hk.without_apply_rng(hk.transform(_encoder_fn))
  reward_network = hk.without_apply_rng(hk.transform(_reward_fn))
  dynamics_network = hk.without_apply_rng(hk.transform(_dynamics_fn))
  critic_network = hk.without_apply_rng(hk.transform(_critic_fn))
  policy_network = hk.without_apply_rng(hk.transform(_actor_fn))
  return TDMPCNetworks(
      encoder_network=encoder_network,
      reward_network=reward_network,
      critic_network=critic_network,
      dynamics_network=dynamics_network,
      policy_network=policy_network,
  )
