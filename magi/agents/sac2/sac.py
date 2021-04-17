import collections
import math
from functools import partial
from typing import Any, List, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax as optix
import reverb
import tensorflow_probability
import tree
from acme import datasets
from acme.adders import reverb as adders
from acme.jax import utils
from haiku import PRNGSequence
from jax import nn
from reverb import rate_limiters

tfp = tensorflow_probability.experimental.substrates.jax
tfd = tfp.distributions
tfb = tfp.bijectors


@jax.jit
def soft_update(
    target_params: hk.Params,
    online_params: hk.Params,
    tau: float,
) -> hk.Params:
  """
    Update target network using Polyak-Ruppert Averaging.
    """
  return jax.tree_multimap(lambda t, s: (1 - tau) * t + tau * s, target_params,
                           online_params)


@jax.jit
def gaussian_log_prob(
    log_std: jnp.ndarray,
    noise: jnp.ndarray,
) -> jnp.ndarray:
  """
    Calculate log probabilities of gaussian distributions.
    """
  return -0.5 * (jnp.square(noise) + 2 * log_std + jnp.log(2 * math.pi))


@jax.jit
def gaussian_and_tanh_log_prob(
    log_std: jnp.ndarray,
    noise: jnp.ndarray,
    action: jnp.ndarray,
) -> jnp.ndarray:
  """
    Calculate log probabilities of gaussian distributions and tanh transformation.
    """
  return gaussian_log_prob(log_std,
                           noise) - jnp.log(nn.relu(1.0 - jnp.square(action)) + 1e-6)


@partial(jax.jit, static_argnums=3)
def reparameterize_gaussian_and_tanh(
    mean: jnp.ndarray,
    log_std: jnp.ndarray,
    key: jnp.ndarray,
    return_log_pi: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """
    Sample from gaussian distributions and tanh transforamation.
    """
  # dist = _output_to_dist(mean, log_std)
  # action = dist.sample(seed=key)
  # if return_log_pi:
  #   return action, dist.log_prob(action)
  # return action
  std = jnp.exp(log_std)
  noise = jax.random.normal(key, std.shape)
  action = jnp.tanh(mean + noise * std)
  if return_log_pi:
    return action, gaussian_and_tanh_log_prob(log_std, noise, action).sum(axis=1)
  else:
    return action


def _output_to_dist(mean, logstd):
  import tensorflow_probability
  tfp = tensorflow_probability.experimental.substrates.jax
  tfd = tfp.distributions
  tfb = tfp.bijectors
  action_tp1_base_dist = tfd.Normal(loc=mean, scale=jnp.exp(logstd))
  action_dist = tfd.TransformedDistribution(action_tp1_base_dist, tfb.Tanh())
  return tfd.Independent(action_dist, 1)


def dummy_from_spec(spec):
  if isinstance(spec, collections.OrderedDict):
    return jax.tree_map(lambda x: x.generate_value(), spec)
  return spec.generate_value()


class SAC:
  name = "SAC"

  def __init__(
      self,
      environment_spec,
      policy_fn,
      critic_fn,
      seed,
      gamma=0.99,
      buffer_size=10**6,
      batch_size=256,
      start_steps=10000,
      tau=5e-3,
      lr_actor=3e-4,
      lr_critic=3e-4,
      lr_alpha=3e-4,
      init_alpha=1.0,
      adam_b1_alpha=0.9,
  ):
    self.rng = PRNGSequence(seed)
    self._action_spec = environment_spec.actions

    self.agent_step = 0
    self.learning_step = 0
    self.gamma = gamma
    replay_table = reverb.Table(name=adders.DEFAULT_PRIORITY_TABLE,
                                sampler=reverb.selectors.Uniform(),
                                remover=reverb.selectors.Fifo(),
                                max_size=buffer_size,
                                rate_limiter=rate_limiters.MinSize(1),
                                signature=adders.NStepTransitionAdder.signature(
                                    environment_spec=environment_spec))
    self._server = reverb.Server([replay_table], port=None)

    # The adder is used to insert observations into replay.
    address = f'localhost:{self._server.port}'
    adder = adders.NStepTransitionAdder(client=reverb.Client(address),
                                        n_step=1,
                                        discount=1.0)
    self._adder = adder

    # The dataset provides an interface to sample from replay.
    dataset = datasets.make_reverb_dataset(server_address=address,
                                           environment_spec=environment_spec,
                                           batch_size=batch_size,
                                           prefetch_size=1,
                                           transition_adder=True)
    self._iterator = dataset.as_numpy_iterator()
    self.batch_size = batch_size
    self.start_steps = start_steps
    self._update_target = jax.jit(partial(optix.incremental_update, step_size=tau))

    # Define fake input for critic.
    dummy_state = tree.map_structure(lambda x: jnp.expand_dims(x, 0),
                                     dummy_from_spec(environment_spec.observations))
    dummy_action = tree.map_structure(lambda x: jnp.expand_dims(x, 0),
                                      dummy_from_spec(environment_spec.actions))

    # Critic.
    self.critic = hk.without_apply_rng(hk.transform(critic_fn))
    self.params_critic = self.params_critic_target = self.critic.init(
        next(self.rng), dummy_state, dummy_action)
    opt_init, self.opt_critic = optix.adam(lr_critic)
    self.opt_state_critic = opt_init(self.params_critic)
    # Actor.
    self.actor = hk.without_apply_rng(hk.transform(policy_fn))
    self.params_actor = self.actor.init(next(self.rng), dummy_state)
    opt_init, self.opt_actor = optix.adam(lr_actor)
    self.opt_state_actor = opt_init(self.params_actor)
    # Entropy coefficient.
    self.target_entropy = -float(environment_spec.actions.shape[0])
    self.log_alpha = jnp.array(np.log(init_alpha), dtype=jnp.float32)
    opt_init, self.opt_alpha = optix.adam(lr_alpha, b1=adam_b1_alpha)
    self.opt_state_alpha = opt_init(self.log_alpha)

    @jax.jit
    def _update_actor(params_actor, opt_state, key, params_critic, log_alpha, state):
      (loss, aux), grad = jax.value_and_grad(self._loss_actor,
                                             has_aux=True)(params_actor, params_critic,
                                                           log_alpha, state, key)
      update, opt_state = self.opt_actor(grad, opt_state)
      params_actor = optix.apply_updates(params_actor, update)
      return params_actor, opt_state, loss, aux

    @jax.jit
    def _update_critic(
        params_critic,
        opt_state,
        key,
        params_critic_target,
        params_actor,
        log_alpha,
        state,
        action,
        reward,
        discount,
        next_state,
        weight,
    ):
      (loss,
       aux), grad = jax.value_and_grad(self._loss_critic,
                                       has_aux=True)(params_critic,
                                                     params_critic_target, params_actor,
                                                     log_alpha, state, action, reward,
                                                     discount, next_state, weight, key)
      update, opt_state = self.opt_critic(grad, opt_state)
      params_critic = optix.apply_updates(params_critic, update)
      return params_critic, opt_state, loss, aux

    @jax.jit
    def _update_alpha(log_alpha, opt_state, mean_log_pi):
      (loss, aux), grad = jax.value_and_grad(self._loss_alpha,
                                             has_aux=True)(log_alpha, mean_log_pi)
      update, opt_state = self.opt_alpha(grad, opt_state)
      log_alpha = optix.apply_updates(log_alpha, update)
      return log_alpha, opt_state, loss, aux

    self._update_actor = _update_actor
    self._update_critic = _update_critic
    self._update_alpha = _update_alpha

  @partial(jax.jit, static_argnums=0)
  def _explore(
      self,
      params_actor: hk.Params,
      state: np.ndarray,
      key: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    mean, log_std = self.actor.apply(params_actor, state)
    return reparameterize_gaussian_and_tanh(mean, log_std, key, False)

  @partial(jax.jit, static_argnums=0)
  def _calculate_value_list(
      self,
      params_critic: hk.Params,
      state: np.ndarray,
      action: np.ndarray,
  ) -> List[jnp.ndarray]:
    return self.critic.apply(params_critic, state, action)

  @partial(jax.jit, static_argnums=0)
  def _calculate_value(
      self,
      params_critic: hk.Params,
      state: np.ndarray,
      action: np.ndarray,
  ) -> jnp.ndarray:
    return jnp.asarray(self._calculate_value_list(params_critic, state,
                                                  action)).min(axis=0)

  @partial(jax.jit, static_argnums=0)
  def _calculate_loss_critic_and_abs_td(
      self,
      value_list: List[jnp.ndarray],
      target: jnp.ndarray,
      weight: np.ndarray,
  ) -> jnp.ndarray:
    abs_td = jnp.abs(target - value_list[0])
    loss_critic = (jnp.square(abs_td) * weight).mean()
    for value in value_list[1:]:
      loss_critic += (jnp.square(target - value) * weight).mean()
    return loss_critic, jax.lax.stop_gradient(abs_td)

  def select_action(self, state, is_eval=True):
    if is_eval:
      action = self._select_action(self.params_actor, utils.add_batch_dim(state))
      return utils.to_numpy_squeeze(action)
    else:
      if self.agent_step < self.start_steps:
        # action = self.action_space.sample()
        action = tfd.Uniform(low=self._action_spec.minimum,
                             high=self._action_spec.maximum).sample(seed=next(self.rng))
      else:
        action = self._explore(self.params_actor, utils.add_batch_dim(state),
                               next(self.rng))
      return utils.to_numpy_squeeze(action)

  @partial(jax.jit, static_argnums=0)
  def _select_action(
      self,
      params_actor: hk.Params,
      state: np.ndarray,
  ) -> jnp.ndarray:
    mean, _ = self.actor.apply(params_actor, state)
    return jnp.tanh(mean)

  def observe_first(self, timestep):
    self._adder.add_first(timestep)

  def observe(self, action, next_timestep):
    self.agent_step += 1
    self._adder.add(action, next_timestep)

  def update(self):
    if self.agent_step < self.start_steps:
      return
    # weight, batch = self.buffer.sample(self.batch_size)
    # state, action, reward, done, next_state = batch
    batch = next(self._iterator)
    state, action, reward, discount, next_state = batch.data
    # No PER for now
    weight = jnp.ones_like(reward)
    discount = discount * self.gamma

    # Update critic.
    self.params_critic, self.opt_state_critic, loss_critic, abs_td = self._update_critic(
        self.params_critic,
        self.opt_state_critic,
        key=next(self.rng),
        params_critic_target=self.params_critic_target,
        params_actor=self.params_actor,
        log_alpha=self.log_alpha,
        state=state,
        action=action,
        reward=reward,
        discount=discount,
        next_state=next_state,
        weight=weight,
    )
    # Update actor
    self.params_actor, self.opt_state_actor, loss_actor, mean_log_pi = self._update_actor(
        self.params_actor,
        self.opt_state_actor,
        key=next(self.rng),
        params_critic=self.params_critic,
        log_alpha=self.log_alpha,
        state=state)
    self.log_alpha, self.opt_state_alpha, loss_alpha, _ = self._update_alpha(
        self.log_alpha,
        self.opt_state_alpha,
        mean_log_pi=mean_log_pi,
    )
    self.learning_step += 1
    # Update target network.
    self.params_critic_target = self._update_target(self.params_critic,
                                                    self.params_critic_target)

  @partial(jax.jit, static_argnums=0)
  def _sample_action(
      self,
      params_actor: hk.Params,
      state: np.ndarray,
      key: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    mean, log_std = self.actor.apply(params_actor, state)
    return reparameterize_gaussian_and_tanh(mean, log_std, key, True)

  @partial(jax.jit, static_argnums=0)
  def _calculate_target(
      self,
      params_critic_target: hk.Params,
      log_alpha: jnp.ndarray,
      reward: np.ndarray,
      discount: np.ndarray,
      next_state: np.ndarray,
      next_action: jnp.ndarray,
      next_log_pi: jnp.ndarray,
  ) -> jnp.ndarray:
    next_q = self._calculate_value(params_critic_target, next_state, next_action)
    next_q -= jnp.exp(log_alpha) * next_log_pi
    assert len(next_q.shape) == 1
    assert len(reward.shape) == 1
    return jax.lax.stop_gradient(reward + discount * next_q)

  @partial(jax.jit, static_argnums=0)
  def _loss_critic(self, params_critic: hk.Params, params_critic_target: hk.Params,
                   params_actor: hk.Params, log_alpha: jnp.ndarray, state: np.ndarray,
                   action: np.ndarray, reward: np.ndarray, discount: np.ndarray,
                   next_state: np.ndarray, weight: np.ndarray or List[jnp.ndarray],
                   key) -> Tuple[jnp.ndarray, jnp.ndarray]:
    next_action, next_log_pi = self._sample_action(params_actor, next_state, key)
    target = self._calculate_target(params_critic_target, log_alpha, reward, discount,
                                    next_state, next_action, next_log_pi)
    q_list = self._calculate_value_list(params_critic, state, action)
    return self._calculate_loss_critic_and_abs_td(q_list, target, weight)

  @partial(jax.jit, static_argnums=0)
  def _loss_actor(self, params_actor: hk.Params, params_critic: hk.Params,
                  log_alpha: jnp.ndarray, state: np.ndarray,
                  key) -> Tuple[jnp.ndarray, jnp.ndarray]:
    action, log_pi = self._sample_action(params_actor, state, key)
    mean_q = self._calculate_value(params_critic, state, action).mean()
    mean_log_pi = log_pi.mean()
    return jax.lax.stop_gradient(
        jnp.exp(log_alpha)) * mean_log_pi - mean_q, jax.lax.stop_gradient(mean_log_pi)

  @partial(jax.jit, static_argnums=0)
  def _loss_alpha(
      self,
      log_alpha: jnp.ndarray,
      mean_log_pi: jnp.ndarray,
  ) -> jnp.ndarray:
    return -log_alpha * (self.target_entropy + mean_log_pi), None
