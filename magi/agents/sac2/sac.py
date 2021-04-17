import math
from functools import partial
from typing import Any, List, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax as optix
from gym.spaces import Box
from haiku import PRNGSequence
from jax import nn
from magi.agents.sac2.buffer import ReplayBuffer
from magi.agents.sac2.network import (ContinuousQFunction, StateDependentGaussianPolicy)


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


@partial(jax.jit, static_argnums=(0, 1))
def optimize(
    fn_loss: Any,
    opt: Any,
    opt_state: Any,
    params_to_update: hk.Params,
    *args,
    **kwargs,
) -> Tuple[Any, hk.Params, jnp.ndarray, Any]:
  (loss, aux), grad = jax.value_and_grad(fn_loss, has_aux=True)(
      params_to_update,
      *args,
      **kwargs,
  )
  update, opt_state = opt(grad, opt_state)
  params_to_update = optix.apply_updates(params_to_update, update)
  return opt_state, params_to_update, loss, aux


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


def fake_state(state_space):
  state = state_space.sample()[None, ...]
  if len(state_space.shape) == 1:
    state = state.astype(np.float32)
  return state


def fake_action(action_space):
  if type(action_space) == Box:
    action = action_space.sample().astype(np.float32)[None, ...]
  else:
    NotImplementedError
  return action


class SAC:
  name = "SAC"

  def __init__(
      self,
      num_agent_steps,
      state_space,
      action_space,
      seed,
      gamma=0.99,
      nstep=1,
      num_critics=2,
      buffer_size=10**6,
      batch_size=256,
      start_steps=10000,
      update_interval=1,
      tau=5e-3,
      lr_actor=3e-4,
      lr_critic=3e-4,
      lr_alpha=3e-4,
      units_actor=(256, 256),
      units_critic=(256, 256),
      log_std_min=-20.0,
      log_std_max=2.0,
      init_alpha=1.0,
      adam_b1_alpha=0.9,
  ):
    np.random.seed(seed)
    self.rng = PRNGSequence(seed)

    self.agent_step = 0
    self.learning_step = 0
    self.num_agent_steps = num_agent_steps
    self.state_space = state_space
    self.action_space = action_space
    self.gamma = gamma
    self.discrete_action = False if type(action_space) == Box else True
    if not hasattr(self, "use_key_critic"):
      self.use_key_critic = True
    if not hasattr(self, "use_key_actor"):
      self.use_key_actor = True

    self.buffer = ReplayBuffer(
        buffer_size=buffer_size,
        state_space=state_space,
        action_space=action_space,
        gamma=gamma,
        nstep=nstep,
    )

    self.discount = gamma**nstep
    self.batch_size = batch_size
    self.start_steps = start_steps
    self.update_interval = update_interval
    # self.update_interval_target = update_interval_target

    self._update_target = jax.jit(partial(soft_update, tau=tau))

    self.num_critics = num_critics
    # Define fake input for critic.
    dummy_state = fake_state(state_space)
    dummy_action = fake_action(action_space)

    def fn_critic(s, a):
      return ContinuousQFunction(
          num_critics=num_critics,
          hidden_units=units_critic,
      )(s, a)

    def fn_actor(s):
      return StateDependentGaussianPolicy(
          action_space=action_space,
          hidden_units=units_actor,
          log_std_min=log_std_min,
          log_std_max=log_std_max,
      )(s)

    # Critic.
    self.critic = hk.without_apply_rng(hk.transform(fn_critic))
    self.params_critic = self.params_critic_target = self.critic.init(
        next(self.rng), dummy_state, dummy_action)
    opt_init, self.opt_critic = optix.adam(lr_critic)
    self.opt_state_critic = opt_init(self.params_critic)
    # Actor.
    self.actor = hk.without_apply_rng(hk.transform(fn_actor))
    self.params_actor = self.actor.init(next(self.rng), dummy_state)
    opt_init, self.opt_actor = optix.adam(lr_actor)
    self.opt_state_actor = opt_init(self.params_actor)
    # Entropy coefficient.
    self.target_entropy = -float(self.action_space.shape[0])
    self.log_alpha = jnp.array(np.log(init_alpha), dtype=jnp.float32)
    opt_init, self.opt_alpha = optix.adam(lr_alpha, b1=adam_b1_alpha)
    self.opt_state_alpha = opt_init(self.log_alpha)

    @jax.jit
    def _update_actor(params_actor, opt_state, key, params_critic, log_alpha, state):
      (loss, aux), grad = jax.value_and_grad(self._loss_actor,
                                             has_aux=True)(params_actor, params_critic,
                                                           log_alpha, state,
                                                           key)
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
        done,
        next_state,
        weight,
    ):
      (loss, aux), grad = jax.value_and_grad(self._loss_critic,
                                             has_aux=True)(params_critic,
                                                           params_critic_target,
                                                           params_actor, log_alpha,
                                                           state, action, reward, done,
                                                           next_state, weight, key)
      update, opt_state = self.opt_critic(grad, opt_state)
      params_critic = optix.apply_updates(params_critic, update)
      return params_critic, opt_state, loss, aux

    @jax.jit
    def _update_alpha(log_alpha, opt_state, mean_log_pi):
      (loss, aux), grad = jax.value_and_grad(self._loss_alpha, has_aux=True)(log_alpha, mean_log_pi)
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
      action = self._select_action(self.params_actor, state[None, ...])
      return np.array(action[0])
    else:
      if self.agent_step < self.start_steps:
        action = self.action_space.sample()
      else:
        action = self._explore(self.params_actor, state[None, ...], next(self.rng))
      return np.array(action[0])

  @partial(jax.jit, static_argnums=0)
  def _select_action(
      self,
      params_actor: hk.Params,
      state: np.ndarray,
  ) -> jnp.ndarray:
    mean, _ = self.actor.apply(params_actor, state)
    return jnp.tanh(mean)

  def observe(self, state, action, reward, next_state, done):
    self.agent_step += 1
    self.buffer.append(state, action, reward, done, next_state)

  def update(self):
    if self.agent_step < self.start_steps:
      return
    self.learning_step += 1
    weight, batch = self.buffer.sample(self.batch_size)
    state, action, reward, done, next_state = batch

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
        done=done,
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
        state=state
    )
    self.log_alpha, self.opt_state_alpha, loss_alpha, _ = self._update_alpha(
        self.log_alpha,
        self.opt_state_alpha,
        mean_log_pi=mean_log_pi,
    )
    # Update target network.
    self.params_critic_target = self._update_target(self.params_critic_target,
                                                    self.params_critic)

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
  def _calculate_log_pi(
      self,
      action: np.ndarray,
      log_pi: np.ndarray,
  ) -> jnp.ndarray:
    return log_pi

  @partial(jax.jit, static_argnums=0)
  def _calculate_target(
      self,
      params_critic_target: hk.Params,
      log_alpha: jnp.ndarray,
      reward: np.ndarray,
      done: np.ndarray,
      next_state: np.ndarray,
      next_action: jnp.ndarray,
      next_log_pi: jnp.ndarray,
  ) -> jnp.ndarray:
    next_q = self._calculate_value(params_critic_target, next_state, next_action)
    next_q -= jnp.exp(log_alpha) * self._calculate_log_pi(next_action, next_log_pi)
    assert len(next_q.shape) == 1
    assert len(reward.shape) == 1
    assert len(done.shape) == 1
    return jax.lax.stop_gradient(reward + (1.0 - done) * self.discount * next_q)

  @partial(jax.jit, static_argnums=0)
  def _loss_critic(self, params_critic: hk.Params, params_critic_target: hk.Params,
                   params_actor: hk.Params, log_alpha: jnp.ndarray, state: np.ndarray,
                   action: np.ndarray, reward: np.ndarray, done: np.ndarray,
                   next_state: np.ndarray, weight: np.ndarray or List[jnp.ndarray],
                   key) -> Tuple[jnp.ndarray, jnp.ndarray]:
    next_action, next_log_pi = self._sample_action(params_actor, next_state, key)
    target = self._calculate_target(params_critic_target, log_alpha, reward, done,
                                    next_state, next_action, next_log_pi)
    q_list = self._calculate_value_list(params_critic, state, action)
    return self._calculate_loss_critic_and_abs_td(q_list, target, weight)

  @partial(jax.jit, static_argnums=0)
  def _loss_actor(self, params_actor: hk.Params, params_critic: hk.Params,
                  log_alpha: jnp.ndarray, state: np.ndarray,
                  key) -> Tuple[jnp.ndarray, jnp.ndarray]:
    action, log_pi = self._sample_action(params_actor, state, key)
    mean_q = self._calculate_value(params_critic, state, action).mean()
    mean_log_pi = self._calculate_log_pi(action, log_pi).mean()
    return jax.lax.stop_gradient(
        jnp.exp(log_alpha)) * mean_log_pi - mean_q, jax.lax.stop_gradient(mean_log_pi)

  @partial(jax.jit, static_argnums=0)
  def _loss_alpha(
      self,
      log_alpha: jnp.ndarray,
      mean_log_pi: jnp.ndarray,
  ) -> jnp.ndarray:
    return -log_alpha * (self.target_entropy + mean_log_pi), None
