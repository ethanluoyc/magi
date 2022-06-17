"""
SAC Version 1.
SAC-V1 includes an additional value network
"""

import functools
import time
from typing import Iterator, NamedTuple, Optional

from acme import core
from acme import types
from acme.jax import networks as networks_lib
from acme.utils import counting
from acme.utils import loggers
import jax
import jax.numpy as jnp
import numpy as np
import optax
import reverb


class TrainingState(NamedTuple):
  """Training state for SAC learner."""

  policy_params: networks_lib.Params
  critic_params: networks_lib.Params
  value_params: networks_lib.Params
  target_value_params: networks_lib.Params
  policy_optimizer_state: optax.OptState
  critic_optimizer_state: optax.OptState
  value_optimizer_state: optax.OptState
  key: networks_lib.PRNGKey
  alpha_optimizer_state: Optional[optax.OptState] = None
  alpha_params: Optional[jnp.ndarray] = None


class SACV1Learner(core.Learner):
  """SAC-V1 learner."""

  _state = TrainingState

  def __init__(
      self,
      policy_network: networks_lib.FeedForwardNetwork,
      critic_network: networks_lib.FeedForwardNetwork,
      value_network: networks_lib.FeedForwardNetwork,
      random_key: networks_lib.PRNGKey,
      dataset: Iterator[reverb.ReplaySample],
      policy_optimizer: optax.GradientTransformation,
      critic_optimizer: optax.GradientTransformation,
      value_optimizer: optax.GradientTransformation,
      alpha_optimizer: Optional[optax.GradientTransformation] = None,
      entropy_coefficient: Optional[float] = None,
      target_entropy: float = 0,
      discount: float = 0.99,
      tau: float = 5e-3,
      init_alpha: float = 1.0,
      logger: Optional[loggers.Logger] = None,
      counter: Optional[counting.Counter] = None,
  ):
    adaptive_entropy_coefficient = entropy_coefficient is None
    if adaptive_entropy_coefficient:
      alpha_optimizer = alpha_optimizer or optax.adam(3e-4)
    else:
      if target_entropy:
        raise ValueError('target_entropy should not be set when '
                         'entropy_coefficient is provided')

    def alpha_loss_fn(log_alpha: jnp.ndarray,
                      entropy: jnp.ndarray) -> jnp.ndarray:
      """Compute the temperature loss for EC-SAC.

      This is absent in the original V1, but we add here for flexibility.
      """
      return log_alpha * (entropy - target_entropy)

    def actor_loss_fn(
        policy_params: networks_lib.Params,
        critic_params: networks_lib.Params,
        alpha: jnp.ndarray,
        observation: jnp.ndarray,
        key: networks_lib.PRNGKey,
    ):
      """Compute the soft actor loss in SAC.

      This corresponds to Eqn (12) in https://arxiv.org/pdf/1801.01290.pdf
      """
      action_dist = policy_network.apply(policy_params, observation)
      actions = action_dist.sample(seed=key)
      log_probs = action_dist.log_prob(actions)

      q1, q2 = critic_network.apply(critic_params, observation, actions)
      q = jnp.minimum(q1, q2)
      entropy = -jnp.mean(log_probs)
      actor_loss = alpha * log_probs - q
      return jnp.mean(actor_loss), {'entropy': entropy}

    def critic_loss_fn(
        critic_params: networks_lib.Params,
        target_value_params: networks_lib.Params,
        transitions: types.Transition,
    ):
      """Compute the soft critic loss in SAC.

      This corresponds to Eqn (7) in https://arxiv.org/pdf/1801.01290.pdf
      """
      data = transitions
      # Compute V^{hat}(s_tp1)
      next_v = value_network.apply(target_value_params, data.next_observation)
      # Eqn 8: Compute Q^{hat}(s_t, a_t) = r_t + gamma * V^{hat}
      target_q = data.reward + data.discount * discount * next_v
      target_q = jax.lax.stop_gradient(target_q)
      # Predict online Q values for the twin critic
      q1, q2 = critic_network.apply(critic_params, data.observation,
                                    data.action)
      critic_loss = jnp.square(target_q - q1) + jnp.square(target_q - q2)
      return jnp.mean(critic_loss), {'q1': q1.mean(), 'q2': q2.mean()}

    def value_loss_fn(
        value_params: networks_lib.Params,
        policy_params: networks_lib.Params,
        critic_params: networks_lib.Params,
        transitions: types.Transition,
        alpha: jnp.ndarray,
        key: networks_lib.PRNGKey,
    ):
      """Compute the soft value loss in SAC.

      This corresponds to Eqn (5) in https://arxiv.org/pdf/1801.01290.pdf
      """
      # Compute Q(s_t, a_t) - logp(a_t|s_t)
      # where a_t ~ pi(s_t)
      dist = policy_network.apply(policy_params, transitions.observation)
      actions = dist.sample(seed=key)
      log_probs = dist.log_prob(actions)
      # Note: in SAC-V1, the online critic is used for estimating the Q(s, a)
      q1, q2 = critic_network.apply(critic_params, transitions.observation,
                                    actions)
      target_v = jnp.minimum(q1, q2) - alpha * log_probs
      # Compute V(s_t)
      v = value_network.apply(value_params, transitions.observation)
      # Compute the objective
      # J = 0.5 * (V(s_t) - E_{a_t} [Q(s_t, a_t) - alpha * logp(a_t | s_t)] )^2
      # we drop the constant 0.5 here
      value_loss = jnp.mean((v - target_v)**2)
      extras = {'value': jnp.mean(v)}
      return value_loss, extras

    critic_grad_fn = jax.value_and_grad(critic_loss_fn, has_aux=True)
    actor_grad_fn = jax.value_and_grad(actor_loss_fn, has_aux=True)
    value_grad_fn = jax.value_and_grad(value_loss_fn, has_aux=True)
    polyak_update = functools.partial(optax.incremental_update, step_size=tau)

    def sgd_step(state: TrainingState, transitions: types.Transition):
      key = state.key
      if adaptive_entropy_coefficient:
        alpha = jnp.exp(state.alpha_params)
      else:
        alpha = entropy_coefficient
      # Update critic
      (critic_loss,
       critic_metrics), critic_grads = critic_grad_fn(state.critic_params,
                                                      state.target_value_params,
                                                      transitions)
      critic_updates, critic_optimizer_state = critic_optimizer.update(
          critic_grads, state.critic_optimizer_state)
      critic_params = optax.apply_updates(state.critic_params, critic_updates)
      actor_key, key = jax.random.split(key)
      # Update policy
      (actor_loss, actor_metrics), policy_grads = actor_grad_fn(
          state.policy_params,
          critic_params,
          alpha,
          transitions.observation,
          actor_key,
      )
      policy_updates, policy_optimizer_state = policy_optimizer.update(
          policy_grads, state.policy_optimizer_state)
      policy_params = optax.apply_updates(state.policy_params, policy_updates)

      value_key, key = jax.random.split(key)
      # Update value network
      (value_loss, value_metrics), value_grads = value_grad_fn(
          state.value_params,
          policy_params,
          critic_params,
          transitions,
          alpha,
          value_key,
      )
      value_updates, value_optimizer_state = critic_optimizer.update(
          value_grads, state.value_optimizer_state)
      value_params = optax.apply_updates(state.value_params, value_updates)

      # Update target value networks
      target_value_params = polyak_update(value_params,
                                          state.target_value_params)

      new_state = TrainingState(
          policy_params=policy_params,
          critic_params=critic_params,
          value_params=value_params,
          target_value_params=target_value_params,
          policy_optimizer_state=policy_optimizer_state,
          critic_optimizer_state=critic_optimizer_state,
          value_optimizer_state=value_optimizer_state,
          key=key,
      )

      metrics = {
          'critic_loss': critic_loss,
          'actor_loss': actor_loss,
          'value_loss': value_loss,
          **critic_metrics,
          **actor_metrics,
          **value_metrics,
      }

      if adaptive_entropy_coefficient:
        # Update temperature
        alpha_loss, alpha_grads = jax.value_and_grad(alpha_loss_fn)(
            state.alpha_params, actor_metrics['entropy'])
        (
            alpha_updates,
            alpha_optimizer_state,
        ) = alpha_optimizer.update(  # pytype: disable=attribute-error
            alpha_grads, state.alpha_optimizer_state)
        alpha_params = optax.apply_updates(state.alpha_params, alpha_updates)
        new_state = new_state._replace(
            alpha_optimizer_state=alpha_optimizer_state,
            alpha_params=alpha_params,
        )
        metrics.update({
            'alpha_loss': alpha_loss,
            'alpha': jnp.exp(alpha_params)
        })

      return new_state, metrics

    def init_state(key):
      init_policy_key, init_critic_key, init_value_key, key = jax.random.split(
          random_key, 4)
      init_policy_params = policy_network.init(init_policy_key)
      init_critic_params = critic_network.init(init_critic_key)
      init_value_params = value_network.init(init_value_key)
      init_policy_optimizer_state = policy_optimizer.init(init_policy_params)
      init_critic_optimizer_state = critic_optimizer.init(init_critic_params)
      init_value_optimizer_state = value_optimizer.init(init_value_params)
      state = TrainingState(
          policy_params=init_policy_params,
          critic_params=init_critic_params,
          value_params=init_value_params,
          target_value_params=init_value_params,
          policy_optimizer_state=init_policy_optimizer_state,
          critic_optimizer_state=init_critic_optimizer_state,
          value_optimizer_state=init_value_optimizer_state,
          key=key,
      )
      if adaptive_entropy_coefficient:
        init_alpha_params = jnp.array(np.log(init_alpha), dtype=jnp.float32)
        init_alpha_optimizer_state = (
            alpha_optimizer.init(  # pytype: disable=attribute-error
                init_alpha_params))
        state = state._replace(
            alpha_optimizer_state=init_alpha_optimizer_state,
            alpha_params=init_alpha_params,
        )
      return state

    self._state = init_state(random_key)

    self._sgd_step = jax.jit(sgd_step)
    self._iterator = dataset
    self._logger = logger or loggers.make_default_logger(
        label='learner', save_data=False)
    self._counter = counter if counter is not None else counting.Counter()
    self._timestamp = None

  def step(self):
    # Get data from replay
    sample = next(self._iterator)
    transitions: types.Transition = sample.data
    # Perform a single learner step
    self._state, metrics = self._sgd_step(self._state, transitions)

    # Compute elapsed time
    timestamp = time.time()
    elapsed_time = timestamp - self._timestamp if self._timestamp else 0
    self._timestamp = timestamp

    # Increment counts and record the current time
    counts = self._counter.increment(steps=1, walltime=elapsed_time)
    # Attempts to write the logs.
    self._logger.write({**metrics, **counts})

  def get_variables(self, names):
    variables = {
        'policy': self._state.policy_params,
        'critic': self._state.critic_params,
        'value': self._state.value_params,
    }
    return [variables[name] for name in names]

  def save(self) -> TrainingState:
    return self._state

  def restore(self, state: TrainingState):
    self._state = state
