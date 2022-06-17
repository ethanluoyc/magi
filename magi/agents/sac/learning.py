"""SAC Learner."""
import time
from typing import Iterator, NamedTuple, Optional

from acme import core
from acme import types
from acme.jax import networks as networks_lib
from acme.jax import types as jax_types
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
  critic_target_params: networks_lib.Params
  policy_optimizer_state: optax.OptState
  critic_optimizer_state: optax.OptState
  key: jax_types.PRNGKey
  alpha_optimizer_state: Optional[optax.OptState] = None
  alpha_params: Optional[jnp.ndarray] = None


class SACLearner(core.Learner):
  """SAC learner."""

  def __init__(
      self,
      policy_network: networks_lib.FeedForwardNetwork,
      critic_network: networks_lib.FeedForwardNetwork,
      random_key: jax_types.PRNGKey,
      dataset: Iterator[reverb.ReplaySample],
      policy_optimizer: optax.GradientTransformation,
      critic_optimizer: optax.GradientTransformation,
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

    def actor_loss(
        policy_params: networks_lib.Params,
        critic_params: networks_lib.Params,
        alpha: jnp.ndarray,
        transitions: types.Transition,
        key: jax_types.PRNGKey,
    ):
      action_dist = policy_network.apply(policy_params, transitions.observation)
      actions = action_dist.sample(seed=key)
      log_probs = action_dist.log_prob(actions)

      q1, q2 = critic_network.apply(critic_params, transitions.observation,
                                    actions)
      q = jnp.minimum(q1, q2)
      entropy = -log_probs.mean()
      actor_loss = alpha * log_probs - q
      return jnp.mean(actor_loss), {'entropy': entropy}

    def critic_loss(
        critic_params: networks_lib.Params,
        critic_target_params: networks_lib.Params,
        policy_params: networks_lib.Params,
        alpha: jnp.ndarray,
        transitions: types.Transition,
        key: jax_types.PRNGKey,
    ):
      next_action_dist = policy_network.apply(policy_params,
                                              transitions.next_observation)
      next_actions = next_action_dist.sample(seed=key)
      next_log_probs = next_action_dist.log_prob(next_actions)

      next_q1, next_q2 = critic_network.apply(critic_target_params,
                                              transitions.next_observation,
                                              next_actions)
      next_q = jnp.minimum(next_q1, next_q2) - alpha * next_log_probs
      target = jax.lax.stop_gradient(transitions.reward +
                                     transitions.discount * discount * next_q)
      q1, q2 = critic_network.apply(critic_params, transitions.observation,
                                    transitions.action)
      critic_loss = jnp.square(target - q1) + jnp.square(target - q2)
      return jnp.mean(critic_loss), {'q1': q1.mean(), 'q2': q2.mean()}

    def alpha_loss(log_alpha: jnp.ndarray, entropy: jnp.ndarray):
      return log_alpha * (entropy - target_entropy), ()

    def sgd_step(state: TrainingState, transitions: types.Transition):
      if adaptive_entropy_coefficient:
        alpha = jnp.exp(state.alpha_params)
      else:
        alpha = entropy_coefficient
      critic_key, policy_key, key = jax.random.split(state.key, 3)
      (critic_loss_value, critic_metrics), critic_grad = jax.value_and_grad(
          critic_loss, has_aux=True)(
              state.critic_params,
              state.critic_target_params,
              state.policy_params,
              alpha,
              transitions,
              critic_key,
          )
      critic_updates, critic_optimizer_state = critic_optimizer.update(
          critic_grad, state.critic_optimizer_state)
      critic_params = optax.apply_updates(state.critic_params, critic_updates)
      (actor_loss_value, actor_metrics), policy_grad = jax.value_and_grad(
          actor_loss, has_aux=True)(
              state.policy_params,
              critic_params,
              alpha,
              transitions,
              policy_key,
          )
      policy_updates, policy_optimizer_state = policy_optimizer.update(
          policy_grad, state.policy_optimizer_state)
      policy_params = optax.apply_updates(state.policy_params, policy_updates)
      critic_target_params = optax.incremental_update(
          critic_params, state.critic_target_params, tau)

      metrics = {
          'critic_loss': critic_loss_value,
          'actor_loss': actor_loss_value,
          **critic_metrics,
          **actor_metrics,
      }

      new_state = TrainingState(
          policy_params=policy_params,
          critic_params=critic_params,
          critic_target_params=critic_target_params,
          policy_optimizer_state=policy_optimizer_state,
          critic_optimizer_state=critic_optimizer_state,
          key=key,
      )
      if adaptive_entropy_coefficient:
        (alpha_loss_value, _), alpha_grad = jax.value_and_grad(
            alpha_loss, has_aux=True)(state.alpha_params,
                                      actor_metrics['entropy'])
        # pytype: disable=attribute-error
        alpha_updates, alpha_optimizer_state = alpha_optimizer.update(
            alpha_grad, state.alpha_optimizer_state)
        # pytype: enable=attribute-error
        alpha_params = optax.apply_updates(state.alpha_params, alpha_updates)
        metrics.update({
            'alpha_loss': alpha_loss_value,
            'alpha': jnp.exp(alpha_params),
        })
        new_state = new_state._replace(
            alpha_params=alpha_params,
            alpha_optimizer_state=alpha_optimizer_state,
        )

      return new_state, metrics

    self._sgd_step = jax.jit(sgd_step)

    def init_state(key):
      init_policy_key, init_critic_key, key = jax.random.split(random_key, 3)
      # Actor.
      init_policy_params = policy_network.init(init_policy_key)
      init_critic_params = critic_network.init(init_critic_key)
      init_policy_optimizer_state = policy_optimizer.init(init_policy_params)
      init_critic_optimizer_state = critic_optimizer.init(init_critic_params)
      state = TrainingState(
          policy_params=init_policy_params,
          critic_params=init_critic_params,
          critic_target_params=init_critic_params,
          policy_optimizer_state=init_policy_optimizer_state,
          critic_optimizer_state=init_critic_optimizer_state,
          key=key,
      )
      if adaptive_entropy_coefficient:
        init_alpha_params = jnp.array(np.log(init_alpha), dtype=jnp.float32)
        # pytype: disable=attribute-error
        init_alpha_optimizer_state = alpha_optimizer.init(init_alpha_params)
        # pytype: enable=attribute-error
        state = state._replace(
            alpha_params=init_alpha_params,
            alpha_optimizer_state=init_alpha_optimizer_state,
        )
      return state

    self._state = init_state(random_key)
    self._iterator = dataset

    self._logger = logger or loggers.make_default_logger(
        label='learner', save_data=False)
    self._counter = counter or counting.Counter()
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
    }
    return [variables[name] for name in names]

  def save(self) -> TrainingState:
    return self._state

  def restore(self, state: TrainingState):
    self._state = state
