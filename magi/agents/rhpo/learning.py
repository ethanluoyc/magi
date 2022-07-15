"""Learner for RHPO agent."""
import functools
import time
from typing import Iterator, NamedTuple, Optional

import acme
from acme import types as acme_types
from acme.jax import losses
from acme.jax import networks as networks_lib
from acme.jax import types as jax_types
from acme.jax import utils
from acme.utils import counting
from acme.utils import loggers
import jax
import jax.numpy as jnp
import optax
import reverb
import rlax

from magi.agents.rhpo import losses


class TrainingState(NamedTuple):
  """Training state for MPO learner."""

  policy_params: networks_lib.Params
  critic_params: networks_lib.Params
  dual_params: losses.MPOParams
  policy_opt_state: optax.OptState
  critic_opt_state: optax.OptState
  dual_opt_state: optax.OptState
  target_policy_params: networks_lib.Params
  target_critic_params: networks_lib.Params
  key: jax_types.PRNGKey
  steps: int


class RHPOLearner(acme.Learner):
  """RHPO learner."""

  def __init__(
      self,
      policy_network: networks_lib.FeedForwardNetwork,
      critic_network: networks_lib.FeedForwardNetwork,
      dataset: Iterator[reverb.ReplaySample],
      random_key: jnp.ndarray,
      policy_optimizer: optax.GradientTransformation,
      critic_optimizer: optax.GradientTransformation,
      dual_optimizer: optax.GradientTransformation,
      num_tasks: int,
      num_components: int,
      discount: float,
      num_samples: int,
      action_dim: int,
      target_policy_update_period: int,
      target_critic_update_period: int,
      policy_loss_fn: Optional[losses.RHPO] = None,
      counter: Optional[counting.Counter] = None,
      logger: Optional[loggers.Logger] = None,
  ):
    policy_loss_fn: losses.RHPO = policy_loss_fn or losses.RHPO(
        epsilon=1e-1,
        epsilon_penalty=1e-3,
        epsilon_mean=1e-3,
        per_dim_constraining=True,
        epsilon_stddev=1e-6,
        epsilon_categorical=1e-6,
        init_log_temperature=1.0,
        init_log_alpha_mean=1.0,
        init_log_alpha_stddev=10.0,
        init_log_alpha_categorical=1.0,
    )

    def compute_loss(
        policy_params: networks_lib.Params,
        mpo_params: losses.MPOParams,
        critic_params: networks_lib.Params,
        target_policy_params: networks_lib.Params,
        target_critic_params: networks_lib.Params,
        transitions: acme_types.Transition,
        key: jax_types.PRNGKey,
    ):
      o_tm1 = transitions.observation
      o_t = transitions.next_observation

      o_tm1 = transitions.observation
      o_t = transitions.next_observation
      batch_size = transitions.discount.shape[0]
      tiled_o_t = utils.tile_nested(o_t, num_samples)  # [N, B, ...]
      critic_loss = 0.

      rewards = transitions.reward
      # Expand reward to in case of single-task
      if rewards.ndim == 1:
        rewards = jnp.expand_dims(rewards, axis=-1)

      for task_i in range(num_tasks):
        task_ids = jnp.full((batch_size,), task_i, dtype=jnp.int32)

        # Get action distributions from policy networks.
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
        )[:, :, task_i]

        q_t = jnp.mean(sampled_q_t, axis=0)  # [B]

        # Compute online critic value of a_tm1 in state o_tm1.
        q_tm1 = critic_network.apply(critic_params, o_tm1,
                                     transitions.action)[:, task_i]  # [B]

        # Critic loss.
        batch_td_learning = jax.vmap(rlax.td_learning)
        td_error = batch_td_learning(q_tm1, rewards[:, task_i],
                                     discount * transitions.discount, q_t)
        critic_loss += jnp.mean(jnp.square(td_error))

      # Actor learning.
      task_key, key = jax.random.split(key)
      sampled_task_ids = jax.random.categorical(
          task_key, logits=jnp.zeros(num_tasks,), shape=(batch_size,))
      online_action_distribution = policy_network.apply(policy_params, o_t,
                                                        sampled_task_ids)
      target_action_distribution = policy_network.apply(target_policy_params,
                                                        o_t, sampled_task_ids)

      # Get sampled actions to evaluate policy; of size [N, B, ...].
      policy_key, key = jax.random.split(key)
      sampled_actions = target_action_distribution.sample(
          num_samples, seed=policy_key)
      # [N, B, T]
      sampled_q_t = jax.vmap(critic_network.apply, (None, 0, 0))(
          target_critic_params,
          tiled_o_t,
          sampled_actions,
      )
      # [N, B]
      sampled_q_t = jnp.squeeze(
          jnp.take_along_axis(
              sampled_q_t,
              jnp.expand_dims(task_ids, axis=(0, -1)),
              axis=-1,
          ),
          axis=-1)

      policy_loss, policy_stats = policy_loss_fn(
          mpo_params,
          online_action_distribution=online_action_distribution,
          target_action_distribution=target_action_distribution,
          actions=sampled_actions,
          q_values=sampled_q_t,
      )
      policy_loss = jnp.mean(policy_loss)
      return (policy_loss, critic_loss), policy_stats

    def sgd_step(state: TrainingState, transitions: acme_types.Transition):
      key, random_key = jax.random.split(state.key)
      compute_loss_with_inputs = functools.partial(
          compute_loss,
          target_policy_params=state.target_policy_params,
          target_critic_params=state.target_critic_params,
          transitions=transitions,
          key=key,
      )
      # Clip the mpo params first
      # This is consistent with the tf implementation see
      # https://github.com/deepmind/acme/blob/master/acme/tf/losses/mpo.py#L193
      dual_params = losses.clip_mpo_params(
          state.dual_params,
          per_dim_constraining=policy_loss_fn.per_dim_constraining,
      )
      # Compute the gradients in a single pass
      ((policy_loss_value, critic_loss_value), vjpfun,
       policy_metrics) = jax.vjp(
           compute_loss_with_inputs,
           state.policy_params,
           dual_params,
           state.critic_params,
           has_aux=True)
      policy_gradients, _, _ = vjpfun((1.0, 0.0))
      _, dual_gradients, _ = vjpfun((1.0, 0.0))
      _, _, critic_gradients = vjpfun((0.0, 1.0))

      # Get optimizer updates and state.
      (
          policy_updates,
          policy_opt_state,
      ) = policy_optimizer.update(policy_gradients, state.policy_opt_state)
      (
          critic_updates,
          critic_opt_state,
      ) = critic_optimizer.update(critic_gradients, state.critic_opt_state)
      dual_updates, dual_opt_state = dual_optimizer.update(
          dual_gradients, state.dual_opt_state)

      # Apply optimizer updates to parameters.
      policy_params = optax.apply_updates(state.policy_params, policy_updates)
      critic_params = optax.apply_updates(state.critic_params, critic_updates)
      dual_params = optax.apply_updates(dual_params, dual_updates)

      steps = state.steps + 1

      # Periodically update target networks.
      target_policy_params = rlax.periodic_update(
          policy_params,
          state.target_policy_params,
          steps,
          target_policy_update_period,
      )
      target_critic_params = rlax.periodic_update(
          critic_params,
          state.target_critic_params,
          steps,
          target_critic_update_period,
      )

      new_state = TrainingState(
          policy_params=policy_params,
          critic_params=critic_params,
          dual_params=dual_params,
          target_policy_params=target_policy_params,
          target_critic_params=target_critic_params,
          policy_opt_state=policy_opt_state,
          critic_opt_state=critic_opt_state,
          dual_opt_state=dual_opt_state,
          key=random_key,
          steps=steps,
      )

      metrics = {
          'policy_loss': policy_loss_value,
          'critic_loss': critic_loss_value,
          **policy_metrics._asdict(),  # pylint: disable=all
      }

      return new_state, metrics

    self._sgd_step = jax.jit(sgd_step)

    self._iterator = dataset

    def make_initial_state(key: jax_types.PRNGKey):
      key1, key2, key = jax.random.split(key, 3)
      policy_params = policy_network.init(key1)
      policy_opt_state = policy_optimizer.init(policy_params)
      critic_params = critic_network.init(key2)
      critic_opt_state = critic_optimizer.init(critic_params)
      dual_params = policy_loss_fn.init_params(num_components, action_dim)
      return TrainingState(
          policy_params=policy_params,
          critic_params=critic_params,
          dual_params=dual_params,
          policy_opt_state=policy_opt_state,
          critic_opt_state=critic_opt_state,
          dual_opt_state=dual_optimizer.init(dual_params),
          target_policy_params=policy_params,
          target_critic_params=critic_params,
          key=key,
          steps=0,
      )

    self._state = make_initial_state(random_key)
    self._timestamp = None

    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.make_default_logger(
        'learner',
        save_data=False,
        asynchronous=True,
        serialize_fn=utils.fetch_devicearray,
    )

  def step(self):
    # Sample replay buffer
    batch = next(self._iterator).data

    self._state, metrics = self._sgd_step(self._state, batch)

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

  def restore(self, state: TrainingState):
    self._state = state

  def save(self):
    return self._state
