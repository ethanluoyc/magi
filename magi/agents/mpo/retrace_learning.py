"""Learner for MPO agent."""
import functools
import time
from typing import Any, Iterator, NamedTuple, Optional

import acme
from acme import types as acme_types
from acme.jax import losses
from acme.jax import networks as networks_lib
from acme.jax import types as jax_types
from acme.jax import utils
from acme.jax.losses import mpo as mpo_losses
from acme.utils import counting
from acme.utils import loggers
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import reverb
import rlax
import tree


class StepOutput(NamedTuple):
  online_action_distribution: Any
  target_action_distribution: Any
  online_samples: acme_types.NestedArray
  target_samples: acme_types.NestedArray
  target_log_probs_behavior_actions: acme_types.NestedArray
  online_log_probs: acme_types.NestedArray
  online_q: acme_types.NestedArray
  target_q: acme_types.NestedArray


class OnlineTargetPiQ:
  """Single step computation for MPO"""

  def __init__(self, policy_network, critic_network):
    self._policy_network = policy_network
    self._critic_network = critic_network

  def __call__(self, policy_params, critic_params, target_policy_params,
               target_critic_params, observations, actions, key, num_samples):
    online_pi_dist = self._policy_network.apply(policy_params, observations)
    target_pi_dist = self._policy_network.apply(target_policy_params,
                                                observations)
    key1, key2 = jax.random.split(key)
    online_samples = online_pi_dist.sample((num_samples,), seed=key1)
    target_samples = target_pi_dist.sample((num_samples,), seed=key2)
    target_log_probs_behavior_actions = target_pi_dist.log_prob(actions)
    online_log_probs = online_pi_dist.log_prob(
        jax.lax.stop_gradient(online_samples))
    online_q_out = self._critic_network.apply(critic_params, observations,
                                              actions)
    target_q_out = self._critic_network.apply(target_critic_params,
                                              observations, actions)
    return StepOutput(
        online_pi_dist,
        target_pi_dist,
        online_samples,
        target_samples,
        target_log_probs_behavior_actions,
        online_log_probs,
        online_q_out,
        target_q_out,
    )


def _nest_stack(list_of_nests, axis=0):
  """Convert a list of nests to a nest of stacked lists."""
  return jax.tree_util.tree_map(lambda *ts: jnp.stack(ts, axis=axis),
                                *list_of_nests)


def static_unroll(fn, unroll_length, inputs, key):
  step_outputs = []
  keys = jax.random.split(key, unroll_length)
  for time_dim in range(unroll_length):
    inputs_t = tree.map_structure(
        lambda t, i_=time_dim: t[i_] if i_ < t.shape[0] else None, inputs)
    step_output = fn(inputs_t, keys[time_dim])
    step_outputs.append(step_output)

  step_outputs = _nest_stack(step_outputs)
  return step_outputs


class TrainingState(NamedTuple):
  """Training state for MPO learner."""

  policy_params: networks_lib.Params
  critic_params: networks_lib.Params
  mpo_params: losses.MPOParams
  policy_opt_state: optax.OptState
  critic_opt_state: optax.OptState
  dual_opt_state: optax.OptState
  target_policy_params: networks_lib.Params
  target_critic_params: networks_lib.Params
  key: jax_types.PRNGKey
  steps: int


class MPORetraceLearner(acme.Learner):
  """MPO learner."""

  def __init__(
      self,
      policy_network: networks_lib.FeedForwardNetwork,
      critic_network: networks_lib.FeedForwardNetwork,
      dataset: Iterator[reverb.ReplaySample],
      random_key: jnp.ndarray,
      policy_optimizer: optax.GradientTransformation,
      critic_optimizer: optax.GradientTransformation,
      dual_optimizer: optax.GradientTransformation,
      discount: float,
      num_samples: int,
      action_dim: int,
      target_policy_update_period: int,
      target_critic_update_period: int,
      policy_loss_fn: Optional[losses.MPO] = None,
      counter: Optional[counting.Counter] = None,
      logger: Optional[loggers.Logger] = None,
  ):
    policy_loss_fn: losses.MPO = policy_loss_fn or losses.MPO(
        epsilon=1e-1,
        epsilon_penalty=1e-3,
        epsilon_mean=1e-3,
        epsilon_stddev=1e-6,
        init_log_temperature=1.0,
        init_log_alpha_mean=1.0,
        init_log_alpha_stddev=10.0,
    )

    def compute_loss(
        policy_params: networks_lib.Params,
        mpo_params: losses.MPOParams,
        critic_params: networks_lib.Params,
        target_policy_params: networks_lib.Params,
        target_critic_params: networks_lib.Params,
        samples,
        key: jax_types.PRNGKey,
    ):
      data = utils.batch_to_sequence(samples)
      observations, actions, rewards, discounts, extra = (data.observation,
                                                          data.action,
                                                          data.reward,
                                                          data.discount,
                                                          data.extras)

      # (T+1, B, *)
      behavior_log_probs = extra['log_prob']

      def fn(inputs, key):
        obs, act = inputs
        step_fn = OnlineTargetPiQ(policy_network, critic_network)
        return step_fn(policy_params, critic_params, target_policy_params,
                       target_critic_params, obs, act, key, num_samples)

      step_outputs: StepOutput = static_unroll(fn, rewards.shape[0],
                                               (observations, actions), key)
      # (S, T + 1, B, A)
      target_pi_samples = utils.batch_to_sequence(step_outputs.target_samples)
      # (S, T + 1, B, O)
      tiled_observations = utils.tile_nested(observations, num_samples)
      # Finally compute target Q values on the new action samples.
      # Shape: [S, T+1, B]
      target_q_target_pi_samples = hk.BatchApply(
          functools.partial(critic_network.apply, target_critic_params),
          3,
      )(tiled_observations, target_pi_samples)
      # Compute the value estimate by averaging over the action dimension.
      # Shape: [T+1, B].
      target_v_target_pi = jnp.mean(target_q_target_pi_samples, axis=0)
      # Split the target V's into the target for learning
      # `value_function_target` and the bootstrap value. Shape: [T, B].
      # value_function_target = target_v_target_pi[:-1]
      # # Shape: [B].
      # bootstrap_value = target_v_target_pi[-1]
      # Get target log probs and behavior log probs from rollout.
      # Shape: [T+1, B].
      target_log_probs_behavior_actions = (
          step_outputs.target_log_probs_behavior_actions)
      rhos = jnp.exp(target_log_probs_behavior_actions - behavior_log_probs)
      # # Filter the importance weights to mask out episode restarts. Ignore the
      # # last action and consider the step type of the next step for masking.
      # # Shape: [T, B].
      # episode_start_mask = tf2_utils.batch_to_sequence(
      #     sample.data.start_of_episode)[1:]
      # rhos = svg0_utils.mask_out_restarting(rhos[:-1], episode_start_mask)
      log_rhos = jnp.log(rhos + 1e-8)
      target_q_values = step_outputs.target_q
      online_q_values = step_outputs.online_q
      batched_retrace = jax.vmap(
          functools.partial(rlax.retrace_continuous, lambda_=1.0),
          in_axes=1,
          out_axes=1)
      retrace_error = batched_retrace(
          online_q_values[:-1],
          target_q_values[1:-1],
          target_v_target_pi[1:],
          rewards[:-1],
          discount * discounts[:-1],
          log_rhos[1:-1],
      )
      critic_loss = 0.5 * jnp.mean(jnp.square(retrace_error))
      # Actor learning
      online_action_distribution = jax.tree_util.tree_map(
          lambda params: jnp.reshape(params, (-1, *params.shape[2:])),
          step_outputs.online_action_distribution[:-1])
      target_action_distribution = jax.tree_util.tree_map(
          lambda params: jnp.reshape(params, (-1, *params.shape[2:])),
          step_outputs.target_action_distribution[:-1])
      sampled_actions = jnp.reshape(
          target_pi_samples[:, :-1],
          (num_samples, -1, target_pi_samples.shape[-1]))
      q_values = jnp.reshape(target_q_target_pi_samples[:, :-1],
                             (num_samples, -1))
      online_action_distribution.log_prob(sampled_actions)
      policy_loss, policy_stats = policy_loss_fn(
          mpo_params,
          online_action_distribution,
          target_action_distribution,
          sampled_actions,
          q_values,
      )
      policy_loss = jnp.mean(policy_loss)
      return (policy_loss, critic_loss), policy_stats

    def sgd_step(state: TrainingState, transitions: acme_types.Transition):
      key, random_key = jax.random.split(state.key)
      compute_loss_with_inputs = functools.partial(
          compute_loss,
          target_policy_params=state.target_policy_params,
          target_critic_params=state.target_critic_params,
          samples=transitions,
          key=key,
      )
      # Clip the mpo params first
      # This is consistent with the tf implementation see
      # https://github.com/deepmind/acme/blob/master/acme/tf/losses/mpo.py#L193
      mpo_params = mpo_losses.clip_mpo_params(
          state.mpo_params,
          per_dim_constraining=policy_loss_fn.per_dim_constraining,
      )
      # Compute the gradients in a single pass
      ((policy_loss_value, critic_loss_value), vjpfun,
       policy_metrics) = jax.vjp(
           compute_loss_with_inputs,
           state.policy_params,
           mpo_params,
           state.critic_params,
           has_aux=True,
       )
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
      mpo_params = optax.apply_updates(mpo_params, dual_updates)

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
          mpo_params=mpo_params,
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
      mpo_params = policy_loss_fn.init_params(action_dim)
      return TrainingState(
          policy_params=policy_params,
          critic_params=critic_params,
          mpo_params=mpo_params,
          policy_opt_state=policy_opt_state,
          critic_opt_state=critic_opt_state,
          dual_opt_state=dual_optimizer.init(mpo_params),
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
        'policy': self._state.target_policy_params,
        'critic': self._state.target_critic_params,
    }
    return [variables[name] for name in names]

  def restore(self, state: TrainingState):
    self._state = state

  def save(self):
    return self._state
