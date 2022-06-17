# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
# Modifications Copyright 2021 Yicheng Luo. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""IMPALA learner."""
import time
from typing import (Callable, Dict, Iterator, List, NamedTuple, Optional,
                    Sequence, Tuple)

import acme
from acme import specs
from acme.jax import networks
from acme.jax import utils
from acme.utils import counting
from acme.utils import loggers
import haiku as hk
import jax
from jax.experimental import optimizers
import jax.numpy as jnp
import numpy as np
import optax
import reverb
import rlax
import tree


def impala_loss(
    unroll_fn: hk.Transformed,
    *,
    discount: float,
    max_abs_reward: float = np.inf,
    baseline_cost: float = 1.0,
    entropy_cost: float = 0.0,
) -> Callable[[hk.Params, reverb.ReplaySample], jnp.DeviceArray]:
  """Builds the standard entropy-regularised IMPALA loss function.

  Args:
    unroll_fn: A `hk.Transformed` object containing a callable which maps
      (params, observations_sequence, initial_state) -> ((logits, value), state)
    discount: The standard geometric discount rate to apply.
    max_abs_reward: Optional symmetric reward clipping to apply.
    baseline_cost: Weighting of the critic loss relative to the policy loss.
    entropy_cost: Weighting of the entropy regulariser relative to policy loss.

  Returns:
    A loss function with signature (params, data) -> loss_scalar.
  """

  def loss_fn(params: hk.Params,
              sample: reverb.ReplaySample) -> jnp.DeviceArray:
    """Batched, entropy-regularised actor-critic loss with V-trace."""

    # Extract the data.
    data = sample.data
    observations, actions, rewards, discounts, extra = (
        data.observation,
        data.action,
        data.reward,
        data.discount,
        data.extras,
    )
    initial_state = tree.map_structure(lambda s: s[0], extra['core_state'])
    behaviour_logits = extra['logits']

    # Apply reward clipping.
    rewards = jnp.clip(rewards, -max_abs_reward, max_abs_reward)

    # Unroll current policy over observations.
    (logits, values), _ = unroll_fn.apply(params, observations, initial_state,
                                          data.start_of_episode)

    # Compute importance sampling weights: current policy / behavior policy.
    rhos = rlax.categorical_importance_sampling_ratios(logits[:-1],
                                                       behaviour_logits[:-1],
                                                       actions[:-1])
    # Mask out invalid transitions from LAST to FIRST
    mask = 1.0 - data.start_of_episode[1:].astype(jnp.float32)

    # Critic loss.
    vtrace_returns = rlax.vtrace_td_error_and_advantage(
        v_tm1=values[:-1],
        v_t=values[1:],
        r_t=rewards[:-1],
        discount_t=discounts[:-1] * discount,
        rho_tm1=rhos,
    )
    critic_loss = jnp.square(vtrace_returns.errors)

    # Policy gradient loss.
    policy_gradient_loss = rlax.policy_gradient_loss(
        logits_t=logits[:-1],
        a_t=actions[:-1],
        adv_t=vtrace_returns.pg_advantage,
        w_t=mask,
    )

    # Entropy regulariser.
    entropy_loss = jnp.mean(rlax.entropy_loss(logits[:-1], mask))
    critic_loss = jnp.mean(critic_loss * mask)
    policy_gradient_loss = jnp.mean(policy_gradient_loss)

    # Combine weighted sum of actor & critic losses, averaged over the sequence.
    mean_loss = (policy_gradient_loss + baseline_cost * critic_loss +
                 entropy_cost * entropy_loss)  # []

    return mean_loss, {
        'entropy_loss': entropy_loss,
        'critic_loss': critic_loss,
        'pg_loss': policy_gradient_loss,
    }

  return utils.mapreduce(loss_fn, in_axes=(None, 0))


_PMAP_AXIS_NAME = 'data'


class TrainingState(NamedTuple):
  """Training state consists of network parameters and optimiser state."""

  params: networks.Params
  opt_state: optax.OptState


class IMPALALearner(acme.Learner):
  """Learner for an importanced-weighted advantage actor-critic."""

  def __init__(
      self,
      obs_spec: specs.Array,
      unroll_fn: networks.PolicyValueRNN,
      initial_state_fn: Callable[[], hk.LSTMState],
      iterator: Iterator[reverb.ReplaySample],
      optimizer: optax.GradientTransformation,
      random_key: networks.PRNGKey,
      discount: float = 0.99,
      entropy_cost: float = 0.0,
      baseline_cost: float = 1.0,
      max_abs_reward: float = np.inf,
      counter: counting.Counter = None,
      logger: loggers.Logger = None,
      devices: Optional[Sequence[jax.xla.Device]] = None,
      prefetch_size: int = 2,
      num_prefetch_threads: Optional[int] = None,
  ):

    local_devices = jax.local_devices()
    self._devices = devices or local_devices
    self._local_devices = [d for d in self._devices if d in local_devices]

    # Transform into pure functions.
    unroll_fn = hk.without_apply_rng(hk.transform(unroll_fn, apply_rng=True))
    initial_state_fn = hk.without_apply_rng(
        hk.transform(initial_state_fn, apply_rng=True))

    loss_fn = impala_loss(
        unroll_fn,
        discount=discount,
        max_abs_reward=max_abs_reward,
        baseline_cost=baseline_cost,
        entropy_cost=entropy_cost,
    )

    @jax.jit
    def sgd_step(
        state: TrainingState, sample: reverb.ReplaySample
    ) -> Tuple[TrainingState, Dict[str, jnp.ndarray]]:
      """Computes an SGD step, returning new state and metrics for logging."""

      # Compute gradients.
      grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
      (loss_value, stats), gradients = grad_fn(state.params, sample)

      # Average gradients over pmap replicas before optimizer update.
      gradients = jax.lax.pmean(gradients, _PMAP_AXIS_NAME)
      grad_norm_unclipped = optax.global_norm(gradients)

      # Apply updates.
      updates, new_opt_state = optimizer.update(gradients, state.opt_state)
      new_params = optax.apply_updates(state.params, updates)
      weight_norm = optimizers.l2_norm(new_params)

      metrics = {
          'loss': loss_value,
          'weight_norm': weight_norm,
          'grad_norm_unclipped': grad_norm_unclipped,
          **stats,
      }

      new_state = TrainingState(params=new_params, opt_state=new_opt_state)

      return new_state, metrics

    def make_initial_state(key: jnp.ndarray) -> TrainingState:
      """Initialises the training state (parameters and optimiser state)."""
      dummy_obs = utils.zeros_like(obs_spec)
      dummy_obs = utils.add_batch_dim(dummy_obs)  # Dummy 'sequence' dim.
      dummy_reset = jnp.array([False])
      initial_state = initial_state_fn.apply(None)
      initial_params = unroll_fn.init(key, dummy_obs, initial_state,
                                      dummy_reset)
      initial_opt_state = optimizer.init(initial_params)
      return TrainingState(params=initial_params, opt_state=initial_opt_state)

    # Initialise training state (parameters and optimiser state).
    state = make_initial_state(random_key)
    self._state = utils.replicate_in_all_devices(state, self._local_devices)

    if num_prefetch_threads is None:
      num_prefetch_threads = len(self._local_devices)
    self._prefetched_iterator = utils.sharded_prefetch(
        iterator,
        buffer_size=prefetch_size,
        devices=self._local_devices,
        num_threads=num_prefetch_threads,
    )

    self._sgd_step = jax.pmap(
        sgd_step, axis_name=_PMAP_AXIS_NAME, devices=self._devices)

    # Set up logging/counting.
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.make_default_logger('learner')

  def step(self):
    """Does a step of SGD and logs the results."""
    samples = next(self._prefetched_iterator)

    # Do a batch of SGD.
    start = time.time()
    self._state, results = self._sgd_step(self._state, samples)

    # Take results from first replica.
    results = utils.get_from_first_device(results)

    # Update our counts and record it.
    counts = self._counter.increment(steps=1, time_elapsed=time.time() - start)

    # Snapshot and attempt to write logs.
    self._logger.write({**results, **counts})

  def get_variables(self, names: Sequence[str]) -> List[networks.Params]:
    del names
    # Return first replica of parameters.
    return [utils.get_from_first_device(self._state.params, as_numpy=False)]

  def save(self) -> TrainingState:
    # Serialize only the first replica of parameters and optimizer state.
    return jax.tree_map(utils.get_from_first_device, self._state)

  def restore(self, state: TrainingState):
    self._state = utils.replicate_in_all_devices(state, self._local_devices)
