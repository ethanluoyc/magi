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
"""IMPALA actor."""
from typing import Callable, Optional, Tuple

from acme import core
from acme.agents.jax.impala import types
from acme.jax import networks
from acme.jax import variable_utils
import dm_env
import haiku as hk
import jax
import jax.numpy as jnp

_LogitsAndValue = Tuple[networks.Logits, networks.Value]
PolicyValueFn = Callable[[hk.Params, types.Observation, hk.LSTMState],
                         Tuple[_LogitsAndValue, hk.LSTMState]]


class IMPALAActor(core.Actor):
  """A recurrent actor."""

  _state: hk.LSTMState
  _prev_state: hk.LSTMState
  _prev_logits: jnp.ndarray

  def __init__(
      self,
      forward_fn: PolicyValueFn,
      initial_state_fn: Callable[[], hk.LSTMState],
      rng: hk.PRNGSequence,
      variable_client: Optional[variable_utils.VariableClient] = None,
      adder=None,
  ):

    # Store these for later use.
    self._adder = adder
    self._variable_client = variable_client
    self._forward = forward_fn
    self._reset_fn_or_none = getattr(forward_fn, 'reset', None)
    self._rng = rng

    # Make sure not to use a random policy after checkpoint restoration by
    # assigning variables before running the environment loop.
    if self._variable_client is not None:
      self._variable_client.update_and_wait()

    self._initial_state = hk.without_apply_rng(
        hk.transform(initial_state_fn, apply_rng=True)).apply(None)

  def select_action(self, observation: types.Observation) -> types.Action:

    if self._state is None:
      self._state = self._initial_state

    # Forward.
    (logits, _), new_state = self._forward(self._params, observation,
                                           self._state)

    self._prev_logits = logits
    self._prev_state = self._state
    self._state = new_state

    action = jax.random.categorical(next(self._rng), logits)

    return action

  def observe_first(self, timestep: dm_env.TimeStep):
    if self._adder:
      self._adder.add_first(timestep)

    # Set the state to None so that we re-initialize at the next policy call.
    self._state = None
    self._should_reset = True

    # Reset state of inference functions that employ stateful wrappers (eg. BIT)
    # at the start of the episode.
    if self._reset_fn_or_none is not None:
      self._reset_fn_or_none()

  def observe(
      self,
      action: types.Action,
      next_timestep: dm_env.TimeStep,
  ):
    if not self._adder:
      return

    extras = {'logits': self._prev_logits, 'core_state': self._prev_state}
    self._adder.add(action, next_timestep, extras)

  def update(self, wait: bool = False):
    if self._variable_client is not None:
      self._variable_client.update(wait)

  @property
  def _params(self) -> Optional[hk.Params]:
    if self._variable_client is None:
      # If self._variable_client is None then we assume self._forward  does not
      # use the parameters it is passed and just return None.
      return None
    return self._variable_client.params
