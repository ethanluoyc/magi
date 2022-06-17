#!/usr/bin/env python3
"""Integration test for the distributed agent."""

from typing import Optional

from absl.testing import absltest
import acme
from acme.testing import fakes
import launchpad as lp
import numpy as np

from magi.agents.impala import agent_distributed as impala_lib
from magi.agents.impala import config
from magi.agents.impala.agent_test import MyNetwork


def network_factory(spec):

  def forward_fn(x, s):
    model = MyNetwork(spec.num_values)
    return model(x, s)

  def initial_state_fn(batch_size: Optional[int] = None):
    model = MyNetwork(spec.num_values)
    return model.initial_state(batch_size)

  def unroll_fn(inputs, state, start_of_episode=None):
    model = MyNetwork(spec.num_values)
    return model.unroll(inputs, state, start_of_episode)

  return {
      'forward': forward_fn,
      'unroll': unroll_fn,
      'initial_state': initial_state_fn,
  }


class DistributedAgentTest(absltest.TestCase):
  """Simple integration/smoke test for the distributed agent."""

  def test_control_suite(self):
    """Tests that the agent can run on the control suite without crashing."""

    def environment_factory():
      environment = fakes.DiscreteEnvironment(
          num_actions=5,
          num_observations=10,
          obs_shape=(5, 10),
          obs_dtype=np.float32,
          episode_length=10,
      )
      return environment

    agent = impala_lib.DistributedIMPALA(
        environment_factory=lambda seed, test: environment_factory(),
        network_factory=network_factory,
        num_actors=2,
        config=config.IMPALAConfig(
            sequence_length=4, sequence_period=4, max_queue_size=1000),
        max_actor_steps=None,
    )
    program = agent.build()

    (learner_node,) = program.groups['learner']
    learner_node.disable_run()

    lp.launch(program, launch_type='test_mt')

    learner: acme.Learner = learner_node.create_handle().dereference()

    for _ in range(5):
      learner.step()


if __name__ == '__main__':
  import tensorflow as tf

  tf.config.experimental.set_visible_devices([], 'GPU')
  absltest.main()
