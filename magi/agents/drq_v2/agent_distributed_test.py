#!/usr/bin/env python3
"""Integration test for the distributed DrQ-V2 agent."""

import acme
import launchpad as lp
import numpy as np
from absl.testing import absltest
from acme import specs
from acme.testing import fakes

from magi.agents.drq_v2 import agent_distributed as drq_v2_lib
from magi.agents.drq_v2 import config
from magi.agents.drq_v2 import networks


class DistributedDrQV2Test(absltest.TestCase):
    """Simple integration/smoke test for the distributed agent."""

    def test_distributed_drq_v2_run(self):
        """Tests that the agent can run on the control suite without crashing."""

        def environment_factory(seed, testing):
            del seed, testing
            environment = fakes.Environment(
                spec=specs.EnvironmentSpec(
                    observations=specs.Array((84, 84, 3), dtype=np.uint8),
                    actions=specs.BoundedArray(
                        (1,), dtype=np.float32, minimum=-1, maximum=1.0
                    ),
                    rewards=specs.Array((), dtype=np.float32),
                    discounts=specs.BoundedArray(
                        (), dtype=np.float32, minimum=0.0, maximum=1.0
                    ),
                )
            )
            return environment

        agent = drq_v2_lib.DistributedDrQV2(
            seed=0,
            environment_factory=environment_factory,
            network_factory=lambda spec: networks.make_networks(
                spec, hidden_size=10, latent_size=10
            ),
            num_actors=2,
            config=config.DrQV2Config(
                batch_size=32,
                min_replay_size=32,
                max_replay_size=1000,
                samples_per_insert=32.0,
                samples_per_insert_tolerance_rate=0.1,
            ),
            max_actor_steps=None,
        )
        program = agent.build()

        (learner_node,) = program.groups["learner"]
        learner_node.disable_run()

        lp.launch(program, launch_type="test_mt")

        learner: acme.Learner = learner_node.create_handle().dereference()

        for _ in range(5):
            learner.step()


if __name__ == "__main__":
    absltest.main()
