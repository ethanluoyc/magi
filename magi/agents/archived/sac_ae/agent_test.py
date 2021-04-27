"""Tests for soft actor critic."""
from absl.testing import absltest
import acme
from acme import specs
from acme.utils import loggers
import haiku as hk

from magi.agents.archived.sac_ae.agent import SACAEAgent
from magi.utils import fakes
from magi.agents.archived.sac_ae import networks


class SACTest(absltest.TestCase):

  def test_sac_ae(self):
    # Create a fake environment to test with.
    environment = fakes.ContinuousVisualEnvironment(action_dim=2,
                                                    observation_shape=(64, 64, 3),
                                                    episode_length=10,
                                                    bounded=True)
    spec = specs.make_environment_spec(environment)

    # Make network purely functional
    policy = hk.without_apply_rng(
        hk.transform(lambda f: networks.Policy(spec.actions.shape[0])(f),
                     apply_rng=True))
    critic = hk.without_apply_rng(
        hk.transform(lambda o, a: networks.Critic()(o, a), apply_rng=True))
    encoder = hk.without_apply_rng(hk.transform(lambda o: networks.Encoder()(o)))
    decoder = hk.without_apply_rng(hk.transform(lambda f: networks.Decoder()(f)))
    linear = hk.without_apply_rng(hk.transform(lambda f: hk.Linear(50)(f)))

    # Construct the agent.
    agent = SACAEAgent(environment_spec=spec,
                       policy=policy,
                       critic=critic,
                       encoder=encoder,
                       decoder=decoder,
                       linear=linear,
                       seed=0,
                       start_steps=10,
                       batch_size=1)

    # Try running the environment loop. We have no assertions here because all
    # we care about is that the agent runs without raising any errors.
    loop = acme.EnvironmentLoop(environment,
                                agent,
                                logger=loggers.make_default_logger(label='environment',
                                                                   save_data=False))
    loop.run(num_episodes=20)


if __name__ == '__main__':
  absltest.main()
