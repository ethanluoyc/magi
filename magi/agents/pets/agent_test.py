"""Tests for PETS agent."""
from absl.testing import absltest
from absl.testing import parameterized
import acme
from acme import specs
from acme.testing import fakes
import jax.numpy as jnp

from magi.agents.pets import builder


def obs_preproc(obs):
    return obs


def obs_postproc(obs, pred):
    return obs + pred


def targ_proc(obs, next_obs):
    return next_obs - obs


def obs_cost_fn(obs):
    return -jnp.exp(-jnp.sum(obs, axis=-1) / (0.6 ** 2))


def ac_cost_fn(acs):
    return 0.01 * jnp.sum(jnp.square(acs), axis=-1)


def terminal_fn(obs, act, goal):
    del act, goal
    return jnp.zeros(obs.shape[0], dtype=jnp.bool_)


def reward_fn(obs, acs, goal):
    del goal
    return -(obs_cost_fn(obs) + ac_cost_fn(acs))


def make_environment():
    """Creates an OpenAI Gym environment."""
    # Load the gym environment.
    environment = fakes.ContinuousEnvironment(
        action_dim=1, observation_dim=2, episode_length=10
    )
    return environment


class PetsTest(parameterized.TestCase):
    """Tests for the PETS agent."""

    @parameterized.parameters(
        ("random",),
        ("cem",),
    )
    def test_run_agent(self, optimizer):
        # Create a fake environment to test with.
        environment = fakes.ContinuousEnvironment(
            action_dim=1, observation_dim=3, episode_length=10, bounded=True
        )
        spec = specs.make_environment_spec(environment)
        agent = builder.make_agent(
            spec,
            reward_fn,
            terminal_fn,
            obs_preproc,
            obs_postproc,
            targ_proc,
            optimizer=optimizer,
            hidden_sizes=[10],
            population_size=100,
            num_ensembles=2,
            planning_horizon=2,
            cem_iterations=1,
            num_particles=1,
        )
        env_loop = acme.EnvironmentLoop(environment, agent)
        env_loop.run(num_episodes=2)


if __name__ == "__main__":
    absltest.main()
