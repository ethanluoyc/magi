from absl.testing import absltest
import acme
from acme import specs
from acme.jax import variable_utils
from acme.testing import fakes
import jax.numpy as jnp

from magi.agents.pets.acting import CEMOptimizerActor
from magi.agents.pets.agent import ModelBasedAgent
from magi.agents.pets import models
from magi.agents.pets.dataset import Dataset
from magi.agents.pets.learning import ModelBasedLearner


def get_config():

  def obs_preproc(obs):
    return obs

  def obs_postproc(obs, pred):
    return obs + pred

  def targ_proc(obs, next_obs):
    return next_obs - obs

  def obs_cost_fn(obs):
    return -jnp.exp(-jnp.sum(obs, axis=-1) / (0.6**2))

  def ac_cost_fn(acs):
    return 0.01 * jnp.sum(jnp.square(acs), axis=-1)

  def cost_fn(obs, acs):
    return obs_cost_fn(obs) + ac_cost_fn(acs)

  return {
      'cost_fn': cost_fn,
      'obs_preproc': obs_preproc,
      'targ_proc': targ_proc,
      'obs_postproc': obs_postproc
  }


def make_environment():
  """Creates an OpenAI Gym environment."""
  # Load the gym environment.
  environment = fakes.ContinuousEnvironment(action_dim=1,
                                            observation_dim=2,
                                            episode_length=10)
  return environment


def make_network(environment_spec):
  output_size = environment_spec.observations.shape[-1]

  def network(x, a):
    input_ = jnp.concatenate([x, a], axis=-1)
    model = models.BNN(output_size, hidden_sizes=[10])
    return model(input_)

  return network


def make_agent(environment_spec: specs.EnvironmentSpec):

  dataset = Dataset()
  config = get_config()
  # Create a learner
  network = make_network(environment_spec)
  learner = ModelBasedLearner(environment_spec, network, dataset,
                              config['obs_preproc'],
                              config['targ_proc'], batch_size=32, num_ensembles=5)
  # Create actor
  variable_client = variable_utils.VariableClient(learner, "")

  actor = CEMOptimizerActor(environment_spec, network, config['cost_fn'],
                            dataset, variable_client, config['obs_preproc'],
                            config['targ_proc'], pop_size=100)
  agent = ModelBasedAgent(actor, learner)
  return agent


class PetsTest(absltest.TestCase):

  def test_run_agent(self):
    # Create a fake environment to test with.
    environment = fakes.ContinuousEnvironment(action_dim=1,
                                              observation_dim=3,
                                              episode_length=10,
                                              bounded=True)
    spec = specs.make_environment_spec(environment)
    agent = make_agent(spec)
    env_loop = acme.EnvironmentLoop(environment, agent)
    env_loop.run(num_episodes=1)


if __name__ == '__main__':
  absltest.main()
