#!/usr/bin/env python
# coding: utf-8

from absl import app
import jax.numpy as jnp
from acme import environment_loop, specs, wrappers
from acme.jax import variable_utils
from acme.utils import loggers
from gym.wrappers.time_limit import TimeLimit

from magi.agents.pets.acting import CEMOptimizerActor
from magi.agents.pets.agent import ModelBasedAgent
from magi.agents.pets import models
from magi.agents.pets.dataset import Dataset
from magi.agents.pets.learning import ModelBasedLearner
from magi.agents.pets.cartpole import CartpoleEnv
from magi.agents.pets import cartpole_config


def make_environment():
  """Creates an OpenAI Gym environment."""
  # Load the gym environment.
  environment = CartpoleEnv()
  environment.seed(0)
  environment = TimeLimit(environment, cartpole_config.TIME_LIMIT)
  environment = wrappers.GymWrapper(environment)
  environment = wrappers.SinglePrecisionWrapper(environment)
  return environment


def make_network(environment_spec):
  output_size = environment_spec.observations.shape[-1]

  def network(x, a):
    input_ = jnp.concatenate([x, a], axis=-1)
    model = models.BNN(output_size, hidden_sizes=[128, 128, 128])
    return model(input_)

  return network


def make_agent(environment_spec: specs.EnvironmentSpec):

  dataset = Dataset()
  # Create a learner
  network = make_network(environment_spec)
  learner = ModelBasedLearner(environment_spec, network, dataset,
                              cartpole_config.obs_preproc, cartpole_config.targ_proc,
                              num_epochs=100)
  # Create actor
  variable_client = variable_utils.VariableClient(learner, "")

  cost_fn = cartpole_config.cost_fn
  actor = CEMOptimizerActor(
      environment_spec,
      network,
      cost_fn,
      dataset,
      variable_client,
      cartpole_config.obs_preproc,
      cartpole_config.obs_postproc
  )
  agent = ModelBasedAgent(actor, learner)
  return agent


def main(unused_argv):
  del unused_argv
  environment = make_environment()
  environment_spec = specs.make_environment_spec(environment)
  actor = make_agent(environment_spec)

  env_loop_logger = loggers.TerminalLogger(label="environment_loop")
  env_loop = environment_loop.EnvironmentLoop(environment,
                                              actor,
                                              logger=env_loop_logger)
  env_loop.run(num_episodes=500)


if __name__ == "__main__":
  app.run(main)
