import argparse

import acme
import numpy as np
from acme import specs
from acme import wrappers
from dm_control import suite
from magi.agents.sac2.sac import SAC

def make_dmc_env(domain_name, task_name):
  env = suite.load(domain_name=domain_name,
                   task_name=task_name,
                   environment_kwargs={"flat_observation": True},
                   task_kwargs={'random': 0})
  # env = GymWrapper(env)
  env = wrappers.CanonicalSpecWrapper(env)
  env = wrappers.SinglePrecisionWrapper(env)
  return env


def run(args):
  env = make_dmc_env(args.domain_name, args.task_name)
  # env_test = make_dmc_env(args.domain_name, args.task_name)

  from magi.agents.sac2.network import (ContinuousQFunction,
                                        StateDependentGaussianPolicy)
  spec = specs.make_environment_spec(env)
  np.random.seed(args.seed)
  print(spec)

  def critic_fn(s, a):
    return ContinuousQFunction(
        num_critics=2,
        hidden_units=(256, 256),
    )(s['observations'], a)

  def policy_fn(s):
    return StateDependentGaussianPolicy(
        action_size=spec.actions.shape[0],
        hidden_units=(256, 256),
        log_std_min=-20,
        log_std_max=2,
    )(s['observations'])

  algo = SAC(
      environment_spec=spec,
      policy_fn=policy_fn,
      critic_fn=critic_fn,
      seed=args.seed,
  )

  loop = acme.EnvironmentLoop(env, algo)
  loop.run(num_steps=int(1e6))


if __name__ == "__main__":
  p = argparse.ArgumentParser()
  p.add_argument('--domain_name', type=str, default='cartpole')
  p.add_argument('--task_name', type=str, default='swingup')
  p.add_argument("--num_agent_steps", type=int, default=750000)
  p.add_argument("--eval_interval", type=int, default=5000)
  p.add_argument("--seed", type=int, default=0)
  args = p.parse_args()
  run(args)
