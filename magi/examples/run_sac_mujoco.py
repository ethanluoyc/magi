"""Run TD3 on Gym Mujoco environments."""
import time

from absl import app
from absl import flags
import acme
from acme import specs
from acme import wrappers
import gym
import haiku as hk
import numpy as np

from magi.agents.sac import networks
from magi.agents.sac.agent import SACAgent
from magi.utils import loggers

FLAGS = flags.FLAGS
flags.DEFINE_string("env", "HalfCheetah-v2", "Gym environment name")
flags.DEFINE_bool("wandb", False, "whether to log result to wandb")
flags.DEFINE_string("wandb_project", "magi", "wandb project name")
flags.DEFINE_string("wandb_entity", "ethanluoyc", "wandb project entity")
flags.DEFINE_integer("num_steps", int(1e6), "Random seed.")
flags.DEFINE_integer("seed", 0, "Random seed.")


def load_env(env_name: str, seed: int):
    env = gym.make(env_name)
    env = wrappers.wrap_all(
        env,
        [
            wrappers.GymWrapper,
            wrappers.CanonicalSpecWrapper,
            wrappers.SinglePrecisionWrapper,
        ],
    )
    env.seed(seed)
    return env


def main(_):
    np.random.seed(FLAGS.seed)
    env = load_env(FLAGS.env, FLAGS.seed)
    spec = specs.make_environment_spec(env)

    def critic_fn(obs, a):
        return networks.DoubleCritic(hidden_units=(256, 256))(obs, a)

    def policy_fn(obs):
        return networks.GaussianPolicy(
            action_size=spec.actions.shape[0], hidden_units=(256, 256)
        )(obs)

    policy = hk.without_apply_rng(hk.transform(policy_fn, apply_rng=True))
    critic = hk.without_apply_rng(hk.transform(critic_fn, apply_rng=True))
    exp_name = f"sac_gym_{FLAGS.env}_{FLAGS.seed}_{int(time.time())}"
    algo = SACAgent(
        environment_spec=spec,
        policy=policy,
        critic=critic,
        seed=FLAGS.seed,
        logger=loggers.make_logger(
            "agent",
            use_wandb=FLAGS.wandb,
            log_frequency=1000,
            wandb_kwargs={
                "project": FLAGS.wandb_project,
                "entity": FLAGS.wandb_entity,
                "name": exp_name,
                "config": FLAGS,
            },
        ),
    )

    loop = acme.EnvironmentLoop(
        env,
        algo,
        logger=loggers.make_logger(label="environment_loop", use_wandb=FLAGS.wandb),
    )
    loop.run(num_steps=FLAGS.num_steps)


if __name__ == "__main__":
    app.run(main)
