"""Run Soft-Actor Critic on dm_control (state observation)."""
import time

from absl import app
from absl import flags
import acme
from acme import specs
from acme import wrappers
from dm_control import suite  # pytype: disable=import-error
import numpy as np
import tensorflow as tf

from magi import wrappers as magi_wrappers
from magi.agents import sac
from magi.utils import loggers

FLAGS = flags.FLAGS
flags.DEFINE_string("domain_name", "cartpole", "dm_control domain")
flags.DEFINE_string("task_name", "swingup", "dm_control task")
flags.DEFINE_bool("wandb", False, "whether to log result to wandb")
flags.DEFINE_string("wandb_project", "magi", "wandb project name")
flags.DEFINE_string("wandb_entity", "ethanluoyc", "wandb project entity")
flags.DEFINE_integer("num_steps", int(1e6), "Random seed.")
flags.DEFINE_integer("seed", 0, "Random seed.")


def load_env(domain_name, task_name, seed):
    env = suite.load(
        domain_name=domain_name,
        task_name=task_name,
        environment_kwargs={"flat_observation": True},
        task_kwargs={"random": seed},
    )
    env = wrappers.CanonicalSpecWrapper(env)
    env = magi_wrappers.TakeKeyWrapper(env, "observations")
    env = wrappers.SinglePrecisionWrapper(env)
    return env


def main(_):
    np.random.seed(FLAGS.seed)
    env = load_env(FLAGS.domain_name, FLAGS.task_name, FLAGS.seed)
    spec = specs.make_environment_spec(env)
    exp_name = (
        f"sac-{FLAGS.domain_name}_{FLAGS.task_name}_{FLAGS.seed}_{int(time.time())}"
    )
    agent_networks = sac.make_networks(spec)
    algo = sac.SACAgent(
        environment_spec=spec,
        networks=agent_networks,
        seed=FLAGS.seed,
        config=sac.SACConfig(target_entropy=sac.target_entropy_from_env_spec(spec)),
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
    tf.config.set_visible_devices([], "GPU")
    app.run(main)
