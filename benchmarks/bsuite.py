"""Run DrQ on bsuite."""
import time

from absl import app
from absl import flags
import acme
from acme import specs
from acme import wrappers
import jax
import numpy as np
import tensorflow as tf

from magi import wrappers as magi_wrappers
from magi.agents import drq
from magi.agents import sac
from magi.utils import loggers
import bsuite


FLAGS = flags.FLAGS
flags.DEFINE_string("logdir", "./logs", "")
flags.DEFINE_integer("num_steps", int(1e6), "")
flags.DEFINE_integer("max_replay_size", 100000, "Maximum replay size")
flags.DEFINE_integer("min_replay_size", 1000, "Minimum replay size")
flags.DEFINE_integer("batch_size", 128, "Batch size")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_string("bsuite_id", "catch/0", "Bsuite id.")
flags.DEFINE_string("results_dir", "/tmp/bsuite", "CSV results directory.")
flags.DEFINE_boolean("overwrite", False, "Whether to overwrite csv results.")


def main(_):
    np.random.seed(FLAGS.seed)

    raw_environment = bsuite.load_and_record_to_csv(
        bsuite_id=FLAGS.bsuite_id,
        results_dir=FLAGS.results_dir,
        overwrite=FLAGS.overwrite,
    )
    env = wrappers.SinglePrecisionWrapper(raw_environment)
    spec = specs.make_environment_spec(env)
    agent_networks = drq.make_networks(spec)
    agent = drq.DrQAgent(
        environment_spec=spec,
        networks=agent_networks,
        config=drq.DrQConfig(
            target_entropy=sac.target_entropy_from_env_spec(spec),
            max_replay_size=FLAGS.max_replay_size,
            min_replay_size=FLAGS.min_replay_size,
            batch_size=FLAGS.batch_size,
            temperature_adam_b1=0.9,
        ),
        seed=FLAGS.seed,
    )
    loop = acme.EnvironmentLoop(env, agent)
    loop.run(num_episodes=env.bsuite_num_episodes)

    loop = acme.EnvironmentLoop(
        env,
        agent,
    )
    loop.run(num_episodes=env.bsuite_num_episodes)  # pytype: disable=attribute-error


if __name__ == "__main__":
    tf.config.set_visible_devices([], "GPU")
    app.run(main)
