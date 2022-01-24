"""Run SAC on bsuite."""

from absl import app
from absl import flags
import acme
from acme import specs
from acme import wrappers
import numpy as np
import tensorflow as tf

from magi.agents import sac, sac_ae
from magi.agents.sac_ae import networks
from magi.agents.sac_ae.agent import SACAEConfig
import bsuite


FLAGS = flags.FLAGS
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

    agent_networks = sac.make_networks(spec)
    agent = sac.SACAgent(
        environment_spec=spec,
        networks=agent_networks,
        seed=FLAGS.seed,
        config=sac.SACConfig(),
    )

    loop = acme.EnvironmentLoop(env, agent)
    loop.run(num_episodes=env.bsuite_num_episodes)  # pytype: disable=attribute-error


if __name__ == "__main__":
    tf.config.set_visible_devices([], "GPU")
    app.run(main)
