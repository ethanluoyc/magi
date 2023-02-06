import functools
import os

import jax
import tensorflow as tf
import wandb
from absl import app
from absl import flags
from absl import logging
from ml_collections import config_flags

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config", None, "experiment configuration.", lock_config=True
)
flags.DEFINE_string("workdir", None, "Work unit directory.")
flags.mark_flags_as_required(["config", "workdir"])


def run(main):
    # Provide access to --jax_backend_target and --jax_xla_backend flags.
    jax.config.config_with_absl()
    app.run(functools.partial(_run_main, main=main))


def _run_main(argv, *, main):
    """Runs the `main` method after some setup."""
    del argv
    # Disable GPU usage for TensorFlow. Otherwise Reverb will take up GPU
    # memory available for JAX
    tf.config.experimental.set_visible_devices([], "GPU")

    rng = jax.random.PRNGKey(FLAGS.config.seed)
    logging.info("RNG: %s", rng)

    # Make workdir if it does not exist
    os.makedirs(FLAGS.workdir, exist_ok=True)
    config = FLAGS.config
    if config.log_to_wandb:
        wandb.init(
            config=config,
            project=config.wandb_project,
            entity=config.wandb_entity,
            dir=FLAGS.workdir,
        )

    main(rng=rng, config=config, workdir=FLAGS.workdir)

    if FLAGS.config.log_to_wandb:
        wandb.finish()
