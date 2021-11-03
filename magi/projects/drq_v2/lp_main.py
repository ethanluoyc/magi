"""Run distributed DrQ-v2"""
import os
import time

from absl import app
from absl import flags
from absl import logging
import jax
import launchpad as lp
from launchpad.nodes.python import local_multi_processing
from ml_collections import config_flags
import tensorflow as tf

from magi.agents import drq_v2
from magi.agents.drq_v2 import agent_distributed as drq_v2_distributed
from magi.projects.drq_v2 import utils

FLAGS = flags.FLAGS
flags.DEFINE_string("workdir", None, "Work unit directory.")
config_flags.DEFINE_config_file("config")
flags.mark_flags_as_required(["workdir", "config"])


def _make_logger_factory(experiment_name, config, workdir):
    def logger_fn(
        label: str,
        save_data: bool,
        **kwargs,
    ):
        return utils.make_default_logger(
            workdir,
            label=label,
            save_data=save_data,
            log_to_wandb=config.log_to_wandb,
            wandb_kwargs={
                "group": experiment_name,
                "name": "/".join([experiment_name, label]),
                "project": config.wandb_project,
                "entity": config.wandb_entity,
                "config": config,
                "job_type": label,
                "dir": workdir,
            },
            **kwargs,
        )

    return logger_fn


def run(rng, config, workdir):
    experiment_name = (
        f"distributed-drqv2-{config.domain_name}-{config.task_name}_"
        f"{config.seed}_{int(time.time())}"
    )
    logger_factory = _make_logger_factory(experiment_name, config, workdir)
    environment_factory = lambda seed, _: utils.make_environment(  # noqa
        config.domain_name,
        config.task_name,
        seed,
        config.frame_stack,
        config.action_repeat,
    )

    agent = drq_v2_distributed.DistributedDrQV2(
        seed=int(jax.random.randint(rng, (), 0, 2 ** 31 - 1)),
        environment_factory=environment_factory,
        network_factory=lambda spec: drq_v2.make_networks(
            spec,
            latent_size=config.latent_size,
        ),
        num_actors=config.num_actors,
        config=drq_v2.DrQV2Config(
            min_replay_size=config.min_replay_size,
            max_replay_size=config.max_replay_size,
            prefetch_size=config.prefetch_size,
            batch_size=config.batch_size,
            discount=config.discount,
            n_step=config.n_step,
            critic_q_soft_update_rate=config.tau,
            learning_rate=config.lr,
            noise_clip=config.noise_clip,
            sigma=(config.sigma_start, config.sigma_end, config.sigma_schedule_steps),
            samples_per_insert=config.samples_per_insert,
        ),
        max_actor_steps=config.num_frames // config.action_repeat,
        logger_fn=logger_factory,
        log_every=config.log_every,
    )
    program = agent.build()

    lp.launch(
        program,
        launch_type=lp.LaunchType.LOCAL_MULTI_PROCESSING,
        local_resources={
            "replay": local_multi_processing.PythonProcess(
                env={"CUDA_VISIBLE_DEVICES": ""}
            ),
            "learner": local_multi_processing.PythonProcess(
                env={
                    "XLA_PYTHON_CLIENT_MEM_FRACTION": ".6",
                    "JAX_PLATFORM_NAME": "gpu",
                },
            ),
            "actor": local_multi_processing.PythonProcess(
                env={
                    "CUDA_VISIBLE_DEVICES": "",
                    "JAX_PLATFORM_NAME": "cpu",
                },
            ),
            "evaluator": local_multi_processing.PythonProcess(
                env={
                    "CUDA_VISIBLE_DEVICES": "",
                    "JAX_PLATFORM_NAME": "cpu",
                },
            ),
            "counter": local_multi_processing.PythonProcess(
                env={
                    "CUDA_VISIBLE_DEVICES": "",
                    "JAX_PLATFORM_NAME": "cpu",
                },
            ),
            "coordinator": local_multi_processing.PythonProcess(
                env={
                    "CUDA_VISIBLE_DEVICES": "",
                    "JAX_PLATFORM_NAME": "cpu",
                },
            ),
        },
        terminal="current_terminal",
    )


def main(argv):
    del argv

    config = FLAGS.config
    rng = jax.random.PRNGKey(config.seed)
    logging.info("RNG: %s", rng)

    # Make workdir if it does not exist
    os.makedirs(FLAGS.workdir, exist_ok=True)
    run(rng=rng, config=config, workdir=FLAGS.workdir)


if __name__ == "__main__":
    # Provide access to --jax_backend_target and --jax_xla_backend flags.
    jax.config.config_with_absl()
    # Binary should use CPU
    jax.config.update("jax_platform_name", "cpu")
    tf.config.experimental.set_visible_devices([], "GPU")
    app.run(main)
