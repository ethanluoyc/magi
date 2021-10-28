import time

from absl import app
from absl import flags
import acme
from acme import specs
from acme import wrappers
from dm_control import suite  # pytype: disable=import-error
from dm_control.suite.wrappers import pixels  # pytype: disable=import-error
import jax
import numpy as np
import optax
import tensorflow as tf

from magi.agents import drq_v2
from magi.utils import loggers
from magi.utils import wrappers as magi_wrappers

FLAGS = flags.FLAGS
flags.DEFINE_string("domain_name", "reacher", "dm_control domain")
flags.DEFINE_string("task_name", "hard", "dm_control task")
flags.DEFINE_bool("wandb", False, "whether to log result to wandb")
flags.DEFINE_string("wandb_project", "magi", "wandb project name")
flags.DEFINE_string("wandb_entity", "ethanluoyc", "wandb project entity")
flags.DEFINE_string("logdir", "./logs", "")
flags.DEFINE_integer("num_frames", int(3e6), "")
flags.DEFINE_integer("eval_freq", 10000, "")
flags.DEFINE_integer("eval_episodes", 10, "")
flags.DEFINE_integer("frame_stack", 3, "")
flags.DEFINE_integer("action_repeat", 2, "")
flags.DEFINE_integer("max_replay_size", 500_000, "Maximum replay size")
flags.DEFINE_integer("min_replay_size", 4000, "Minimum replay size")
flags.DEFINE_integer("batch_size", 256, "Batch size")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_float("lr", 1e-4, "")
flags.DEFINE_float("sigma_start", 1.0, "")
flags.DEFINE_float("sigma_end", 0.1, "")
flags.DEFINE_integer("sigma_schedule_steps", 500000, "")
flags.DEFINE_integer("prefetch_size", None, "")
flags.DEFINE_integer("latent_size", 50, "")
flags.DEFINE_integer("observations_per_step", 2, "")
flags.DEFINE_integer("learner_log_freq", 5000, "")
flags.DEFINE_integer("env_log_freq", 1, "")
flags.DEFINE_integer("num_seed_frames", 4000, "")
flags.DEFINE_integer("num_expl_steps", 2000, "")


def load_env(domain_name, task_name, seed, frame_stack, action_repeat):
    env = suite.load(
        domain_name=domain_name,
        task_name=task_name,
        environment_kwargs={"flat_observation": True},
        task_kwargs={"random": seed},
    )
    camera_id = 2 if domain_name == "quadruped" else 0
    env = pixels.Wrapper(
        env,
        pixels_only=True,
        render_kwargs={"width": 84, "height": 84, "camera_id": camera_id},
    )
    env = wrappers.CanonicalSpecWrapper(env)
    env = magi_wrappers.TakeKeyWrapper(env, "pixels")
    env = wrappers.ActionRepeatWrapper(env, action_repeat)
    env = magi_wrappers.FrameStackingWrapper(env, num_frames=frame_stack)
    env = wrappers.SinglePrecisionWrapper(env)
    return env


def main(_):
    np.random.seed(FLAGS.seed)
    if FLAGS.wandb:
        import wandb  # pylint: disable=import-outside-toplevel

        experiment_name = (
            f"drq-v2-{FLAGS.domain_name}-{FLAGS.task_name}_"
            f"{FLAGS.seed}_{int(time.time())}"
        )
        wandb.init(
            project=FLAGS.wandb_project,
            entity=FLAGS.wandb_entity,
            name=experiment_name,
            config=FLAGS,
            dir=FLAGS.logdir,
        )
    env = load_env(
        FLAGS.domain_name,
        FLAGS.task_name,
        FLAGS.seed,
        FLAGS.frame_stack,
        FLAGS.action_repeat,
    )
    test_env = load_env(
        FLAGS.domain_name,
        FLAGS.task_name,
        FLAGS.seed + 42,
        FLAGS.frame_stack,
        FLAGS.action_repeat,
    )
    spec = specs.make_environment_spec(env)
    agent_networks = drq_v2.make_networks(spec, latent_size=FLAGS.latent_size)
    agent = drq_v2.DrQV2(
        environment_spec=spec,
        networks=agent_networks,
        config=drq_v2.DrQV2Config(
            min_replay_size=FLAGS.min_replay_size,
            max_replay_size=FLAGS.max_replay_size,
            batch_size=FLAGS.batch_size,
            sigma=(FLAGS.sigma_start, FLAGS.sigma_end, FLAGS.sigma_schedule_steps),
            learning_rate=FLAGS.lr,
            prefetch_size=FLAGS.prefetch_size,
            samples_per_insert=FLAGS.batch_size / FLAGS.observations_per_step,
        ),
        seed=FLAGS.seed,
        logger=loggers.make_logger(
            label="learner",
            log_frequency=FLAGS.learner_log_freq,
            use_wandb=FLAGS.wandb,
        ),
    )
    evaluator_network = drq_v2.get_default_behavior_policy(
        agent_networks,
        spec.actions,
        # Turn off action noise in evaluator_network
        optax.constant_schedule(0.0),
    )
    eval_actor = agent.builder.make_actor(
        jax.random.PRNGKey(FLAGS.seed + 10),
        evaluator_network,
        variable_source=agent,
    )

    train_loop = acme.EnvironmentLoop(
        env,
        agent,
        logger=loggers.make_logger(
            label="train_loop", log_frequency=FLAGS.env_log_freq, use_wandb=FLAGS.wandb
        ),
    )
    eval_loop = acme.EnvironmentLoop(
        test_env,
        eval_actor,
        logger=loggers.make_logger(label="eval_loop", use_wandb=FLAGS.wandb),
    )
    num_steps = FLAGS.num_frames // FLAGS.action_repeat
    assert FLAGS.num_frames % FLAGS.action_repeat == 0
    for _ in range(num_steps // FLAGS.eval_freq):
        train_loop.run(num_steps=FLAGS.eval_freq)
        eval_actor.update(wait=True)
        eval_loop.run(num_episodes=FLAGS.eval_episodes)

    if FLAGS.wandb:
        wandb.finish()


if __name__ == "__main__":
    tf.config.set_visible_devices([], "GPU")
    app.run(main)
