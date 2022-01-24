"""Run DrQ on dm_control suite."""
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
flags.DEFINE_string("domain_name", "cheetah", "dm_control domain")
flags.DEFINE_string("task_name", "run", "dm_control task")
flags.DEFINE_bool("wandb", False, "whether to log result to wandb")
flags.DEFINE_string("wandb_project", "magi", "wandb project name")
flags.DEFINE_string("wandb_entity", "ethanluoyc", "wandb project entity")
flags.DEFINE_string("logdir", "./logs", "")
flags.DEFINE_integer("num_steps", int(1e6), "")
flags.DEFINE_integer("eval_freq", 5000, "")
flags.DEFINE_integer("eval_episodes", 10, "")
flags.DEFINE_integer("frame_stack", 3, "")
flags.DEFINE_integer("action_repeat", None, "")
flags.DEFINE_integer("max_replay_size", 100000, "Maximum replay size")
flags.DEFINE_integer("min_replay_size", 1000, "Minimum replay size")
flags.DEFINE_integer("batch_size", 128, "Batch size")
flags.DEFINE_integer("seed", 42, "Random seed.")

PLANET_ACTION_REPEAT = {
    "cartpole-swingup": 8,
    "reacher-easy": 4,
    "cheetah-run": 4,
    "finger-spin": 2,
    "ball_in_cup-catch": 4,
    "walker-walk": 2,
}


def load_env(domain_name, task_name, seed, frame_stack, action_repeat):
    env = bsuite.load_from_id('catch/0')
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
            f"drq-{FLAGS.domain_name}-{FLAGS.task_name}_"
            f"{FLAGS.seed}_{int(time.time())}"
        )
        wandb.init(
            project=FLAGS.wandb_project,
            entity=FLAGS.wandb_entity,
            name=experiment_name,
            config=FLAGS,
            dir=FLAGS.logdir,
        )
    if FLAGS.action_repeat is None:
        action_repeat = PLANET_ACTION_REPEAT.get(
            f"{FLAGS.domain_name}-{FLAGS.task_name}", None
        )
        if action_repeat is None:
            print(
                "Unable to find action repeat configuration from PlaNet, default to 2"
            )
            action_repeat = 2
    else:
        action_repeat = FLAGS.action_repeat
    env = load_env(
        FLAGS.domain_name, FLAGS.task_name, FLAGS.seed, FLAGS.frame_stack, action_repeat
    )
    test_env = load_env(
        FLAGS.domain_name,
        FLAGS.task_name,
        FLAGS.seed + 42,
        FLAGS.frame_stack,
        action_repeat,
    )
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
        logger=loggers.make_logger(
            label="learner", log_frequency=1000, use_wandb=FLAGS.wandb
        ),
    )
    evaluator_network = drq.apply_policy_sample(agent_networks, eval_mode=True)
    evaluator = agent.builder.make_actor(
        jax.random.PRNGKey(FLAGS.seed + 10),
        evaluator_network,
        variable_source=agent,
    )

    train_loop = acme.EnvironmentLoop(
        env,
        agent,
        logger=loggers.make_logger(label="train_loop", use_wandb=FLAGS.wandb),
    )
    eval_loop = acme.EnvironmentLoop(
        test_env,
        evaluator,
        logger=loggers.make_logger(label="eval_loop", use_wandb=FLAGS.wandb),
    )
    for _ in range(FLAGS.num_steps // FLAGS.eval_freq):
        train_loop.run(num_steps=FLAGS.eval_freq)
        evaluator.update(wait=True)
        eval_loop.run(num_episodes=FLAGS.eval_episodes)

    if FLAGS.wandb:
        wandb.finish()


if __name__ == "__main__":
    tf.config.set_visible_devices([], "GPU")
    app.run(main)
