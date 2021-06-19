import time

from absl import app
from absl import flags
import acme
from acme import specs
from acme.utils import counting
from acme.wrappers import gym_wrapper
import numpy as np

from magi.agents import drq
from magi.agents.drq import networks
from magi.agents.drq.agent import DrQConfig
from magi.experimental.environments import bullet_kuka_env
from magi.utils import loggers
import utils

FLAGS = flags.FLAGS
flags.DEFINE_bool("wandb", False, "whether to log result to wandb")
flags.DEFINE_string("wandb_project", "magi", "wandb project name")
flags.DEFINE_string("wandb_entity", "ethanluoyc", "wandb project entity")
flags.DEFINE_integer("num_steps", int(5e5), "Random seed.")
flags.DEFINE_integer("min_num_steps", int(1e3), "Random seed.")
flags.DEFINE_integer("batch_size", 128, "Batch size.")
flags.DEFINE_string("logdir", "./logs", "")
flags.DEFINE_integer("eval_freq", 1000, "")
flags.DEFINE_integer("eval_episodes", 10, "")
flags.DEFINE_integer("max_replay_size", 100_000, "Minimum replay size")
flags.DEFINE_integer("seed", 42, "Random seed.")


def load_env(seed):
    # Update this if necessary for change the environment
    env = bullet_kuka_env.KukaDiverseObjectEnv(
        renders=False,
        isDiscrete=False,
        width=84,
        height=84,
        numObjects=1,
        maxSteps=8,
        blockRandom=0,
        cameraRandom=0,
    )
    env.seed(seed)
    env = gym_wrapper.GymWrapper(env)
    return env


def main(_):
    np.random.seed(FLAGS.seed)
    if FLAGS.wandb:
        import wandb  # pylint: disable=import-outside-toplevel

        experiment_name = f"drq_kuka-grasp_" f"{FLAGS.seed}_{int(time.time())}"
        wandb.init(
            project=FLAGS.wandb_project,
            entity=FLAGS.wandb_entity,
            name=experiment_name,
            config=FLAGS,
            dir=FLAGS.logdir,
        )
    env = load_env(FLAGS.seed)
    spec = specs.make_environment_spec(env)
    network_spec = networks.make_default_networks(spec)

    agent = drq.DrQAgent(
        environment_spec=spec,
        networks=network_spec,
        config=DrQConfig(
            max_replay_size=FLAGS.max_replay_size,
            batch_size=FLAGS.batch_size,
            temperature_adam_b1=0.9,
            initial_num_steps=FLAGS.min_num_steps,
        ),
        seed=FLAGS.seed,
        logger=loggers.make_logger(
            label="learner", log_frequency=500, use_wandb=FLAGS.wandb
        ),
    )
    eval_actor = agent.make_actor(is_eval=True)
    counter = counting.Counter()
    loop = acme.EnvironmentLoop(
        env,
        agent,
        logger=loggers.make_logger(
            label="environment_loop", log_frequency=10, use_wandb=FLAGS.wandb
        ),
        counter=counter,
    )
    eval_logger = loggers.make_logger(label="evaluation", use_wandb=FLAGS.wandb)
    for _ in range(FLAGS.num_steps // FLAGS.eval_freq):
        loop.run(num_steps=FLAGS.eval_freq)
        eval_stats = utils.evaluate(eval_actor, env)
        eval_logger.write({**eval_stats, "steps": counter.get_counts()["steps"]})
    if FLAGS.wandb:
        wandb.finish()


if __name__ == "__main__":
    app.run(main)
