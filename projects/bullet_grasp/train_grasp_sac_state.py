import time

from absl import app
from absl import flags
import acme
from acme import specs
from acme import wrappers
from acme.utils import counting
from acme.wrappers import gym_wrapper
import haiku as hk
import numpy as np

from environments import kuka_state_env
from magi.agents.sac.agent import SACAgent
from magi.utils import loggers
import utils

FLAGS = flags.FLAGS
flags.DEFINE_bool("wandb", False, "whether to log result to wandb")
flags.DEFINE_string("name", "kuka_grasp", "experiment name")
flags.DEFINE_string("wandb_project", "magi", "wandb project name")
flags.DEFINE_string("wandb_entity", "ethanluoyc", "wandb project entity")
flags.DEFINE_string("wandb_group", "", "wandb project entity")
flags.DEFINE_integer("num_steps", int(5e5), "Random seed.")
flags.DEFINE_integer("min_num_steps", int(1e3), "Random seed.")
flags.DEFINE_integer("eval_every", int(1000), "Random seed.")
flags.DEFINE_integer("batch_size", 128, "Random seed.")
flags.DEFINE_integer("seed", 0, "Random seed.")


def load_env(seed):
    # Update this if necessary for change the environment
    env = kuka_state_env.KukaDiverseObjectEnv(
        renders=False,
        isDiscrete=False,
        width=64,
        height=64,
        numObjects=1,
        maxSteps=8,
        blockRandom=1.0,
        cameraRandom=0,
    )
    env.seed(seed)
    env = gym_wrapper.GymWrapper(env)
    env = wrappers.SinglePrecisionWrapper(env)
    env = wrappers.CanonicalSpecWrapper(env)
    return env


def main(_):
    np.random.seed(FLAGS.seed)
    if FLAGS.wandb:
        import wandb

        wandb.init(
            project=FLAGS.wandb_project,
            entity=FLAGS.wandb_entity,
            name=f"{FLAGS.name}_{time.time()}",
            group=FLAGS.wandb_group,
            config=FLAGS,
        )
    env = load_env(FLAGS.seed)
    spec = specs.make_environment_spec(env)

    from magi.agents.sac import networks

    def critic_fn(s, a):
        return networks.DoubleCritic(
            hidden_units=(256, 256),
        )(s, a)

    def policy_fn(s):
        return networks.GaussianPolicy(
            action_size=spec.actions.shape[0],
            hidden_units=(256, 256),
        )(s)

    policy = hk.without_apply_rng(hk.transform(policy_fn, apply_rng=True))
    critic = hk.without_apply_rng(hk.transform(critic_fn, apply_rng=True))

    agent = SACAgent(
        environment_spec=spec,
        policy=policy,
        critic=critic,
        seed=FLAGS.seed,
        logger=loggers.make_logger(
            label="learner", log_frequency=500, use_wandb=FLAGS.wandb
        ),
        initial_num_steps=FLAGS.min_num_steps,
        batch_size=FLAGS.batch_size,
    )
    eval_actor = agent.make_actor(is_eval=False)
    counter = counting.Counter()
    loop = acme.EnvironmentLoop(
        env,
        agent,
        logger=loggers.make_logger(label="environment_loop", use_wandb=FLAGS.wandb),
        counter=counter,
    )
    eval_logger = loggers.make_logger(label="evaluation", use_wandb=FLAGS.wandb)
    for _ in range(FLAGS.num_steps // FLAGS.eval_every):
        loop.run(num_steps=FLAGS.eval_every)
        eval_stats = utils.evaluate(eval_actor, env)
        eval_logger.write({**eval_stats, "steps": counter.get_counts()["steps"]})
    if FLAGS.wandb:
        wandb.finish()


if __name__ == "__main__":
    app.run(main)
