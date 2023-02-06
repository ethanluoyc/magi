"""Run TD3-BC on D4RL."""

import d4rl  # type: ignore
import d4rl_dataset
import gym
import jax
import numpy as np
import optax
import tensorflow as tf
import wandb
from absl import app
from absl import flags
from absl import logging
from acme import specs
from acme import wrappers
from acme.agents.jax import actor_core
from acme.agents.jax import actors
from acme.jax import variable_utils

from magi.agents import td3
from magi.agents import td3_bc

FLAGS = flags.FLAGS
flags.DEFINE_string("policy", "TD3_BC", "Policy name")
flags.DEFINE_string("env", "hopper-medium-v0", "OpenAI gym environment name")
flags.DEFINE_integer("seed", 0, "seed")
flags.DEFINE_integer("eval_freq", int(5e3), "evaluation frequency")
flags.DEFINE_integer("max_timesteps", int(1e6), "maximum number of steps")
flags.DEFINE_integer("batch_size", 256, "batch size")
flags.DEFINE_float("discount", 0.99, "discount")
flags.DEFINE_float("tau", 0.005, "target network update rate")
flags.DEFINE_float(
    "policy_noise", 0.2, "Noise added to target policy during critic update"
)
flags.DEFINE_float("noise_clip", 0.5, "Range to clip target policy noise")
flags.DEFINE_integer("policy_freq", 2, "Frequency of delayed policy updates")
# TD3 + BC
flags.DEFINE_float("alpha", 2.5, "alpha")
flags.DEFINE_bool("normalize", True, "normalize data")
flags.DEFINE_bool("wandb", False, "whether to use W&B")


def evaluate(actor, env_name, seed, mean, std, seed_offset=100, eval_episodes=10):
    """Evaluate the policy
    Runs policy for X episodes and returns average reward.
    A fixed seed is used for the eval environment
    """
    eval_env = make_environment(env_name)
    eval_env.seed(seed + seed_offset)
    actor.update(wait=True)
    avg_reward = 0.0
    for _ in range(eval_episodes):
        timestep = eval_env.reset()
        actor.observe_first(timestep)
        while not timestep.last():
            obs = (np.array(timestep.observation) - mean) / std
            action = actor.select_action(obs)
            timestep = eval_env.step(action)
            actor.observe(action, timestep)
            avg_reward += timestep.reward

    avg_reward /= eval_episodes
    d4rl_score = eval_env.get_normalized_score(avg_reward)

    logging.info("---------------------------------------")
    logging.info("Evaluation over %d episodes: %.3f", eval_episodes, d4rl_score)
    logging.info("---------------------------------------")
    return d4rl_score


def make_environment(name):
    environment = gym.make(name)
    environment = wrappers.GymWrapper(environment)
    return wrappers.SinglePrecisionWrapper(environment)


def main(_):
    # Disable TF GPU
    tf.config.set_visible_devices([], "GPU")
    if FLAGS.wandb:
        wandb.init(project="magi", entity="ethanluoyc", name="td3_bc")
    logging.info("---------------------------------------")
    logging.info("Policy: %s, Env: %s, Seed: %s", FLAGS.policy, FLAGS.env, FLAGS.seed)
    logging.info("---------------------------------------")

    np.random.seed(FLAGS.seed)
    env = make_environment(FLAGS.env)
    environment_spec = specs.make_environment_spec(env)
    env.seed(FLAGS.seed)

    max_action = environment_spec.actions.maximum[0]
    assert max_action == 1.0
    agent_networks = td3.make_networks(environment_spec)
    data = d4rl.qlearning_dataset(env)
    if FLAGS.normalize:
        data, mean, std = d4rl_dataset.normalize_obs(data)
    else:
        mean, std = 0, 1
    data_iterator = d4rl_dataset.make_tf_data_iterator(
        data, batch_size=FLAGS.batch_size
    ).as_numpy_iterator()
    random_key = jax.random.PRNGKey(FLAGS.seed)
    learner_key, actor_key = jax.random.split(random_key)
    learner = td3_bc.TD3BCLearner(
        agent_networks["policy"],
        agent_networks["critic"],
        iterator=data_iterator,
        random_key=learner_key,
        policy_optimizer=optax.adam(3e-4),
        critic_optimizer=optax.adam(3e-4),
        discount=FLAGS.discount,
        tau=FLAGS.tau,
        policy_noise=FLAGS.policy_noise * max_action,
        noise_clip=FLAGS.noise_clip * max_action,
        policy_update_period=FLAGS.policy_freq,
        alpha=FLAGS.alpha,
    )

    evaluator_network = td3.apply_policy_sample(agent_networks, eval_mode=True)
    evaluator = actors.GenericActor(
        actor_core.batched_feed_forward_to_actor_core(evaluator_network),
        random_key=actor_key,
        variable_client=variable_utils.VariableClient(learner, "policy", device="cpu"),
    )
    evaluations = []
    for t in range(int(FLAGS.max_timesteps)):
        learner.step()
        # Evaluate episode
        if (t + 1) % FLAGS.eval_freq == 0:
            logging.info("Time steps: %d", t + 1)
            evaluations.append(evaluate(evaluator, FLAGS.env, FLAGS.seed, mean, std))
            if FLAGS.wandb:
                wandb.log({"step": t, "eval_returns": evaluations[-1]})


if __name__ == "__main__":
    FLAGS.logtostderr = True
    app.run(main)
