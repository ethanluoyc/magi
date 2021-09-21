"""Run CRR on D4RL."""

import copy
from typing import Sequence

from absl import app
from absl import flags
from absl import logging
from acme import specs
from acme import wrappers
from acme.agents.tf import actors as acting
from acme.tf import networks
from acme.tf import utils
from acme.tf import variable_utils
import d4rl  # type: ignore
import gym
import numpy as np
import sonnet as snt  # type: ignore
import tensorflow as tf
import wandb

from magi.agents.crr import tf_learning as learning
from magi.examples.offline import d4rl_dataset
from magi.utils import loggers

FLAGS = flags.FLAGS
flags.DEFINE_string("policy", "CRR(TF)", "Policy name")
flags.DEFINE_string("env", "hopper-medium-v0", "OpenAI gym environment name")
flags.DEFINE_integer("seed", 0, "seed")
flags.DEFINE_integer("log_freq", 500, "log frequency")
flags.DEFINE_integer("eval_freq", int(5e3), "evaluation frequency")
flags.DEFINE_integer("max_timesteps", int(1e6), "maximum number of steps")
flags.DEFINE_integer("eval_episodes", int(100), "maximum number of steps")
flags.DEFINE_integer("batch_size", 256, "batch size")
flags.DEFINE_float("discount", 0.99, "discount")
flags.DEFINE_bool("normalize", False, "normalize data")
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


def make_actor(action_spec, policy_network, variable_source):
    variable_client = variable_utils.VariableClient(
        client=variable_source,
        variables={"policy": policy_network.variables},
    )
    variable_client.update_and_wait()
    policy = snt.Sequential(
        [
            policy_network,
            networks.StochasticMeanHead(),
            networks.ClipToSpec(action_spec),
        ]
    )
    return acting.FeedForwardActor(policy, variable_client=variable_client)


def make_networks(
    environment_spec: specs.EnvironmentSpec,
    policy_layer_sizes: Sequence[int] = (256, 256, 256),
    critic_layer_sizes: Sequence[int] = (512, 512, 256),
    vmin: float = -5.0,
    vmax: float = 5.0,
    num_atoms: int = 51,
):
    action_spec = environment_spec.actions
    # Get total number of action dimensions from action spec.
    num_dimensions = np.prod(action_spec.shape, dtype=int)
    # # Create the shared observation network; here simply a state-less operation.
    # observation_network = tf2_utils.batch_concat

    # Create the policy network.
    policy_network = snt.Sequential(
        [
            networks.LayerNormMLP(policy_layer_sizes, activate_final=True),
            networks.MultivariateNormalDiagHead(
                num_dimensions,
                tanh_mean=True,
                min_scale=0.3,
                init_scale=0.7,
                fixed_scale=False,
                use_tfd_independent=False,
            ),
        ]
    )

    # Create the critic network.
    critic_network = snt.Sequential(
        [
            # The multiplexer concatenates the observations/actions.
            networks.CriticMultiplexer(),
            networks.LayerNormMLP(critic_layer_sizes, activate_final=True),
            networks.DiscreteValuedHead(vmin, vmax, num_atoms),
        ]
    )

    _ = utils.create_variables(policy_network, [environment_spec.observations])
    _ = utils.create_variables(
        critic_network,
        [environment_spec.observations, environment_spec.actions],
    )
    return {"policy": policy_network, "critic": critic_network}


def main(_):
    # Disable TF GPU
    # tf.config.set_visible_devices([], "GPU")
    if FLAGS.wandb:
        wandb.init(project="magi", entity="ethanluoyc", name="CRR (TF)")
    logging.info("---------------------------------------")
    logging.info("Policy: %s, Env: %s, Seed: %s", FLAGS.policy, FLAGS.env, FLAGS.seed)
    logging.info("---------------------------------------")

    np.random.seed(FLAGS.seed)

    tf.random.set_seed(FLAGS.seed)
    env = make_environment(FLAGS.env)
    environment_spec = specs.make_environment_spec(env)
    env.seed(FLAGS.seed)

    agent_networks = make_networks(environment_spec)
    target_networks = copy.deepcopy(agent_networks)
    data = d4rl.qlearning_dataset(env)
    if FLAGS.normalize:
        data, mean, std = d4rl_dataset.normalize_obs(data)
    else:
        mean, std = 0, 1
    data_iterator = d4rl_dataset.make_tf_data_iterator(
        data, batch_size=FLAGS.batch_size
    )
    learner = learning.CRRLearner(
        policy_network=agent_networks["policy"],
        critic_network=agent_networks["critic"],
        target_policy_network=target_networks["policy"],
        target_critic_network=target_networks["critic"],
        dataset=data_iterator,
        discount=FLAGS.discount,
        logger=loggers.make_logger(
            "learner",
            log_frequency=FLAGS.log_freq,
            use_wandb=FLAGS.wandb,
            wandb_kwargs={"config": FLAGS},
        ),
    )

    evaluator = make_actor(environment_spec.actions, agent_networks["policy"], learner)
    evaluations = []
    for t in range(int(FLAGS.max_timesteps)):
        learner.step()
        # Evaluate episode
        if (t + 1) % FLAGS.eval_freq == 0:
            logging.info("Time steps: %d", t + 1)
            evaluations.append(
                evaluate(
                    evaluator,
                    FLAGS.env,
                    FLAGS.seed,
                    mean,
                    std,
                    eval_episodes=FLAGS.eval_episodes,
                )
            )
            if FLAGS.wandb:
                wandb.log({"step": t, "eval_returns": evaluations[-1]})


if __name__ == "__main__":
    FLAGS.logtostderr = True
    app.run(main)
