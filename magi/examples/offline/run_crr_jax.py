"""Run CRR on D4RL."""
from typing import Sequence

from absl import app
from absl import flags
from absl import logging
from acme import specs
from acme import wrappers
from acme.agents.jax import actors as acting
from acme.jax import networks as acme_networks
from acme.jax import variable_utils
import d4rl  # type: ignore
import gym
import haiku as hk
import jax
import numpy as np
import tensorflow as tf
import wandb

from magi.agents.crr import learning
from magi.agents.crr import networks
from magi.examples.offline import d4rl_dataset
from magi.utils import loggers

FLAGS = flags.FLAGS
flags.DEFINE_string("policy", "CRR (JAX)", "Policy name")
flags.DEFINE_string("env", "hopper-medium-v0", "OpenAI gym environment name")
flags.DEFINE_integer("seed", 0, "seed")
flags.DEFINE_integer("log_freq", 500, "log frequency")
flags.DEFINE_integer("eval_freq", int(5e3), "evaluation frequency")
flags.DEFINE_integer("max_timesteps", int(1e6), "maximum number of steps")
flags.DEFINE_integer("batch_size", 256, "batch size")
flags.DEFINE_integer("eval_episodes", 10, "number of evaluation episodes")
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


def make_actor(action_spec, policy_network, variable_source, random_key):
    variable_client = variable_utils.VariableClient(
        client=variable_source,
        key="policy",
    )
    variable_client.update_and_wait()

    def behavior_policy(params, key, obs):
        del key
        action_dist = policy_network.apply(params, obs)
        clip_fn = hk.transform(lambda a: acme_networks.ClipToSpec(action_spec)(a))
        return clip_fn.apply(None, None, action_dist.mean())

    return acting.FeedForwardActor(
        behavior_policy, random_key, variable_client=variable_client
    )


def make_networks(
    environment_spec: specs.EnvironmentSpec,
    policy_layer_sizes: Sequence[int] = (256, 256, 256),
    critic_layer_sizes: Sequence[int] = (256, 256, 256),
    vmin: float = -150.0,
    vmax: float = 150.0,
    num_atoms: int = 51,
):
    action_spec = environment_spec.actions
    # Get total number of action dimensions from action spec.
    num_dimensions = int(np.prod(action_spec.shape, dtype=int))
    # # Create the shared observation network; here simply a state-less operation.
    # observation_network = tf2_utils.batch_concat

    def policy_fn(observations):
        # Create the policy network.
        policy_network = hk.Sequential(
            [
                acme_networks.LayerNormMLP(policy_layer_sizes, activate_final=True),
                networks.MultivariateNormalDiagHead(
                    num_dimensions,
                    tanh_mean=True,
                    init_scale=0.2,
                    fixed_scale=True,
                    use_tfd_independent=False,
                ),
            ]
        )
        return policy_network(observations)

    def critic_fn(observations, actions):
        # Create the critic network.
        critic_network = hk.Sequential(
            [
                # The multiplexer concatenates the observations/actions.
                acme_networks.CriticMultiplexer(),
                acme_networks.LayerNormMLP(critic_layer_sizes, activate_final=True),
                networks.DiscreteValuedHead(vmin, vmax, num_atoms),
            ]
        )
        return critic_network(observations, actions)

    return {
        "policy": hk.without_apply_rng(hk.transform(policy_fn)),
        "critic": hk.without_apply_rng(hk.transform(critic_fn)),
    }


def main(_):
    # Disable TF GPU

    tf.config.set_visible_devices([], "GPU")
    if FLAGS.wandb:
        wandb.init(project="magi", entity="ethanluoyc", name="CRR (JAX)")
    logging.info("---------------------------------------")
    logging.info("Policy: %s, Env: %s, Seed: %s", FLAGS.policy, FLAGS.env, FLAGS.seed)
    logging.info("---------------------------------------")

    np.random.seed(FLAGS.seed)
    env = make_environment(FLAGS.env)
    environment_spec = specs.make_environment_spec(env)
    env.seed(FLAGS.seed)

    agent_networks = make_networks(environment_spec)
    data = d4rl.qlearning_dataset(env)
    if FLAGS.normalize:
        data, mean, std = d4rl_dataset.normalize_obs(data)
    else:
        mean, std = 0, 1
    data_iterator = d4rl_dataset.make_tf_data_iterator(
        data, batch_size=FLAGS.batch_size
    ).as_numpy_iterator()
    learner_key, actor_key = jax.random.split(jax.random.PRNGKey(FLAGS.seed))
    learner = learning.CRRLearner(
        environment_spec=environment_spec,
        policy_network=agent_networks["policy"],
        critic_network=agent_networks["critic"],
        random_key=learner_key,
        dataset=data_iterator,
        discount=FLAGS.discount,
        logger=loggers.make_logger(
            "learner",
            log_frequency=FLAGS.log_freq,
            use_wandb=FLAGS.wandb,
            wandb_kwargs={"config": FLAGS},
        ),
    )

    evaluator = make_actor(
        environment_spec.actions,
        agent_networks["policy"],
        learner,
        actor_key,
    )
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
