"""Run Behavior-Cloning (BC) baseline on D4RL."""

from absl import app
from absl import flags
from absl import logging
from acme import specs
from acme import types
from acme import wrappers
from acme.agents.jax import actors as acting_lib
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.jax import variable_utils
import d4rl  # type: ignore
import gym
import haiku as hk
import jax
import numpy as np
import optax
import tensorflow as tf
import wandb

from magi.agents import bc
from magi.examples.offline import d4rl_dataset

FLAGS = flags.FLAGS
flags.DEFINE_string("policy", "TD3_BC", "Policy name")
flags.DEFINE_string("env", "hopper-medium-v0", "OpenAI gym environment name")
flags.DEFINE_integer("seed", 0, "seed")
flags.DEFINE_integer("eval_freq", int(5e3), "evaluation frequency")
flags.DEFINE_integer("max_timesteps", int(1e6), "maximum number of steps")
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
    logging.info(
        "Evaluation over %d episodes: %.3f (unormalized %.3f)",
        eval_episodes,
        d4rl_score,
        avg_reward,
    )
    logging.info("---------------------------------------")
    return d4rl_score


def make_environment(name):
    environment = gym.make(name)
    environment = wrappers.GymWrapper(environment)
    return wrappers.SinglePrecisionWrapper(environment)


def make_actor(policy_network, variable_source, random_key):
    variable_client = variable_utils.VariableClient(
        variable_source, "policy", device="cpu"
    )
    variable_client.update_and_wait()

    def make_policy(network):
        def policy(params, key, obs):
            del key
            return network.apply(params, obs).mode()  # TODO: Support stochastic eval

        return policy

    return acting_lib.FeedForwardActor(
        make_policy(policy_network),
        random_key,
        variable_client=variable_client,
    )


def make_network(
    environment_spec: specs.EnvironmentSpec,
    policy_layer_sizes=(256, 256),
):
    """Make default networks used by TD3."""
    action_size = np.prod(environment_spec.actions.shape, dtype=int)

    def _policy_fn(obs, is_training=False, key=None):
        del is_training
        del key
        return hk.Sequential(
            [
                hk.nets.MLP(policy_layer_sizes, activate_final=True),
                networks_lib.NormalTanhDistribution(action_size),
            ]
        )(obs)

    policy = hk.without_apply_rng(hk.transform(_policy_fn, apply_rng=True))
    dummy_obs = utils.zeros_like(environment_spec.observations)
    dummy_obs = utils.add_batch_dim(dummy_obs)
    network = networks_lib.FeedForwardNetwork(
        lambda key: policy.init(key, dummy_obs), policy.apply
    )
    return network


def main(_):
    # Disable TF GPU
    tf.config.set_visible_devices([], "GPU")
    if FLAGS.wandb:
        wandb.init(project="magi", entity="ethanluoyc", name="bc")
    logging.info("---------------------------------------")
    logging.info("Policy: %s, Env: %s, Seed: %s", FLAGS.policy, FLAGS.env, FLAGS.seed)
    logging.info("---------------------------------------")

    np.random.seed(FLAGS.seed)
    env = make_environment(FLAGS.env)
    environment_spec = specs.make_environment_spec(env)
    env.seed(FLAGS.seed)

    policy_network = make_network(environment_spec)
    data = d4rl.qlearning_dataset(env)
    if FLAGS.normalize:
        data, mean, std = d4rl_dataset.normalize_obs(data)
    else:
        mean, std = 0, 1
    # data_iterator = d4rl_dataset.make_tf_data_iterator(
    #     data, batch_size=FLAGS.batch_size
    # ).as_numpy_iterator()
    # def _wrap_iterator(data_iterator):
    #     for d in data_iterator:
    #         yield d.data
    # data_iterator = _wrap_iterator(data_iterator)

    def get_iterator(data, batch_size):
        # Unlike the data iterator above,
        # this does not sample a mini-batch uniformly from the offline dataset.
        ds = tf.data.Dataset.from_tensor_slices(
            types.Transition(
                observation=data["observations"],
                action=data["actions"],
                reward=data["rewards"],
                discount=1 - tf.cast(data["terminals"], tf.float32),
                next_observation=data["next_observations"],
                extras=(),
            )
        )
        return (
            ds.repeat()
            .shuffle(len(data["rewards"]))
            .batch(batch_size, drop_remainder=True)
        )

    data_iterator = get_iterator(data, FLAGS.batch_size)
    data_iterator = data_iterator.as_numpy_iterator()
    random_key = jax.random.PRNGKey(FLAGS.seed)
    learner_key, actor_key = jax.random.split(random_key)
    loss_fn = bc.logp(
        lambda dist_params, actions: dist_params.log_prob(actions)
    )  # noqa
    learner = bc.BCLearner(
        policy_network,
        random_key=learner_key,
        loss_fn=loss_fn,
        optimizer=optax.adam(0.005),
        demonstrations=data_iterator,
        num_sgd_steps_per_step=1,
    )

    evaluator = make_actor(policy_network, learner, actor_key)
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
