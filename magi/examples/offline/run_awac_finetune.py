"""Run AWAC D4RL fine-tuning experiment."""

from absl import app
from absl import flags
from absl import logging
import acme
from acme import specs
from acme import types
from acme import wrappers
from acme.adders import reverb as adders_reverb
from acme.agents import agent as agent_lib
from acme.utils import counting
from acme.utils import loggers
import d4rl  # type: ignore
import gym
import haiku as hk
import jax
import numpy as np
import reverb
import tensorflow as tf
import wandb

from magi.agents import awac
from magi.agents.awac import networks as network_lib

FLAGS = flags.FLAGS
flags.DEFINE_string("policy", "AWAC", "Policy name")
flags.DEFINE_string("env", "hopper-medium-v0", "OpenAI gym environment name")
flags.DEFINE_integer("seed", 0, "seed")
flags.DEFINE_integer("eval_freq", int(5e3), "evaluation frequency")
flags.DEFINE_integer("num_offline_steps", int(1e6), "maximum number of steps")
flags.DEFINE_integer("num_steps", int(500_000), "maximum number of steps")
flags.DEFINE_integer("replay_size", int(1e6), "maximum number of steps")
flags.DEFINE_integer("batch_size", 256, "batch size")
flags.DEFINE_float("discount", 0.99, "discount")
flags.DEFINE_float("tau", 0.005, "target network update rate")
flags.DEFINE_integer("target_update_period", 2, "Frequency of delayed policy updates")
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


def make_networks(environment_spec: specs.EnvironmentSpec):
    action_dim = environment_spec.actions.shape[0]
    policy = hk.without_apply_rng(
        hk.transform(lambda obs: network_lib.Actor(action_dim)(obs))
    )
    critic = hk.without_apply_rng(
        hk.transform(lambda obs, a: network_lib.DoubleCritic()(obs, a))
    )
    return {"policy": policy, "critic": critic}


def make_environment(name):
    """Create an instance of environment"""
    environment = gym.make(name)
    environment = wrappers.GymWrapper(environment)
    return wrappers.SinglePrecisionWrapper(environment)


def make_policy(policy_network, is_eval=True):
    """Make policy function from network definition."""

    def policy(params, key, obs):
        action_dist = policy_network.apply(params, obs)
        if is_eval:
            return action_dist.sample(seed=key)
        return action_dist.mode()

    return policy


def load_dataset_into_reverb(server, data):
    """Load offline dataset into reverb"""
    logging.info("Populating reverb with offline data")
    replay_client = reverb.Client(f"localhost:{server.port}")
    total_size = data["rewards"].shape[0]
    with replay_client.trajectory_writer(num_keep_alive_refs=2) as writer:
        for i in range(total_size):
            blob = types.Transition(
                observation=data["observations"][i].astype(np.float32),
                action=data["actions"][i].astype(np.float32),
                reward=data["rewards"][i].astype(np.float32),
                discount=(1.0 - data["terminals"][i]).astype(np.float32),
                next_observation=data["next_observations"][i].astype(np.float32),
            )
            writer.append(blob)
            item = types.Transition(
                observation=writer.history.observation[-1],
                action=writer.history.action[-1],
                reward=writer.history.reward[-1],
                discount=writer.history.discount[-1],
                next_observation=writer.history.next_observation[-1],
            )
            writer.create_item(
                table=adders_reverb.DEFAULT_PRIORITY_TABLE,
                priority=1.0,
                trajectory=item,
            )


def main(_):
    # Disable TF GPU
    tf.config.set_visible_devices([], "GPU")
    if FLAGS.wandb:
        wandb.init(project="magi", entity="ethanluoyc", name="awac")
    logging.info("---------------------------------------")
    logging.info("Policy: %s, Env: %s, Seed: %s", FLAGS.policy, FLAGS.env, FLAGS.seed)
    logging.info("---------------------------------------")

    np.random.seed(FLAGS.seed)
    env = make_environment(FLAGS.env)
    environment_spec = specs.make_environment_spec(env)
    assert (environment_spec.actions.maximum == 1.0).all()
    assert (environment_spec.actions.minimum == -1.0).all()
    env.seed(FLAGS.seed)

    agent_networks = make_networks(environment_spec)
    data = d4rl.qlearning_dataset(env)
    logging.info("Offline dataset size is %d", data["rewards"].shape[0])
    random_key = jax.random.PRNGKey(FLAGS.seed)
    builder = awac.AWACBuilder(
        config=awac.AWACConfig(
            discount=FLAGS.discount,
            tau=FLAGS.tau,
            target_update_period=FLAGS.target_update_period,
            batch_size=FLAGS.batch_size,
            max_replay_size=max(FLAGS.replay_size, data["rewards"].shape[0]),
        )
    )
    # Load offline dataset into Reverb
    replay_tables = builder.make_replay_tables(environment_spec)
    replay_server = reverb.Server(replay_tables, port=None)
    load_dataset_into_reverb(replay_server, data)

    # Offline training
    logging.info("Start offline training")
    replay_client = reverb.Client(f"localhost:{replay_server.port}")
    data_iterator = builder.make_dataset_iterator(replay_client)
    learner_key, actor_key = jax.random.split(random_key)
    learner = builder.make_learner(
        environment_spec, learner_key, agent_networks, data_iterator
    )
    evaluator = builder.make_actor(
        random_key=actor_key,
        policy_network=make_policy(agent_networks["policy"], is_eval=True),
        variable_source=learner,
    )
    evaluations = []
    for t in range(int(FLAGS.num_offline_steps)):
        learner.step()
        # Evaluate episode
        if (t + 1) % FLAGS.eval_freq == 0:
            logging.info("Offline time steps: %d", t + 1)
            evaluations.append(evaluate(evaluator, FLAGS.env, FLAGS.seed, 0.0, 1.0))
            if FLAGS.wandb:
                wandb.log({"offline/step": t, "offline/eval_returns": evaluations[-1]})
    # Online fine-tuning
    logging.info("Start online fine-tuning")
    actor = builder.make_actor(
        random_key,
        make_policy(agent_networks["policy"], is_eval=False),
        adder=builder.make_adder(replay_client),
        variable_source=learner,
    )
    online_agent = agent_lib.Agent(
        actor,
        learner,
        min_observations=0,
        observations_per_step=1,
    )

    counter = counter = counting.Counter()
    train_loop = acme.EnvironmentLoop(
        env, online_agent, logger=loggers.NoOpLogger(), counter=counter
    )
    for t in range(int(FLAGS.num_steps) // FLAGS.eval_freq):
        train_loop.run(num_steps=int(FLAGS.num_steps) // FLAGS.eval_freq)
        logging.info(
            "Offline epoch: %d, steps: %d", t + 1, counter.get_counts()["steps"]
        )
        evaluations.append(evaluate(evaluator, FLAGS.env, FLAGS.seed, 0.0, 1.0))
        if FLAGS.wandb:
            wandb.log(
                {
                    "online/epoch": t,
                    "online/eval_returns": evaluations[-1],
                    **counter.get_counts(),
                }
            )


if __name__ == "__main__":
    FLAGS.logtostderr = True
    app.run(main)
