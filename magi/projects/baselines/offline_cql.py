"""Run Offline IQL learning."""
import os
from typing import Tuple

from absl import app
from absl import flags
from absl import logging
from acme import specs
from acme import wrappers
from acme.agents.jax import actor_core
from acme.agents.jax import actors
from acme.jax import variable_utils
from acme.utils import counting
import gym
import jax
import ml_collections
from ml_collections import config_flags
import numpy as np
import optax
import yaml

from magi.agents import cql
from magi.agents import sac
from magi.projects.baselines import dataset_utils
from magi.projects.baselines import logger_utils

config_flags.DEFINE_config_file("config", default="configs/cql_antmaze_offline.py")
flags.DEFINE_string("workdir", "./tmp", "Where to save results")
flags.DEFINE_bool("log_to_wandb", False, "If true, log to WandB")

FLAGS = flags.FLAGS


def evaluate(actor, environment, eval_episodes=10):
    actor.update(wait=True)
    avg_reward = 0.0
    for _ in range(eval_episodes):
        timestep = environment.reset()
        actor.observe_first(timestep)
        while not timestep.last():
            action = actor.select_action(timestep.observation)
            timestep = environment.step(action)
            actor.observe(action, timestep)
            avg_reward += timestep.reward

    avg_reward /= eval_episodes
    d4rl_score = environment.get_normalized_score(avg_reward)

    logging.info("---------------------------------------")
    logging.info("Evaluation over %d episodes: %.3f", eval_episodes, d4rl_score)
    logging.info("---------------------------------------")
    return d4rl_score


def normalize(dataset):

    trajs = dataset_utils.split_into_trajectories(
        dataset.observations,
        dataset.actions,
        dataset.rewards,
        dataset.masks,
        dataset.dones_float,
        dataset.next_observations,
    )

    def compute_returns(traj):
        episode_return = 0
        for _, _, rew, _, _, _ in traj:
            episode_return += rew

        return episode_return

    trajs.sort(key=compute_returns)

    dataset.rewards /= compute_returns(trajs[-1]) - compute_returns(trajs[0])
    dataset.rewards *= 1000.0


def _make_dataset_iterator(dataset, batch_size: int):
    while True:
        batch = dataset.sample(batch_size)
        yield batch


def make_env_and_dataset(
    env_name: str, seed: int, batch_size: int
) -> Tuple[gym.Env, dataset_utils.D4RLDataset]:
    env = gym.make(env_name)
    env.seed(seed)
    env = wrappers.wrap_all(
        env,
        [
            wrappers.GymWrapper,
            wrappers.SinglePrecisionWrapper,
        ],
    )

    dataset = dataset_utils.D4RLDataset(env)

    if "antmaze" in env_name:
        dataset.rewards -= (dataset.rewards - 0.5) * 4.0
    elif "halfcheetah" in env_name or "walker2d" in env_name or "hopper" in env_name:
        normalize(dataset)

    return env, _make_dataset_iterator(dataset, batch_size)


def main(_):
    config: ml_collections.ConfigDict = FLAGS.config
    workdir = FLAGS.workdir
    log_to_wandb = FLAGS.log_to_wandb
    # Fix global random seed
    np.random.seed(config.seed)

    # Create working directory
    os.makedirs(workdir, exist_ok=True)

    # Save configuration
    with open(os.path.join(workdir, "config.yaml"), "wt") as f:
        yaml.dump(config.to_dict(), f)

    # Create dataset and environment
    environment, dataset = make_env_and_dataset(
        config.env_name, config.seed, config.batch_size
    )

    # Setup learner and evaluator
    spec = specs.make_environment_spec(environment)
    agent_networks = sac.make_networks(
        spec,
        policy_layer_sizes=config.hidden_dims,
        critic_layer_sizes=config.hidden_dims,
    )
    random_key = jax.random.PRNGKey(config.seed)
    learner_key, evaluator_key = jax.random.split(random_key)
    counter = counting.Counter(time_delta=0)
    eval_logger = logger_utils.make_default_logger(
        workdir,
        "evaluation",
        save_data=True,
        log_to_wandb=log_to_wandb,
    )
    learner = cql.CQLLearner(
        agent_networks["policy"],
        agent_networks["critic"],
        learner_key,
        dataset,
        policy_optimizer=optax.adam(config.actor_lr),
        critic_optimizer=optax.adam(config.critic_lr),
        alpha_optimizer=optax.adam(config.critic_lr),
        target_entropy=sac.target_entropy_from_env_spec(spec),
        discount=config.discount,
        tau=config.tau,
        init_alpha=config.init_alpha,
        policy_eval_start=config.policy_eval_start,
        temp=config.temp,
        init_alpha_prime=config.init_alpha_prime,
        max_q_backup=config.max_q_backup,
        deterministic_backup=config.deterministic_backup,
        num_random=config.num_random,
        with_lagrange=config.with_lagrange,
        lagrange_thresh=config.lagrange_thresh,
        counter=counting.Counter(counter, prefix="learner", time_delta=0),
        logger=logger_utils.make_default_logger(
            workdir,
            "learner",
            time_delta=5.0,
            save_data=True,
            log_to_wandb=log_to_wandb,
        ),
    )

    evaluator_network = sac.apply_policy_and_sample(
        agent_networks, spec.actions, eval_mode=True
    )
    evaluator = actors.GenericActor(
        actor_core.batched_feed_forward_to_actor_core(evaluator_network),
        random_key=evaluator_key,
        variable_client=variable_utils.VariableClient(learner, "policy"),
        backend=None,
    )

    # Run training loop
    assert config.num_steps % config.eval_interval == 0
    for _ in range(config.num_steps // config.eval_interval):
        for _ in range(config.eval_interval):
            learner.step()
        normalized_score = evaluate(evaluator, environment, config.eval_episodes)
        counts = counter.increment(steps=config.eval_interval)
        eval_logger.write({"normalized_score": normalized_score, **counts})


if __name__ == "__main__":
    app.run(main)
