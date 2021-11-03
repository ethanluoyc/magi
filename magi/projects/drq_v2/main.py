"""Entry point for the DrQ-v2 dm_control baseline"""
import acme
from acme import specs
import jax
import jax.numpy as jnp
import ml_collections

from magi.projects.drq_v2 import agents
from magi.projects.drq_v2 import app
from magi.projects.drq_v2 import utils


def main(rng: jnp.ndarray, config: ml_collections.ConfigDict, workdir: str):
    train_key, eval_key, agent_key = jax.random.split(rng, 3)
    environment = utils.make_environment(
        config.domain_name,
        config.task_name,
        train_key,
        config.frame_stack,
        config.action_repeat,
    )
    eval_environment = utils.make_environment(
        config.domain_name,
        config.task_name,
        eval_key,
        config.frame_stack,
        config.action_repeat,
    )
    spec = specs.make_environment_spec(environment)
    agent_logger = utils.make_default_logger(
        workdir=workdir,
        label="learner",
        time_delta=config.learner_log_time,
        log_to_wandb=config.log_to_wandb,
    )
    # TODO(yl): this is the extension point for plugging in different agents
    agent, eval_actor = agents.drq_v2_agent(config, agent_key, spec, agent_logger)

    train_logger = utils.make_default_logger(
        workdir=workdir,
        label="train_loop",
        time_delta=config.train_log_time,
        log_to_wandb=config.log_to_wandb,
    )
    train_loop = acme.EnvironmentLoop(environment, agent, logger=train_logger)
    eval_logger = utils.make_default_logger(
        workdir=workdir,
        label="eval_loop",
        time_delta=config.eval_log_time,
        log_to_wandb=config.log_to_wandb,
    )
    eval_loop = acme.EnvironmentLoop(eval_environment, eval_actor, logger=eval_logger)
    num_steps = config.num_frames // config.action_repeat
    assert (
        config.num_frames % config.action_repeat == 0
    ), "Number of environment frames must be divisible by action repeat"
    for _ in range(num_steps // config.eval_freq):
        train_loop.run(num_steps=config.eval_freq)
        eval_actor.update(wait=True)
        eval_loop.run(num_episodes=config.eval_episodes)


if __name__ == "__main__":
    app.run(main)
