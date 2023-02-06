"""Run Offline IQL learning."""
import experiment
import jax
import logger_utils
import ml_collections
import optax
from acme import specs
from acme.agents.jax import actor_core
from acme.agents.jax import actors
from acme.jax import variable_utils
from acme.utils import counting

from magi.agents import iql


def main(config: ml_collections.ConfigDict, workdir: str):
    # Create dataset and environment
    environment, dataset = experiment.make_env_and_dataset(
        config.env_name, config.seed, config.batch_size
    )

    # Setup learner and evaluator
    spec = specs.make_environment_spec(environment)
    agent_networks = iql.make_networks(
        spec,
        hidden_dims=config.hidden_dims,
        dropout_rate=config.dropout_rate,
    )
    random_key = jax.random.PRNGKey(config.seed)
    learner_key, evaluator_key = jax.random.split(random_key)
    counter = counting.Counter(time_delta=0)
    eval_logger = logger_utils.make_default_logger(
        workdir,
        "evaluation",
        save_data=True,
        log_to_wandb=config.log_to_wandb,
    )
    learner = iql.IQLLearner(
        learner_key,
        agent_networks,
        dataset,
        # Use cosine schedule as in the original implementation
        policy_optimizer=optax.chain(
            optax.scale_by_adam(),
            optax.scale_by_schedule(
                optax.cosine_decay_schedule(-config.actor_lr, config.num_steps)
            ),
        ),
        critic_optimizer=optax.adam(config.critic_lr),
        value_optimizer=optax.adam(config.value_lr),
        discount=config.discount,
        tau=config.tau,
        expectile=config.expectile,
        temperature=config.temperature,
        counter=counting.Counter(counter, prefix="learner", time_delta=0),
        logger=logger_utils.make_default_logger(
            workdir,
            "learner",
            time_delta=5.0,
            save_data=True,
            log_to_wandb=config.log_to_wandb,
        ),
    )

    evaluator_network = iql.apply_policy_and_sample(
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
        normalized_score = experiment.evaluate(
            evaluator, environment, config.eval_episodes
        )
        counts = counter.increment(steps=config.eval_interval)
        eval_logger.write({"normalized_score": normalized_score, **counts})


if __name__ == "__main__":
    experiment.run(main)
