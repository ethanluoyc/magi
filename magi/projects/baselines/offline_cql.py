"""Run Offline CQL learning."""
from acme import specs
from acme.agents.jax import actor_core
from acme.agents.jax import actors
from acme.jax import variable_utils
from acme.utils import counting
import jax
import ml_collections
import optax

from magi.agents import cql
from magi.agents import sac
from magi.projects.baselines import experiment
from magi.projects.baselines import logger_utils


def main(config: ml_collections.ConfigDict, workdir: str):
  # Create dataset and environment
  environment, dataset = experiment.make_env_and_dataset(
      config.env_name, config.seed, config.batch_size)

  # Setup learner and evaluator
  spec = specs.make_environment_spec(environment)
  agent_networks = cql.make_default_networks(
      spec,
      policy_layer_sizes=config.policy_dims,
      critic_layer_sizes=config.critic_dims,
  )
  random_key = jax.random.PRNGKey(config.seed)
  learner_key, evaluator_key = jax.random.split(random_key)
  counter = counting.Counter(time_delta=0)
  eval_logger = logger_utils.make_default_logger(
      workdir,
      'evaluation',
      save_data=True,
      log_to_wandb=config.log_to_wandb,
  )
  learner = cql.CQLLearner(
      agent_networks['policy'],
      agent_networks['critic'],
      learner_key,
      dataset,
      policy_optimizer=optax.adam(config.actor_lr),
      critic_optimizer=optax.adam(config.critic_lr),
      alpha_optimizer=optax.adam(config.critic_lr),
      target_entropy=sac.target_entropy_from_env_spec(spec),
      discount=config.discount,
      tau=config.tau,
      init_alpha=config.init_alpha,
      num_bc_steps=config.num_bc_steps,
      softmax_temperature=config.softmax_temperature,
      cql_alpha=config.cql_alpha,
      max_q_backup=config.max_q_backup,
      deterministic_backup=config.deterministic_backup,
      num_cql_samples=config.num_cql_samples,
      with_lagrange=config.with_lagrange,
      target_action_gap=config.target_action_gap,
      counter=counting.Counter(counter, prefix='learner', time_delta=0),
      logger=logger_utils.make_default_logger(
          workdir,
          'learner',
          time_delta=5.0,
          save_data=True,
          log_to_wandb=config.log_to_wandb,
      ),
  )

  evaluator_network = sac.apply_policy_sample(agent_networks, eval_mode=True)
  evaluator = actors.GenericActor(
      actor_core.batched_feed_forward_to_actor_core(evaluator_network),
      random_key=evaluator_key,
      variable_client=variable_utils.VariableClient(learner, 'policy'),
      backend=None,
  )

  # Run training loop
  assert config.num_steps % config.eval_interval == 0
  for _ in range(config.num_steps // config.eval_interval):
    for _ in range(config.eval_interval):
      learner.step()
    normalized_score = experiment.evaluate(evaluator, environment,
                                           config.eval_episodes)
    counts = counter.increment(steps=config.eval_interval)
    eval_logger.write({'normalized_score': normalized_score, **counts})


if __name__ == '__main__':
  experiment.run(main)
