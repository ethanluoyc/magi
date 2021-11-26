"""Run Offline IQL learning."""
from typing import Sequence

from acme import specs
from acme.agents.jax import actor_core
from acme.agents.jax import actors
from acme.jax import networks as networks_lib
from acme.jax import utils as jax_utils
from acme.jax import variable_utils
from acme.jax.networks import distributional
from acme.utils import counting
import haiku as hk
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import tensorflow_probability

from magi.agents import cql
from magi.agents import sac
from magi.projects.baselines import experiment
from magi.projects.baselines import logger_utils

hk_init = hk.initializers
tfp = tensorflow_probability.substrates.jax
tfd = tfp.distributions

Initializer = hk.initializers.Initializer


class NormalTanhDistribution(hk.Module):
    """Module that produces a TanhTransformedDistribution distribution."""

    def __init__(
        self,
        num_dimensions: int,
        min_scale: float = 1e-3,
        w_init: hk_init.Initializer = hk_init.VarianceScaling(1.0, "fan_in", "uniform"),
        b_init: hk_init.Initializer = hk_init.Constant(0.0),
    ):
        """Initialization.

        Args:
          num_dimensions: Number of dimensions of a distribution.
          min_scale: Minimum standard deviation.
          w_init: Initialization for linear layer weights.
          b_init: Initialization for linear layer biases.
        """
        super().__init__(name="Normal")
        self._min_scale = min_scale
        self._loc_layer = hk.Linear(num_dimensions, w_init=w_init, b_init=b_init)
        self._scale_layer = hk.Linear(num_dimensions, w_init=w_init, b_init=b_init)

    def __call__(self, inputs: jnp.ndarray) -> tfd.Distribution:
        loc = self._loc_layer(inputs)
        scale = self._scale_layer(inputs)
        scale = jnp.exp(jnp.clip(scale, -20.0, 2.0))
        distribution = tfd.Normal(loc=loc, scale=scale)
        return tfd.Independent(
            distributional.TanhTransformedDistribution(distribution),
            reinterpreted_batch_ndims=1,
        )


def make_networks(
    spec: specs.EnvironmentSpec,
    policy_layer_sizes: Sequence[int] = (256, 256),
    critic_layer_sizes: Sequence[int] = (256, 256),
):
    num_dimensions = np.prod(spec.actions.shape, dtype=int)

    def _actor_fn(obs):
        network = hk.Sequential(
            [
                hk.nets.MLP(
                    list(policy_layer_sizes),
                    w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "uniform"),
                    activation=jax.nn.relu,
                    activate_final=True,
                ),
                NormalTanhDistribution(num_dimensions),
            ]
        )
        return network(obs)

    def _critic_fn(obs, action):
        network1 = hk.nets.MLP(
            list(critic_layer_sizes) + [1],
            w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "uniform"),
            activation=jax.nn.relu,
            activate_final=False,
        )
        network2 = hk.nets.MLP(
            list(critic_layer_sizes) + [1],
            w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "uniform"),
            activation=jax.nn.relu,
            activate_final=False,
        )
        input_ = jnp.concatenate([obs, action], axis=-1)
        value1 = network1(input_)
        value2 = network2(input_)
        return jnp.squeeze(value1, axis=-1), jnp.squeeze(value2, axis=-1)

    dummy_action = jax_utils.zeros_like(spec.actions)
    dummy_obs = jax_utils.zeros_like(spec.observations)
    dummy_action = jax_utils.add_batch_dim(dummy_action)
    dummy_obs = jax_utils.add_batch_dim(dummy_obs)

    policy = hk.without_apply_rng(hk.transform(_actor_fn, apply_rng=True))
    critic = hk.without_apply_rng(hk.transform(_critic_fn, apply_rng=True))

    return {
        "policy": networks_lib.FeedForwardNetwork(
            lambda key: policy.init(key, dummy_obs), policy.apply
        ),
        "critic": networks_lib.FeedForwardNetwork(
            lambda key: critic.init(key, dummy_obs, dummy_action), critic.apply
        ),
    }


def main(config: ml_collections.ConfigDict, workdir: str):
    # Create dataset and environment
    environment, dataset = experiment.make_env_and_dataset(
        config.env_name, config.seed, config.batch_size
    )

    # Setup learner and evaluator
    spec = specs.make_environment_spec(environment)
    agent_networks = make_networks(
        spec,
        policy_layer_sizes=config.policy_dims,
        critic_layer_sizes=config.critic_dims,
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
        num_bc_steps=config.num_bc_steps,
        softmax_temperature=config.softmax_temperature,
        cql_alpha=config.cql_alpha,
        max_q_backup=config.max_q_backup,
        deterministic_backup=config.deterministic_backup,
        num_cql_samples=config.num_cql_samples,
        with_lagrange=config.with_lagrange,
        target_action_gap=config.target_action_gap,
        counter=counting.Counter(counter, prefix="learner", time_delta=0),
        logger=logger_utils.make_default_logger(
            workdir,
            "learner",
            time_delta=5.0,
            save_data=True,
            log_to_wandb=config.log_to_wandb,
        ),
    )

    evaluator_network = sac.apply_policy_sample(agent_networks, eval_mode=True)
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
