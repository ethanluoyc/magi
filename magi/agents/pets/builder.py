from acme import specs
from acme.jax import variable_utils
import jax
import numpy as np
import optax
import tensorflow_probability as tfp

from magi.agents.pets import acting
from magi.agents.pets import models
from magi.agents.pets.agent import ModelBasedAgent
from magi.agents.pets.learning import ModelBasedLearner
from magi.agents.pets.replay import ReplayBuffer

tfd = tfp.experimental.substrates.jax.distributions


def make_network(
    environment_spec,
    hidden_sizes,
    activation=jax.nn.swish,
    min_logvar=-10.0,
    max_logvar=0.5,
):
    output_size = environment_spec.observations.shape[-1]

    def network(network_input):
        model = models.GaussianMLP(
            output_size,
            hidden_sizes=hidden_sizes,
            activation=activation,
            min_logvar=min_logvar,
            max_logvar=max_logvar,
        )
        return model(network_input)

    return network


def make_agent(
    environment_spec: specs.EnvironmentSpec,
    cost_fn,
    terminal_fn,
    obs_preprocess,
    obs_postprocess,
    target_process,
    # Model and Learner
    *,
    hidden_sizes=(200, 200, 200),
    activation=jax.nn.swish,
    min_logvar=-10.0,
    max_logvar=0.5,
    num_ensembles=5,
    batch_size=32,
    planning_horizon=25,
    lr=1e-3,
    num_epochs=100,
    seed=1,
    min_delta=0.1,
    patience=5,
    val_ratio=0,
    weight_decay=1e-5,
    # Actor
    optimizer="cem",
    population_size=100,
    cem_iterations=5,
    cem_elite_frac=0.1,
    cem_alpha=0.1,
    cem_return_mean_elites=True,
    num_particles=20,
    replay_capacity=int(1e6),
    logger=None,
    counter=None,
):
    rng = jax.random.PRNGKey(seed)
    actor_rng, learner_rng, replay_rng = jax.random.split(rng, 3)
    replay = ReplayBuffer(
        replay_capacity,
        obs_shape=environment_spec.observations.shape,
        action_shape=environment_spec.actions.shape,
        rng=np.random.default_rng(
            int(jax.random.randint(replay_rng, (), 0, 2 ** 31 - 1))
        ),
    )
    network = make_network(
        environment_spec,
        hidden_sizes,
        activation=activation,
        min_logvar=min_logvar,
        max_logvar=max_logvar,
    )

    # Create an ensemble model
    model = models.EnsembleModel(
        network, obs_preprocess, obs_postprocess, target_process, num_ensembles
    )
    opt = optax.chain(
        optax.add_decayed_weights(weight_decay),
        optax.scale_by_adam(),
        optax.scale(-lr),
    )

    # Create a learner
    learner = ModelBasedLearner(
        environment_spec,
        model,
        replay,
        opt,
        batch_size=batch_size,
        num_epochs=num_epochs,
        seed=learner_rng,
        min_delta=min_delta,
        val_ratio=val_ratio,
        patience=patience,
        logger=logger,
        counter=counter,
    )

    # Create actor
    model_env = models.ModelEnv(model, cost_fn, terminal_fn)
    variable_client = variable_utils.VariableClient(learner, "")

    if optimizer == "cem":
        actor = acting.CEMOptimizerActor(
            environment_spec,
            model_env,
            replay,
            variable_client,
            population_size=population_size,
            planning_horizon=planning_horizon,
            n_iterations=cem_iterations,
            elite_frac=cem_elite_frac,
            alpha=cem_alpha,
            return_mean_elites=cem_return_mean_elites,
            num_particles=num_particles,
            seed=actor_rng,
        )
    elif optimizer == "random":
        actor = acting.RandomOptimizerActor(
            environment_spec,
            model_env,
            replay,
            variable_client,
            num_samples=population_size,
            planning_horizon=planning_horizon,
            num_particles=num_particles,
            seed=actor_rng,
        )

    agent = ModelBasedAgent(actor, learner)
    return agent
