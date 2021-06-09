from acme.jax import variable_utils
from acme import specs
import haiku as hk
import jax
import jax.numpy as jnp
import tensorflow_probability as tfp
import tree

from magi.agents.pets import acting
from magi.agents.pets.agent import ModelBasedAgent
from magi.agents.pets.dataset import Dataset
from magi.agents.pets.learning import ModelBasedLearner
from magi.agents.pets import models

tfd = tfp.experimental.substrates.jax.distributions


def make_network(environment_spec,
                 hidden_sizes,
                 activation=jax.nn.swish,
                 min_logvar=-10.,
                 max_logvar=0.5):
  output_size = environment_spec.observations.shape[-1]

  def network(x, a):
    input_ = jnp.concatenate([x, a], axis=-1)
    model = models.GaussianMLP(
        output_size,
        hidden_sizes=hidden_sizes,
        activation=activation,
        min_logvar=min_logvar,
        max_logvar=max_logvar,
    )
    return model(input_)

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
    min_logvar=-10.,
    max_logvar=0.5,
    num_ensembles=5,
    batch_size=32,
    time_horizon=25,
    lr=1e-3,
    num_epochs=100,
    seed=1,
    min_delta=0.1,
    patience=5,
    val_ratio=0,
    weight_decay=1e-5,
    # Actor
    optimizer='cem',
    population_size=100,
    cem_iterations=5,
    cem_elite_frac=0.1,
    cem_alpha=0.1,
    cem_return_mean_elites=True,
    num_particles=20,
    logger=None,
    counter=None,
):

  dataset = Dataset()
  network = make_network(environment_spec,
                         hidden_sizes,
                         activation=activation,
                         min_logvar=min_logvar,
                         max_logvar=max_logvar)
  # Create a learner

  transformed = hk.without_apply_rng(hk.transform(network))

  def model_init_fn(rng, x, a):
    vmapped_init_fn = jax.vmap(transformed.init, (0, None, None))
    ensem_params = vmapped_init_fn(jax.random.split(rng, num_ensembles),
                                   obs_preprocess(x), a)
    return ensem_params

  loss_fn, evaluate_fn = make_loss_and_eval_fn(transformed.apply,
                                               obs_preprocess,
                                               target_process,
                                               weight_decay=weight_decay)

  learner = ModelBasedLearner(
      environment_spec,
      model_init_fn,
      loss_fn,
      evaluate_fn,
      dataset,
      lr=lr,
      batch_size=batch_size,
      num_epochs=num_epochs,
      seed=seed + 1000,
      min_delta=min_delta,
      val_ratio=val_ratio,
      patience=patience,
      logger=logger,
      counter=counter,
  )

  forward_fn = make_forward_fn(network, obs_preprocess, obs_postprocess)
  # Create actor
  variable_client = variable_utils.VariableClient(learner, '')

  if optimizer == 'cem':
    actor = acting.CEMOptimizerActor(
        environment_spec,
        forward_fn,
        cost_fn,
        terminal_fn,
        dataset,
        variable_client,
        pop_size=population_size,
        time_horizon=time_horizon,
        n_iterations=cem_iterations,
        elite_frac=cem_elite_frac,
        alpha=cem_alpha,
        return_mean_elites=cem_return_mean_elites,
        num_particles=num_particles,
        seed=seed + 1001,
    )
  elif optimizer == 'random':
    actor = acting.RandomOptimizerActor(environment_spec,
                                        forward_fn,
                                        cost_fn,
                                        terminal_fn,
                                        dataset,
                                        variable_client,
                                        num_samples=population_size,
                                        time_horizon=time_horizon,
                                        num_particles=num_particles,
                                        seed=seed + 1001)

  agent = ModelBasedAgent(actor, learner)
  return agent


def make_forward_fn(network, obs_preprocess, obs_postprocess, shuffle=True):
  # NOTE: this currently corresponds to the TSInf formulation in PETS
  # TODO: Add the other options
  transformed = hk.without_apply_rng(hk.transform(network))

  def init_fn(ensem_params, key, observations):
    num_ensembles = tree.flatten(ensem_params)[0].shape[0]
    batch_size = observations.shape[0]
    if batch_size % num_ensembles:
      raise NotImplementedError('Ragged batch not supported.')
    indices = jnp.arange(batch_size)
    if shuffle:
      indices = jax.random.permutation(key, indices)
    return indices

  def forward_fn(ensem_params, state, key, observations, actions):
    # TODO(yl): Add sampling propagation indices instead of being fully deterministic
    shuffle_indices = state
    observations = observations[shuffle_indices]
    actions = actions[shuffle_indices]

    states = obs_preprocess(observations)
    num_ensembles = tree.flatten(ensem_params)[0].shape[0]
    batch_size = states.shape[0]
    new_batch_size, ragged = divmod(batch_size, num_ensembles)
    if ragged:
      raise NotImplementedError(
          f'Ragged batch not supported. ({batch_size} % {num_ensembles} == {ragged})')
    reshaped_states, reshaped_act = tree.map_structure(
        lambda x: x.reshape((num_ensembles, new_batch_size, *x.shape[1:])),
        (states, actions))
    mean, std = jax.vmap(transformed.apply)(ensem_params, reshaped_states, reshaped_act)
    mean, std = tree.map_structure(lambda x: x.reshape((-1, *x.shape[2:])), (mean, std))
    output = jax.random.normal(key, shape=mean.shape) * std + mean
    output = obs_postprocess(observations, output)
    # Shuffle back
    output = jax.ops.index_update(output, shuffle_indices, output)
    return output

  return init_fn, forward_fn


def make_loss_and_eval_fn(forward_fn,
                          obs_preprocess,
                          target_postprocess,
                          weight_decay=0.00001):
  # TODO(yl): Consider internalizing the loss and score functions

  def loss(params, x, a, xnext) -> jnp.ndarray:
    """Compute the loss of the network, including L2."""
    proc_x = obs_preprocess(x)
    mean, std = forward_fn(params, proc_x, a)
    dist = tfd.Independent(tfd.Normal(loc=mean, scale=std), 1)
    target = target_postprocess(x, xnext)
    logp_loss = -jnp.mean(dist.log_prob(target), axis=0)
    model_params = hk.data_structures.filter(
        lambda _, name, __: name not in ['min_logvar', 'max_logvar'], params)
    var_params = hk.data_structures.filter(
        lambda _, name, __: name in ['min_logvar', 'max_logvar'], params)
    l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(model_params))
    var_loss = 0.01 * (jnp.sum(var_params['gaussian_mlp']['max_logvar'])
                       - jnp.sum(var_params['gaussian_mlp']['min_logvar']))
    return logp_loss + weight_decay * l2_loss + var_loss

  @jax.jit
  def batched_loss(ensem_params, x, a, xnext) -> jnp.ndarray:
    """Compute the loss of the network, including L2."""
    losses = jax.vmap(loss, (0, None, None, None))(ensem_params, x, a, xnext)
    return jnp.mean(losses)

  def evaluate(params, x, a, xnext) -> jnp.ndarray:
    """Compute the loss of the network, including L2."""
    # x = self.obs_preprocess(x)
    proc_x = obs_preprocess(x)
    mean, std = forward_fn(params, proc_x, a)
    dist = tfd.Independent(tfd.Normal(loc=mean, scale=std), 1)
    target = target_postprocess(x, xnext)
    logp_loss = -jnp.mean(dist.log_prob(target), axis=0)
    return logp_loss

  @jax.jit
  def batched_eval(ensem_params, x, a, xnext):
    losses = jax.vmap(evaluate, (0, None, None, None))(ensem_params, x, a, xnext)
    return jnp.mean(losses)

  return batched_loss, batched_eval
