from acme.jax import variable_utils
from acme import specs
import haiku as hk
import jax
import jax.numpy as jnp
import tensorflow_probability as tfp
import tree
import optax

from magi.agents.pets import acting
from magi.agents.pets.agent import ModelBasedAgent
from magi.agents.pets.dataset import ReplayBuffer
from magi.agents.pets.learning import ModelBasedLearner
from magi.agents.pets import models

tfd = tfp.experimental.substrates.jax.distributions


def make_network(environment_spec,
                 hidden_sizes,
                 activation=jax.nn.swish,
                 min_logvar=-10.,
                 max_logvar=0.5):
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
  rng = jax.random.PRNGKey(seed)
  dataset = ReplayBuffer(
      int(1e6),
      obs_shape=environment_spec.observations.shape,
      action_shape=environment_spec.actions.shape,
      max_trajectory_length=1000,
  )
  network = make_network(environment_spec,
                         hidden_sizes,
                         activation=activation,
                         min_logvar=min_logvar,
                         max_logvar=max_logvar)
  transformed = hk.without_apply_rng(hk.transform(network))

  loss_fn, evaluate_fn = make_loss_and_eval_fn(transformed.apply,
                                               obs_preprocess,
                                               target_process,
                                               weight_decay=weight_decay)
  learner_rng, actor_rng = jax.random.split(rng)
  opt = optax.adamw(lr, weight_decay=weight_decay, eps=1e-5)
  # Create a learner
  learner = ModelBasedLearner(
      environment_spec,
      transformed.init,
      loss_fn,
      num_ensembles,
      evaluate_fn,
      obs_preprocess,
      dataset,
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
        seed=actor_rng,
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
                                        seed=actor_rng)

  agent = ModelBasedAgent(actor, learner)
  return agent


def make_forward_fn(network, obs_preprocess, obs_postprocess, shuffle=True):
  # NOTE: this currently corresponds to the TSInf formulation in PETS
  # TODO: Add the other options
  transformed = hk.without_apply_rng(hk.transform(network))

  def init_fn(ensem_params, normalizer, key, observations):
    num_ensembles = tree.flatten(ensem_params)[0].shape[0]
    batch_size = observations.shape[0]
    if batch_size % num_ensembles:
      raise NotImplementedError('Ragged batch not supported.')
    indices = jnp.arange(batch_size)
    if shuffle:
      indices = jax.random.permutation(key, indices)
    return (normalizer, indices)

  def forward_fn(ensem_params, state, key, observations, actions):
    # TODO(yl): Add sampling propagation indices instead of being fully deterministic
    assert observations.ndim == 2
    assert actions.ndim == 2
    normalizer, shuffle_indices = state
    batch_size = observations.shape[0]

    shuffled_observations = observations[shuffle_indices]
    shuffled_actions = actions[shuffle_indices]

    shuffled_states = obs_preprocess(shuffled_observations)
    num_ensembles = tree.flatten(ensem_params)[0].shape[0]
    new_batch_size, ragged = divmod(batch_size, num_ensembles)
    shuffled_inputs = jnp.concatenate([shuffled_states, shuffled_actions], axis=-1)
    normalized_inputs = normalizer(shuffled_inputs)
    # from jax.experimental import host_callback as hcb
    if ragged:
      raise NotImplementedError(
          f'Ragged batch not supported. ({batch_size} % {num_ensembles} == {ragged})')
    reshaped_inputs = tree.map_structure(
        lambda x: x.reshape((num_ensembles, new_batch_size, *x.shape[1:])),
        normalized_inputs)
    mean, logvar = jax.vmap(transformed.apply, in_axes=(0, 0))(ensem_params,
                                                               reshaped_inputs)
    # hcb.id_print((mean[0, 0], logvar[0, 0]), what='in')
    std = jnp.exp(logvar * 0.5)
    mean, std = tree.map_structure(lambda x: x.reshape((batch_size, mean.shape[-1])),
                                   (mean, std))
    # import pdb; pdb.set_trace()
    # print("Outputs", mean.shape, jnp.mean(mean, axis=0))
    shuffled_predictions = jax.random.normal(key, shape=mean.shape) * std + mean
    shuffled_output = obs_postprocess(shuffled_observations, shuffled_predictions)
    # Shuffle back
    output = jax.ops.index_update(shuffled_output, shuffle_indices, shuffled_output)
    return output

  return init_fn, forward_fn


def make_loss_and_eval_fn(forward_fn,
                          obs_preprocess,
                          target_postprocess,
                          weight_decay=0.00001):
  # TODO(yl): Consider internalizing the loss and score functions
  # pylint: disable=redefined-builtin

  def gaussian_nll(params, normalizer, x, a, xnext) -> jnp.ndarray:
    """Compute the loss of the network, including L2."""
    proc_x = obs_preprocess(x)
    input = jnp.concatenate([proc_x, a], axis=-1)
    input = normalizer(input)
    mean, logvar = forward_fn(params, input)
    target = target_postprocess(x, xnext)
    inv_var = jnp.exp(-logvar)
    assert mean.shape == target.shape == logvar.shape
    mse_loss = jnp.square(mean - target)
    logp_loss = mse_loss * inv_var + logvar
    # mean along batch and output dimension
    # logp_loss = logp_loss.mean(-1).mean(-1)
    return logp_loss

  # @jax.jit
  def batched_loss(ensem_params, normalizer, x, a, xnext) -> jnp.ndarray:
    """Compute the loss of the network, including L2."""
    nll_loss = jax.vmap(gaussian_nll, (0, None, 0, 0, 0))(ensem_params, normalizer, x,
                                                          a, xnext)
    return nll_loss.mean(axis=(1, 2)).sum()

  def evaluate(params, normalizer, x, a, xnext) -> jnp.ndarray:
    """Compute the validation loss of a single network, MSE.
    """
    # Validation is MSE
    proc_x = obs_preprocess(x)
    input = jnp.concatenate([proc_x, a], axis=-1)
    input = normalizer(input)
    mean, _ = forward_fn(params, input)
    # dist = tfd.Independent(tfd.Normal(loc=mean, scale=std), 1)
    target = target_postprocess(x, xnext)
    mse_loss = jnp.mean(jnp.square(target - mean).mean(axis=-1), axis=-1)
    return mse_loss

  @jax.jit
  def batched_eval(ensem_params, normalizer, x, a, xnext):
    """Compute the validation loss for the ensembles, MSE
    Args:
      params: ensemble parameters of shape [E, ...]
      normalizer: normalizer for normalizing the inputs
      x, a, x: training data of shape [E, B, ...]
    Returns:
      mse_loss: mean squared error of shape [E] from the ensembles
    """
    losses = jax.vmap(evaluate, (0, None, None, None, None))(ensem_params, normalizer,
                                                             x, a, xnext)
    # Return the validation MSE per ensemble
    return losses

  return batched_loss, batched_eval
