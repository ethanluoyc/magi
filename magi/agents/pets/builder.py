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


def make_network(environment_spec, hidden_sizes):
  output_size = environment_spec.observations.shape[-1]

  def network(x, a):
    input_ = jnp.concatenate([x, a], axis=-1)
    model = models.GaussianMLP(output_size, hidden_sizes=hidden_sizes)
    return model(input_)

  return network


def make_agent(environment_spec: specs.EnvironmentSpec,
               cost_fn,
               obs_preprocess,
               obs_postprocess,
               target_process,
               optimizer='cem',
               hidden_sizes=(10,),
               num_ensembles=5,
               batch_size=32,
               population_size=100,
               logger=None,
               counter=None):

  dataset = Dataset()
  network = make_network(environment_spec, hidden_sizes)
  # Create a learner

  transformed = hk.without_apply_rng(hk.transform(network))

  def model_init_fn(rng, x, a):
    vmapped_init_fn = jax.vmap(transformed.init, (0, None, None))
    ensem_params = vmapped_init_fn(jax.random.split(rng, num_ensembles),
                                   obs_preprocess(x), a)
    return ensem_params

  loss_fn, evaluate_fn = make_loss_and_eval_fn(transformed.apply, obs_preprocess,
                                               target_process)

  learner = ModelBasedLearner(environment_spec,
                              model_init_fn,
                              loss_fn,
                              evaluate_fn,
                              dataset,
                              batch_size=batch_size,
                              logger=logger,
                              counter=counter)

  forward_fn = make_forward_fn(network, obs_preprocess, obs_postprocess)
  # Create actor
  variable_client = variable_utils.VariableClient(learner, '')

  if optimizer == 'cem':
    actor = acting.CEMOptimizerActor(environment_spec,
                                     forward_fn,
                                     cost_fn,
                                     dataset,
                                     variable_client,
                                     pop_size=population_size)
  elif optimizer == 'random':
    actor = acting.RandomOptimizerActor(environment_spec,
                                        forward_fn,
                                        cost_fn,
                                        dataset,
                                        variable_client,
                                        num_samples=population_size)

  agent = ModelBasedAgent(actor, learner)
  return agent


def make_forward_fn(network, obs_preprocess, obs_postprocess):
  transformed = hk.without_apply_rng(hk.transform(network))

  def forward_fn(ensem_params, key, observations, actions):
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
    return obs_postprocess(observations, output)

  return forward_fn


def make_loss_and_eval_fn(forward_fn, obs_preprocess, target_postprocess):

  def loss(params, x, a, xnext) -> jnp.ndarray:
    """Compute the loss of the network, including L2."""
    # x = self.obs_preprocess(x)
    proc_x = obs_preprocess(x)
    mean, std = forward_fn(params, proc_x, a)
    target = target_postprocess(x, xnext)
    logp_loss = -jnp.mean(jnp.sum(tfd.Normal(loc=mean, scale=std).log_prob(target), -1))
    model_params = hk.data_structures.filter(
        lambda module_name, name, value: name not in ['min_logvar', 'max_logvar'],
        params)
    var_params = hk.data_structures.filter(
        lambda module_name, name, value: name in ['min_logvar', 'max_logvar'], params)
    l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(model_params))
    var_loss = (0.01 * jnp.sum(var_params['gaussian_mlp']['max_logvar'])
                - 0.01 * jnp.sum(var_params['gaussian_mlp']['min_logvar']))
    return logp_loss + 0.0001 * l2_loss + var_loss

  @jax.jit
  def batched_loss(ensem_params, x, a, xnext) -> jnp.ndarray:
    """Compute the loss of the network, including L2."""
    losses = jax.vmap(loss, (0, None, None, None))(ensem_params, x, a, xnext)
    return jnp.mean(losses)

  def metric(params, x, a, xnext) -> jnp.ndarray:
    """Compute the loss of the network, including L2."""
    # x = self.obs_preprocess(x)
    proc_x = obs_preprocess(x)
    mean, std = forward_fn(params, proc_x, a)
    target = target_postprocess(x, xnext)
    logp_loss = -jnp.mean(jnp.sum(tfd.Normal(loc=mean, scale=std).log_prob(target), -1))
    return logp_loss

  @jax.jit
  def batched_metric(ensem_params, x, a, xnext):
    losses = jax.vmap(metric, (0, None, None, None))(ensem_params, x, a, xnext)
    return jnp.mean(losses)

  return batched_loss, batched_metric
