from typing import List

import haiku as hk
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
import tensorflow_probability as tfp
from acme import core, specs
from acme.jax import utils
from acme.utils import loggers

from magi.agents.pets import data

tfd = tfp.experimental.substrates.jax.distributions


class ModelBasedLearner(core.Learner):

  def __init__(
      self,
      spec: specs.EnvironmentSpec,
      net_fn,
      dataset: data.Dataset,
      obs_preprocess,
      target_postprocess,
      lr: float = 1e-3,
      batch_size: int = 32,
      num_epochs: int = 100,
      seed: int = 1,
      logger: loggers.Logger = None,
      num_ensembles=5,
  ):
    self.net = hk.without_apply_rng(hk.transform(net_fn))
    self.opt = optax.adam(lr)
    self.obs_preprocess = obs_preprocess
    self.target_postprocess = target_postprocess

    def loss(params, x, a, xnext) -> jnp.ndarray:
      """Compute the loss of the network, including L2."""
      # x = self.obs_preprocess(x)
      proc_x = self.obs_preprocess(x)
      dmean, std = self.net.apply(params, proc_x, a)
      xpred = dmean + x
      logp_loss = -jnp.mean(
          jnp.sum(tfd.Normal(loc=xpred, scale=std).log_prob(xnext), -1))
      l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))
      # TODO handle this proper
      # varloss = 0.01 * (jnp.sum(params['bnn']['max_logvar']) -
      # jnp.sum(params['bnn']['min_logvar']))
      return logp_loss + 0.0001 * l2_loss  # + var_loss

    def batched_loss(ensem_params, x, a, xnext) -> jnp.ndarray:
      """Compute the loss of the network, including L2."""
      losses = jax.vmap(loss, (0, None, None, None))(ensem_params, x, a, xnext)
      return jnp.mean(losses)

    @jax.jit
    def update(ensem_params, opt_state, x, a, xnext):
      """Learning rule (stochastic gradient descent)."""
      value, grads = jax.value_and_grad(batched_loss)(ensem_params, x, a, xnext)
      updates, opt_state = self.opt.update(grads, opt_state)
      new_params = optax.apply_updates(ensem_params, updates)
      return new_params, opt_state, value

    self._spec = spec
    self.loss = loss
    self.update_fn = update
    self.dataset = dataset
    self.batch_size = batch_size
    self.num_epochs = num_epochs

    # Initialize state
    # self.params = self.net.init(
    #     jax.random.PRNGKey(seed),
    #     utils.add_batch_dim(utils.zeros_like(self._spec.observations)),
    #     utils.add_batch_dim(utils.zeros_like(self._spec.actions)),
    # )
    self.ensem_params = jax.vmap(self.net.init, (0, None, None))(
        jax.random.split(jax.random.PRNGKey(seed), num_ensembles),
        self.obs_preprocess(
            utils.add_batch_dim(utils.zeros_like(self._spec.observations))),
        utils.add_batch_dim(utils.zeros_like(self._spec.actions)),
    )

    self.opt_state = self.opt.init(self.ensem_params)
    self._logger = logger or loggers.TerminalLogger('learner', time_delta=1.0)

  def step(self):
    """Perform an update step of the learner's parameters."""
    self._train_model(self.dataset)

  def _train_model(self, dataset):
    # At the end of the episode, train the dynamics model
    obs_t, obs_tp1, actions, rewards = dataset.collect()
    ds = (tf.data.Dataset.from_tensor_slices(
        (obs_t, obs_tp1, actions,
         rewards)).shuffle(self.batch_size).batch(self.batch_size))
    for epoch in range(self.num_epochs):
      for batch in ds.as_numpy_iterator():
        obs_t_b, obs_tp1_b, actions_b, _ = map(lambda x: jnp.array(x), batch)
        self.ensem_params, self.opt_state, loss = self.update_fn(
            self.ensem_params, self.opt_state, obs_t_b, actions_b, obs_tp1_b)
        self._logger.write({'model_loss': loss, 'epoch': epoch})

  def get_variables(self, names: List[str]) -> List[hk.Params]:
    del names
    return [self.ensem_params]

  def run(self):
    """Run the update loop; typically an infinite loop which calls step."""
    while True:
      self.step()
