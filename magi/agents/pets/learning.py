from typing import List

from acme import core
from acme import specs
from acme.jax import utils
from acme.utils import loggers
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_probability as tfp

from magi.agents.pets import dataset as dataset_lib

tfd = tfp.experimental.substrates.jax.distributions


class EarlyStopping:
  """
  Args:
    min_delta: Minimum change in the monitored quantity
        to qualify as an improvement, i.e. an absolute
        change of less than min_delta, will count as no
        improvement.
    patience: Number of epochs with no improvement
        after which training will be stopped.
    verbose: verbosity mode.
  """

  def __init__(self, min_delta: float = 0., patience: int = 0):
    self.patience = patience
    self.min_delta = abs(min_delta)
    self.wait = 0
    self.stopped_epoch = 0

  def reset(self):
    # Allow instances to be re-used
    self.wait = 0
    self.stopped_epoch = 0
    self.best = np.Inf
    self.best_weights = None

  def update(self, current, epoch):
    if current is None:
      return False

    self.wait += 1
    if self._is_improvement(current, self.best):
      print('Epoch {} New best {}, old {}'.format(epoch, current, self.best))
      self.best = current
      # If current is smaller the previous best, reset wait
      self.wait = 0

    if self.wait >= self.patience:
      return True
    return False

  def _is_improvement(self, monitor_value, reference_value):
    return np.less(monitor_value, reference_value - self.min_delta)


class ModelBasedLearner(core.Learner):

  def __init__(
      self,
      spec: specs.EnvironmentSpec,
      net_fn,
      dataset: dataset_lib.Dataset,
      obs_preprocess,
      target_postprocess,
      lr: float = 1e-3,
      batch_size: int = 32,
      num_epochs: int = 100,
      seed: int = 1,
      min_delta=0.1,
      patience=5,
      logger: loggers.Logger = None,
      num_ensembles=5,
  ):
    self.net = hk.without_apply_rng(hk.transform(net_fn))
    self.opt = optax.adam(lr)
    self.obs_preprocess = obs_preprocess
    self.target_postprocess = target_postprocess
    self._early_stopper = EarlyStopping(min_delta=min_delta, patience=patience)

    def loss(params, x, a, xnext) -> jnp.ndarray:
      """Compute the loss of the network, including L2."""
      # x = self.obs_preprocess(x)
      proc_x = self.obs_preprocess(x)
      mean, std = self.net.apply(params, proc_x, a)
      # xpred = dmean + x
      target = target_postprocess(x, xnext)
      logp_loss = -jnp.mean(
          jnp.sum(tfd.Normal(loc=mean, scale=std).log_prob(target), -1))
      model_params = hk.data_structures.filter(
          lambda module_name, name, value: name not in ['min_logvar', 'max_logvar'],
          params)
      var_params = hk.data_structures.filter(
          lambda module_name, name, value: name in ['min_logvar', 'max_logvar'], params)
      l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(model_params))
      var_loss = (0.01 * jnp.sum(var_params['bnn']['max_logvar'])
                  - 0.01 * jnp.sum(var_params['bnn']['min_logvar']))
      return logp_loss + 0.0001 * l2_loss + var_loss

    @jax.jit
    def batched_loss(ensem_params, x, a, xnext) -> jnp.ndarray:
      """Compute the loss of the network, including L2."""
      losses = jax.vmap(loss, (0, None, None, None))(ensem_params, x, a, xnext)
      return jnp.mean(losses)

    def metric(params, x, a, xnext) -> jnp.ndarray:
      """Compute the loss of the network, including L2."""
      # x = self.obs_preprocess(x)
      proc_x = self.obs_preprocess(x)
      mean, std = self.net.apply(params, proc_x, a)
      target = target_postprocess(x, xnext)
      logp_loss = -jnp.mean(
          jnp.sum(tfd.Normal(loc=mean, scale=std).log_prob(target), -1))
      return logp_loss

    def batched_metric(ensem_params, x, a, xnext):
      losses = jax.vmap(metric, (0, None, None, None))(ensem_params, x, a, xnext)
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
    self._batched_metric_fn = batched_metric

    self.ensem_params = jax.vmap(self.net.init, (0, None, None))(
        jax.random.split(jax.random.PRNGKey(seed), num_ensembles),
        self.obs_preprocess(
            utils.add_batch_dim(utils.zeros_like(self._spec.observations))),
        utils.add_batch_dim(utils.zeros_like(self._spec.actions)),
    )

    self.opt_state = self.opt.init(self.ensem_params)
    self._logger = logger or loggers.TerminalLogger('learner', time_delta=0.01)

  def step(self):
    """Perform an update step of the learner's parameters."""
    self._train_model(self.dataset)

  def _evaluate(self, params, iterator):
    total_loss = 0.
    count = 0
    for batch in iterator:
      batch_loss = self._batched_metric_fn(params, batch.o_tm1, batch.a_t, batch.o_t)
      total_loss += batch_loss * len(batch.a_t)
      count += len(batch.a_t)
    return total_loss / count

  def _train_model(self, dataset):
    # At the end of the episode, train the dynamics model
    # obs_t, obs_tp1, actions, rewards = dataset.collect()
    self._early_stopper.reset()
    train_iterator, val_iterator = dataset.get_iterators(self.batch_size, val_ratio=0.1)
    for epoch in range(self.num_epochs):
      # Train
      for batch in train_iterator:
        self.ensem_params, self.opt_state, _ = self.update_fn(
            self.ensem_params, self.opt_state, batch.o_tm1, batch.a_t, batch.o_t)
        # self._logger.write({'model_loss': loss, 'epoch': epoch})
      # Evaluate on validation set
      validation_loss = self._evaluate(self.ensem_params, val_iterator)
      should_stop = self._early_stopper.update(validation_loss, epoch)
      if should_stop:
        print('Early stopping')
        break
    # self._logger.write({"epoch": epoch, "validation_loss": validation_loss})

  def get_variables(self, names: List[str]) -> List[hk.Params]:
    del names
    return [self.ensem_params]
