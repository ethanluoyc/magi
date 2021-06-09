from typing import List, NamedTuple

import chex
from absl import logging
from acme import core
from acme.jax import utils
from acme import specs
from acme.utils import counting
from acme.utils import loggers
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

from magi.agents.pets import dataset as dataset_lib


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

  def __init__(self, min_delta: float = 0.01, patience: int = 0):
    self.patience = patience
    self.min_delta = abs(min_delta)
    self.wait = 0
    self.stopped_epoch = 0

  def reset(self):
    # Allow instances to be re-used
    self.wait = 0
    self.stopped_epoch = 0
    self.best = np.Inf
    self.best_params = None

  def update(self, current, epoch, params):
    if current is None:
      return False

    self.wait += 1
    if self._is_improvement(current, self.best):
      print('Epoch {} New best {}, old {}'.format(epoch, current, self.best))
      self.best = current
      self.best_params = params
      # If current is smaller the previous best, reset wait
      self.wait = 0

    if self.wait >= self.patience:
      return True
    return False

  def _is_improvement(self, val_loss, best_val_loss):
    # if not np.isinf(val_loss) and np.isinf(best_val_loss):
    #   return True
    improvement = (best_val_loss - val_loss) / best_val_loss
    improved = (improvement > self.min_delta).any().item()
    return improved


@chex.dataclass
class Normalizer:
  mean: jnp.ndarray
  std: jnp.ndarray

  def __call__(self, input):
    # NOTE(yl) The normalization is causing trouble compared to the original impl.
    # when normalization is used, the dynamics rollout explodes, consequently
    # CEM fails (the costs go up to >1e10)
    # This is probably a precision issue, not sure at the moment
    # return input
    return (input - self.mean) / (self.std)


class TraningState(NamedTuple):
  params: hk.Params
  opt_state: optax.OptState
  normalizer: Normalizer


class ModelBasedLearner(core.Learner):

  def __init__(
      self,
      spec: specs.EnvironmentSpec,
      init_fn,
      loss_fn,
      num_ensembles,
      evaluate_fn,
      obs_preprocess,
      dataset: dataset_lib.ReplayBuffer,
      optimizer,
      batch_size: int = 32,
      num_epochs: int = 100,
      seed: jnp.ndarray = None,
      min_delta=0.1,
      patience=5,
      val_ratio=0,
      logger: loggers.Logger = None,
      counter: counting.Counter = None,
  ):
    self.opt = optimizer
    self.dataset = dataset
    self.batch_size = batch_size
    self.num_epochs = num_epochs
    self._num_ensembles = num_ensembles
    self._val_ratio = val_ratio
    self._early_stopper = EarlyStopping(min_delta=min_delta, patience=patience)
    init_key, iter_key = jax.random.split(seed)
    self._rng = np.random.default_rng(np.asarray(iter_key))

    def model_init_fn(rng, x, a):
      inputs = jnp.concatenate([obs_preprocess(x), a], axis=-1)
      params_list = []
      rngs = jax.random.split(rng, num_ensembles)
      for r in rngs:
        params_list.append(init_fn(r, inputs))
      # pylint: disable=no-value-for-parameter
      ensem_params = jax.tree_multimap(lambda *x: jnp.stack(x), *params_list)
      return ensem_params

    def init():
      params = model_init_fn(
          init_key,
          utils.add_batch_dim(utils.zeros_like(spec.observations)),
          utils.add_batch_dim(utils.zeros_like(spec.actions)),
      )
      processed_dummy_input = obs_preprocess(utils.zeros_like(spec.observations))
      norm = jnp.concatenate(
          [processed_dummy_input, utils.zeros_like(spec.actions)], axis=-1)
      mean = jnp.zeros(norm.shape, dtype=jnp.float32)
      std = jnp.ones(norm.shape, dtype=jnp.float32)
      return TraningState(params, self.opt.init(params), Normalizer(mean=mean, std=std))

    def update_normalizer(state: TraningState, x, a, xnext):
      """Learning rule (stochastic gradient descent)."""
      del xnext
      new_input = jnp.concatenate([obs_preprocess(x), a], axis=-1)
      new_input = np.asarray(new_input)
      new_mean = np.mean(new_input, axis=0)
      new_std = np.std(new_input, axis=0, dtype=np.float64)
      # We are using a larger eps here for handling observation dims
      # that do not change during training. The original implementation uses
      # 1e-12, which is okay only if the inputs are float64, but is too small
      # for float32 which JAX uses by default.
      #
      # Without this, environments such as reacher or pusher will not work as
      # the observation includes positions of goal which do not change.
      # This needs to be investigated further. In particular, simply changing the eps
      # here does not seem to fix problems.
      # affect how we normalize. While the original impl simply does
      # (o - mean) / std. In the case of small std, the normalized inputs will explode.
      new_std[new_std < 1e-12] = 1.0
      new_mean = jnp.array(new_mean.astype(np.float32))
      new_std = jnp.array(new_std.astype(np.float32))
      return state._replace(normalizer=Normalizer(mean=new_mean, std=new_std))

    @jax.jit
    def update(state: TraningState, x, a, xnext):
      """Learning rule (stochastic gradient descent)."""
      value, grads = jax.value_and_grad(loss_fn)(state.params, state.normalizer, x, a,
                                                 xnext)
      updates, new_opt_state = self.opt.update(grads, state.opt_state, state.params)
      new_params = optax.apply_updates(state.params, updates)
      return TraningState(new_params, new_opt_state, state.normalizer), value

    self._update = update
    self._update_normalizer = update_normalizer
    self._score = evaluate_fn

    self._state = init()
    self._logger = logger or loggers.TerminalLogger("learner", time_delta=0.01)
    self._counter = counter or counting.Counter("learner")

  def step(self):
    """Perform an update step of the learner's parameters."""
    self._train_model(self.dataset)

  def _evaluate(self, params, normalizer, iterator):
    if isinstance(iterator, dataset_lib.BootstrapIterator):
      iterator.toggle_bootstrap()

    losses = []
    for batch in iterator:
      batch_loss = self._score(params, normalizer, batch.obs, batch.act, batch.next_obs)
      losses.append(batch_loss)
    if isinstance(iterator, dataset_lib.BootstrapIterator):
      iterator.toggle_bootstrap()

    return jnp.stack(losses).mean(axis=0)

  def _train_model(self, dataset: dataset_lib.ReplayBuffer):
    # At the end of the episode, train the dynamics model
    transitions = dataset.get_all()
    self._state = self._update_normalizer(self._state, transitions.obs, transitions.act,
                                          transitions.next_obs)
    logging.info("Normalizer mean %s", self._state.normalizer.mean)
    logging.info("Normalizer std %s", self._state.normalizer.std)
    train_iterator, val_iterator = dataset.get_iterators(
        self.batch_size,
        train_ensemble=True,
        ensemble_size=self._num_ensembles,
        bootstrap_permutes=False,
        val_ratio=self._val_ratio)
    if val_iterator is None:
      val_iterator = train_iterator
    validation_loss = self._evaluate(self._state.params, self._state.normalizer,
                                     val_iterator)
    logging.info("Start loss %s", validation_loss)
    for epoch in range(self.num_epochs):
      # Train
      batch_losses = []
      for batch in train_iterator:
        self._state, loss = self._update(self._state, batch.obs, batch.act,
                                         batch.next_obs)
        batch_losses.append(loss)
      batch_loss = jnp.mean(jnp.array(batch_losses), axis=0)
      # Evaluate on validation set
      validation_loss = self._evaluate(self._state.params, self._state.normalizer,
                                       val_iterator)
    logging.info("Epoch %d Train loss %s, Validation loss %s", epoch, batch_loss,
                 validation_loss)

  def get_variables(self, names: List[str]) -> List[hk.Params]:
    del names
    return [{"params": self._state.params, "state": self._state.normalizer}]
