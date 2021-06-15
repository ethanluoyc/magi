from typing import List, NamedTuple

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
from magi.agents.pets import models


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
      logging.info('Epoch %d New best %.3f, old %.3f', epoch, current, self.best)
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


class TraningState(NamedTuple):
  params: hk.Params
  opt_state: optax.OptState
  normalizer: models.Normalizer


class ModelBasedLearner(core.Learner):

  def __init__(
      self,
      spec: specs.EnvironmentSpec,
      model: models.EnsembleModel,
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
    del min_delta, patience  # cross-validation not used for now
    self.opt = optimizer
    self.dataset = dataset
    self.batch_size = batch_size
    self.num_epochs = num_epochs
    self._val_ratio = val_ratio
    self._model = model
    self._logger = logger or loggers.TerminalLogger("learner", time_delta=0.01)
    self._counter = counter or counting.Counter("learner")

    init_key, iter_key = jax.random.split(seed)
    self._rng = np.random.default_rng(np.asarray(iter_key))

    def init():
      obs = utils.add_batch_dim(utils.zeros_like(spec.observations))
      act = utils.add_batch_dim(utils.zeros_like(spec.actions))
      params, normalizer = model.init(init_key, obs, act)
      return TraningState(params, self.opt.init(params), normalizer)

    @jax.jit
    def update(state: TraningState, x, a, xnext):
      """Learning rule (stochastic gradient descent)."""
      value, grads = jax.value_and_grad(model.loss)(state.params, state.normalizer, x,
                                                    a, xnext)
      updates, new_opt_state = self.opt.update(grads, state.opt_state, state.params)
      new_params = optax.apply_updates(state.params, updates)
      return TraningState(new_params, new_opt_state, state.normalizer), value

    self._update = update

    self._state = init()

  def _evaluate(self, params, normalizer, iterator):
    if isinstance(iterator, dataset_lib.BootstrapIterator):
      iterator.toggle_bootstrap()

    losses = []
    for batch in iterator:
      batch_loss = self._model.evaluate(params, normalizer, batch.obs, batch.act,
                                        batch.next_obs)
      losses.append(batch_loss)
    if isinstance(iterator, dataset_lib.BootstrapIterator):
      iterator.toggle_bootstrap()

    return jnp.stack(losses).mean(axis=0)

  def _train(self, dataset: dataset_lib.ReplayBuffer):
    # At the end of the episode, train the dynamics model
    transitions = dataset.get_all()
    new_normalizer = self._model.update_normalizer(transitions.obs, transitions.act,
                                                   transitions.next_obs)
    self._state = self._state._replace(normalizer=new_normalizer)
    logging.info("Normalizer mean %s", self._state.normalizer.mean)
    logging.info("Normalizer std %s", self._state.normalizer.std)
    train_iterator, val_iterator = dataset.get_iterators(
        self.batch_size,
        train_ensemble=True,
        ensemble_size=self._model.num_ensembles,
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

  def step(self):
    """Perform an update step of the learner's parameters."""
    self._train(self.dataset)

  def get_variables(self, names: List[str]) -> List[hk.Params]:
    del names
    return [{"params": self._state.params, "state": self._state.normalizer}]
