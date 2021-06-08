from typing import List, NamedTuple

from acme import core
from acme.jax import utils
from acme import specs
from acme.utils import counting
from acme.utils import loggers
import haiku as hk
import jax
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


class TraningState(NamedTuple):
  params: hk.Params
  opt_state: optax.OptState


class ModelBasedLearner(core.Learner):

  def __init__(
      self,
      spec: specs.EnvironmentSpec,
      model_init_fn,
      loss_fn,
      evaluate_fn,
      dataset: dataset_lib.Dataset,
      lr: float = 1e-3,
      batch_size: int = 32,
      num_epochs: int = 100,
      seed: int = 1,
      min_delta=0.1,
      patience=5,
      logger: loggers.Logger = None,
      counter: counting.Counter = None,
  ):
    self.opt = optax.adam(lr)
    self.dataset = dataset
    self.batch_size = batch_size
    self.num_epochs = num_epochs
    self._early_stopper = EarlyStopping(min_delta=min_delta, patience=patience)

    def init():
      params = model_init_fn(
          jax.random.PRNGKey(seed),
          utils.add_batch_dim(utils.zeros_like(spec.observations)),
          utils.add_batch_dim(utils.zeros_like(spec.actions)),
      )
      return TraningState(params, self.opt.init(params))

    @jax.jit
    def update(state: TraningState, x, a, xnext):
      """Learning rule (stochastic gradient descent)."""
      value, grads = jax.value_and_grad(loss_fn)(state.params, x, a, xnext)
      updates, new_opt_state = self.opt.update(grads, state.opt_state)
      new_params = optax.apply_updates(state.params, updates)
      return TraningState(new_params, new_opt_state), value

    self._update = update
    self._score = evaluate_fn

    self._state = init()
    self._logger = logger or loggers.TerminalLogger('learner', time_delta=0.01)
    self._counter = counter or counting.Counter('learner')

  def step(self):
    """Perform an update step of the learner's parameters."""
    self._train_model(self.dataset)

  def _evaluate(self, params, iterator):
    total_loss = 0.
    count = 0
    for batch in iterator:
      batch_loss = self._score(params, batch.o_tm1, batch.a_t, batch.o_t)
      total_loss += batch_loss * len(batch.a_t)
      count += len(batch.a_t)
    return total_loss / count

  def _train_model(self, dataset):
    # At the end of the episode, train the dynamics model
    train_iterator, val_iterator = dataset.get_iterators(self.batch_size, val_ratio=0)
    if val_iterator is not None:
      self._early_stopper.reset()
    for epoch in range(self.num_epochs):
      # Train
      for batch in train_iterator:
        self._state, _ = self._update(self._state, batch.o_tm1, batch.a_t, batch.o_t)
      # Evaluate on validation set
      if val_iterator is not None:
        validation_loss = self._evaluate(self._state.params, val_iterator)
        should_stop = self._early_stopper.update(validation_loss, epoch)
        if should_stop:
          print('Early stopping')
          break
    # self._logger.write({"epoch": epoch, "validation_loss": validation_loss})

  def get_variables(self, names: List[str]) -> List[hk.Params]:
    del names
    return [self._state.params]
