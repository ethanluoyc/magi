"""Logging for RL experiments."""
from typing import Any, Optional

from absl import logging
from acme.utils.loggers import aggregators
from acme.utils.loggers import asynchronous as async_logger
from acme.utils.loggers import base
from acme.utils.loggers import filters
from acme.utils.loggers import terminal

try:
  import wandb
except ImportError:
  wandb = None


class WandbLogger(base.Logger):
  """Logging results to weights and biases"""

  def __init__(
      self,
      label: Optional[str] = None,
      steps_key: Optional[str] = None,
      *,
      project: Optional[str] = None,
      entity: Optional[str] = None,
      dir: Optional[str] = None,  # pylint: disable=redefined-builtin
      name: Optional[str] = None,
      group: Optional[str] = None,
      config: Optional[Any] = None,
      **wandb_kwargs,
  ):
    if wandb is None:
      raise ImportError(
          'Logger not supported as `wandb` logger is not installed yet,'
          ' install it with `pip install wandb`.')
    self._label = label
    self._iter = 0
    self._steps_key = steps_key
    if wandb.run is None:
      self._run = wandb.init(
          project=project,
          dir=dir,
          entity=entity,
          name=name,
          group=group,
          config=config,
          **wandb_kwargs,
      )
    else:
      self._run = wandb.run
    # define default x-axis (for latest wandb versions)
    if steps_key and getattr(self._run, 'define_metric', None):
      prefix = f'{self._label}/*' if self._label else '*'
      self._run.define_metric(
          prefix, step_metric=f'{self._label}/{self._steps_key}')

  @property
  def run(self):
    """Return the current wandb run."""
    return self._run

  def write(self, data: base.LoggingData):
    data = base.to_numpy(data)
    if self._steps_key is not None and self._steps_key not in data:
      logging.warn('steps key %s not found. Skip logging.', self._steps_key)
      return
    if self._label:
      stats = {f'{self._label}/{k}': v for k, v in data.items()}
    else:
      stats = data
    self._run.log(stats)

    self._iter += 1

  def close(self):
    pass


def make_logger(
    label: str,
    log_frequency: int = 1,
    asynchronous: bool = False,
    print_fn=None,
    serialize_fn=base.to_numpy,
    steps_key: str = 'steps',
    use_wandb=True,
    wandb_kwargs=None,
) -> base.Logger:
  """Make a default Acme logger.

  Args:
    label: Name to give to the logger.
    save_data: Whether to persist data.
    time_delta: Time (in seconds) between logging events.
    asynchronous: Whether the write function should block or not.
    print_fn: How to print to terminal (defaults to print).
    serialize_fn: An optional function to apply to the write inputs before
      passing them to the various loggers.
    steps_key: Ignored.

  Returns:
    A logger object that responds to logger.write(some_dict).
  """
  if not print_fn:
    print_fn = print
  terminal_logger = terminal.TerminalLogger(label=label, print_fn=print_fn)

  loggers = [terminal_logger]
  if use_wandb:
    if wandb_kwargs is None:
      wandb_kwargs = {}
    loggers.append(
        WandbLogger(label=label, steps_key=steps_key, **wandb_kwargs))

  # Dispatch to all writers and filter Nones and by time.
  logger = aggregators.Dispatcher(loggers, serialize_fn)
  logger = filters.NoneFilter(logger)
  if asynchronous:
    logger = async_logger.AsyncLogger(logger)
  logger = filters.GatedFilter(
      logger, gating_fn=lambda t: t % log_frequency == 0)

  return logger
