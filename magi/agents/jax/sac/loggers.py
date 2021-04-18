"""Logging for RL experiments."""
import wandb
from acme.utils.loggers import base, terminal, aggregators, filters
from acme.utils.loggers import asynchronous as async_logger


class WandbLogger(base.Logger):
  """Logging results to weights and biases"""

  def __init__(self, label=None, log_every_n_steps=1):
    self._label = label
    self._log_every_n_steps = log_every_n_steps
    self._run = wandb.run
    self._step = 0
    assert self._run is not None

  @property
  def run(self):
    """Return the current wandb run."""
    return self._run

  def write(self, data: base.LoggingData):
    data = base.to_numpy(data)
    if self._step % self._log_every_n_steps == 0:
      if self._label:
        stats = {f'{self._label}/{k}': v for k, v in data.items()}
      self._run.log(stats)
    self._step += 1


def make_logger(label: str,
                time_delta: float = 1.0,
                asynchronous: bool = False,
                print_fn=None,
                serialize_fn=base.to_numpy,
                steps_key: str = 'steps',
                use_wandb=True,
                wandb_log_every_n_steps=1):
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
  del steps_key
  if not print_fn:
    print_fn = print
  terminal_logger = terminal.TerminalLogger(label=label, print_fn=print_fn)

  loggers = [terminal_logger]
  if use_wandb:
    loggers.append(WandbLogger(label, log_every_n_steps=wandb_log_every_n_steps))

  # Dispatch to all writers and filter Nones and by time.
  logger = aggregators.Dispatcher(loggers, serialize_fn)
  logger = filters.NoneFilter(logger)
  if asynchronous:
    logger = async_logger.AsyncLogger(logger)
  logger = filters.TimeFilter(logger, time_delta)

  return logger
