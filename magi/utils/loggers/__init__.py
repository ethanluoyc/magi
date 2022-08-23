"""Logging for RL experiments."""
from absl import logging
from acme.utils.loggers import aggregators
from acme.utils.loggers import asynchronous as async_logger
from acme.utils.loggers import base
from acme.utils.loggers import filters
from acme.utils.loggers import terminal

from magi.utils.loggers import wandb


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
    print_fn = logging.info
  terminal_logger = terminal.TerminalLogger(label=label, print_fn=print_fn)

  loggers = [terminal_logger]
  if use_wandb:
    if wandb_kwargs is None:
      wandb_kwargs = {}
    loggers.append(
        wandb.WandbLogger(label=label, steps_key=steps_key, **wandb_kwargs))

  # Dispatch to all writers and filter Nones and by time.
  logger = aggregators.Dispatcher(loggers, serialize_fn)
  logger = filters.NoneFilter(logger)
  if asynchronous:
    logger = async_logger.AsyncLogger(logger)
  logger = filters.GatedFilter(
      logger, gating_fn=lambda t: t % log_frequency == 0)

  return logger
