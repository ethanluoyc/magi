"""Utilieis Logging for RL experiments in Acme."""
from typing import Any, Dict, Mapping, Optional

from absl import logging
from acme.jax import utils as jax_utils
from acme.utils import loggers as loggers_lib
from acme.utils.loggers import base

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
      run=None,
      **wandb_kwargs,
  ):
    if wandb is None:
      raise ImportError(
          "Logger not supported as `wandb` logger is not installed yet,"  # pragma: no-cover
          " install it with `pip install wandb`.")
    self._label = label
    self._iter = 0
    self._steps_key = steps_key
    if run is None:
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
    else:
      self._run = run
    # define default x-axis (for latest wandb versions)
    if steps_key and getattr(self._run, "define_metric", None):
      prefix = f"{self._label}/*" if self._label else "*"
      self._run.define_metric(
          prefix, step_metric=f"{self._label}/{self._steps_key}")

  @property
  def run(self):
    """Return the current wandb run."""
    return self._run

  def write(self, data: base.LoggingData):
    data = base.to_numpy(data)
    if self._steps_key is not None and self._steps_key not in data:
      logging.warn("steps key %s not found. Skip logging.", self._steps_key)
      return
    if self._label:
      stats = {f"{self._label}/{k}": v for k, v in data.items()}
    else:
      stats = data
    self._run.log(stats)

    self._iter += 1

  def close(self):
    pass


def _get_time_delta(time_delta: Optional[float], default_time_delta: float):
  if time_delta is not None:
    return time_delta
  else:
    return default_time_delta


class LoggerFactory:
  """Factory for creating loggers used in magi RL experiments."""

  def __init__(
      self,
      workdir: Optional[str] = None,
      add_uid: bool = True,
      log_to_wandb: bool = False,
      wandb_kwargs: Optional[Mapping[str, Any]] = None,
      time_delta: float = 1.0,
      actor_time_delta: Optional[float] = None,
      learner_time_delta: Optional[float] = None,
      evaluator_time_delta: Optional[float] = None,
      async_learner_logger: bool = False,
  ):
    """Create a logger factory."""

    wandb_kwargs = wandb_kwargs or {}
    self._log_to_wandb = log_to_wandb
    self._run = None
    if log_to_wandb and self._run is None:
      import wandb  # pylint: disable=import-outside-toplevel

      wandb.require("service")
      self._run = wandb.init(**wandb_kwargs)
    self._add_uid = add_uid
    self._workdir = workdir
    self._time_delta = time_delta
    self._actor_time_delta = actor_time_delta
    self._learner_time_delta = learner_time_delta
    self._evaluator_time_delta = evaluator_time_delta
    self._async_learner_logger = async_learner_logger

  @property
  def run(self):
    return self._run

  def __call__(self,
               label: str,
               steps_key: Optional[str] = None,
               task_instance: int = 0):
    """Create an experiment logger."""
    if steps_key is None:
      steps_key = f"{label}_steps"

    # Binding the experiment logger factory ensures that
    # the wandb run associated with the launching process gets
    # serialized and passed to workers correctly.
    if label == "learner":
      return self.make_default_logger(
          label=label,
          asynchronous=self._async_learner_logger,
          time_delta=_get_time_delta(self._learner_time_delta,
                                     self._time_delta),
          serialize_fn=jax_utils.fetch_devicearray,
          workdir=self._workdir,
          log_to_wandb=self._log_to_wandb,
          steps_key=steps_key,
          wandb_run=self._run,
          add_uid=self._add_uid,
      )
    elif label in ("evaluator", "eval_loop", "evaluation", "eval"):
      return self.make_default_logger(
          label=label,
          time_delta=_get_time_delta(self._evaluator_time_delta,
                                     self._time_delta),
          steps_key=steps_key,
          workdir=self._workdir,
          log_to_wandb=self._log_to_wandb,
          wandb_run=self._run,
          add_uid=self._add_uid,
      )
    elif label in ("actor", "train_loop", "train"):
      return self.make_default_logger(
          label=label,
          save_data=task_instance == 0,
          time_delta=_get_time_delta(self._evaluator_time_delta,
                                     self._time_delta),
          steps_key=steps_key,
          workdir=self._workdir,
          log_to_wandb=self._log_to_wandb,
          wandb_run=self._run,
          add_uid=self._add_uid,
      )
    else:
      logging.warning("Unknown label %s. Fallback to default.", label)
      return self.make_default_logger(
          label=label,
          steps_key=steps_key,
          time_delta=self._time_delta,
          workdir=self._workdir,
          log_to_wandb=self._log_to_wandb,
          wandb_run=self._run,
          add_uid=self._add_uid,
      )

  @staticmethod
  def make_default_logger(
      label: str,
      save_data: bool = True,
      time_delta: float = 1.0,
      asynchronous: bool = False,
      print_fn=None,
      workdir: Optional[str] = None,
      serialize_fn=base.to_numpy,
      steps_key: str = "steps",
      log_to_wandb: bool = False,
      wandb_kwargs: Optional[Dict[str, Any]] = None,
      add_uid: bool = False,
      wandb_run=None,
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
    terminal_logger = loggers_lib.TerminalLogger(label=label, print_fn=print_fn)

    loggers = [terminal_logger]
    if save_data and workdir is not None:
      loggers.append(
          loggers_lib.CSVLogger(workdir, label=label, add_uid=add_uid))
    if save_data and log_to_wandb:
      if wandb_kwargs is None:
        wandb_kwargs = {}
      loggers.append(
          WandbLogger(
              label=label, steps_key=steps_key, run=wandb_run, **wandb_kwargs))

    # Dispatch to all writers and filter Nones and by time.
    logger = loggers_lib.Dispatcher(loggers, serialize_fn)
    logger = loggers_lib.NoneFilter(logger)
    if asynchronous:
      logger = loggers_lib.AsyncLogger(logger)
    logger = loggers_lib.TimeFilter(logger, time_delta)
    return loggers_lib.AutoCloseLogger(logger)
