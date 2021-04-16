"""Logging for RL experiments."""
import wandb
from acme.utils.loggers import base


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
    if self._step % self._log_every_n_steps == 0:
      if self._label:
        stats = {f'{self._label}/{k}': v for k, v in data.items()}
      self._run.log(stats)
    self._step += 1
