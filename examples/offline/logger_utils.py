from typing import Any, Dict, Optional

from absl import logging
from acme.utils.loggers import aggregators
from acme.utils.loggers import asynchronous as async_logger
from acme.utils.loggers import base
from acme.utils.loggers import csv
from acme.utils.loggers import filters
from acme.utils.loggers import terminal

from magi.utils import loggers as magi_loggers


def make_default_logger(
    workdir: str,
    label: str,
    save_data: bool = True,
    time_delta: float = 1.0,
    asynchronous: bool = False,
    print_fn=None,
    serialize_fn=base.to_numpy,
    steps_key: Optional[str] = None,
    log_to_wandb: bool = False,
    add_uid: bool = False,
    wandb_kwargs: Optional[Dict[str, Any]] = None,
) -> base.Logger:
    if not print_fn:
        print_fn = logging.info
    terminal_logger = terminal.TerminalLogger(label=label, print_fn=print_fn)

    loggers = [terminal_logger]

    if save_data:
        loggers.append(csv.CSVLogger(workdir, label=label, add_uid=add_uid))

    if save_data and log_to_wandb:
        wandb_kwargs = wandb_kwargs or {}
        loggers.append(
            magi_loggers.WandbLogger(label=label, steps_key=steps_key, **wandb_kwargs)
        )

    # Dispatch to all writers and filter Nones and by time.
    logger = aggregators.Dispatcher(loggers, serialize_fn)
    logger = filters.NoneFilter(logger)
    if asynchronous:
        logger = async_logger.AsyncLogger(logger)
    logger = filters.TimeFilter(logger, time_delta)

    return logger
