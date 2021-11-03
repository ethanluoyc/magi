from typing import Any, Dict, Optional

from absl import logging
from acme import wrappers
from acme.utils.loggers import aggregators
from acme.utils.loggers import asynchronous as async_logger
from acme.utils.loggers import base
from acme.utils.loggers import csv
from acme.utils.loggers import filters
from acme.utils.loggers import terminal
from dm_control import suite  # pytype: disable=import-error
from dm_control.suite.wrappers import pixels  # pytype: disable=import-error

from magi import wrappers as magi_wrappers
from magi.utils import loggers as magi_loggers


def make_environment(domain_name, task_name, rng, frame_stack, action_repeat):
    """Create a visual DMC environment"""
    env = suite.load(
        domain_name=domain_name,
        task_name=task_name,
        environment_kwargs={"flat_observation": True},
        task_kwargs={"random": rng},
    )
    camera_id = 2 if domain_name == "quadruped" else 0
    env = pixels.Wrapper(
        env,
        pixels_only=True,
        render_kwargs={"width": 84, "height": 84, "camera_id": camera_id},
    )
    env = wrappers.CanonicalSpecWrapper(env)
    env = magi_wrappers.TakeKeyWrapper(env, "pixels")
    env = wrappers.ActionRepeatWrapper(env, action_repeat)
    env = magi_wrappers.FrameStackingWrapper(env, num_frames=frame_stack)
    env = wrappers.SinglePrecisionWrapper(env)
    return env


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
