"""Example running TD-MPC on dm_control.

Run this example with

```shell
python -m magi.agents.tdmpc.run_tdmpc \
  --config magi/agents/tdmpc/configs/walker.py \
  --config.task=walker-walk
```

See configs/ for configurations for other environments.

"""
import functools
import os

from absl import app
from absl import flags
from absl import logging

os.environ["MUJOCO_GL"] = "egl"
# pylint: disable=wrong-import-position

from acme import wrappers
from acme.jax import experiments
from ml_collections import config_flags
import optax
import tensorflow as tf

from magi.agents import tdmpc
from magi.experiments import experiment_logger

_CONFIG = config_flags.DEFINE_config_file("config", None)
_WORKDIR = flags.DEFINE_string("workdir", None, "Where to store artifacts")
flags.mark_flag_as_required("config")


def make_logger_factory(config):
  wandb_kwargs = dict(
      name=config.wandb_name,
      entity=config.wandb_entity,
      project=config.wandb_project,
      config=config.to_dict(),
      tags=[config.task],
  )

  logger_factory = experiment_logger.LoggerFactory(
      log_to_wandb=config.get("use_wandb", False),
      workdir=_WORKDIR.value,
      learner_time_delta=10.0,
      wandb_kwargs=wandb_kwargs,
  )

  return logger_factory


def make_environment_factory(config):

  def environment_factory(seed):
    # pylint: disable=import-outside-toplevel
    from dm_control import suite

    domain, task = config.task.replace("-", "_").split("_", 1)
    domain = dict(cup="ball_in_cup").get(domain, domain)
    assert (domain, task) in suite.ALL_TASKS
    env = suite.load(
        domain, task, task_kwargs={"random": seed}, visualize_reward=False)
    env = wrappers.ConcatObservationWrapper(env)
    env = wrappers.ActionRepeatWrapper(env, config.action_repeat)
    env = wrappers.CanonicalSpecWrapper(env)
    env = wrappers.SinglePrecisionWrapper(env)
    return env

  return environment_factory


def _make_schedule(config):
  return getattr(optax, config.name)(**config.kwargs)


def make_experiment_config(config):
  environment_factory = make_environment_factory(config)
  logger_factory = make_logger_factory(config)
  networks_factory = functools.partial(
      tdmpc.make_networks,
      latent_size=config.latent_dim,
      encoder_hidden_size=config.enc_dim,
      mlp_hidden_size=config.mlp_dim,
  )
  optimizer = optax.chain(
      optax.clip_by_global_norm(config.grad_clip_norm),
      optax.adam(config.lr),
  )
  std_schedule = _make_schedule(config.std_schedule)
  horizon_schedule = _make_schedule(config.horizon_schedule)
  builder = tdmpc.TDMPCBuilder(
      tdmpc.TDMPCConfig(
          std_schedule=std_schedule,
          horizon_schedule=horizon_schedule,
          optimizer=optimizer,
          batch_size=config.batch_size,
          # One update per actor step.
          samples_per_insert=config.batch_size,
          samples_per_insert_tolerance_rate=0.1,
          max_replay_size=config.max_buffer_size,
          variable_update_period=config.variable_update_period,
          per_alpha=config.per_alpha,
          per_beta=config.per_beta,
          discount=config.discount,
          num_samples=config.num_samples,
          min_std=config.min_std,
          temperature=config.temperature,
          momentum=config.momentum,
          num_elites=config.num_elites,
          iterations=config.iterations,
          tau=config.tau,
          seed_steps=config.seed_steps,
          mixture_coef=config.mixture_coef,
          horizon=config.horizon,
          consistency_coef=config.consistency_coef,
          reward_coef=config.reward_coef,
          value_coef=config.value_coef,
          rho=config.rho,
      ))

  return experiments.ExperimentConfig(
      builder=builder,
      network_factory=networks_factory,
      environment_factory=environment_factory,
      max_num_actor_steps=config.train_steps,
      seed=config.seed,
      logger_factory=logger_factory,
      checkpointing=None,
  )


def main(_):
  tf.config.set_visible_devices([], "GPU")
  config = _CONFIG.value
  logging.info("Config:\n%s", config)
  experiment_config = make_experiment_config(config)
  experiments.run_experiment(
      experiment_config,
      eval_every=config.eval_freq,
      num_eval_episodes=config.eval_episodes,
  )


if __name__ == "__main__":
  app.run(main)
