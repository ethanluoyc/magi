"""CQL configuration for D4RL mujoco locomotion"""
import base_config


def get_config():
  config = base_config.get_base_config()
  config.actor_lr = 1e-4
  config.value_lr = 3e-4
  config.critic_lr = 3e-4

  config.env_name = 'hopper-medium-v2'
  config.policy_dims = (256, 256, 256)
  config.critic_dims = (256, 256, 256)

  config.discount = 0.99

  config.tau = 0.005  # For soft target updates.
  config.init_alpha = 1.0
  config.num_bc_steps = int(10e3)
  config.softmax_temperature = 1.0
  config.cql_alpha = 5.0
  config.max_q_backup = False
  config.deterministic_backup = True
  config.num_cql_samples = 10
  config.with_lagrange = False
  config.target_action_gap = 5.0

  return config
