import base_config


def get_config():
  config = base_config.get_base_config()
  base_config.override_medium(config)
  config.domain_name = 'quadruped'
  config.task_name = 'run'
  # Reduce the replay size according to Appendix B
  config.max_replay_size = int(1e5)
  return config
