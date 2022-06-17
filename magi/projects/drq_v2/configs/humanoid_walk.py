from magi.projects.drq_v2.configs import default


def get_config():
  config = default.get_base_config()
  default.override_hard(config)
  config.domain_name = 'humanoid'
  config.task_name = 'walk'
  config.lr = 8e-5
  config.latent_size = 100
  return config
