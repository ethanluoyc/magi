import base_config

def get_config():
  config = base_config.get_base_config()
  base_config.override_hard(config)
  config.domain_name = 'humanoid'
  config.task_name = 'walk'
  config.lr = 8e-5
  config.latent_size = 100
  return config
