from magi.projects.drq_v2.configs import humanoid_walk


def get_config():
  config = humanoid_walk.get_config()
  config.samples_per_insert = 32.0
  config.num_actors = 4
  config.log_every = 10.0
  return config
