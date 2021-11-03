from magi.projects.drq_v2.configs import reacher_hard


def get_config():
    config = reacher_hard.get_config()
    config.samples_per_insert = 32.0
    config.num_actors = 4
    config.log_every = 10.0
    return config
