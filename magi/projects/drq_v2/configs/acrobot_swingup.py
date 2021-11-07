from magi.projects.drq_v2.configs import default


def get_config():
    config = default.get_base_config()
    default.override_medium(config)
    config.domain_name = "acrobot"
    config.task_name = "swingup"
    return config
