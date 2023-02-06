import base_config


def get_config():
    config = base_config.get_base_config()
    base_config.override_medium(config)
    config.domain_name = "reacher"
    config.task_name = "hard"
    return config
