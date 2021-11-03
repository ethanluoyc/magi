from magi.projects.drq_v2.configs import default


def get_config():
    config = default.get_base_config()
    default.override_medium(config)
    config.domain_name = "quadruped"
    config.task_name = "run"
    # Reduce the replay size according to Appendix B
    config.max_replay_size = int(1e5)
    return config
