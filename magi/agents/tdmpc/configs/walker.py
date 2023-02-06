from magi.agents.tdmpc.configs import default


def get_config():
    config = default.get_config()
    config.task = "walker-walk"
    config.action_repeat = 2

    return config
