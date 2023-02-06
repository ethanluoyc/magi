from magi.agents.tdmpc.configs import default


def get_config():
    config = default.get_config()
    config.task = "dog-walk"

    config.lr = 3e-4
    config.action_repeat = 2
    config.train_steps = 5000000 // config.get_oneway_ref("action_repeat")
    config.iterations = 8
    config.latent_dim = 100
    config.batch_size = 2048

    return config
