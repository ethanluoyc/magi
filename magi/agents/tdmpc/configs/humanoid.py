from magi.agents.tdmpc.configs import default


def get_config():
  config = default.get_config()
  config.task = "humanoid-walk"

  config.action_repeat = 2
  config.train_steps = 3000000 / config.get_oneway_ref("action_repeat")
  config.iterations = 12
  config.latent_dim = 100

  return config
