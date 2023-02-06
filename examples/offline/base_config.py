"""Base configuration for offline experiments."""
import ml_collections


def get_base_config():
    """Get base configutation."""
    config = ml_collections.ConfigDict()

    config.seed = 42
    config.env_name = "antmaze-medium-play-v0"

    # Number of episodes used for evaluation.
    config.eval_episodes = 10
    # Eval interval.
    config.eval_interval = 5000
    # Mini batch size.
    config.batch_size = 256
    # Number of training steps.
    config.num_steps = int(1e6)

    config.log_to_wandb = False

    return config
