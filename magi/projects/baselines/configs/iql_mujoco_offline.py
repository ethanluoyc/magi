"""IQL configuration for D4RL mujoco locomotion"""
import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 42
    config.env_name = "halfcheetah-expert-v2"

    # Number of episodes used for evaluation.
    config.eval_episodes = 10
    # Eval interval.
    config.eval_interval = 5000
    # Mini batch size.
    config.batch_size = 256
    # Number of training steps.
    config.num_steps = int(1e6)

    config.actor_lr = 3e-4
    config.value_lr = 3e-4
    config.critic_lr = 3e-4

    config.hidden_dims = (256, 256)

    config.discount = 0.99

    config.expectile = 0.7  # The actual tau for expectiles.
    config.temperature = 3.0
    # Not used for now
    # config.dropout_rate = None

    config.tau = 0.005  # For soft target updates.

    return config
