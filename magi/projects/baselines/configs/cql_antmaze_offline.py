"""IQL configuration for D4RL mujoco locomotion"""
import ml_collections


def get_config():
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

    config.actor_lr = 1e-4
    config.value_lr = 3e-4
    config.critic_lr = 3e-4

    config.hidden_dims = (256, 256, 256)

    config.discount = 0.99

    config.tau = 0.005  # For soft target updates.
    config.init_alpha = 1.0
    config.policy_eval_start = 40000
    config.temp = 1
    config.init_alpha_prime = 1
    config.max_q_backup = False
    config.deterministic_backup = True
    config.num_random = 10
    config.with_lagrange = True
    config.lagrange_thresh = 5.0

    return config
