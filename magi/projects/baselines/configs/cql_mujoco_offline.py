"""IQL configuration for D4RL mujoco locomotion"""
import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 42
    config.env_name = "halfcheetah-medium-v2"

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

    config.policy_dims = (256, 256, 256)
    config.critic_dims = (256, 256, 256)

    config.discount = 0.99

    config.tau = 0.005  # For soft target updates.
    config.init_alpha = 1.0
    config.num_bc_steps = int(10e3)
    config.softmax_temperature = 1.0
    config.cql_alpha = 5.0
    config.max_q_backup = False
    config.deterministic_backup = True
    config.num_cql_samples = 10
    config.with_lagrange = False
    config.target_action_gap = 5.0

    # Logging
    config.log_to_wandb = False

    return config
