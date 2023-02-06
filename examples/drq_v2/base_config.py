import ml_collections


def get_base_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()

    config.seed = 42

    # Environment config
    config.domain_name = "reacher"
    config.task_name = "hard"
    config.frame_stack = 3
    config.action_repeat = 2
    config.discount = 0.99

    config.num_frames = int(3e6)

    # Evaluation config
    config.eval_freq = 10000
    config.eval_episodes = 10

    # Replay config
    config.max_replay_size = 1000000
    config.min_replay_size = 2000
    config.batch_size = 256
    config.prefetch_size = 4

    # sigma schedule
    config.sigma_start = 1.0
    config.sigma_end = 0.1
    config.sigma_schedule_steps = 500000

    config.lr = 1e-4
    config.samples_per_insert = 128.0
    # Dimensionality of the latent feature
    config.latent_size = 50
    config.n_step = 3
    config.tau = 0.01
    config.noise_clip = 0.3

    # Logging config
    config.log_every = 10.0
    config.learner_log_time = 5.0
    config.train_log_time = 1.0
    config.eval_log_time = 1.0
    config.log_to_wandb = False  # whether to log result to wandb
    config.wandb_project = "magi"  # wandb project to log to
    config.wandb_entity = "ethanluoyc"  # entity of the runner
    return config


def override_easy(config):
    config.num_frames = int(1e6)
    config.sigma_schedule_steps = 100000


def override_medium(config):
    config.num_frames = int(3e6)
    config.sigma_schedule_steps = 500000


def override_hard(config):
    config.num_frames = int(30e6)
    config.sigma_schedule_steps = 2000000
