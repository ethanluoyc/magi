from ml_collections import config_dict


def get_config():
    config = config_dict.ConfigDict()

    config.task = "quadruped-run"
    config.action_repeat = 4
    action_repeat = config.get_oneway_ref("action_repeat")
    config.discount = 0.99
    config.episode_length = 1000 // action_repeat
    config.train_steps = 500000 // action_repeat

    # planning
    config.iterations = 6
    config.num_samples = 512
    config.num_elites = 64
    config.mixture_coef = 0.05
    config.min_std = 0.05
    config.temperature = 0.5
    config.momentum = 0.1

    # learning
    config.batch_size = 512
    config.max_buffer_size = 1000000
    config.horizon = 5
    config.reward_coef = 0.5
    config.value_coef = 0.1
    config.consistency_coef = 2
    config.rho = 0.5
    config.kappa = 0.1
    config.lr = 1e-3
    schedule_steps = 25000
    config.std_schedule = dict(
        name="linear_schedule",
        kwargs=dict(
            init_value=0.5,
            end_value=config.get_oneway_ref("min_std"),
            transition_steps=schedule_steps,
        ),
    )
    # TODO(yl): Investigate if setting to episode_length works equally well.
    # i.e.
    # config.variable_update_period = config.get_oneway_ref('episode_length')
    config.variable_update_period = 1
    config.horizon_schedule = dict(
        name="linear_schedule",
        kwargs=dict(
            init_value=1,
            end_value=config.get_oneway_ref("horizon"),
            transition_steps=schedule_steps,
        ),
    )
    config.per_alpha = 0.6
    config.per_beta = 0.4
    config.grad_clip_norm = 10
    config.seed_steps = 5000
    config.update_freq = 2
    config.tau = 0.01

    # architecture
    config.enc_dim = 256
    config.mlp_dim = 512
    config.latent_dim = 50

    # wandb (insert your own)
    config.use_wandb = False
    config.wandb_project = None
    config.wandb_entity = None
    config.wandb_name = None

    # misc
    config.seed = 1
    config.exp_name = "default"
    config.eval_freq = 20000
    config.eval_episodes = 10
    config.save_video = False
    config.save_model = False
    return config
