"""Configuration for running PETS pusher expeirment"""
import base_config
import jax
import jax.numpy as np


def get_config():
    config = base_config.get_base_config()

    config.task_horizon = 150
    config.planning_horizon = 30
    config.activation = jax.nn.swish
    config.lr = 1e-3
    config.weight_decay = 4e-4
    config.population_size = 500
    config.num_epochs = 5
    config.patience = 5

    def get_goal(env):
        return env.ac_goal_pos

    def obs_preproc(obs):
        return obs

    def obs_postproc(obs, pred):
        return obs + pred

    def targ_proc(obs, next_obs):
        return next_obs - obs

    def obs_cost_fn(obs, goal):
        to_w, og_w = 0.5, 1.25
        tip_pos, obj_pos, goal_pos = obs[:, 14:17], obs[:, 17:20], goal

        tip_obj_dist = np.sum(np.abs(tip_pos - obj_pos), axis=1)
        obj_goal_dist = np.sum(np.abs(goal_pos - obj_pos), axis=1)
        return to_w * tip_obj_dist + og_w * obj_goal_dist

    def ac_cost_fn(acs):
        return 0.1 * np.sum(np.square(acs), axis=1)

    def reward_fn(obs, acs, goal):
        return -(obs_cost_fn(obs, goal) + ac_cost_fn(acs))

    config.env_name = "pusher"
    config.get_goal = get_goal
    config.obs_preproc = obs_preproc
    config.obs_postproc = obs_postproc
    config.targ_proc = targ_proc
    config.reward_fn = reward_fn
    return config
