import numpy as np


def evaluate(actor, env, num_episodes=200):
    actor.update(wait=True)
    episode_lengths = []
    episode_returns = []

    for _ in range(num_episodes):
        ep_step = 0
        ep_ret = 0
        timestep = env.reset()
        actor.observe_first(timestep)
        while not timestep.last():
            action = actor.select_action(timestep.observation)
            timestep = env.step(action)
            ep_step += 1
            ep_ret += timestep.reward
        episode_lengths.append(ep_step)
        episode_returns.append(ep_ret)
    return {
        "eval_average_episode_length": np.mean(episode_lengths),
        "eval_average_episode_return": np.mean(episode_returns),
    }
