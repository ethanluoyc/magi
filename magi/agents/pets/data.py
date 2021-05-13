import jax.numpy as jnp

# TODO: make this acme.adders compatible.


class Dataset:

  def __init__(self):
    self.observations_t = []
    self.observations_tp1 = []
    self.rewards = []
    self.actions = []

  def add(self, obs_t, obs_tp1, action, reward):
    self.observations_t.append(obs_t)
    self.observations_tp1.append(obs_tp1)
    self.rewards.append(reward)
    self.actions.append(action)

  def collect(self):
    return (
        jnp.stack(self.observations_t),
        jnp.stack(self.observations_tp1),
        jnp.stack(self.actions),
        jnp.stack(self.rewards),
    )

  def reset(self):
    self.observations_t = []
    self.observations_tp1 = []
    self.rewards = []
    self.actions = []
