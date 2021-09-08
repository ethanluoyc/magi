"""Default network architectures for TD3."""
import haiku as hk
import jax
import jax.numpy as jnp


class Actor(hk.Module):
    def __init__(self, action_dim, max_action):
        super().__init__()

        self.l1 = hk.Linear(256)
        self.l2 = hk.Linear(256)
        self.l3 = hk.Linear(action_dim)

        self.max_action = max_action

    def __call__(self, state):
        a = jax.nn.relu(self.l1(state))
        a = jax.nn.relu(self.l2(a))
        return self.max_action * jnp.tanh(self.l3(a))


class Critic(hk.Module):
    def __init__(self):
        super().__init__()
        # Q1 architecture
        self.l1 = hk.Linear(256)
        self.l2 = hk.Linear(256)
        self.l3 = hk.Linear(1)

        # Q2 architecture
        self.l4 = hk.Linear(256)
        self.l5 = hk.Linear(256)
        self.l6 = hk.Linear(1)

    def __call__(self, state, action):
        sa = jnp.concatenate([state, action], axis=1)

        q1 = jax.nn.relu(self.l1(sa))
        q1 = jax.nn.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = jax.nn.relu(self.l4(sa))
        q2 = jax.nn.relu(self.l5(q2))
        q2 = self.l6(q2)
        return jnp.squeeze(q1, axis=-1), jnp.squeeze(q2, axis=-1)

