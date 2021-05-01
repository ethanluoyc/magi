from typing import Sequence

from magi.agents.sac_ae import networks


def make_default_networks(
    environment_spec,
    num_critics: int = 2,
    critic_hidden_sizes: Sequence[int] = (256, 256),
    actor_hidden_sizes: Sequence[int] = (256, 256),
    latent_size: int = 50,
    log_std_min: float = -10.,
    log_std_max: float = 2.,
    num_filters: int = 32,
    num_layers: int = 4,
):

  def critic(x, a):
    x = networks.SACLinear(feature_dim=latent_size)(x)
    return networks.ContinuousQFunction(
        num_critics=num_critics,
        hidden_units=critic_hidden_sizes,
    )(x, a)

  def actor(x):
    x = networks.SACLinear(feature_dim=latent_size)(x)
    return networks.StateDependentGaussianPolicy(
        action_size=environment_spec.actions.shape[0],
        hidden_units=actor_hidden_sizes,
        log_std_min=log_std_min,
        log_std_max=log_std_max,
        clip_log_std=False,
    )(x)

  def encoder(x):
    return networks.SACEncoder(num_filters=num_filters, num_layers=num_layers)(x)

  # Encoder.
  return {
      "encoder": encoder,
      "critic": critic,
      "actor": actor,
  }
