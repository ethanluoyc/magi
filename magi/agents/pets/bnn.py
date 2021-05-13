import haiku as hk
import jax
import jax.numpy as jnp


class BNN(hk.Module):

  def __init__(self, output_size: int, name=None):
    super().__init__(name=name)
    self.output_size = output_size

    self.mlp = hk.Sequential(
        [hk.Flatten(),
         hk.Linear(50), jax.nn.swish,
         hk.Linear(50), jax.nn.swish],
        name='mlp')
    self.min_logvar = hk.get_parameter('min_logvar', (output_size,),
                                       init=hk.initializers.Constant(-10.))
    self.max_logvar = hk.get_parameter('max_logvar', (output_size,),
                                       init=hk.initializers.Constant(0.5))
    self.linear_mean = hk.Linear(self.output_size)
    self.linear_logvar = hk.Linear(self.output_size)

  def __call__(self, x):
    h = self.mlp(x)
    mean = self.linear_mean(h)
    logvar = self.linear_logvar(h)
    # This is the implementation detail discussed in the appendix
    logvar = self.max_logvar - jax.nn.softplus(self.max_logvar - logvar)
    logvar = self.min_logvar + jax.nn.softplus(logvar - self.min_logvar)
    std = jnp.sqrt(jnp.exp(logvar))
    return mean, std
