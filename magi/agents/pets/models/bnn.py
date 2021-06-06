from typing import Optional, Sequence
import haiku as hk
import jax
import jax.numpy as jnp


class BNN(hk.Module):

  def __init__(self,
               output_size: int,
               hidden_sizes: Sequence[int],
               activation=jax.nn.swish,
               min_logvar: float = -10.,
               max_logvar: float = 0.5,
               name: Optional[str] = None):
    super().__init__(name=name)
    self.output_size = output_size
    self.mlp = hk.nets.MLP(hidden_sizes, activation=activation, activate_final=True)
    self.min_logvar = hk.get_parameter('min_logvar', (output_size,),
                                       init=hk.initializers.Constant(output_size // 2
                                                                     * min_logvar))
    self.max_logvar = hk.get_parameter('max_logvar', (output_size,),
                                       init=hk.initializers.Constant(output_size // 2
                                                                     * max_logvar))
    self.linear_mean = hk.Linear(self.output_size, name='mean')
    self.linear_logvar = hk.Linear(self.output_size, name='logvar')

  def __call__(self, x):
    h = self.mlp(x)
    mean = self.linear_mean(h)
    logvar = self.linear_logvar(h)
    # This is the implementation detail discussed in the appendix
    logvar = self.max_logvar - jax.nn.softplus(self.max_logvar - logvar)
    logvar = self.min_logvar + jax.nn.softplus(logvar - self.min_logvar)
    std = jnp.sqrt(jnp.exp(logvar))
    return mean, std
