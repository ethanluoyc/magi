from typing import Optional, Sequence

import haiku as hk
import jax
import jax.numpy as jnp


def ensemble_transform(fn, ensemble_size: int):
  fn_transformed = hk.transform(fn)

  def init(rng, *args):
    """Initialize the params for each network in the ensemble."""
    rngs = jax.random.split(rng, ensemble_size)
    in_axes = (0,) + (None,) * len(args)
    return jax.vmap(fn_transformed.init, in_axes=in_axes)(rngs, *args)

  def apply(params, rng, *args):
    """Apply the inputs with each network in the ensembles"""
    rngs = jax.random.split(rng, ensemble_size)
    in_axes = (0, 0) + (None,) * len(args)
    return jax.vmap(fn_transformed.apply, in_axes=in_axes)(params, rngs, *args)

  return hk.Transformed(init, apply)


def ensemble_transform_with_state(fn, ensemble_size: int):
  fn_transformed = hk.transform_with_state(fn)

  def init(rng, *args):
    """Initialize the params and state for each network in the ensemble."""
    rngs = jax.random.split(rng, ensemble_size)
    in_axes = (0,) + (None,) * len(args)
    return jax.vmap(fn_transformed.init, in_axes=in_axes)(rngs, *args)

  def apply(params, state, rng, *args):
    """Apply the inputs with each network in the ensembles"""
    rngs = jax.random.split(rng, ensemble_size)
    in_axes = (0, 0, 0) + (None,) * len(args)
    return jax.vmap(fn_transformed.apply, in_axes=in_axes)(params, state, rngs, *args)

  return hk.Transformed(init, apply)


class LayerNormMLP(hk.Module):
  """Simple feedforward MLP torso with initial layer-norm.
  This module is an MLP which uses LayerNorm (with a tanh normalizer) on the
  first layer and non-linearities (elu) on all but the last remaining layers.

  TODO(yl): investigate if using layer-norm allows us to remove normalization
  in the original PETS implementation.
  """

  def __init__(self,
               layer_sizes: Sequence[int],
               w_init=None,
               activation=jax.nn.elu,
               activate_final: bool = False):
    """Construct the MLP.
    Args:
      layer_sizes: a sequence of ints specifying the size of each layer.
      activate_final: whether or not to use the activation function on the final
        layer of the neural network.
    """
    super().__init__(name='feedforward_mlp_torso')
    if w_init is None:
      w_init = hk.initializers.UniformScaling(scale=0.333)

    self._network = hk.Sequential([
        hk.Linear(layer_sizes[0], w_init=w_init),
        hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
        jax.lax.tanh,
        hk.nets.MLP(layer_sizes[1:],
                    w_init=w_init,
                    activation=activation,
                    activate_final=activate_final),
    ])

  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    """Forwards the policy network."""
    return self._network(inputs)


def _truncated_normal_w_init(shape, dtype):
  assert len(shape) == 2
  fan_in = shape[0]
  std = 1. / (2 * jnp.sqrt(fan_in))
  mean = jax.lax.convert_element_type(0., dtype)
  std = jax.lax.convert_element_type(std, dtype)
  unscaled = jax.random.truncated_normal(hk.next_rng_key(),
                                         shape=shape,
                                         lower=-2.,
                                         upper=2.)
  return unscaled * std + mean


class GaussianMLP(hk.Module):

  def __init__(self,
               output_size: int,
               hidden_sizes: Sequence[int],
               *,
               activation=jax.nn.swish,
               min_logvar: float = -10.,
               max_logvar: float = 0.5,
               name: Optional[str] = None):
    super().__init__(name=name)
    self.output_size = output_size
    # self.mlp = LayerNormMLP(layer_sizes=hidden_sizes,
    #                         w_init=hk.initializers.VarianceScaling(
    #                             scale=1.0, distribution='truncated_normal'),
    #                         activation=activation,
    #                         activate_final=True)
    w_init = _truncated_normal_w_init
    self.mlp = hk.nets.MLP(hidden_sizes,
                           w_init=w_init,
                           activation=activation,
                           activate_final=True)
    # NOTE(yl): we currently have one set of min/max logvar parameters for each
    # network ensemble. In the orignal implementation, only a single set
    # is used. I am not sure this empirically affects the performance.
    # To implement the original behaviour, we need to ensemble transform
    # first and then create a single set of parameters for max/min logvar.
    # This can probably be achived by creating a haiku module that uses hk.vmap
    # to create the ensembles. I haven't tested if this works.
    # self.min_logvar = hk.get_parameter('min_logvar', (output_size,),
    #                                    init=hk.initializers.Constant(min_logvar))
    # self.max_logvar = hk.get_parameter('max_logvar', (output_size,),
    #                                    init=hk.initializers.Constant(max_logvar))
    self.min_logvar = jnp.ones(output_size) * min_logvar
    self.max_logvar = jnp.ones(output_size) * max_logvar
    # self.linear_mean = hk.Linear(self.output_size, name='mean')
    self.mean_and_logvar = hk.Linear(self.output_size * 2,
                                     w_init=w_init,
                                     name='mean_and_logvar')

  def __call__(self, x):
    h = self.mlp(x)
    # mean = self.linear_mean(h)
    # logvar = self.linear_logvar(h)
    mean, logvar = jnp.split(self.mean_and_logvar(h), 2, axis=-1)
    # This is the implementation detail discussed in the appendix
    logvar = self.max_logvar - jax.nn.softplus(self.max_logvar - logvar)
    logvar = self.min_logvar + jax.nn.softplus(logvar - self.min_logvar)
    return mean, logvar
