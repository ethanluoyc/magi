"""Types used in DrQ"""
from typing import Callable

from acme import types
import jax.numpy as jnp

DataAugmentation = Callable[[jnp.ndarray, types.NestedArray], types.NestedArray]
