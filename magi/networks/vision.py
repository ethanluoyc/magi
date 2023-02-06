"""Common torso networks used for processing image observations"""
from typing import Callable, Sequence

import haiku as hk
import jax
import jax.numpy as jnp


class DrQTorso(hk.Module):
    """DrQ Torso inspired by the second DrQ paper [Yarats et al., 2021].
    [Yarats et al., 2021] https://arxiv.org/abs/2107.09645
    """

    def __init__(
        self,
        activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
        data_format: str = "NHWC",
        name: str = "drq_torso",
    ):
        super().__init__(name=name)

        gain = 2**0.5 if activation is jax.nn.relu else 1.0

        def build_conv_layer(
            name: str,
            output_channels: int = 32,
            kernel_shape: Sequence[int] = (3, 3),
            stride: int = 1,
        ):
            return hk.Conv2D(
                output_channels=output_channels,
                kernel_shape=kernel_shape,
                stride=stride,
                padding="SAME",
                data_format=data_format,
                w_init=hk.initializers.Orthogonal(scale=gain),
                b_init=jnp.zeros,
                name=name,
            )

        self._network = hk.Sequential(
            [
                build_conv_layer("conv_0", stride=2),
                activation,
                build_conv_layer("conv_1", stride=1),
                activation,
                build_conv_layer("conv_2", stride=1),
                activation,
                build_conv_layer("conv_3", stride=1),
                activation,
                hk.Flatten(),
            ]
        )

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        if not jnp.issubdtype(inputs, jnp.floating):
            raise ValueError(
                "Expect inputs to be float pixel values normalized between 0 to 1."
            )
        # Normalize to -0.5 to 0.5
        preprocessed_inputs = inputs - 0.5
        torso_output = self._network(preprocessed_inputs)

        return torso_output
