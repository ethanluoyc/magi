"""Behavior-cloning agent.

TODO(yl): this is simply a backport of a newer version of Acme's newer learner.
Remove this when after 0.2.3 is released
"""
from acme.agents.jax.bc.losses import logp  # noqa
from acme.agents.jax.bc.losses import mse  # noqa
from acme.agents.jax.bc.losses import peerbc  # noqa

from magi.agents.bc.learning import BCLearner  # noqa
