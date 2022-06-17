"""Common types used throughout Magi."""
# pytype: disable=not-supported-yet
from typing import Any, Iterable, Mapping, Union

from dm_env import specs

NestedArray = Any
NestedTensor = Any

# pytype: disable=not-supported-yet
NestedSpec = Union[specs.Array, Iterable['NestedSpec'], Mapping[Any,
                                                                'NestedSpec']]
# pytype: enable=not-supported-yet
