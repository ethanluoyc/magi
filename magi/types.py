"""Common types used throughout Magi."""
# pytype: disable=not-supported-yet
from typing import Any, Iterable, Mapping, Union

from dm_env import specs

NestedArray = Any
NestedTensor = Any

NestedSpec = Union[
    specs.Array,
    Iterable["NestedSpec"],  # pytype: disable=not-supported-yet
    Mapping[Any, "NestedSpec"],  # pytype: disable=not-supported-yet
]
