"""Common types used throughout Magi."""

from typing import Any, Iterable, Mapping, Union

from dm_env import specs

NestedArray = Any
NestedTensor = Any

NestedSpec = Union[
    specs.Array,
    Iterable["NestedSpec"],
    Mapping[Any, "NestedSpec"],  # pytype: disable=not-supported-yet
]
