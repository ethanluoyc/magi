"""Loggers for logging data."""
import abc
from typing import Any, Mapping


class Logger(abc.ABC):
    """logger interface"""

    @abc.abstractmethod
    def write(self, data: Mapping[str, Any]):
        """Log data to the logger."""
