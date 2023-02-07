"""Magi is a JAX RL library."""


try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version

    __version__ = version("magi")

    del version, PackageNotFoundError
except PackageNotFoundError:
    # package is not installed
    pass
