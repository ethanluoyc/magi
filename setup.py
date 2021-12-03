#!/usr/bin/env python3
"""Install scripts for setuptools."""
from importlib import util as import_util

from setuptools import find_packages
from setuptools import setup

spec = import_util.spec_from_file_location("_metadata", "magi/_metadata.py")
_metadata = import_util.module_from_spec(spec)
spec.loader.exec_module(_metadata)
version = _metadata.__version__


with open("README.md") as f:
    long_description = f.read()

setup(
    name="magi",
    version=version,
    description="Reinforcement Learning in JAX",
    author="Yicheng Luo",
    author_email="ethanluoyc@gmail.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.7, <3.9",
    install_requires=[
        "absl-py",
        "numpy",
        # Tensorflow
        "tensorflow",
        "tensorflow_probability",
        # JAX
        "jax",
        # DeepMind JAX eco-system
        "chex",
        "optax",
        "rlax",
        "dm-haiku",
        "dm-reverb",
        "dm-launchpad",
        "dm-acme[jax,launchpad]>=0.2.3",
        "dm-tree",
        "ml_collections",
        "gym",
        "PyYAML",
    ],
)
