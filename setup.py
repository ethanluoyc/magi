#!/usr/bin/env python3
"""Install scripts for setuptools."""
from importlib import util as import_util

from setuptools import find_packages
from setuptools import setup

spec = import_util.spec_from_file_location("_metadata", "magi/_metadata.py")
_metadata = import_util.module_from_spec(spec)
spec.loader.exec_module(_metadata)
version = _metadata.__version__

with open("README.md", "rt", encoding="utf-8") as f:
    long_description = f.read()

tf_requirements = [
    "tensorflow",
    "tensorflow_probability",
    "rlds",
    # Reverb and launchpad must pinned against specific versions of TF
    "dm-reverb",
    "dm-launchpad",
]

jax_requirements = [
    # Pinned with jax<0.4 as the pinned version of TF/TFP is not compatible
    # with jax>=0.4. However, if you use later versions of TFP/TF magi should work fine.
    "jax<0.4",
    "jaxlib",
    "chex",
    "optax",
    "dm-haiku",
    "rlax",
] + tf_requirements

setup(
    name="magi",
    version=version,
    description="Reinforcement Learning in JAX",
    license="Apache License, Version 2.0",
    url="https://github.com/ethanluoyc/magi/",
    author="Yicheng Luo",
    author_email="ethanluoyc@gmail.com",
    keywords="reinforcement-learning machine learning jax ai",
    project_urls={
        "Source": "https://github.com/ethanluoyc/magi/",
        "Tracker": "https://github.com/ethanluoyc/magi/issues",
    },
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8, <3.10",
    install_requires=["absl-py", "dm-tree", "numpy", "dm-acme", "gym"],
    extras_require={"jax": jax_requirements},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
