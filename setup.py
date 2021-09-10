#!/usr/bin/env python3
import setuptools

if __name__ == "__main__":
    setuptools.setup(
        install_requires=[
            "jax",
            "jaxlib",
            "dm-acme[jax]==0.2.2",
            "wandb",
            "ml_collections",
        ],
        extras_require={"envs": ["gym", "bsuite", "dm-control"]},
    )
