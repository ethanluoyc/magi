"""Test automation with nox"""
import multiprocessing
import os

import nox

DEFAULT_PYTHON_VERSIONS = ["3.8", "3.9"]
PYTHON_VERSIONS = os.environ.get(
    "NOX_PYTHON_VERSIONS", ",".join(DEFAULT_PYTHON_VERSIONS)
).split(",")


@nox.session(python=PYTHON_VERSIONS)
def lint(session):
    session.install("-r", "requirements/dev.txt")
    session.run("black", "--check", "--diff", "magi")
    session.run("isort", "magi", "--check", "--diff")
    session.run("pylint", "magi")


@nox.session(python=PYTHON_VERSIONS)
def test(session):
    # Make GPU unavailable (Testing runs on CPU only).
    session.env["CUDA_VISIBLE_DEVICES"] = ""
    session.env["JAX_PLATFORM_NAME"] = "cpu"
    session.install("pytest", "pytest-xdist")
    session.install(
        "git+https://github.com/deepmind/acme.git#egg=dm-acme[jax,tf,examples]",
        "jax[cpu]<0.4",
    )
    # TODO(yl): Test installed copy instead of editable install
    session.install("-e", ".[jax]")
    session.install("pipdeptree")
    session.run("pipdeptree", "-p", "magi[jax]")
    session.run("pytest", "-n", str(min(multiprocessing.cpu_count(), 16)))
