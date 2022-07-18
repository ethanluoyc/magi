"""Test automation with nox"""
import nox
import os

DEFAULT_PYTHON_VERSIONS = ["3.8", "3.9"]
PYTHON_VERSIONS = os.environ.get("NOX_PYTHON_VERSIONS",
                                 ",".join(DEFAULT_PYTHON_VERSIONS)).split(",")


def _install_acme(session):
  session.install("git+https://github.com/deepmind/acme.git#egg=dm-acme[jax]")


@nox.session(python=PYTHON_VERSIONS)
def lint(session):
  session.install("-r", "requirements/dev.txt")
  session.run("yapf", "-r", "--diff", "magi")
  session.run("isort", "magi", "--check", "--diff")
  session.run("pylint", "magi")


@nox.session(python=PYTHON_VERSIONS)
def typecheck(session):
  session.install("-r", "requirements/dev.txt")
  _install_acme(session)
  session.install(".")
  session.run("pytype", "-k", "-j", "8", "magi")
  examples = os.listdir("examples")
  for example in examples:
    if os.path.isdir(os.path.join("examples", example)):
      session.run("pytype", "-k", "-j", "8", "-d", "import-error",
                  os.path.join("examples", example))


@nox.session(python=PYTHON_VERSIONS)
def test(session):
  # Make GPU unavailable (Testing runs on CPU only).
  session.env["CUDA_VISIBLE_DEVICES"] = ""
  session.install("pytest", "pytest-xdist")
  _install_acme(session)
  # TODO(yl): Test installed copy instead of editable install
  session.install("-e", ".")
  session.install("pipdeptree")
  session.run("pipdeptree", "-p", "magi")
  session.run("pytest")
