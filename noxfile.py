"""Test automation with nox"""
import nox


@nox.session
def tests(session):
    session.env["CUDA_VISIBLE_DEVICES"] = ""
    session.install("-U", "pip", "wheel", "setuptools")
    session.install("pytest", "pytest-xdist")
    session.install("-r", "requirements.txt")
    session.install("-e", ".", "--no-deps")
    session.run("make", "test", external=True)
