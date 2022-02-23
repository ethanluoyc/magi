"""Test automation with nox"""
import nox


@nox.session
def tests(session):
    session.env["CUDA_VISIBLE_DEVICES"] = ""
    session.install(
        "-r",
        "requirements/tests.txt",
    )
    session.run("make", "integration-test", external=True)


@nox.session
def lint(session):
    session.install("-r", "requirements/base.txt", "-r", "requirements/dev.txt")
    # session.run("pylint", "magi")
    session.run("pytype", "--config", "pytype.cfg", "-j", "4", "magi")
