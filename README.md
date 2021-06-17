# Magi
Reinforcement learning library in JAX built on top of [Acme](https://github.com/deepmind/acme).

[![pytest](https://github.com/ethanluoyc/magi/actions/workflows/ci.yaml/badge.svg?branch=develop)](https://github.com/ethanluoyc/magi/actions/workflows/ci.yaml)

## Installation
Create a new Python virtual environment and update
```python
python3 -m venv venv
source venv/bin/activate
python -m pip install -U setuptools wheel pip
pip install -e .
```

## Developing
We use [Poetry](https://python-poetry.org/)
for managing dependencies and building the package. If you are interested
in extending the library, first install Poetry following the information on their website.

Once Poetry is installed, navigate to the repository's root directory and run
```
poetry install
```
This will install all the dependencies needed and install magi as an editable package for local
development.
There are some optional dependencies.
For example, we do not install the packages
for the RL environments. Those can be install by running `poetry intall -E envs'`.

Also note that the installation process above will only install the CPU version of jaxlib.
To install the GPU versison, you need to use the packages distributed by the JAX team. 
Follow the instruction on JAX GitHub for how to install GPU-enabled jaxlib.

## Testing
On Linux, you can run tests with
```
poetry run pytest -n `grep -c ^processor /proc/cpuinfo` magi
```
