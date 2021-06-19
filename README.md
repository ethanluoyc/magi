# Magi
Reinforcement learning library in JAX built on top of [Acme](https://github.com/deepmind/acme).

[![pytest](https://github.com/ethanluoyc/magi/actions/workflows/ci.yaml/badge.svg?branch=develop)](https://github.com/ethanluoyc/magi/actions/workflows/ci.yaml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Installation
1. Create a new Python virtual environment
```python
python3 -m venv venv
```

2. Install dependencies and the package in editable mode by running

```python
pip install -f https://storage.googleapis.com/jax-releases/jax_releases.html -e .
pip install -e '.[dev,envs]'
```

## Testing
On Linux, you can run tests with
```
pytest -n `grep -c ^processor /proc/cpuinfo` magi
```
