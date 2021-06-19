# Magi RL library in JAX

Magi is a RL library developed on top of [Acme](https://github.com/deepmind/acme).

**[Installation](#installation)** |
**[Agents](./magi/agents)** |
**[Contributing](./CONTRIBUTING.md)**
<!-- **[Examples]** -->

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

If for some reason installation fails, first check out GitHub Actions
badge to see if this fails on the latest CI run. If the CI is successful,
then it's likely that there are some issues to setting up your own environment.
Refer to [.github/workflows/ci.yaml](.github/workflows/ci.yaml) as the official source
for how to set up the environment.

## Testing
On Linux, you can run tests with
```
pytest -n `grep -c ^processor /proc/cpuinfo` magi
```

## Contributing
Refer to [CONTRIBUTING.md](./CONTRIBUTING.md)
