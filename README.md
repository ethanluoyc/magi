# Magi RL library in JAX

**[Installation](#installation)** |
**[Agents](./magi/agents)** |
**[Examples](./magi/examples)** |
**[Contributing](./CONTRIBUTING.md)** |
**[Documentation](./docs)**

[![pytest](https://github.com/ethanluoyc/magi/actions/workflows/ci.yaml/badge.svg?branch=develop)](https://github.com/ethanluoyc/magi/actions/workflows/ci.yaml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Magi is a RL library developed on top of [Acme](https://github.com/deepmind/acme).

_Note_: Magi is in alpha development so expect breaking changes!

## Installation
1. Create a new Python virtual environment
```python
python3 -m venv venv
```

2. Install dependencies and the package in editable mode by running

```python
pip install jax==0.2.11 jaxlib==0.1.64
pip install -e '.[dev,envs]'
```

Magi is tested against jax==0.2.11 and jaxlib==0.1.64 but it should also work
with more recent versions of jax and jaxlib.
Note that the installation above uses the CPU verison of JAX.
If you want to use JAX with the CUDA backend, follow the instructions
on JAX's GitHub repo for how to install the CUDA-enabled version of jaxlib.
For CUDA 11.3, you can replace the commands above with

```bash
pip install jax==0.2.11 jaxlib==0.1.64+cuda111 -f https://storage.googleapis.com/jax-releases/jax_releases.html # will use the jaxlib version with CUDA 11.3 support
```

If for some reason installation fails, first check out GitHub Actions
badge to see if this fails on the latest CI run. If the CI is successful,
then it's likely that there are some issues to setting up your own environment.
Refer to [.github/workflows/ci.yaml](.github/workflows/ci.yaml) as the official source
for how to set up the environment.

## Agents
magi includes popular RL algorithm implementation such as SAC, DrQ, SAC-AE and PETS.
Refer to [magi/agents](./magi/agents) for a full list of agents.

## Examples
Check out [magi/examples](./magi/examples) where
we include examples of using our RL agents on popular benchmark tasks.

## Testing
On Linux, you can run tests with
```
JAX_PLATFORM_NAME=cpu pytest -n `grep -c ^processor /proc/cpuinfo` magi
```

## Contributing
Refer to [CONTRIBUTING.md](./CONTRIBUTING.md).

## Acknowledgements
Magi is inspired by many of the open-source RL projects out there. Since we
have a very small number of developers, we take advantage of bits and pieces of
good implementation from other repositories.

## License
TODO: open-source timeline not yet decided.
