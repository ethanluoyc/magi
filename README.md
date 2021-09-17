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
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies and the package in editable mode by running

```bash
pip install -r requirements.txt # This uses pinned dependencies, you may adjust this for your needs.
pip install -e .
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
