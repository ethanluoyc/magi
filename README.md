# Magi RL library in JAX

**[Installation](#installation)** |
**[Agents](./magi/agents)** |
**[Examples](./magi/examples)** |
**[Contributing](./CONTRIBUTING.md)** |
**[Documentation](./docs)**


[![pytest](https://github.com/ethanluoyc/magi/actions/workflows/ci.yaml/badge.svg?branch=develop)](https://github.com/ethanluoyc/magi/actions/workflows/ci.yaml)

> **Note**
> Future development of JAX agents in Magi have moved to [Corax](https://github.com/ethanluoyc/corax)

Magi is a RL library in JAX that is fully compatible with [Acme](https://github.com/deepmind/acme).

In addition to the features provided by Acme, Magi offers implementation of RL
agents that are not found in the Acme repository as well as providing useful tools
for integrating experiment logging services such as WandB.

_Note_:
Magi is in alpha development so expect breaking changes!

Magi currently depends on HEAD version of dm-acme instead of the latest
release version on PyPI which is fairly old.

## Installation
1. Create a new Python virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies with the following commands.

```bash
pip install -U pip setuptools wheel
# Magi depends on latest version of dm-acme.
# The dependencies in setup.py are abstract which allows you to pin
# a specific version of dm-acme.
# The following command installs the latest version of dm-acme
pip install 'git+https://github.com/deepmind/acme.git#egg=dm-acme[jax,tf,examples]'
# Install magi in editable mode, with additional dependencies.
# In case you need to run examples on GPU, you should install the
# GPU version of JAX with a command like the following
pip install 'jax[cuda]<0.4' -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -e '.[jax]'
```

The base installation for magi does not list TensorFlow/JAX as a dependency.
However, note that JAX requires platform-specific installation
(CPU/GPU and CUDA versions). Furthermore, Acme depends on Reverb and LaunchPad
which requires them to be pinned against specific versions of TensorFlow. This should
be handled if you use install dm-acme with [jax,tf] extras. However, you can also
use install with different versions of TensorFlow/Reverb/Launchpad. In that case,
you should omit the extras and find compatible versions and pin those versions
accordingly.

If for some reason installation fails, first check out GitHub Actions
badge to see if this fails on the latest CI run. If the CI is successful,
then it's likely that there are some issues to setting up your own environment.
Refer to [.github/workflows/ci.yaml](.github/workflows/ci.yaml) as the official source
for how to set up the environment.

## Agents
magi includes popular RL algorithm implementation such as SAC, DrQ, SAC-AE and PETS.
Refer to [magi/agents](./magi/agents) for a full list of agents.

## Examples
Check out [examples](./examples) where
we include examples of using our RL agents on popular benchmark tasks.

## Testing
On Linux, you can run tests with
```
nox test
```

## Contributing
Refer to [CONTRIBUTING.md](./CONTRIBUTING.md).

## Acknowledgements
Magi is inspired by many of the open-source RL projects out there.
Here is a (non-exhaustive) list of related libraries and packages that Magi references:

* https://github.com/deepmind/acme
* https://github.com/ikostrikov/jaxrl
* https://github.com/tensorflow/agents
* https://github.com/rail-berkeley/rlkit


## License
[Apache License 2.0](https://github.com/ethanluoyc/magi/blob/develop/LICENSE)

## Citation
If you use Magi in your work,
please cite us according to the [CITATION](/CITATION.cff) file.
You may learn more about the CITATION file from [here](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/creating-a-repository-on-github/about-citation-files).
