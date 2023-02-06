# Visual Continuous Control with DrQ-v2

This project includes experimental training scripts for running single-process
and distributed implmentation of DrQ-v2 from

> Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning,
Denis Yarats, Rob Fergus, Alessandro Lazaric, Lerrel Pinto,
https://arxiv.org/abs/2107.09645

## Installation
1. Follow the [installation instructions](/README.md) of magi.
2. Follow the
[installation instructions](https://github.com/deepmind/dm_control/tree/master) of dm_control.

## Running the training script
For single process implementation use

```python
export MUJOCO_GL=egl
magi/projects/drq_v2/main.py \
    --config magi/projects/drq_v2/configs/reacher_hard.py \
    --workdir /tmp/drq_v2
```

For distributed implementation with launchpad use

```python
export MUJOCO_GL=egl
JAX_PLATFORM_NAME=cpu magi/projects/drq_v2/lp_main.py \
    --config magi/projects/drq_v2/configs/reacher_hard_distributed.py
    --workdir /tmp/drq_v2
```

## Configuration

We use [ml_collections](https://github.com/google/ml_collections) for managing configuration, you may want to look
at the ConfigDict referenced by the training script for additional hyperparameters.

You can configure the training script to log to Weights and Biases
by setting `config.log_to_wandb` to true and configure the `config.wandb_project`
and config.wandb_entity properly.

## Troubleshooting

1. Issues to failing to initialize CUDA.

   You may need to adjust the JAX GPU memory allocation settings.
   When GPU rendering backend in dm_control is used, the default JAX GPU memory preallocation may fail. Refer to https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html for more details. For example, adjust JAX to use the amount of pre-allocated memory by running
   ```
   export XLA_PYTHON_CLIENT_MEM_FRACTION=.6
   ```

2. Errors relating to importing tensorflow, launchpad or reverb.

   Make sure that you are installing the pinned versions of dm-launchpad-nightly,
   dm-reverb, etc. from [requirements.txt](/requirements.txt)

   Here is a list of related issues and solutions that may be of interest

   - https://github.com/deepmind/acme/issues/47

3. Issues trying to load the dm_control environments.

   Consult the documentation of dm_control for solutions.

4. Issues with running distributed agents.

   First, make sure that dm-launchpad-nightly has been installed properly. While we have
   disabled TensorFlow from consuming GPU memory, it may be possible that our solution
   is not robust. In that case, substitute the version of tensorflow with the same version
   of tensorflow-cpu may solve the problem. For example, suppose you have tensorflow==2.6.0
   installed, you should run
   ```
   pip uninstall tensorflow
   pip install tensorflow-cpu==2.6.0
   ```

5. Other GPU related issues.

   Other GPU related issues were most likely not due to how we implement the agents,
   please try to troubleshoot these problems with the libraries.

## Potential issues
1. Not enough memory

   Since we use Reverb as the replay backend, which stores all replay data
   in memory, running the DrQ agents with 1M replay may be prohibitively expensive.
   For a 1M replay, assuming images of size 84 x 84 x 3 and frame stacking of 3,
   storing 1M observations require 84 x 84 x 3 x 3 x 1e6 bytes ~ 64GB of memory.
   Storing the images in the transitions would thus take ~128GB of memory.
   However, since Reverb does compression under the hood, the actual memory usage
   is usually much smaller than this.
   We have not yet investigated how to reduce memory consumption further.
   Nevertheless, if you do run into issues with not having enought memory,
   the only solution right now is to run with a smaller replay buffer.
