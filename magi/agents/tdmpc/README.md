# TD-MPC implementation in JAX

This folder contains an implementation of TD-MPC introduced in ([Hansen et al., 2022a]).
The implementation is based on the original [PyTorch version](https://github.com/nicklashansen/tdmpc)
with a few notable differences:

1. We use Reverb as the replay buffer implementation. In particular, the SequenceAdder
is used for inserting sequences of experience that are consumed by the learner. New
items are inserted with a priority value of 1.0 instead of the current maximum priority.
It is difficult to add new items with the current maximum priority using a distributed
replay buffer implementation such as Reverb.
2. The learner is updated continuously instead of at the end of every episode.
This is more consistent with the typical way agents are implemented in Acme.
The rate the learner is updated can be controlled by Reverb's SPI parameters.
3. The loss for the model and policy is computed in a single SGD step, with
`jax.lax.stop_gradient` inserted in the appropriate places to be consistent with the
original implementation.
4. The target network is updated at every learner step instead of every two steps.
As a result, it may be useful to adjust the default target network update rate.

The implementation was tested on a few environments (e.g. walker, humanoid, dog) and
the was found to match the results in the paper. None of the modification above
was found to hurt performance. However, more thorough benchmark may be needed.
Please only use this implementation as a starting point for your project and
refer to the original implementation for reproducing results.

## TODOs
1. Support pixel observations. This should be relatively easy as most of the
code is written without assuming the modality of the observations. However,
we do need to add options for passing in data augmentations to the agent. Also,
we may want to reduce memory footprint of the pixel observations by employing
agent-side observation stacking as in the Acme MPO.
2. Support incorporating additional demonstrations as in MoDem ([Hansen et al., 2022b]).
This should also
be relatively easy by adding a `make_demonstration` factory and creating the
correct dataset iterator.
3. Expose more logging on the actor side for easier debugging.

[Hansen et al., 2022a]: (https://proceedings.mlr.press/v162/hansen22a.html)
[Hansen et al., 2022b]: (https://arxiv.org/abs/2212.05698)
