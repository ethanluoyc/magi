# DrQ-v2: Improved Data-Augmented RL Agent

This is a JAX implementation of

> Yarats, D., Fergus, R., Lazaric, A., & Pinto, L. (2021).
_Mastering visual continuous control: Improved data-augmented reinforcement learning_. arXiv preprint [arXiv:2107.09645](https://arxiv.org/abs/2107.09645).

The official PyTorch implementation can be found at

    https://github.com/facebookresearch/drqv2


## TODOs
1. Implement the data augmentation used by the original implementation. Right now
  we use the random crop augmentation from
  https://github.com/ikostrikov/jax-rl/blob/main/jax_rl/agents/drq/augmentations.py.
  This version does not perform bilinear interpolation after cropping the image,
  which was found by the authors to further boost performance.
2. Investigate better initialization for the policy. It's quite common to initialize
  the weights of the last layer of the policy to be near zero, as done in Acme's D4PG.
  It's worth investigating if this addition can further improve performance.

<!-- References -->
[Yarats et al., 2021]: https://arxiv.org/abs/2107.09645
