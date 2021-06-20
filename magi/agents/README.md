# Agents

Magi includes a few RL agents implemented in JAX. The JAX agents
use the [DeepMind JAX stack](https://deepmind.com/blog/article/using-jax-to-accelerate-our-research) and
is compatible with [Acme](https://github.com/deepmind/acme).

The implementation aims to be simple and self-contained whenever it is possible.
It is best to simply fork the agents for your own purpose.

For the JAX agents, we use

1. [dm-haiku](https://github.com/deepmind/dm-haiku) for implementing the neural network modules,
2, [optax](https://github.com/deepmind/optax) for building optimizers,
3. [rlax](https://github.com/deepmind/rlax/tree/master/rlax) for RL loss functions,
4. TensorFlow Probability's (TFP) JAX [substrates](https://www.tensorflow.org/probability/api_docs/python/tfp/substrates/jax) and [Distrax](https://github.com/deepmind/distrax) to build
probability distributions primitives,
5. [dm-reverb](https://github.com/deepmind/reverb) for scalable replay buffer implementation.

Here is a list of the agents that's currently available in Magi. Refer to the code
of individual agents which include examples of running them on benchmarks.

## Model-free

Agent                                                          | Paper                    | Code
-------------------------------------------------------------- | ----------------------:  | ---:
Soft Actor-Critic (SAC)                                        | [Haarnoja et al., 2018]  | [sac](./sac)
Soft Actor-Critic Auto-encoder (SAC-AE)                        | [Yarats et al., 2020]    | [sac_ae](./sac_ae)
Data Regulalized Q (DrQ)                                       | [Kostrikov et al., 2020] | [drq](./drq)


## Model-based

Agent                                                          | Paper                    | Code
-------------------------------------------------------------- | ----------------------:  | ---:
Probabilistic Ensembles with Trajectory Sampling (PETS)        | [Chua et al., 2018]      | [pets](./pets)

## TODOs
If possible, we can make some of the agents we have here upstream as part of Acme.

<!-- References -->
[Haarnoja et al., 2018]: https://arxiv.org/abs/1801.01290
[Yarats et al., 2020]: https://arxiv.org/abs/1910.01741
[Kostrikov et al., 2020]: https://arxiv.org/abs/2004.13649
[Chua et al., 2018]: https://arxiv.org/abs/1805.12114
