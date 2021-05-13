# Probabilistic Ensembles with Trajectory Sampling (PETS)

## Overview
This directory includes an implementation of
PETS [1] in JAX.

This implementation is WIP, and does not match the original implementation,
so it should not used as a baseline for providing performance of PETs.
At this point, it is used to illustrate how to implement NN ensembles and model-based agents in JAX.

## TODOs
1. Match the architecture and loss function in as well as the training details in [1]
2. Run experiments on the original benchmarks proposed in [1].

[1] Kurtland Chua, Roberto Calandra, Rowan McAllister, Sergey Levine:
Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models. NeurIPS 2018: 4759-4770, https://sites.google.com/view/drl-in-a-handful-of-trials
