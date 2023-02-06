# Probabilistic Ensembles with Trajectory Sampling (PETS)

## Overview
This directory includes an implementation of
PETS [1] in JAX.

This implementation tries to match the implementation in mbrl-lib.
Currently, only TSInf propagation is implemented, but extension to other
propagation method should be straightforward.

## Examples
[magi.examples.pets](../../examples) include examples of running this implementation
of PETS on the environments used in [1].

[1] Kurtland Chua, Roberto Calandra, Rowan McAllister, Sergey Levine:
Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models. NeurIPS 2018: 4759-4770, https://sites.google.com/view/drl-in-a-handful-of-trials
