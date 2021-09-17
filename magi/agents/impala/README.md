# Importance-weighted actor-learner architecture (IMPALA)

This agent is an implementation of the algorithm described in _IMPALA: Scalable
Distributed Deep-RL with Importance Weighted Actor-Learner Architectures_
([Espeholt et al., 2018]).

In magi, we include both a single-process implementation of IMPALA as well as
a distributed implementation (with dm-launchpad). Specifically:

* [agent.py](./agent.py) includes the single process implementation. This is a direct
fork of the implementation in Acme, with additional support for allowing truncated
sequences to be added to the Reverb queue.
* [agent_distributed.py](./agent_distributed.py) includes the distributed implementation

The construction of an IMPALA agent is performed by the `Builder`, which knows how to
construct the individual components of the IMPALA architecture (actors, learners, replays etc).
The builder is then used by a `Layout`, which knows how to construct agents in the single-process
case as well as in the distributed case.
These interfaces may be further lifted out of the impala subdirectory as we implement more distributed agents,
and earliers implementation of other agents may be refactored to be like the IMPALA implementation
to support both single-process and distributed runtime.

[Espeholt et al., 2018]: https://arxiv.org/abs/1802.01561
