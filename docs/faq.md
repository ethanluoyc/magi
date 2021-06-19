# FAQ

- __Why dm_env instead of gym?__ While gym.Env is the most popular
interface for defining RL environments, since we use a lot of Acme's
abstraction for building our agents, it is easier to interface with dm_env
instead of gym.

    In addition, nested observation and action spaces are represented
    as nested trees of dm_env.ArraySpec and this makes it easy to use a tree-processing
    library like dm-tree to map over them.

    Furthermore, we found that the Environment class in dm_env has better semantics. For example, in dm_env, the TimeStep distinguishes
    between final steps that correspond to a terminal step and steps that
    correspond to truncated episodes. Providing such a distinction makes it
    easier to implement agents for continuous control tasks. Adapting gym.Env to be compatible
    is easy: simple use the Acme GymWrappers

- __How should I use magi for my own project?__.
    We recommend either using magi either as a dependency or creating a dedicated package for your project under [projects/](../projects).
    This makes it easier to differentiate changes that are project-specific
    from changes that would make general RL research easier. However, should you find
    some changes in magi to be really useful and believe that everyone would benefit
    from it as a whole, then we would really love you to contribute those changes
    upstream. Refer to [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines on how to
    contribute to Magi.

    For extending an existing agent, it is best to fork an agent for your
    own development. You may do so by copying the agent from magi/agents
    to your dedicated project directory. This of course requires you to rewrite
    the import statements and you can do automatically with a few `sed` commands.
    The agents in Magi are meant to be baselines for research projects to extend from
    so we try to make them as simple for the general case as possible.
