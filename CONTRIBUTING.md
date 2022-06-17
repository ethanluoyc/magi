# Contributing to Magi

Thanks for your interest in contributing to Magi!

We are actively looking for contributions to improve Magi.
1. __Contributing new agents__.
We are looking for contributions for new RL agent implementation.
They should follow the styles for the agents in magi.
Have a look at the agents implemented in magi to have a sense of how to create new agents. We aim for
the implementation to be self-contained so that forking the agents
should be easy. This means that the new agents should only
incorporate code that _only_ implements the core RL algorithm,
without additions that are tied to particular environments or benchmarks.
Handling environment-specific configuration should be performed in benchmark code instead.
TODO: add some more detailed notes on how to contribute.
2. __Contributing new benchmarks__.
This can be reports of the performance of the agent implementation in magi on popular RL environments.
For example, the SAC (and variant) agents have benchmark scripts
for running with the dm_control suite.
3. __Contributing new RL components__.
We are interested in designing
general-purpose components that can be used to fascilitate rapid and
reproducible RL research.
4. __Improving documentation__. Fixing typos, adding tutorials and
improving the documentation are really appreciated.
---

Be sure to first discuss the proposed change that you
would like to make with a GitHub issues.
All contributions should be made via Pull Requests (PRs) and will be reviewed before
being merged. We will try to review the PRs as fast as we can.

To develop locally, first install the additional list of development dependencies by running:
```
pip install -r requirements/dev.txt
```

For the changes to be merged, we expect that the code conforms to the style guide
adopted by this repository.

There use the following tools to ensure that the code conform to the styles.

1. [pytype](https://google.github.io/pytype/) is used for type checking the code.
Whenever possible, the public interface should be annotated with types.
2. [pylint](http://pylint.pycqa.org/en/latest/) catches common style issues and
potential errors. We use the configuration from the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).
3. [yapf](https://github.com/google/yapf) is used to autoformat the code.
4. [isort](https://pycqa.github.io/isort/) is used for sorting import lines automatically.
isort is configured to use the Google style.

You may want to set up [pre-commit](https://pre-commit.com/) hooks to ensure that your commits
conform to the styles. To set up the hooks, run

```
pre-commit install
```

Then every time you make a new commit, the commit hooks will run.

New functionality should be accompanied by unit tests. The tests should be placed
next to the modules implementing the functionality.
We use [pytest](https://docs.pytest.org/) to run the test suites.
You can run all of the tests in magi with
```bash
pytest magi
```
In addition, you can run the tests in parallel with
```bash
pytest -n <number of parallel runners> magi
```

The public APIs should be documented with the [Google style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings). The public APIs should also
be type annotated. The types will be checked with pytype.


## Tips for Development.
We find VSCode to be quite convenient to development. Here are some tips
for developing Magi with VSCode.

1. Use the VSCode Python Extension https://code.visualstudio.com/docs/languages/python.
2. The following settings.json file can be used
```json
{
    "python.linting.flake8Enabled": false,
    "python.linting.mypyEnabled": false,
    "python.linting.pylintEnabled": true,
    "python.languageServer": "Pylance",
    "python.linting.enabled": true,
    "python.formatting.provider": "yapf",
    "editor.rulers": [80],
    "editor.tabSize": 4,
    "editor.detectIndentation": false,
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true,
    },
}
```
