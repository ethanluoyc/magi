# Developer Guide
## Tips about development

To install the project as in development mode, go to the project root
and run
```python
pip install -e .
```

For development, you need to install additional dependencies for testing
and developing. You can do this by

```python
pip install -e '.[test,dev]'
```

To contribute to this project, you need to ensure that the commits are
conform to the standard of this repository.
You can install the pre-commit hooks which will check that your code
conforms to the style guides of this repo. Install the pre-commit
hooks by `pre-commit install` after you install the dev and test dependencies.

## Notes about dependencies
Currently, we do not pin the versions of the dependencies, this is
likely to break some of our code as the vendor libraries upgrade. Once
this issue surfaces, we can start introducing constraints to the dependencies. It may also be a good idea to not list dependencies to
tensorflow, tfp, jax or jaxlib, and we should probably provide guidelines
for installing these dependencies in future.
