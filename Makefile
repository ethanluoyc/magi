SRC = magi

.PHONY: typecheck
typecheck:
	pytype $(shell grep -c ^processor /proc/cpuinfo) $(SRC)

.PHONY: test
test:
	pytest --color=yes -rf --ignore=magi/agents/archived --ignore=magi/experimental \
	 $(SRC)

.PHONY: isort
isort:
	isort $(SRC)

.PHONY: clean
clean:
	find . -type d -name  "__pycache__" -exec rm -r {} +
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .pytype

.PHONY: lint
lint:
	flake8 $(SRC)
	pylint $(SRC)

.PHONY: install
install:
	pip install -U pip setuptools wheel
	pip install -U --upgrade-strategy eager -f https://storage.googleapis.com/jax-releases/jax_releases.html -e .
	pip install -U --upgrade-strategy eager -e '.[envs,dev]'
