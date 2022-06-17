SRC = magi
N_CPUS ?= $(shell grep -c ^processor /proc/cpuinfo)

.PHONY: typecheck
typecheck:
	pytype -k -j $(N_CPUS) $(SRC)

.PHONY: test
test:
	JAX_PLATFORM_NAME=cpu pytest -n $(N_CPUS) \
		--color=yes \
		-rf \
		--ignore-glob="*/agent_test.py" \
		--ignore-glob="*/agent_distributed_test.py" \
		--durations=10 \
		$(SRC)

.PHONY: integration-test
integration-test:
	JAX_PLATFORM_NAME=cpu pytest -n $(N_CPUS) \
		--color=yes \
		-rf \
		--durations=10 \
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
	pylint $(SRC)

.PHONY: install
install:
	pip install -U pip setuptools wheel
	pip install -r requirements/dev.txt
	pip install -e .
