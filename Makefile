typecheck:
	pytype magi

test:
	pytest --ignore=magi/agents/archived --ignore=magi/experimental magi

isort:
	isort magi

clean:
	find . -type d -name  "__pycache__" -exec rm -r {} +

lint:
	pylint magi
