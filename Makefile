typecheck:
	pytype magi

test:
	pytest .

isort:
	isort magi

clean:
	find . -type d -name  "__pycache__" -exec rm -r {} +
