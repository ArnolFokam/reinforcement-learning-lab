.PHONY: test clean dev-env

clean:
	rm -rf .pytest_cache

test: clean dev-env
	pytest tests -v

dev-env:
	pip install -r requirements.devel.txt

format: dev-env
	isort .
	black .
	flake8 --extend-ignore E501 .
