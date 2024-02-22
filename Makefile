lint: ## check style with flake8
	flake8 rsnl

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	rm -fr docs/_build
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -fr {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/

test: ## run tests quickly with the default Python
	PYTHONPATH=$$PYTHONPATH:. pytest

coverage: ## check code coverage quickly with the default Python
	PYTHONPATH=$$PYTHONPATH:. pytest --cov=rsnl

install: clean ## install the package to the active Python's site-packages
	python -m pip install numpy
	python -m pip install -e . --use-pep517

dev: install ## install the development requirements to the active Python's site-packages
	pip install -r requirements-dev.txt
