.PHONY: env jupyter test test-coverage run-pipeline

# Set up and activate virtual-environment, and install dependencies
env:
	python3.12 -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

# Run Jupyterlab
jupyter:
	.venv/bin/jupyter lab notebooks/

# Run unit tests
test:
	.venv/bin/pytest

# Run unit tests and generate code-coverage report
test-coverage:
	.venv/bin/pytest --cov=src --cov-report=xml

run-pipeline:
	PYTHONPATH=src python src/pipeline/preprocess.py

