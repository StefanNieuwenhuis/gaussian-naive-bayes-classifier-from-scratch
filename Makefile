.PHONY: env jupyter test

# Set up and activate virtual-environment, and install dependencies
env:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

# Run Jupyterlab
jupyter:
	.venv/bin/jupyter lab notebooks/

# Run unit tests
test:
	.venv/bin/pytest