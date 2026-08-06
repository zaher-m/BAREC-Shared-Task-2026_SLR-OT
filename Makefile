PY ?= .venv/bin/python
ARGS ?=
export PYTHONPATH := src

help:
	@echo "make verify        recompute every reported number from the committed artifacts"
	@echo "make verify ARGS=--quick   the same minus the slow selection refits (~15s)"
	@echo "make test          threshold optimiser against the reference implementation"
	@echo "make setup         venv + CPU-side dependencies (enough for verify and experiments)"
	@echo "make clean-pyc     remove __pycache__ and *.pyc"

verify:
	$(PY) -m slra_ot.verify $(ARGS)

test:
	$(PY) tests/test_thresholds.py

setup:
	python3 -m venv .venv
	.venv/bin/pip install -U pip
	.venv/bin/pip install numpy scipy scikit-learn pandas pyarrow pyyaml

clean-pyc:
	find . -name __pycache__ -type d -not -path './.venv/*' -exec rm -rf {} +
	find . -name '*.py[cod]' -not -path './.venv/*' -delete

.PHONY: help verify test setup clean-pyc
