# Gallops — developer entry points.
#
# `make check` is the gate: it is what you run before opening a PR and what CI
# runs on every PR, in a clean environment with no site gallops.toml. The two
# must agree, so anything that can reject work belongs in this target and not
# in a workflow's own command list. That rule is paddock's, and it is the reason
# a green local run and a green PR are the same claim there.
#
# Every target runs against whichever environment is active — an activated
# venv, an activated conda env, or the repo's own .venv as a fallback.
# Override explicitly with:  make test PY=/path/to/python

PY := $(shell \
  if [ -n "$$VIRTUAL_ENV" ]; then echo "$$VIRTUAL_ENV/bin/python"; \
  elif [ -n "$$CONDA_PREFIX" ] && [ "$$CONDA_DEFAULT_ENV" != "base" ]; then echo "$$CONDA_PREFIX/bin/python"; \
  elif [ -x .venv/bin/python ]; then echo ".venv/bin/python"; \
  else echo "python3"; fi)

.PHONY: help setup check test lint fmt typecheck which clean

help:
	@echo "make setup      — create .venv and install gallops (uv; the default)"
	@echo "make check      — lint + tests + portability gate. Run before every PR."
	@echo "make test       — pytest only"
	@echo "make lint       — ruff only"
	@echo "make fmt        — ruff format + autofix"
	@echo "make typecheck  — mypy. ADVISORY: not in 'make check', never blocks a PR."
	@echo "make which      — show which interpreter these targets will use"

which:
	@echo "$(PY)"
	$(PY) -c "import sys; print(sys.version)"

# The base install deliberately omits torch, so this is fast and works on a
# machine with no GPU. Add the training stack explicitly:
#   uv pip install --python .venv/bin/python -e ".[train]"
setup:
	uv venv --python 3.11 .venv
	uv pip install --python .venv/bin/python -e ".[dev]"
	@echo
	@echo "Next:"
	@echo "  source .venv/bin/activate"
	@echo "  cp gallops.toml.example gallops.toml && \$$EDITOR gallops.toml"
	@echo "  gallops config"

check: lint test

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests

# Advisory. Deliberately not a dependency of `check`, matching paddock: what
# blocks locally must be what blocks in CI, and a type gate over research code
# fails PRs that touch none of it. The exit code is mypy's own, not swallowed.
typecheck:
	$(PY) -m mypy $(or $(PATHS),src/gallops)

fmt:
	$(PY) -m ruff check --fix src tests
	$(PY) -m ruff format src tests

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache dist build
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
