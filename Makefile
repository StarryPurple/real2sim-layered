SHELL := /usr/bin/env bash
.SHELLFLAGS := -euo pipefail -c

PYTHON := .venv/bin/python
RUFF := .venv/bin/ruff
PYTEST := .venv/bin/pytest
VULTURE := .venv/bin/vulture
DEPTRY := .venv/bin/deptry
AST_GREP := .venv/bin/ast-grep
SHELL_SCRIPTS := scripts/env.sh

.PHONY: help format lint docs test check check-env hooks

help:
	@echo "Targets:"
	@echo "  make format     Format Python (ruff) and authored shell scripts (shfmt)"
	@echo "  make lint       Run read-only code checks (ruff, pyright, tach, ast-grep, vulture, deptry, shellcheck, shfmt -d, gitleaks)"
	@echo "  make docs       Run deterministic better-docs check gate"
	@echo "  make test       Run pytest test suite"
	@echo "  make check      Run full verification (lint + docs + test)"
	@echo "  make check-env  Verify PyTorch CUDA, MuJoCo EGL, and Blender 4.2"
	@echo "  make hooks      Install prek git hooks"

format:
	$(RUFF) format src scripts tests
	shfmt -w -i 2 -ci $(SHELL_SCRIPTS)

lint:
	$(RUFF) format --check src scripts tests
	$(RUFF) check src scripts tests
	pyright
	tach check
	$(AST_GREP) scan --error
	$(VULTURE)
	$(DEPTRY) src scripts
	shfmt -d -i 2 -ci $(SHELL_SCRIPTS)
	shellcheck $(SHELL_SCRIPTS)
	gitleaks git --no-banner --redact

docs:
	./scripts/check-docs.sh

test:
	$(PYTEST)

check: lint docs test

check-env:
	source scripts/env.sh && $(PYTHON) scripts/check_env.py

hooks:
	prek install
