"""Smoke tests for package initialization."""

from __future__ import annotations

import real2sim_layered


def test_package_exposes_version() -> None:
    assert isinstance(real2sim_layered.__version__, str)
    assert real2sim_layered.__version__
