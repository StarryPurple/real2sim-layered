# Architecture Contract

This document defines the repository ownership boundaries, declarative module layering, engineering invariants, and tool exclusions for `real2sim-layered`. Unified check commands live in [AGENTS.md](../AGENTS.md); research hypotheses live in [docs/theory.md](theory.md).

## Ownership map

| Path | Owner / Responsibility | Current state |
| :--- | :--- | :--- |
| `src/real2sim_layered/` | Core Python library for layered Real2Sim reasoning, simulation adapters, and evaluation | Package root initialized (`__version__`) |
| `scripts/` | Environment activation (`env.sh`), environment verification (`check_env.py`), dataset acquisition (`download_mvoik4d.py`), and vendored doc gate (`check-docs.sh`) | Operational |
| `tests/` | Automated `pytest` verification suite | Smoke tests initialized |
| `docs/` | Shipped contracts (`AGENTS.md`, `architecture.md`) and working formulation (`theory.md`) | Active |
| `.agents/notes/` & `.agents/records/` | Durable decisions and bounded empirical study records | Active |

## Declarative module layering

As subpackages are introduced under `src/real2sim_layered/`, register each module in [tach.toml](../tach.toml) so `tach check` enforces downward-only dependencies:

| Layer | Planned responsibility | Allowed dependencies |
| :--- | :--- | :--- |
| `cli` / `harness` | CLI entrypoints, benchmark runner, experiment orchestration | `agent`, `eval`, `sim`, `schemas` |
| `agent` | Layered common-sense planner, physics-escalation policy, anomaly fallback | `sim`, `tools`, `schemas` |
| `sim` & `tools` | MuJoCo and Blender simulation backends, vision/3D/4D external tool adapters | `schemas` |
| `eval` | MVOIK-4D metric computation and rendering comparison | `sim`, `schemas` |
| `schemas` | Pure Pydantic data models, interaction graph types, and configuration contracts | Standard library and `pydantic` only |

Do not create empty placeholder subpackages ahead of real implementation. Add each module to [tach.toml](../tach.toml) in the same change that introduces it.

## Golden principles and domain invariants

1. **Deterministic runtime activation**: GPU PyTorch, headless MuJoCo (`MUJOCO_GL=egl`), and bundled Blender 4.2 (`tools/blender-4.2.9-linux-x64` plus `tools/syslibs`) are activated through `source scripts/env.sh`. Never hardcode machine-local paths inside `src/real2sim_layered/`.
2. **Single-source configuration and secrets**: load API keys and external endpoints through one dedicated config module from environment variables (`.env*` is gitignored); never commit credentials or scatter `os.environ` lookups across core logic.
3. **Explicit edge contracts**: validate external inputs (VLM outputs, dataset manifests, simulation parameters) once at module boundaries via Pydantic schemas. Ban silent exception swallowing (`except Exception: pass`) without an explicit justification.
4. **Suppression hygiene**: every `# noqa`, `# pyright: ignore`, or `# shellcheck disable=` directive must include a concrete inline reason; fix root causes before adding suppressions.
5. **Reproducible artifact separation**: keep datasets (`data/`), model caches (`cache/`), bundled binaries (`tools/`), temporary scratch (`tmp/`, `.agents/scratch/`), and experiment runs (`outputs/`) outside git tracking.

## Exclusions and vendored files

- **Gitignored local directories**: `.venv/`, `cache/`, `tools/`, `data/`, `outputs/`, `tmp/`, `.agents/scratch/`.
- **Vendored upstream gate**: `scripts/check-docs.sh` is copied verbatim from `better-docs` (`kit_version=0.1.0`) and is excluded from local `shfmt`/`shellcheck` edits so upstream upgrades stay clean.
