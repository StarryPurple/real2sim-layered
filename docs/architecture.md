# Architecture Contract

This document defines the repository ownership boundaries, declarative module layering, engineering invariants, and tool exclusions for `real2sim-layered`. Unified check commands live in [AGENTS.md](../AGENTS.md); research hypotheses live in [docs/theory.md](theory.md).

## Ownership map

| Path | Owner / Responsibility | Current state |
| :--- | :--- | :--- |
| `src/real2sim_layered/schemas/` | Pure Pydantic contracts (`config`, `dataset`, `eval`, `ikg`) | Operational (`tach`-checked) |
| `src/real2sim_layered/tools/` | MVOIK-4D RGBA loader, 5-case probe manifest, OpenAI-compatible VLM client (`mvoik4d`, `vlm`) | Operational (`tach`-checked) |
| `src/real2sim_layered/sim/` | Deterministic MuJoCo physical diagnostics and offscreen EGL RGBA rendering (`diagnostics`) | Operational (`tach`-checked) |
| `src/real2sim_layered/eval/` | Dual-axis visual + physical evaluation and Markdown/JSON reporting (`metrics`) | Operational (`tach`-checked) |
| `src/real2sim_layered/cli.py` | CLI entrypoints (`probe-inspect`, `eval-smoke`, `extract-ikg`) | Operational (`tach`-checked) |
| `scripts/` | Environment setup (`env.sh`), checks (`check_env.py`, `check-docs.sh`), dataset download (`download_mvoik4d.py`), local VLM server (`serve_vlm.py`) | Operational |
| `tests/` | Automated `pytest` verification suite | 13 unit & data tests |
| `docs/` | Shipped contracts (`AGENTS.md`, `architecture.md`) and working formulation (`theory.md`) | Active |
| `.agents/notes/` & `.agents/records/` | Durable decisions and bounded empirical study records | Active |

## Enforced module layering

[tach.toml](../tach.toml) enforces downward-only dependencies across shipped modules (and will register `agent` once Layer 1–3 orchestration lands):

| Layer | Responsibility | Allowed dependencies |
| :--- | :--- | :--- |
| `cli` | CLI entrypoints and evaluation/extraction orchestration | `schemas`, `tools`, `eval` |
| `agent` *(planned)* | Layered common-sense planner, local physics escalation, anomaly fallback | `sim`, `tools`, `schemas` |
| `eval` | Multi-view RGBA metrics and dual-axis physical+visual report synthesis | `schemas`, `sim` |
| `sim` & `tools` | MuJoCo/Blender diagnostics and MVOIK-4D / VLM adapters | `schemas` |
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
