# Architecture Contract

This document defines the repository ownership boundaries, declarative module layering, engineering invariants, and tool exclusions for `real2sim-layered`. Unified check commands live in [AGENTS.md](../AGENTS.md); research hypotheses live in [docs/theory.md](theory.md).

## Ownership map

| Path | Owner / Responsibility | Current state |
| :--- | :--- | :--- |
| `src/real2sim_layered/schemas/` | Pure Pydantic domain contracts (`models`) and single-source config/env loader (`config`) | Operational (`tach`-checked) |
| `src/real2sim_layered/tools/` | MVOIK-4D RGBA loader, 5-case probe manifest, OpenAI-compatible VLM client (`mvoik4d`, `vlm`), and TripoSR+CoACD mesh lifter (`image_to_3d`) | Operational (`tach`-checked) |
| `src/real2sim_layered/sim/` | MuJoCo MJCF compiler & diagnostics (`_mujoco`, `diagnostics`, `mujoco_builder`) and headless Blender 4.2 worker (`blender_builder`, `blender_worker`) | Operational (`tach`-checked) |
| `src/real2sim_layered/eval/` | Dual-axis visual + physical evaluation and Markdown/JSON reporting (`metrics`) | Operational (`tach`-checked) |
| `src/real2sim_layered/agent/` | Layer-1 synthesis (`layer1`), Layer-2 local physics escalation (`layer2`), Layer-3 anomaly fallback (`layer3`) | Operational (`tach`-checked) |
| `src/real2sim_layered/cli.py` | CLI entrypoints (`probe-inspect`, `eval-smoke`, `extract-ikg`, `run-layer1`, `run-ablation`) | Operational (`tach`-checked) |
| `scripts/` | Environment setup (`env.sh`), checks (`check_env.py`, `check-docs.sh`), dataset download (`download_mvoik4d.py`), local VLM server (`serve_vlm.py`) | Operational |
| `rules/ast-grep/` & `sgconfig.yml` | Mechanical AST invariant rules (`no-scattered-environ`, `no-silent-except-pass`) | Operational |
| `tests/` | Automated `pytest` verification suite | Operational |
| `docs/` | Shipped contracts (`AGENTS.md`, `architecture.md`) and working formulation (`theory.md`) | Active |
| `.agents/notes/` & `.agents/records/` | Durable decisions and bounded empirical study records | Active |

## Enforced module layering

[tach.toml](../tach.toml) enforces downward-only dependencies across shipped modules:

| Layer | Responsibility | Allowed dependencies |
| :--- | :--- | :--- |
| `cli` | CLI entrypoints and evaluation/extraction/layered ablation orchestration | `schemas`, `tools`, `eval`, `agent` |
| `agent` | Layer-1 synthesis (`layer1`), Layer-2 physics escalation (`layer2`), Layer-3 anomaly fallback (`layer3`) | `sim`, `tools`, `schemas` |
| `eval` | Multi-view RGBA metrics and dual-axis physical+visual report synthesis | `schemas`, `sim` |
| `sim` & `tools` | MuJoCo/Blender builders & diagnostics and MVOIK-4D / VLM adapters | `schemas` |
| `schemas` | Pure Pydantic data models (`models`) and single-source configuration (`config`) | Standard library and `pydantic` only |

Do not create empty placeholder subpackages ahead of real implementation. Add each module to [tach.toml](../tach.toml) in the same change that introduces it.

## Golden principles and domain invariants

1. **Deterministic runtime activation**: GPU PyTorch, headless MuJoCo (`MUJOCO_GL=egl`), bundled Blender 4.2 (`tools/blender-4.2.9-linux-x64` plus `tools/syslibs`), and upstream TripoSR sources (`tools/TripoSR`, shimmed with `scikit-image` marching cubes) are activated through `source scripts/env.sh`. Never hardcode machine-local paths inside `src/real2sim_layered/`.
2. **Single-source configuration and secrets**: load API keys, `.env`, proxy overrides, and external binary paths through `real2sim_layered.schemas.config` (`.env*` is gitignored); `ast-grep` bans scattered `os.environ`/`os.getenv` calls across `src/real2sim_layered/`.
3. **Explicit edge contracts**: validate external inputs (VLM outputs, dataset manifests, simulation parameters) once at module boundaries via Pydantic schemas. `ast-grep` bans silent `except ...: pass` swallowing.
4. **Suppression hygiene**: every `# noqa`, `# pyright: ignore`, or `# shellcheck disable=` directive must include a concrete inline reason; eliminate root causes before adding suppressions (`ruff` `RUF100` enforces zero stale suppressions).
5. **Reproducible artifact separation**: keep datasets (`data/`), model caches (`cache/`), bundled binaries (`tools/`), temporary scratch (`tmp/`, `.agents/scratch/`), and experiment runs (`outputs/`) outside git tracking.

## Exclusions and vendored files

- **Gitignored local directories**: `.venv/`, `cache/`, `tools/`, `data/`, `outputs/`, `tmp/`, `.agents/scratch/`.
- **Blender embedded runtime**: `src/real2sim_layered/sim/blender_worker.py` executes inside Blender 4.2's embedded Python (`bpy`/`mathutils`), so `pyright`, `tach`, `vulture`, and `deptry` exclude it while `ruff` formats and lints it.
- **Vendored upstream gate**: `scripts/check-docs.sh` is copied verbatim from `better-docs` (`kit_version=0.1.0`) and is excluded from local `shfmt`/`shellcheck` edits so upstream upgrades stay clean.
