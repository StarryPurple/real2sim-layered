# real2sim-layered

Layered common-sense Real2Sim agent with local physics escalation and anomaly fallback for reconstructing dynamic 4D multi-object interactions from monocular video on **MVOIK-4D**.

> **Project status**: Repository scaffolding, GPU/simulation environment, dataset downloader, and code/documentation quality gates are operational. Core agent, simulation, and evaluation modules under `src/real2sim_layered/` are in active development.

## Quick start

1. Activate the local environment (PFS caches, `.venv`, bundled Blender 4.2, `syslibs`, `MUJOCO_GL=egl`):

```sh
source scripts/env.sh
```

2. Verify PyTorch CUDA, headless MuJoCo EGL offscreen rendering, and headless Blender 4.2 rigid-body simulation:

```sh
make check-env
```

3. Download the MVOIK-4D dataset into `data/MVOIK-4D/` (resumable):

```sh
python scripts/download_mvoik4d.py
```

4. Run code quality, documentation, and test checks:

```sh
make check
```

## Repository layout

```text
src/real2sim_layered/  Core Python package
scripts/               env.sh, check_env.py, download_mvoik4d.py, check-docs.sh
tests/                 Automated pytest suite
docs/                  Documentation contract (AGENTS.md), architecture.md, theory.md (working formulation)
ROADMAP.md             Agreed near-term task board
basic_instruction.md   Original Chinese research topic brief
.agents/notes/         Durable decision and empirical conclusion notes
.agents/records/       Bounded empirical study records
```

## Key references

- [AGENTS.md](AGENTS.md): Standing rules and verification commands.
- [docs/architecture.md](docs/architecture.md): Module boundaries, invariants, and exclusions.
- [docs/theory.md](docs/theory.md): Working formulation (research hypotheses and assumptions; not current implementation spec).
- [ROADMAP.md](ROADMAP.md): Agreed near-term task board.
- [basic_instruction.md](basic_instruction.md): Original topic description and reference links (HAT-4D, BVB, AHa-3D, L4GM, DreamScene4D).
