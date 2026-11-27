# real2sim-layered

Layered common-sense Real2Sim agent with local physics escalation and anomaly fallback for reconstructing dynamic 4D multi-object interactions from monocular video on **MVOIK-4D**.

> **Project status**: MVOIK-4D `with_gt` loader, 5-case cross-regime probe set, delivered-trajectory MuJoCo+RGBA dual-axis evaluator, local `Qwen/Qwen3.5-4B` OpenAI-compatible VLM pipeline, and the full **Layer-1 → Layer-2 (local physics escalation) → Layer-3 (pre-commit gated anomaly fallback)** pipeline across all three geometry backends (`sim_proxy`, `procedural_blender`, and TripoSR+CoACD `image_to_3d_sim`) are operational and benchmarked across all 30 `with_gt` cases (360 runs).

## Quick start

1. Activate the local environment (PFS caches, `.venv`, bundled Blender 4.2, `syslibs`, `MUJOCO_GL=egl`):

```sh
source scripts/env.sh
```

2. Verify PyTorch CUDA, headless MuJoCo EGL offscreen rendering, and headless Blender 4.2 rigid-body simulation:

```sh
make check-env
```

3. Download MVOIK-4D (`with_gt` first or full dataset, resumable):

```sh
python scripts/download_mvoik4d.py --no-proxy --workers 24 --allow-patterns "with_gt/**"
```

4. Inspect the 5-case cross-regime probe set and run the dual-axis smoke evaluation:

```sh
python -m real2sim_layered.cli probe-inspect
python -m real2sim_layered.cli eval-smoke --out outputs/eval_smoke_probe.json
```

5. Serve local `Qwen/Qwen3.5-4B` (OpenAI-compatible), extract `G_int`, and run the Layer-1 baseline or 4-way layering ablation:

```sh
python scripts/serve_vlm.py --model Qwen/Qwen3.5-4B --device cuda:2 --port 8000 --no-proxy
python -m real2sim_layered.cli extract-ikg --max-frames 6 --out-dir outputs/ikg_probe_qwen35_4b_roi
python -m real2sim_layered.cli run-layer1 --backend all --ikg-dir outputs/ikg_probe_qwen35_4b_roi --max-frames 6
python -m real2sim_layered.cli run-ablation --backend sim_proxy --ikg-dir outputs/ikg_probe_qwen35_4b_roi --max-frames 6
python -m real2sim_layered.cli run-ablation --backend image_to_3d_sim --ikg-dir outputs/ikg_probe_qwen35_4b_roi --max-frames 6
```

6. Run code quality, documentation, and test checks:

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
