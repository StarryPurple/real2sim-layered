# Roadmap

Agreed near-term tasks for `real2sim-layered` (tier governance and task-line rules live in [docs/AGENTS.md](docs/AGENTS.md)).

- **Human collaborators**: any collaborator may add agreed tasks, claim or update `@owner`, tick `[x]`, reorder, or remove cancelled/superseded rows directly.
- **AI agents**: require explicit collaborator authorization before adding, rewording, or reassigning rows (a request to implement work does not authorize adding a row); an agent may tick `[x]` on a uniquely matched assigned row after verifying its done condition.

## Active

- [x] Build MVOIK-4D dataset loader, 5-case cross-regime probe manifest, and dual-axis evaluator (multi-view RGBA/IoU/LPIPS + MuJoCo/Blender physical diagnostics) — verified by unit tests and a reproducible baseline report on the 5 probe cases.
- [x] Deploy local `Qwen3.5-4B-VL` OpenAI-compatible serving endpoint and implement unified VLM client with Pydantic structured output — verified by deterministic `G_int` extraction tests on the 5 probe cases.
- [x] Implement Layer-1 coarse Real2Sim baseline (`G_int -> S_0`) with pluggable `procedural_blender` and `sim_proxy` geometry backends — produces executable `.blend`/MJCF scenes and logs baseline scores under `.agents/records/`.

## Up next

- [ ] Implement Layer-2 local physics escalation (`E_phys`) and simulator-in-the-loop verifier (`D_phys`) in MuJoCo/Blender — improves physical metrics over L1-only on rigid and occlusion probe cases.
- [ ] Implement Layer-3 anomaly detection and fallback router (`R_anom`) for deformable, topological-split, and multiphysics regimes — prevents solver-divergence regressions on non-rigid probe cases.
- [ ] Integrate Image-to-3D mesh generation plus convex decomposition (`image_to_3d_sim`) and benchmark all three `B_asset` geometry backends on `with_gt/`.
- [ ] Run full 30-case `MVOIK-4D/with_gt/` ablations (layering, feedback modality, geometry backend, and `Qwen3.5-4B-VL` vs. larger VLM scaling) and seal empirical studies under `.agents/records/`.
- [ ] Prepare target submission for **ICARA 2027** (IEEE Xplore full paper; fallback: ICRCA 2027 / IEEE ICM 2027 WIP) with external baseline comparisons (BVB-style and 4DGS heuristic smoothing), figures, and tables.
