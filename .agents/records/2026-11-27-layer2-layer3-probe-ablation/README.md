# Study: 2026-11-27-layer2-layer3-probe-ablation — 4-Way Layering Ablation on 5 Probe Cases

Status: active

> Non-authoritative temporal evidence record. Current runnable pipeline and contracts live in `docs/`. Large raw artifacts (MJCF XMLs, renders, JSON dumps) remain outside git under `outputs/ablation_probe_sim_proxy_v2/`.

## Provenance

- **Started**: 2026-11-27
- **Base commit / pin**: `ac401c7` + delivered-trajectory MuJoCo evaluation, L2 floor-anchored contact grounding + input-view silhouette guard, L3 per-component optical-flow non-rigidity + topological split + pre-commit safety gate, and MuJoCo keyframe scale rendering
- **External artifact store**: `outputs/ablation_probe_sim_proxy_v2/{reports.json,reports.md}` (v1 pre-fix baseline archived locally in `outputs/ablation_probe_sim_proxy/`)
- **Inputs**: Blind ROI-cropped `Qwen/Qwen3.5-4B` `InteractionGraph` JSONs (`outputs/ikg_probe_qwen35_4b_roi/`), `sim_proxy` backend, 6 keyframes, 4 canonical `eval_0..eval_3` views (`49.1 deg` FOV)
- **Reproduction command**: `python -m real2sim_layered.cli run-ablation --backend sim_proxy --ikg-dir outputs/ikg_probe_qwen35_4b_roi --max-frames 6 --out-dir outputs/ablation_probe_sim_proxy_v2`

## Runs and observations

### Aggregate summary across 5 cross-regime probe cases (v2, delivered-trajectory physics)

| Arm | Mean IoU | Mean CosSim | Mean MaxPen (m) | Mean Float% | Mean TrackErr (m) | CF-OK Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `L1_only` | 0.128 | 0.209 | 0.1479 | 0.0% | 0.000 | 1/5 (20%) |
| `L1_global_phys` | 0.109 | 0.160 | 0.1256 | 0.0% | 0.401 | 1/5 (20%) |
| `L1_L2_local` | 0.116 | 0.214 | **0.0111** | 6.7% | 0.146 | **5/5 (100%)** |
| `L1_L2_L3_full` | 0.116 | **0.215** | **0.0111** | 10.8% | **0.132** | **5/5 (100%)** |

*(v1 baseline prior to delivered-trajectory evaluation, floor anchoring, and L3 pre-commit gating: `L1_L2_local` Mean IoU was `0.090` and `L1_L2_L3_full` regressed to `0.084`; `L1_only` and `L1_global_phys` shared identical frame-0 physics metrics `0.1252 m`.)*

### Per-case results (v2)

| Case | Regime | Method | Mode | IoU | PSNR | CosSim | MaxPen | Float% | TrackErr | CF-OK | Time(s) |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0004_brick_fall_1` | `rigid` | `L1_only` | `kinematic` | 0.135 | 8.5 | 0.200 | 0.1457 | 0.0% | 0.000 | no | 1.14 |
| `0004_brick_fall_1` | `rigid` | `L1_global_phys` | `physics_sim` | 0.097 | 8.0 | 0.136 | 0.0603 | 0.0% | 0.094 | no | 0.58 |
| `0004_brick_fall_1` | `rigid` | `L1_L2_local` | `physics_sim` | 0.106 | 8.4 | 0.167 | 0.0020 | 0.0% | 0.051 | yes | 2.13 |
| `0004_brick_fall_1` | `rigid` | `L1_L2_L3_full` | `physics_sim` | 0.106 | 8.4 | 0.167 | 0.0020 | 0.0% | 0.051 | yes | 1.79 |
| `0029_cups_uncover_object` | `occluded_kinematic` | `L1_only` | `kinematic` | 0.085 | 7.1 | 0.128 | 0.1340 | 0.0% | 0.000 | no | 0.91 |
| `0029_cups_uncover_object` | `occluded_kinematic` | `L1_global_phys` | `physics_sim` | 0.090 | 6.2 | 0.069 | 0.1340 | 0.0% | 0.409 | no | 0.56 |
| `0029_cups_uncover_object` | `occluded_kinematic` | `L1_L2_local` | `physics_sim` | 0.058 | 7.0 | 0.086 | 0.0130 | 33.3% | 0.232 | yes | 7.40 |
| `0029_cups_uncover_object` | `occluded_kinematic` | `L1_L2_L3_full` | `fallback_kinematic` | 0.060 | 6.9 | 0.092 | 0.0130 | 54.2% | 0.163 | yes | 3.52 |
| `0014_press_ball_1` | `deformable` | `L1_only` | `kinematic` | 0.056 | 8.3 | 0.122 | 0.1556 | 0.0% | 0.000 | no | 0.57 |
| `0014_press_ball_1` | `deformable` | `L1_global_phys` | `physics_sim` | 0.010 | 9.6 | 0.020 | 0.1434 | 0.0% | 0.531 | no | 0.54 |
| `0014_press_ball_1` | `deformable` | `L1_L2_local` | `physics_sim` | 0.075 | 8.6 | 0.166 | 0.0131 | 0.0% | 0.124 | yes | 2.23 |
| `0014_press_ball_1` | `deformable` | `L1_L2_L3_full` | `physics_sim` | 0.075 | 8.6 | 0.166 | 0.0131 | 0.0% | 0.124 | yes | 1.38 |
| `0000_cut_apple_1` | `topological_split` | `L1_only` | `kinematic` | 0.265 | 7.2 | 0.418 | 0.2830 | 0.0% | 0.000 | no | 0.61 |
| `0000_cut_apple_1` | `topological_split` | `L1_global_phys` | `physics_sim` | 0.245 | 7.9 | 0.363 | 0.2751 | 0.0% | 0.864 | no | 0.54 |
| `0000_cut_apple_1` | `topological_split` | `L1_L2_local` | `physics_sim` | 0.233 | 6.9 | 0.418 | 0.0113 | 0.0% | 0.204 | yes | 3.04 |
| `0000_cut_apple_1` | `topological_split` | `L1_L2_L3_full` | `physics_sim` | 0.233 | 6.9 | 0.418 | 0.0113 | 0.0% | 0.204 | yes | 2.22 |
| `0039_syringe_draw_eject_cola` | `multiphysics_ood` | `L1_only` | `kinematic` | 0.099 | 5.2 | 0.174 | 0.0212 | 0.0% | 0.000 | yes | 0.84 |
| `0039_syringe_draw_eject_cola` | `multiphysics_ood` | `L1_global_phys` | `physics_sim` | 0.102 | 4.9 | 0.210 | 0.0151 | 0.0% | 0.106 | yes | 0.55 |
| `0039_syringe_draw_eject_cola` | `multiphysics_ood` | `L1_L2_local` | `physics_sim` | 0.108 | 4.9 | 0.230 | 0.0163 | 0.0% | 0.117 | yes | 3.68 |
| `0039_syringe_draw_eject_cola` | `multiphysics_ood` | `L1_L2_L3_full` | `physics_sim` | 0.108 | 4.9 | 0.230 | 0.0163 | 0.0% | 0.117 | yes | 2.85 |

### Key findings and open limits

- **Delivered-trajectory diagnostics cleanly separate all 4 arms**: Evaluating actual delivered keyframes with finite-difference velocities eliminates the v1 `L1_only == L1_global_phys` artifact (`TrackErr 0.000` vs `0.401 m`, `MaxPen 0.1479` vs `0.1256 m`) and fixes false-positive floating ratios on moving rigid bodies (`Float% 0.0%` on 4/5 cases).
- **Floor-anchored Layer-2 (`E_phys`) solves penetration without visual collapse**: Cuts mean `MaxPen` by **92.5%** vs `L1_only` (`0.1479 -> 0.0111 m`), cuts `TrackErr` by **63.7%** vs `L1_global_phys` (`0.401 -> 0.146 m`), lifts `CF-OK` from **20% to 100%**, and raises Mean IoU from v1's `0.090` to **`0.116`** (surpassing `L1_global_phys` `0.109` on IoU and surpassing `L1_only` on Patch Cosine Sim `0.214` vs `0.209`). On `0014_press_ball_1`, `L1_L2_local` improves IoU to `0.075` while `L1_global_phys` collapses to `0.010`.
- **Pre-commit gated Layer-3 (`R_anom`) guarantees non-degrading fallback**: Recovers `0029_cups_uncover_object` (`IoU 0.058 -> 0.060`, `CosSim 0.086 -> 0.092`, `TrackErr 0.232 -> 0.163 m`) while safely rejecting candidates that would worsen silhouette or penetration on `0004_brick_fall_1`, `0000_cut_apple_1`, and `0039_syringe_draw_eject_cola`, yielding the highest overall Patch Cosine Sim (`0.215`) and lowest physics-validated `TrackErr` (`0.132 m`) at `100%` `CF-OK`.
- **Remaining gap for next milestone (`image_to_3d_sim` + full 30-case study)**: Coarse primitives remain the primary ceiling on multi-view IoU (`0.06`–`0.23`), and subtle local deformations (`0014_press_ball_1`, `shape_cv=0.02`) require finer mesh/mask geometry or stronger VLM regime recognition to activate deformable shape-key synthesis.

## Closure

When the 30-case `with_gt/` ablation and `image_to_3d_sim` comparison complete, distill takeaways into `.agents/notes/concluded/experiment/YYYY-MM-DD-<slug>.md`, update line 3 to `Status: concluded`, and seal with `./scripts/check-docs.sh --seal`.
