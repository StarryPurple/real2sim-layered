# Study: 2026-11-27-layer2-layer3-probe-ablation — 4-Way Layering Ablation on 5 Probe Cases

Status: active

> Non-authoritative temporal evidence record. Current runnable pipeline and contracts live in `docs/`. Large raw artifacts (MJCF XMLs, renders, JSON dumps) remain outside git under `outputs/ablation_probe_sim_proxy/`.

## Provenance

- **Started**: 2026-11-27
- **Base commit / pin**: `6bc3f57` + hardened per-body MuJoCo diagnostics, Layer-2 local physics escalation (`E_phys`), and Layer-3 anomaly fallback (`R_anom`)
- **External artifact store**: `outputs/ablation_probe_sim_proxy/{reports.json,reports.md}`
- **Inputs**: Blind ROI-cropped `Qwen/Qwen3.5-4B` `InteractionGraph` JSONs (`outputs/ikg_probe_qwen35_4b_roi/`), `sim_proxy` backend, 6 keyframes, 4 canonical `eval_0..eval_3` views (`49.1 deg` FOV)
- **Reproduction command**: `python -m real2sim_layered.cli run-ablation --backend sim_proxy --ikg-dir outputs/ikg_probe_qwen35_4b_roi --max-frames 6 --out-dir outputs/ablation_probe_sim_proxy`

## Runs and observations

### Aggregate summary across 5 cross-regime probe cases

| Arm | Mean IoU | Mean MaxPen (m) | Mean Float% | Mean TrackErr (m) | CF-OK Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `L1_only` | 0.125 | 0.1252 | 0.0% | 0.360 | 1/5 (20%) |
| `L1_global_phys` | 0.106 | 0.1252 | 0.0% | 0.360 | 1/5 (20%) |
| `L1_L2_local` | 0.090 | **0.0117** | 0.3% | **0.162** | **5/5 (100%)** |
| `L1_L2_L3_full` | 0.084 | **0.0117** | 0.3% | **0.162** | **5/5 (100%)** |

### Per-case results

| Case | Regime | Method | Mode | IoU | PSNR | CosSim | MaxPen | Float% | TrackErr | CF-OK | Time(s) |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0004_brick_fall_1` | `rigid` | `L1_only` | `kinematic` | 0.135 | 8.5 | 0.200 | 0.0603 | 0.0% | 0.075 | no | 1.14 |
| `0004_brick_fall_1` | `rigid` | `L1_global_phys` | `physics_sim` | 0.097 | 8.0 | 0.136 | 0.0603 | 0.0% | 0.075 | no | 0.58 |
| `0004_brick_fall_1` | `rigid` | `L1_L2_local` | `physics_sim` | 0.083 | 7.8 | 0.121 | 0.0020 | 0.0% | 0.075 | yes | 0.66 |
| `0004_brick_fall_1` | `rigid` | `L1_L2_L3_full` | `physics_sim` | 0.083 | 7.8 | 0.121 | 0.0020 | 0.0% | 0.075 | yes | 1.52 |
| `0029_cups_uncover_object` | `occluded_kinematic` | `L1_only` | `kinematic` | 0.070 | 7.0 | 0.112 | 0.1323 | 0.0% | 0.322 | no | 0.89 |
| `0029_cups_uncover_object` | `occluded_kinematic` | `L1_global_phys` | `physics_sim` | 0.074 | 6.8 | 0.123 | 0.1323 | 0.0% | 0.322 | no | 0.56 |
| `0029_cups_uncover_object` | `occluded_kinematic` | `L1_L2_local` | `physics_sim` | 0.051 | 6.6 | 0.069 | 0.0156 | 0.6% | 0.291 | yes | 0.95 |
| `0029_cups_uncover_object` | `occluded_kinematic` | `L1_L2_L3_full` | `fallback_kinematic` | 0.060 | 7.0 | 0.088 | 0.0156 | 0.6% | 0.291 | yes | 2.13 |
| `0014_press_ball_1` | `deformable` | `L1_only` | `kinematic` | 0.056 | 8.3 | 0.122 | 0.1434 | 0.0% | 0.486 | no | 0.58 |
| `0014_press_ball_1` | `deformable` | `L1_global_phys` | `physics_sim` | 0.010 | 9.6 | 0.020 | 0.1434 | 0.0% | 0.486 | no | 0.56 |
| `0014_press_ball_1` | `deformable` | `L1_L2_local` | `physics_sim` | 0.041 | 8.4 | 0.093 | 0.0131 | 0.5% | 0.125 | yes | 0.58 |
| `0014_press_ball_1` | `deformable` | `L1_L2_L3_full` | `physics_sim` | 0.041 | 8.4 | 0.093 | 0.0131 | 0.5% | 0.125 | yes | 1.38 |
| `0000_cut_apple_1` | `topological_split` | `L1_only` | `kinematic` | 0.265 | 7.2 | 0.418 | 0.2751 | 0.0% | 0.807 | no | 0.61 |
| `0000_cut_apple_1` | `topological_split` | `L1_global_phys` | `physics_sim` | 0.245 | 7.9 | 0.363 | 0.2751 | 0.0% | 0.807 | no | 0.57 |
| `0000_cut_apple_1` | `topological_split` | `L1_L2_local` | `physics_sim` | 0.173 | 6.5 | 0.320 | 0.0113 | 0.2% | 0.210 | yes | 0.61 |
| `0000_cut_apple_1` | `topological_split` | `L1_L2_L3_full` | `fallback_topological` | 0.159 | 6.2 | 0.281 | 0.0113 | 0.2% | 0.210 | yes | 1.40 |
| `0039_syringe_draw_eject_cola` | `multiphysics_ood` | `L1_only` | `kinematic` | 0.099 | 5.2 | 0.174 | 0.0151 | 0.0% | 0.109 | yes | 0.97 |
| `0039_syringe_draw_eject_cola` | `multiphysics_ood` | `L1_global_phys` | `physics_sim` | 0.102 | 4.9 | 0.210 | 0.0151 | 0.0% | 0.109 | yes | 0.53 |
| `0039_syringe_draw_eject_cola` | `multiphysics_ood` | `L1_L2_local` | `physics_sim` | 0.098 | 5.0 | 0.197 | 0.0163 | 0.0% | 0.108 | yes | 0.96 |
| `0039_syringe_draw_eject_cola` | `multiphysics_ood` | `L1_L2_L3_full` | `fallback_kinematic` | 0.078 | 5.2 | 0.137 | 0.0163 | 0.0% | 0.108 | yes | 2.28 |

### Key findings and open limits

- **Layer-2 contact grounding & local escalation (`E_phys`)**: Cuts mean `MaxPen` by **90.7%** (`0.1252 m -> 0.0117 m`), cuts mean `TrackErr` by **55.0%** (`0.360 m -> 0.162 m`), and lifts counterfactual survival (`CF-OK`) from **20% to 100%** across all 5 probe cases.
- **Naive global rigid physics failure (`L1_global_phys`)**: Collapses `0014_press_ball_1` visual IoU from `0.056` to `0.010` (`-82%`) while failing counterfactual checks on 4/5 cases.
- **Layer-3 anomaly routing (`R_anom`)**: Keeps pure rigid (`0004_brick_fall_1`) in `physics_sim` (zero false fallback), routes `0000_cut_apple_1` (`fallback_topological`) and `0039_syringe_draw_eject_cola` (`fallback_kinematic`), and recovers silhouette fidelity on drifting `0029_cups_uncover_object` (`IoU 0.051 -> 0.060`, `CosSim 0.069 -> 0.088`).
- **Remaining gap for next milestone (`image_to_3d_sim` + full 30-case study)**: Coarse primitives limit multi-view IoU (`0.04`–`0.27`), and `0014_press_ball_1` needs internal optical-flow / local contour deformation cues so L3 triggers `fallback_deformable` even when the 4B VLM misclassifies the object as `rigid`.

## Closure

When the 30-case `with_gt/` ablation and `image_to_3d_sim` comparison complete, distill takeaways into `.agents/notes/concluded/experiment/YYYY-MM-DD-<slug>.md`, update line 3 to `Status: concluded`, and seal with `./scripts/check-docs.sh --seal`.
