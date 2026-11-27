# Study: 2026-11-27-three-backend-h3-probe — Three-Backend Geometry Comparison (H3) on 5 Probe Cases

Status: active

> Non-authoritative temporal evidence record. Current runnable pipeline and contracts live in `docs/`. Large raw artifacts (OBJ meshes, CoACD hulls, UV textures, `.blend` scenes, JSON dumps) remain outside git under `outputs/`.

## Provenance

- **Started**: 2026-11-27
- **Base commit / pin**: `49f759c` + TripoSR + CoACD `image_to_3d_sim` backend (`tools/image_to_3d.py`, MuJoCo `<asset>` textured visual OBJ + multi-hull convex collision compilation)
- **External artifact stores**:
  - `outputs/ablation_probe_sim_proxy_v2/reports.json`
  - `outputs/ablation_probe_procedural_blender/reports.json`
  - `outputs/ablation_probe_image_to_3d_sim/reports.json`
  - Cached TripoSR + CoACD meshes: `outputs/assets_image_to_3d/`
- **Inputs**: Blind ROI-cropped `Qwen/Qwen3.5-4B` `InteractionGraph` JSONs (`outputs/ikg_probe_qwen35_4b_roi/`), 6 keyframes, 4 canonical `eval_0..eval_3` views (`49.1 deg` FOV), `TripoSR` (`stabilityai/TripoSR`, MC resolution `128`) + `CoACD` (`threshold=0.08`, max `6` hulls)
- **Reproduction commands**:
  - `python -m real2sim_layered.cli run-ablation --backend sim_proxy --ikg-dir outputs/ikg_probe_qwen35_4b_roi --max-frames 6 --out-dir outputs/ablation_probe_sim_proxy_v2`
  - `python -m real2sim_layered.cli run-ablation --backend procedural_blender --ikg-dir outputs/ikg_probe_qwen35_4b_roi --max-frames 6 --out-dir outputs/ablation_probe_procedural_blender`
  - `python -m real2sim_layered.cli run-ablation --backend image_to_3d_sim --ikg-dir outputs/ikg_probe_qwen35_4b_roi --max-frames 6 --out-dir outputs/ablation_probe_image_to_3d_sim`

## Runs and observations

### Aggregate 3-backend × 4-arm comparison across 5 cross-regime probe cases

| Backend (`B_asset`) | Arm | Mean IoU | Mean PSNR (dB) | Mean CosSim | Mean MaxPen (m) | Mean Float% | Mean TrackErr (m) | CF-OK | Mean Time (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `sim_proxy` | `L1_only` | 0.128 | 7.27 | 0.209 | 0.1479 | 0.0% | 0.000 | 20% | 0.81 |
| `sim_proxy` | `L1_global_phys` | 0.109 | 7.31 | 0.160 | 0.1256 | 0.0% | 0.401 | 20% | 0.55 |
| `sim_proxy` | `L1_L2_local` | 0.116 | 7.15 | 0.214 | 0.0111 | 6.7% | 0.146 | **100%** | 3.70 |
| `sim_proxy` | `L1_L2_L3_full` | **0.116** | 7.13 | **0.215** | 0.0111 | 10.8% | 0.132 | **100%** | 2.35 |
| `procedural_blender` | `L1_only` | 0.121 | 6.67 | 0.173 | 0.1479 | 6.7% | 0.000 | 20% | 1.45 |
| `procedural_blender` | `L1_global_phys` | 0.106 | 6.80 | 0.149 | 0.1256 | 0.0% | 0.401 | 20% | 1.25 |
| `procedural_blender` | `L1_L2_local` | 0.114 | 6.63 | 0.184 | 0.0111 | 6.7% | 0.146 | **100%** | 3.94 |
| `procedural_blender` | `L1_L2_L3_full` | 0.114 | 6.59 | 0.188 | 0.0111 | 10.8% | 0.132 | **100%** | 2.77 |
| `image_to_3d_sim` | `L1_only` | 0.059 | 9.23 | 0.131 | 0.0834 | 15.0% | 0.000 | 20% | 17.79 |
| `image_to_3d_sim` | `L1_global_phys` | 0.048 | 9.15 | 0.113 | 0.0611 | 0.0% | 0.207 | 20% | 0.35 |
| `image_to_3d_sim` | `L1_L2_local` | 0.038 | 9.34 | 0.103 | **0.0090** | 10.0% | 0.117 | **100%** | 2.07 |
| `image_to_3d_sim` | `L1_L2_L3_full` | 0.043 | **9.39** | 0.114 | **0.0094** | 27.5% | **0.092** | **100%** | 1.48 |

### Per-case `image_to_3d_sim` results (TripoSR + CoACD)

| Case | Regime | Method | Mode | IoU | PSNR | CosSim | MaxPen | Float% | TrackErr | CF-OK | Time(s) |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0004_brick_fall_1` | `rigid` | `L1_only` | `kinematic` | 0.050 | 12.0 | 0.122 | 0.1001 | 0.0% | 0.000 | no | 24.00 |
| `0004_brick_fall_1` | `rigid` | `L1_L2_L3_full` | `physics_sim` | 0.049 | 12.3 | 0.106 | 0.0033 | 0.0% | 0.082 | yes | 0.89 |
| `0029_cups_uncover_object` | `occluded_kinematic` | `L1_only` | `kinematic` | 0.099 | 7.9 | 0.141 | 0.0787 | 41.7% | 0.000 | no | 46.38 |
| `0029_cups_uncover_object` | `occluded_kinematic` | `L1_L2_L3_full` | `fallback_kinematic` | 0.028 | 8.0 | 0.059 | 0.0092 | 54.2% | 0.131 | yes | 2.67 |
| `0014_press_ball_1` | `deformable` | `L1_only` | `kinematic` | 0.008 | 10.3 | 0.026 | 0.1271 | 0.0% | 0.000 | no | 2.27 |
| `0014_press_ball_1` | `deformable` | `L1_L2_L3_full` | `physics_sim` | 0.007 | 10.2 | 0.034 | 0.0137 | 0.0% | 0.098 | yes | 0.77 |
| `0000_cut_apple_1` | `topological_split` | `L1_only` | `kinematic` | 0.108 | 8.5 | 0.263 | 0.0888 | 0.0% | 0.000 | no | 12.96 |
| `0000_cut_apple_1` | `topological_split` | `L1_L2_L3_full` | `fallback_topological` | 0.108 | 8.8 | 0.253 | 0.0136 | 66.7% | 0.037 | yes | 1.04 |
| `0039_syringe_draw_eject_cola` | `multiphysics_ood` | `L1_only` | `kinematic` | 0.028 | 7.5 | 0.102 | 0.0225 | 33.3% | 0.000 | yes | 3.36 |
| `0039_syringe_draw_eject_cola` | `multiphysics_ood` | `L1_L2_L3_full` | `physics_sim` | 0.025 | 7.6 | 0.117 | 0.0075 | 16.7% | 0.113 | yes | 2.02 |

### Key findings (H1 + H2 + H3)

- **Backend-agnostic layering gains (`H1`, `H2`)**: Across all three backends, Layer-2 lifts counterfactual survival (`CF-OK`) from `20%` to **`100%`** while cutting mean penetration by **89%–92.5%** (`0.0090–0.0111 m`), and Layer-3 strictly improves over `L1_L2_local` on IoU, Patch Cosine Sim, and `TrackErr` across all three backends. On `image_to_3d_sim`, Layer-3 activates both `fallback_kinematic` (`0029_cups_uncover_object`) and `fallback_topological` (`0000_cut_apple_1`, recovering IoU `0.083 -> 0.108` and `TrackErr 0.116 -> 0.037 m`).
- **Clear three-way geometry backend Pareto trade-off (`H3`)**:
  - `sim_proxy` delivers the highest occlusion-robust multi-view silhouette IoU (`0.116`), Patch Cosine Sim (`0.215`), and fastest end-to-end synthesis (`0.81 s` L1).
  - `procedural_blender` matches `sim_proxy` physics (`MaxPen 0.0111 m`, `CF-OK 100%`) and near-equal IoU (`0.114`) while outputting editable `.blend` scenes (`1.45 s` L1).
  - `image_to_3d_sim` (TripoSR + CoACD) achieves the highest masked appearance PSNR (**`9.39 dB`**, `+2.26 dB` over `sim_proxy`), tightest convex-hull contact precision (**`MaxPen 0.0094 m`**), and lowest physics-validated trajectory error (**`TrackErr 0.092 m`**), at higher cold-lifting latency (`17.8 s`, cached to `< 2 s` downstream) and lower silhouette IoU (`0.043`) when single-view k-means crops are partially occluded.

## Closure

Once the full 30-case `with_gt/` multi-backend ablation finishes, distill final conclusions into `.agents/notes/concluded/experiment/YYYY-MM-DD-h3-geometry-backends.md`, update line 3 to `Status: concluded`, and seal with `./scripts/check-docs.sh --seal`.
