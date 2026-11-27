# Study: 2026-11-27-full30-three-backend-ablation — Full 30-Case MVOIK-4D Benchmark Across Three Geometry Backends

Status: active

> Non-authoritative temporal evidence record. Current runnable pipeline and contracts live in `docs/`. Large raw artifacts (360 rendered/simulated scene folders, TripoSR+CoACD meshes, PNG plots, JSON dumps) remain outside git under `outputs/`.

## Provenance

- **Started**: 2026-11-27
- **Base commit / pin**: `4636b41` + `extract-ikg --all-cases` and lazy `coacd` import fix (`tools/image_to_3d.py`)
- **Dataset coverage**: Full `MVOIK-4D/with_gt/` split (`n = 30` cases: `topological_split=12`, `occluded_kinematic=8`, `deformable=6`, `rigid=2`, `multiphysics_ood=2`)
- **External artifact stores**:
  - Blind ROI-cropped `Qwen/Qwen3.5-4B` IKGs (30/30): `outputs/ikg_all30_qwen35_4b_roi/`
  - 30-case 4-arm ablation JSON/MD reports:
    - `outputs/ablation_all30_sim_proxy/{reports.json,reports.md}`
    - `outputs/ablation_all30_procedural_blender/{reports.json,reports.md}`
    - `outputs/ablation_all30_image_to_3d_sim/{reports.json,reports.md}`
  - Publication figures: `outputs/figures_all30/{fig1_all30_three_backends_ablation.png,fig2_h3_pareto_all30.png}`
- **Reproduction commands**:
  - `python -m real2sim_layered.cli extract-ikg --all-cases --max-frames 6 --out-dir outputs/ikg_all30_qwen35_4b_roi`
  - `python -m real2sim_layered.cli run-ablation --backend sim_proxy --all-cases --ikg-dir outputs/ikg_all30_qwen35_4b_roi --max-frames 6 --out-dir outputs/ablation_all30_sim_proxy`
  - `python -m real2sim_layered.cli run-ablation --backend procedural_blender --all-cases --ikg-dir outputs/ikg_all30_qwen35_4b_roi --max-frames 6 --out-dir outputs/ablation_all30_procedural_blender`
  - `python -m real2sim_layered.cli run-ablation --backend image_to_3d_sim --all-cases --ikg-dir outputs/ikg_all30_qwen35_4b_roi --max-frames 6 --out-dir outputs/ablation_all30_image_to_3d_sim`

## Runs and observations

### Blind VLM (`Qwen/Qwen3.5-4B`) 30-case regime recognition and overconfidence

With anonymized `clip_XXXX` prompts and foreground ROI cropping, `Qwen/Qwen3.5-4B` produced `30/30` (`100%`) schema-valid `InteractionGraph` JSONs (`10.0`–`27.8 s` per clip):

| Regime | Cases (`n`) | Blind Accuracy | Mean Confidence (Correct / Wrong) |
| :--- | :---: | :---: | :---: |
| `topological_split` | 12 | 11/12 (91.7%) | — |
| `multiphysics_ood` | 2 | 2/2 (100.0%) | — |
| `rigid` | 2 | 1/2 (50.0%) | — |
| `deformable` | 6 | 1/6 (16.7%) | — |
| `occluded_kinematic` | 8 | 0/8 (0.0%) | — |
| **Overall** | **30** | **15/30 (50.0%)** | **0.95 / 0.95** |

Identical confidence (`0.95`) on correct and misclassified clips confirms that small-VLM priors are uncalibrated and require VLM-independent runtime escalation triggers plus pre-commit gating.

### Full 30-case aggregate results across all 3 backends × 4 layering arms (360 runs)

| Backend (`B_asset`) | Arm | Mean IoU | Mean PSNR (dB) | Mean CosSim | Mean MaxPen (m) | Mean Float% | Mean TrackErr (m) | CF-OK | L3 Triggers | Mean Time (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `sim_proxy` | `L1_only` | 0.129 ± 0.077 | 7.47 | 0.221 | 0.2085 | 5.7% | 0.000 | 6.7% | 0/30 | 0.50 |
| `sim_proxy` | `L1_global_phys` | 0.083 ± 0.078 | 7.58 | 0.143 | 0.1789 | 0.0% | 0.841 | 6.7% | 0/30 | 0.33 |
| `sim_proxy` | `L1_L2_local` | **0.132 ± 0.075** | 7.38 | **0.235** | 0.0469 | 14.4% | 0.200 | **100.0%** | 0/30 | 2.24 |
| `sim_proxy` | `L1_L2_L3_full` | 0.128 ± 0.069 | 7.34 | 0.231 | **0.0307** | 17.1% | **0.190** | 96.7% | 7/30 | 1.50 |
| `procedural_blender` | `L1_only` | 0.127 ± 0.079 | 6.66 | 0.197 | 0.2085 | 5.7% | 0.000 | 6.7% | 0/30 | 1.36 |
| `procedural_blender` | `L1_global_phys` | 0.082 ± 0.080 | 6.88 | 0.137 | 0.1789 | 0.0% | 0.841 | 6.7% | 0/30 | 1.27 |
| `procedural_blender` | `L1_L2_local` | **0.134 ± 0.079** | 6.60 | **0.219** | 0.0469 | 14.4% | 0.200 | **100.0%** | 0/30 | 3.17 |
| `procedural_blender` | `L1_L2_L3_full` | 0.129 ± 0.072 | 6.56 | 0.214 | **0.0307** | 17.1% | **0.190** | 96.7% | 7/30 | 2.41 |
| `image_to_3d_sim` | `L1_only` | 0.069 ± 0.041 | 9.23 | 0.149 | 0.1216 | 9.5% | 0.000 | 16.7% | 0/30 | 15.82 |
| `image_to_3d_sim` | `L1_global_phys` | 0.054 ± 0.043 | 9.23 | 0.122 | 0.0936 | 0.0% | 0.248 | 16.7% | 0/30 | 0.40 |
| `image_to_3d_sim` | `L1_L2_local` | **0.073 ± 0.047** | 9.25 | **0.157** | 0.0184 | 19.5% | 0.154 | **100.0%** | 0/30 | 2.02 |
| `image_to_3d_sim` | `L1_L2_L3_full` | 0.064 ± 0.045 | **9.33** | 0.138 | **0.0147** | 27.5% | **0.116** | **100.0%** | 16/30 | 1.39 |

### Paired Wilcoxon signed-rank significance (`n = 30`)

| Backend | `MaxPen (L2 < L1)` | `MaxPen (L2 < Global)` | `TrackErr (L2 < Global)` | `IoU (L2 > Global)` | `MaxPen` Reduction (L3 vs L1) | `TrackErr` Reduction (L3 vs Global) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `sim_proxy` | `p = 1.30e-08` | `p = 1.64e-07` | `p = 9.31e-09` | `p = 7.98e-04` | **-85.3%** | **-77.4%** |
| `procedural_blender` | `p = 1.30e-08` | `p = 1.64e-07` | `p = 9.31e-09` | `p = 4.18e-04` | **-85.3%** | **-77.4%** |
| `image_to_3d_sim` | `p = 1.86e-09` | `p = 1.02e-07` | `p = 1.40e-05` | `p = 2.38e-03` | **-87.9%** | **-53.3%** |

### Per-regime breakdown (`sim_proxy` vs `image_to_3d_sim`)

| Regime (`n`) | Backend | `L1_only` (IoU / Pen / CF) | `L1_global_phys` (IoU / Pen / CF) | `L1_L2_local` (IoU / Pen / CF) | `L1_L2_L3_full` (IoU / Pen / CF) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `rigid` (2) | `sim_proxy` | 0.172 / 0.162 / 0% | 0.131 / 0.119 / 0% | 0.137 / 0.008 / 100% | 0.137 / 0.008 / 100% |
| `occluded_kinematic` (8) | `sim_proxy` | 0.132 / 0.202 / 0% | 0.097 / 0.168 / 0% | 0.123 / 0.094 / 100% | 0.117 / 0.055 / 88% |
| `deformable` (6) | `sim_proxy` | 0.177 / 0.188 / 0% | 0.130 / 0.146 / 0% | 0.179 / 0.078 / 100% | 0.166 / 0.049 / 100% |
| `topological_split` (12) | `sim_proxy` | 0.108 / 0.263 / 0% | 0.045 / 0.240 / 0% | 0.126 / 0.012 / 100% | 0.126 / 0.012 / 100% |
| `multiphysics_ood` (2) | `sim_proxy` | 0.050 / 0.015 / 100% | 0.059 / 0.012 / 100% | 0.056 / 0.011 / 100% | 0.056 / 0.011 / 100% |
| `rigid` (2) | `image_to_3d_sim` | 0.086 / 0.110 / 0% | 0.092 / 0.090 / 0% | 0.085 / 0.011 / 100% | 0.064 / 0.008 / 100% |
| `occluded_kinematic` (8) | `image_to_3d_sim` | 0.087 / 0.102 / 25% | 0.077 / 0.076 / 25% | 0.089 / 0.015 / 100% | 0.073 / 0.013 / 100% |
| `deformable` (6) | `image_to_3d_sim` | 0.078 / 0.134 / 17% | 0.053 / 0.086 / 17% | 0.085 / 0.025 / 100% | 0.088 / 0.024 / 100% |
| `topological_split` (12) | `image_to_3d_sim` | 0.060 / 0.148 / 0% | 0.037 / 0.123 / 0% | 0.064 / 0.021 / 100% | 0.055 / 0.013 / 100% |
| `multiphysics_ood` (2) | `image_to_3d_sim` | 0.014 / 0.016 / 100% | 0.028 / 0.013 / 100% | 0.012 / 0.007 / 100% | 0.012 / 0.007 / 100% |

### H2 feedback-modality ablation (`sim_proxy`, `n = 30`, with counterfactual pre-commit guard)

Adding `counterfactual_survival` preservation to the Layer-3 pre-commit gate (`agent/layer3.py`) lifts `dual_axis` (`L1_L2_L3_full`) to `100.0%` (`30/30`) `CF-OK` (`outputs/ablation_all30_feedback_h2/reports.json`, `scripts/run_feedback_ablation.py`):

| Feedback Modality | Mean IoU | Mean PSNR (dB) | Mean CosSim | Mean MaxPen (m) | Mean TrackErr (m) | CF-OK |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `visual_only` | 0.1535 | 7.57 | 0.2600 | 0.2328 | 0.0540 | 3.3% (1/30) |
| `physics_only` | 0.1309 | 7.32 | 0.2321 | **0.0221** | 0.2363 | 96.7% (29/30) |
| `dual_axis` (Ours) | 0.1280 | 7.34 | 0.2312 | **0.0298** (`-87.2%`, `p=1.86e-09`) | **0.1883** (`-20.3%`) | **100.0% (30/30)** |

### Key findings (H1 + H2 + H3 at full `n = 30` scale)

- **`H1` (Local vs. Global Physics)**: On all 3 backends, `L1_global_phys` severely degrades visual IoU (`-22.5%` to `-35.7%`) and leaves `83%`–`93%` of scenes failing counterfactual checks. `L1_L2_local` lifts `CF-OK` to **`100%` (`30/30`)** on all 3 backends while **exceeding `L1_only` on Mean IoU and Patch Cosine Sim** across all 3 backends (`p <= 2.38e-03` vs. global physics; `p <= 1.86e-09` on penetration).
- **`H2` (Failure-Aware Layer-3 Fallback)**: Layer-3 activates all three fallback modes across the 30-case split (`7/30` on `sim_proxy`/`procedural_blender`; `16/30` on `image_to_3d_sim`), further slashing residual penetration by **20%–35%** (`0.0469 -> 0.0307 m` and `0.0184 -> 0.0147 m`) and cutting trajectory drift (`0.200 -> 0.190 m` and `0.154 -> 0.116 m`). One edge case (`0021_shuffle_2` under `sim_proxy`) reveals an input-view vs. multi-view gating nuance where a VLM misclassified `topological_split` caused a slight `CF-OK` drop (`100% -> 96.7%`), suggesting adding a `counterfactual_survival` check to the Layer-3 pre-commit gate.
- **`H3` (Three-Backend Pareto Frontier)**: `sim_proxy` (`IoU 0.132`, `CosSim 0.235`, `0.50 s` L1) and `procedural_blender` (`IoU 0.134`, editable `.blend`, `1.36 s` L1) lead occlusion-robust silhouette coverage and latency; `image_to_3d_sim` (TripoSR + CoACD) leads masked appearance fidelity (**`PSNR 9.33 dB`**), convex-hull contact tightness (**`MaxPen 0.0147 m`**), and lowest physics-validated trajectory error (**`TrackErr 0.116 m`**) at `100%` `CF-OK`.

## Closure

When paper drafting begins (after optional feedback-modality ablation and larger-VLM scaling check), distill final claims into `.agents/notes/concluded/experiment/YYYY-MM-DD-full30-mvoik4d-benchmark.md`, update line 3 to `Status: concluded`, and seal with `./scripts/check-docs.sh --seal`.
