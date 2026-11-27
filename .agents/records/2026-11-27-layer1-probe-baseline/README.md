# Study: 2026-11-27-layer1-probe-baseline — Layer-1 Coarse Real2Sim Baseline on 5 Probe Cases

Status: active

> Non-authoritative temporal evidence record. Current runnable pipeline and contracts live in `docs/`. Large raw artifacts (`.blend` files, MJCF XMLs, rendered PNGs) remain outside git under `outputs/layer1_probe/`.

## Provenance

- **Started**: 2026-11-27
- **Base commit / pin**: `63e1000` + Layer-1 synthesizer (`sim_proxy` & `procedural_blender`)
- **External artifact store**: `outputs/layer1_probe/reports.json`, `outputs/layer1_probe/{sim_proxy,procedural_blender}/<case_id>/{scene_spec.json,scene.xml,build_scene.py,scene.blend}`
- **Inputs**: Blind ROI-cropped `Qwen/Qwen3.5-4B` `InteractionGraph` JSONs from `outputs/ikg_probe_qwen35_4b_roi/`
- **Permission / retention**: Local PFS workspace; reproduce via `python -m real2sim_layered.cli run-layer1 --backend both --ikg-dir outputs/ikg_probe_qwen35_4b_roi --max-frames 6 --out-dir outputs/layer1_probe`

## Runs and observations

| Case | Regime | Method | IoU | PSNR (dB) | CosSim | Stable | MaxPen (m) | Float% | CF-OK |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0004_brick_fall_1` | `rigid` | `L1_sim_proxy` | 0.152 | 8.2 | 0.257 | yes | 0.0548 | 1.0% | no |
| `0004_brick_fall_1` | `rigid` | `L1_procedural_blender` | 0.144 | 7.6 | 0.203 | yes | 0.0548 | 1.0% | no |
| `0029_cups_uncover_object` | `occluded_kinematic` | `L1_sim_proxy` | 0.095 | 6.7 | 0.190 | yes | 0.1077 | 0.0% | no |
| `0029_cups_uncover_object` | `occluded_kinematic` | `L1_procedural_blender` | 0.096 | 6.4 | 0.159 | yes | 0.1077 | 0.0% | no |
| `0014_press_ball_1` | `deformable` | `L1_sim_proxy` | 0.045 | 8.3 | 0.116 | yes | 0.0233 | 0.0% | yes |
| `0014_press_ball_1` | `deformable` | `L1_procedural_blender` | 0.044 | 6.9 | 0.103 | yes | 0.0233 | 0.0% | yes |
| `0000_cut_apple_1` | `topological_split` | `L1_sim_proxy` | 0.276 | 6.9 | 0.412 | yes | 0.0311 | 0.0% | yes |
| `0000_cut_apple_1` | `topological_split` | `L1_procedural_blender` | 0.271 | 7.3 | 0.374 | yes | 0.0311 | 0.0% | yes |
| `0039_syringe_draw_eject_cola` | `multiphysics_ood` | `L1_sim_proxy` | 0.111 | 4.8 | 0.236 | yes | 0.0064 | 0.5% | yes |
| `0039_syringe_draw_eject_cola` | `multiphysics_ood` | `L1_procedural_blender` | 0.098 | 5.3 | 0.202 | yes | 0.0064 | 0.5% | yes |

### Key baseline observations

- **Severe L1 physical violations on rigid & multi-object scenes**: Both `0004_brick_fall_1` (`MaxPen=0.0548 m`) and `0029_cups_uncover_object` (`MaxPen=0.1077 m` across 4 overlapping bodies) fail counterfactual perturbation survival (`CF-OK=no`), providing a clear quantitative target for Layer-2 local physics escalation (`E_phys`).
- **Backend parity**: `sim_proxy` (MuJoCo EGL) and `procedural_blender` (headless Blender 4.2 Cycles + `.blend`) produce closely aligned silhouettes across all 5 cases (`IoU` delta `<= 0.013`) while exposing appearance/geometry limits of coarse primitives on thin/curved objects (`0014_press_ball_1`, `0039_syringe_draw_eject_cola`).

## Closure

When Layer-1 baseline comparisons complete (including `image_to_3d_sim`), distill takeaways into `.agents/notes/concluded/experiment/YYYY-MM-DD-<slug>.md`, change line 3 above to `Status: concluded`, and seal in place with `./scripts/check-docs.sh --seal`.
