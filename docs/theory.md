# Theoretical Architecture

Status: working formulation (tier rules live in [docs/AGENTS.md](AGENTS.md); source code and shipped docs own `HEAD` implementation state).

## Problem formulation

Given a monocular RGB(A) video `V = {I_1, ..., I_T}` observing dynamic multi-object interactions, reconstruct an **executable, editable, physically grounded 4D simulation scene** `S = (A, Theta, X_{1:T}, M)` consisting of 3D object assets `A`, physical parameters `Theta`, temporal trajectories/deformations `X_{1:T}`, and per-interval execution modes `M`, evaluated on **MVOIK-4D** (`with_gt` 30 cases, expandable to `with_gt_memory` and `with_gt_gaussian_ls`).

Real-world interaction videos span **five physically heterogeneous regimes** present in MVOIK-4D `with_gt`:

1. **Rigid-body dynamics** (gravity, tipping, collision, support transfer; e.g., `0004_brick_fall_1`).
2. **Multi-object occlusion and permanence** (sliding, covering, uncovering; e.g., `0029_cups_uncover_object`).
3. **Soft-body / elastic deformation** (contact compression and recovery; e.g., `0014_press_ball_1`).
4. **Topological splitting / state transition** (cutting and peeling into sub-parts; e.g., `0000_cut_apple_1`).
5. **Multiphysics out-of-distribution (OOD)** (combustion, fluid ejection; e.g., `0038_match_lit`, `0039_syringe_draw_eject_cola`).

This heterogeneity exposes a fundamental dilemma in current Real2Sim agents:

- Pure visual or keyframed programmatic reconstruction (e.g., BVB-style `VLM -> Blender`) matches coarse layout but violates gravity, contact, support, and object permanence.
- Heuristic 4D Gaussian post-processing (e.g., rigid alignment and voxel smoothing on neural 4DGS sequences) lacks equations of motion, contact forces, and interactive simulatability.
- Uniformly forcing a rigid-body simulator across all frames causes solver divergence or visual destruction on non-rigid, topological-split, and multiphysics cases.

## Definitions and notation

| Symbol / Term | Definition | Domain / Constraints |
| :--- | :--- | :--- |
| `V` | Input monocular foreground-segmented RGBA sequence `{I_1, ..., I_T}` | Single camera, dynamic multi-object interaction |
| `G_int` | Structured Interaction Knowledge Graph (entities, contact/occlusion intervals, regime tags `r`) | `r` in `{rigid, occluded_kinematic, deformable, topological_split, multiphysics_ood}` |
| `B_asset` | Pluggable 3D geometry backend producing scene assets `A` | `{procedural_blender, sim_proxy, image_to_3d_sim}` |
| `S_0` | Layer-1 common-sense coarse 4D scene hypothesis | Editable Blender / MJCF scene with kinematic trajectories |
| `E_phys` | Layer-2 local physics escalation operator (MuJoCo / Blender simulation + parameter inversion) | Invoked selectively on critical contact, support, or collision windows |
| `D_phys` | Deterministic physical-visual verifier report | Penetration depth, floating ratio, support stability, permanence check, silhouette IoU |
| `R_anom` | Layer-3 failure-aware anomaly detector and fallback router | Compares simulated vs. observed residuals to switch solver/mode |

## Core assumptions and hypotheses

- **A1 (Layered sufficiency)**: Most non-contact transport segments can be initialized from a compact local VLM (`Qwen3.5-4B-VL` baseline, scalable) producing `G_int` and coarse `S_0`, without global physics optimization on every frame — *Falsification criterion*: L1 initialization errors routinely fall outside the convergence basin of local physics refinement on MVOIK-4D contact windows.
- **A2 (Simulator-in-the-loop discriminability)**: Quantitative diagnostics `D_phys` from MuJoCo/Blender on local contact intervals expose physical failures invisible to visual-only VLM critique and drive autonomous repair without human-in-the-loop annotation — *Falsification criterion*: simulator feedback fails to improve contact/stability or multi-view accuracy over visual-only refinement loops on MVOIK-4D.
- **A3 (Detectable solver-model mismatch)**: When the observed interaction violates active simulator assumptions (soft-body compression, cutting, combustion, fluid jet), coupled physics-divergence and visual-reprojection residuals separate cleanly enough for `R_anom` to trigger fallback before simulation corrupts the scene — *Falsification criterion*: residual gating causes excessive false fallbacks on rigid cases or fails to prevent regressions on non-rigid/topological/OOD cases relative to global physics.
- **H1 (Selective escalation efficiency)**: Selective `L1 -> L2 -> L3` escalation outperforms both visual-only reconstruction and uniform global rigid-body simulation across the 5 MVOIK-4D regimes at lower simulation and agent-turn cost *(open hypothesis)*.
- **H2 (Small VLM + deterministic physics leverage)**: Grounding a compact local VLM (`Qwen3.5-4B-VL`) with structured `G_int` and deterministic `D_phys` feedback narrows or closes the gap to larger visual-only VLMs on physical accuracy and contact timing *(open hypothesis)*.
- **H3 (Geometry-backend frontier)**: Comparing `procedural_blender`, `sim_proxy`, and `image_to_3d_sim` (Image-to-3D mesh + convex decomposition) reveals a measurable appearance-vs-simulatability trade-off where sim-ready mesh processing governs contact stability *(open hypothesis)*.

## Theoretical architecture

1. **Layer 1 — Common-sense structuring and coarse synthesis (`G_int`, `S_0`)**: A lightweight local VLM (`Qwen3.5-4B-VL`, swappable via a unified OpenAI-compatible client) parses sampled frames of `V` into an interaction graph `G_int` with per-interval regime tags and constructs an editable coarse scene `S_0` using a selectable geometry backend `B_asset`.
2. **Layer 2 — Local physics escalation and verification (`E_phys`, `D_phys`)**: On contact, collision, support-transfer, and occlusion intervals flagged by `G_int`, the harness runs local MuJoCo or Blender simulation (`E_phys`), computes structured physical and silhouette diagnostics (`D_phys`), and feeds actionable error summaries back to the agent for bounded iterative refinement.
3. **Layer 3 — Failure-aware anomaly detection and fallback (`R_anom`)**: Monitors simulation instability and persistent simulation-vs-observation silhouette/motion mismatch. When solver assumptions break, `R_anom` routes the affected object or time interval to a regime-appropriate fallback (compliant/deformable proxy, event-triggered asset split, or constrained kinematic/appearance replay) while preserving physics-verified neighboring segments.

### Scope boundary

This architecture targets **single-video, per-scene executable Real2Sim reconstruction and verification**. It is intentionally orthogonal to downstream robot-policy training and cross-task Recursive Self-Improvement (RSI) loops, and produces simulatable MJCF / `.blend` scenes rather than static neural 4DGS sequences.

## Experimental protocol and ablation axes

- **5-case cross-regime probe set** (rapid iteration): `0004_brick_fall_1` (rigid), `0029_cups_uncover_object` (occlusion), `0014_press_ball_1` (deformable), `0000_cut_apple_1` (topological split), `0039_syringe_draw_eject_cola` (multiphysics OOD).
- **Full core evaluation**: all 30 cases in `data/MVOIK-4D/with_gt/` evaluated across 4 canonical views (`eval_0..eval_3`) plus physical plausibility metrics (penetration, floating, stability, counterfactual perturbation survival) and compute/token cost.
- **Controlled ablation axes**:
  1. **Layering**: `L1 only` vs. `L1 + global rigid physics` vs. `L1 + L2 local escalation` vs. `L1 + L2 + L3 anomaly fallback`.
  2. **Feedback modality**: open-loop vs. visual-only VLM critique vs. structured `D_phys` diagnostics + error overlays.
  3. **Geometry backend (`B_asset`)**: `procedural_blender` vs. `sim_proxy` vs. `image_to_3d_sim`.
  4. **VLM scaling**: `Qwen3.5-4B-VL` baseline vs. larger local or API VLM backbones.

## Evidence and revision trail

- **Evidence owner**: MVOIK-4D benchmark runs logged under [.agents/records/](../.agents/records/README.md).
- **Falsified-assumption log**: [.agents/notes/concluded/](../.agents/notes/README.md) paired with sealed study directories in `.agents/records/`.
