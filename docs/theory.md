# Theoretical Architecture

Status: working formulation (tier rules live in [docs/AGENTS.md](AGENTS.md); source code and shipped docs own `HEAD` implementation state).

## Problem formulation

Given a monocular RGB video observing dynamic multi-object interactions (grasping, collision, insertion, folding, stretching), reconstruct an editable, physically plausible 4D simulation scene evaluated on the **MVOIK-4D** benchmark. The target system bridges multimodal video understanding with programmatic scene construction (Blender) and local physics simulation (MuJoCo / Blender rigid-body) through a layered agent harness.

## Definitions and notation

| Symbol / Term | Definition | Domain / Constraints |
| :--- | :--- | :--- |
| `V` | Input monocular RGB video sequence `{I_1, ..., I_T}` | Single camera, dynamic multi-object interaction |
| `G_int` | Structured interaction knowledge graph (objects, actions, contact intervals, temporal transitions) | Extracted from `V` before or alongside 3D synthesis |
| `S_0` | Coarse common-sense 4D scene hypothesis (geometry proxies, kinematics, contact schedule) | Lightweight programmatic representation |
| `E_phys` | Local physics escalation operator (MuJoCo / Blender simulation + contact/stability residual check) | Invoked on interaction intervals with physical ambiguity or violation |
| `R_anom` | Anomaly detector and fallback policy triggered when physical or visual residuals exceed tolerance | Switches solver fidelity, tool backend, or falls back to constrained kinematics |

## Core assumptions and hypotheses

- **A1 (Layered sufficiency)**: Most non-contact or coarse-motion segments can be initialized from common-sense VLM priors and programmatic geometry (`S_0`) without running full-scene high-cost physics optimization at every frame — Falsification criterion: coarse initialization error consistently prevents local physics refinement from converging on MVOIK-4D contact intervals.
- **A2 (Local physics discriminability)**: Targeted physical verification (`E_phys`) on contact, collision, and support events exposes penetration, floating, and instability errors that visual-only reconstruction misses — Falsification criterion: physics-verified reconstructions fail to improve contact/stability or 4D accuracy metrics over visual-only baselines on MVOIK-4D.
- **A3 (Graceful anomaly fallback)**: Deformable, heavily occluded, or non-rigid events where rigid-body physics models break can be detected by residual spikes and routed to an explicit fallback path (`R_anom`) without corrupting surrounding segments — Falsification criterion: anomaly detection produces excessive false triggers or degrades end-to-end reconstruction quality compared to no fallback.
- **H1 (Layered escalation efficiency)**: Selective local physics escalation achieves comparable or higher 4D interaction fidelity than uniform global physics optimization at lower compute and agent-turn cost *(open hypothesis)*.
- **H2 (Interaction-graph grounding)**: Conditioning tool selection and escalation boundaries on `G_int` reduces multi-object occlusion and contact-ordering failures relative to flat prompt-to-Blender baselines *(open hypothesis)*.

## Theoretical architecture

1. **Layer 1 — Common-sense perception and structuring**: Sample key frames from `V`, build an interaction graph `G_int`, and synthesize an initial editable scene hypothesis `S_0`.
2. **Layer 2 — Local physics escalation**: Identify physically critical sub-intervals (contact onset, support transfer, collision, insertion) from `G_int` and verify/refine them via targeted MuJoCo or Blender simulation (`E_phys`).
3. **Layer 3 — Anomaly detection and fallback**: Monitor simulation divergence, severe visual-reprojection mismatch, or unsupported deformation modes; invoke `R_anom` to select an alternative 3D/4D tool or constrained kinematic repair.

## Evidence and revision trail

- **Evidence owner**: MVOIK-4D benchmark runs logged under [.agents/records/](../.agents/records/README.md).
- **Falsified-assumption log**: [.agents/notes/concluded/](../.agents/notes/README.md) paired with sealed study directories in `.agents/records/`.
