# Study: 2026-11-27-probe-vlm-ikg — Blind Qwen/Qwen3.5-4B IKG Extraction on 5 Probe Cases

Status: active

> Non-authoritative temporal evidence record. Current runnable pipeline and contracts live in `docs/`. Large raw artifacts (extracted JSON dumps and renders) remain outside git under `outputs/`.

## Provenance

- **Started**: 2026-11-27
- **Base commit / pin**: `89dfb84` + MVOIK-4D loader / VLM client / dual-axis evaluator
- **External artifact store**: `outputs/eval_smoke_probe.json`, `outputs/ikg_probe_qwen35_4b_blind/`, `outputs/ikg_probe_qwen35_4b_roi/`
- **Permission / retention**: Local PFS research workspace; regenerate anytime via `python -m real2sim_layered.cli`

## Runs and observations

| Date | Run / Source ID | Config / Commit | Observed Outcome | Collaborator Notes / Limits |
| :--- | :--- | :--- | :--- | :--- |
| 2026-11-27 | `eval_smoke_probe` | `smoke_oracle_check`, 5 probe cases, 4 views (`160x120`), 120 MuJoCo steps | 5/5 cases valid; IoU `1.000`, PSNR `50.0 dB`, CosSim `1.000`, MaxPen `0.0024 m`, Float `0.8%`, CF-OK `yes` | Upper-bound sanity check verifying loader, RGBA metrics, and MuJoCo diagnostics |
| 2026-11-27 | `ikg_blind_fullframe` | `Qwen/Qwen3.5-4B` (`cuda:2`, bf16, `temperature=0`, `max_frames=6`, blind `clip_XXXX`, full-frame 512px) | 5/5 valid Pydantic JSON on 1st attempt (`9.8`–`15.8s`, `1799`–`2116` tok); blind regime match `1/5` (`0000_cut_apple_1`); high confidence (`0.85`–`0.95`) even on misclassified regimes | Foreground occupies only 2%–12% of full frame, crushing small objects (`0014_press_ball_1`, `0039_syringe_draw_eject_cola`) |
| 2026-11-27 | `ikg_blind_roi_crop` | Same as above + union-alpha foreground ROI crop (`pad_ratio=0.15`) | 5/5 valid Pydantic JSON (`9.1`–`15.9s`, `1299`–`2524` tok, ~25% fewer tokens on small-FG clips); blind regime match `2/5` (`0000_cut_apple_1`, `0039_syringe_draw_eject_cola`); 4/4 objects detected on `0029_cups_uncover_object` | Overconfident (`conf=0.95`) regime errors persist on `0014_press_ball_1` (wire coil) and `0004_brick_fall_1`, confirming need for Layer 2/3 runtime physics-residual verification |

## Closure

When this study finishes or pivots, distill the takeaway into `.agents/notes/concluded/experiment/YYYY-MM-DD-<slug>.md`, change line 3 above to `Status: concluded`, and seal in place with `./scripts/check-docs.sh --seal`.
