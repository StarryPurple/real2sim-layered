# Empirical Study Records

This directory holds bounded empirical studies (MVOIK-4D benchmark runs, ablations, physics-escalation diagnostics) as `<study-id>/README.md` plus compact summary tables.

- Keep large raw artifacts (videos, meshes, rendered frames, checkpoints, raw logs) outside git under `outputs/` or `data/`.
- Start every study with `Status: active` on line 3 of `<study-id>/README.md`.
- When a study finishes or pivots, distill the takeaway into `.agents/notes/concluded/<class>/YYYY-MM-DD-<slug>.md`, update line 3 of `<study-id>/README.md` to `Status: concluded`, and seal in place with `./scripts/check-docs.sh --seal`.
- Concluded studies are immutable historical evidence verified by [manifest.sha256](../notes/archived/manifest.sha256).
