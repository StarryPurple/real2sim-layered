# AGENTS.md

`real2sim-layered` builds a layered common-sense Real2Sim agent with local physics escalation (MuJoCo and Blender) and anomaly fallback for reconstructing dynamic 4D multi-object interactions from monocular video on MVOIK-4D. Follow [docs/AGENTS.md](docs/AGENTS.md) for documentation tiers, prose standards, and budgets, and [docs/architecture.md](docs/architecture.md) for code boundaries and invariants.

## Repository layout

```text
src/real2sim_layered/  Core Python package
scripts/               Environment setup (env.sh), checks (check_env.py, check-docs.sh), dataset download
tests/                 Automated pytest suite
docs/                  Documentation contract (AGENTS.md), architecture contract, working theory (theory.md)
ROADMAP.md             Agreed near-term task board (human-editable; agent edits require authorization)
basic_instruction.md   Original Chinese course/research topic brief (read-only reference)
.agents/notes/         Durable decision and conclusion records; archived/ is SHA-256 sealed
.agents/records/       Bounded empirical study records (large artifacts remain outside git)
```

## Commands

```sh
make check          # Run formatting check, lint, typecheck, boundary check, shell check, docs check, and tests
make format         # Format Python (ruff) and authored Bash scripts (shfmt)
make lint           # Run read-only code checks (ruff, pyright, tach, ast-grep, vulture, deptry, shellcheck, shfmt -d, gitleaks)
make test           # Run pytest test suite
make docs           # Run ./scripts/check-docs.sh
make check-env      # Verify GPU PyTorch, headless MuJoCo EGL, and headless Blender 4.2
```

## Standing rules

- **Current-state docs only**: update affected `README.md` and current-state `docs/` in the same change as code edits. Apply `prose-standard` (no CoT leakage, change narration, or defensive filler).
- **One home per fact**: place every fact in its canonical tier ([docs/AGENTS.md](docs/AGENTS.md)) and link everywhere else.
- **Working theory vs. shipped code**: [docs/theory.md](docs/theory.md) holds the living research formulation and hypotheses; source code and shipped `docs/` own `HEAD` behavior.
- **Notes, records, and scratch**: keep `.agents/scratch/` untracked (`BD001`). Record durable decisions in [.agents/notes/](.agents/notes/README.md) and bounded experiment runs in [.agents/records/](.agents/records/README.md); sealed files in `.agents/notes/archived/` and concluded studies are frozen.
- **Roadmap discipline**: never add, reword, or reassign rows in [ROADMAP.md](ROADMAP.md) without explicit collaborator authorization; tick `[x]` only on a uniquely matched assigned row after verifying its done condition.
- **Code health and skills**: read [docs/architecture.md](docs/architecture.md) before introducing new modules or changing dependencies. Use `codebase-review` on substantive diffs, `codebase-gc` periodically for entropy cleanup, and `doc-garden` / `doc-verify` for documentation upkeep.
- **Verify before finishing**: run `make check` (and `./scripts/check-docs.sh --probes` when editing docs); report only commands actually executed.
