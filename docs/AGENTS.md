# AGENTS.md — Documentation Standard

This file defines document tiers, language rules, and maintenance contracts for `real2sim-layered`. Use `prose-standard` for writing/trimming, `doc-garden` for periodic maintenance, and `doc-verify` for fresh-context fact-checking.

## Verification authority

- **Deliverable authority**: `src/real2sim_layered/`, `scripts/`, `tests/`, `pyproject.toml`, `tach.toml`, `Makefile`, and executed `make check` / `make check-env` runs.
- **Non-code authorities**: [basic_instruction.md](../basic_instruction.md) (original topic brief), upstream MVOIK-4D dataset and reference papers, and bounded empirical logs in [.agents/records/](../.agents/records/README.md).

## Tier taxonomy: one home per fact

| Tier | Job | Does NOT belong there |
| :--- | :--- | :--- |
| Root `AGENTS.md` | Always-loaded standing orders, layout, and commands (1–3 lines per rule, linking owners) | Worked examples, situational procedures, restated reference docs |
| `docs/AGENTS.md` | Tier boundaries, fact-to-home map, language mode, and maintenance rules | Subsystem internals, runbooks, historical decisions |
| `README.md` | Entry point: current capabilities, environment setup, quick start, layout map | Unbuilt feature specs, exhaustive references, changelog dumps |
| `docs/` *(current-state)* | Shipped architecture ([architecture.md](architecture.md)), guides, and reference contracts | Unimplemented proposals (→ `.agents/notes/proposed/`), failed experiments |
| `docs/theory.md` | Living research formulation: problem definition, notation, assumptions, hypotheses, falsification criteria; rewrite in place as theory evolves | Shipped code specs (→ `docs/`), raw run logs (→ `.agents/records/`), falsified-assumption history (→ `.agents/notes/concluded/`) |
| `ROADMAP.md` | Agreed near-term tasks and `@owner` assignment; human-editable, agent-authorized; prune completed rows once shipped | Unagreed proposals (→ `.agents/notes/proposed/`), architectural rationale (→ `.agents/notes/implemented/`) |
| `.agents/notes/` | Durable decision and empirical conclusion records (`proposed/`, `implemented/`, `concluded/`, `rejected/`, `archived/`) | Ephemeral review dumps (→ `.agents/scratch/`), current reference specs (→ `docs/`) |
| `.agents/records/` | Bounded empirical studies (`<study-id>/README.md` + compact summaries; large artifacts stay outside git); sealed in place once `Status: concluded` | Current runnable pipeline docs (→ `docs/`), raw datasets or checkpoints |
| `.agents/scratch/` *(gitignored)* | Ephemeral session scratch and review dumps | Anything committed to git (`BD001` fails on tracked files) |

## Non-current-state tier rules

- **`docs/theory.md`**: update in place when assumptions tighten or mechanisms evolve. When an experiment falsifies an assumption, remove or revise the claim in `docs/theory.md` and log the falsification in `.agents/notes/concluded/<class>/YYYY-MM-DD-<slug>.md` citing the sealed study in `.agents/records/`.
- **`ROADMAP.md`**: task lines use `- [ ] <concise task> — <done condition> [— @owner] [[spec](path)]`. Only collaborator-confirmed tasks enter `## Active` or `## Up next`; move completed rows out after shipping.

## Language and i18n

- Mode: `single-en` (primary: `en`).
- Exception: [basic_instruction.md](../basic_instruction.md) is preserved verbatim in Chinese as the original read-only topic prompt.

## Writing rules

- State current truth at `HEAD`; never copy AI-slop prose or narrate past revisions on current-state pages.
- Retrieval surface first: answer first, inventory then expand, path before concept, load-bearing facts on scan anchors.
- Preserve complete propositions (actor, timing, modality, negative guarantees, failure modes) when condensing.
- UTF-8, LF endings, no trailing whitespace, single trailing newline.

## Wordcount budgets

[docs-check.conf](../docs-check.conf) pins token ceilings for `AGENTS.md`, `docs/AGENTS.md`, `README.md`, `docs/architecture.md`, `docs/theory.md`, and `ROADMAP.md`. When `./scripts/check-docs.sh` fails on `BD010`:

1. **Relocate** content owned by a narrower tier and leave a one-line link.
2. **Condense** prose while preserving every factual proposition.
3. **Raise** the ceiling in [docs-check.conf](../docs-check.conf) only when required facts genuinely need the space, and justify the diff in the commit.
