# Agent Notes

An Agent Note records a durable decision, proposal, or empirical conclusion that affects `real2sim-layered`—the rationale, alternatives rejected, consequences, and verification that code and reference docs do not carry. Mechanical checks run via [scripts/check-docs.sh](../../scripts/check-docs.sh).

## Layout and naming

Every active Agent Note path uses lowercase kebab-case (`[a-z0-9-]+`) for `{lifecycle}`, `{class}`, and `slug`: `{lifecycle}/{class}/YYYY-MM-DD-slug.md`.

- `proposed/`: reviewed proposals for unbuilt work.
- `implemented/`: shipped decisions written in present tense and kept current with shipped facts.
- `concluded/`: distilled empirical findings and falsified-assumption takeaways citing sealed studies under [.agents/records/](../records/README.md).
- `rejected/`: considered and declined proposals; keep only while the rationale prevents a plausible, tempting mistake, otherwise delete outright.
- `archived/`: frozen historical snapshots sealed in [archived/manifest.sha256](archived/manifest.sha256).

Allowed classes are configured in [docs-check.conf](../../docs-check.conf) (`feature,bug-fix,architecture,simplification,process,testing,hypothesis,experiment,pipeline`). Do not add a centralized `INDEX.md`.

## Header and required sections

First four lines of every active note (line 2 and line 4 must be blank):

```markdown
# Agent Note: <title>

Status: proposed

```

Exact line-3 `Status:` forms:
- `proposed/`: `Status: proposed`
- `implemented/`: `Status: implemented`
- `concluded/`: `Status: concluded`
- `rejected/`: `Status: rejected — <non-whitespace reason>`

Required `##` sections:
- `proposed/`: `## Problem`, `## Proposal`, `## Alternatives considered`, `## Acceptance criteria`, `## Risks`.
- `implemented/`: `## Problem`, `## Decision`, `## Alternatives considered`, `## Consequences` (no proposal-era headings).
- `concluded/`: `## Hypothesis / Question`, `## Evidence`, `## Conclusion`, `## Direction impact`, `## Alternatives considered`.
- `rejected/`: `## Problem`, `## Proposal`, `## Alternatives considered`.

## Supersession and deletion filter

- Every new note triggers a supersession scan over active notes. Full supersession absorbs unique rationale into the new owner and deletes or archives the superseded note; partial supersession keeps both active and cross-linked.
- Delete discarded incomplete reasoning or draft mistakes completely; never place them in `rejected/` or `archived/`.
- Mechanical or local edits do not require an Agent Note.
