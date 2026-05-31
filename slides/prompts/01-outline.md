You are helping me prepare the CAiSE 2026 talk for pmf-tsfm.

## Inputs
- `paper.pdf` — full paper
- `SLIDES.md` — narrative brief, hard constraints, locked outline
- `talk_design/outline.md` — beat-by-beat content (if exists)
- `talk_design/breif.md` — one-page reference

## What to do

If `talk_design/outline.md` exists, your job is to PRESSURE-TEST the existing outline
against the paper, not recreate it. Look for:
- Mismatches between paper findings and slide claims
- Slides where the question they answer is unclear
- Missing transitions between beats
- Beats whose time budget can't fit the planned visual + speech

If `talk_design/outline.md` does NOT exist, produce one from scratch.

## For each beat give

- Title and time allocation (must sum to session minus buffer)
- One-sentence purpose
- The question the audience is asking in their head as the slide opens
- The spoken transition INTO this beat from the previous one
- The visual artifact (chart, diagram, animation, gif)

## End the outline with

- The single sentence the audience should remember walking out
- Two or three places where audience attention is most at risk and why
- One slide that, if cut, would most weaken the talk

## Style

Be opinionated. The paper and the talk are different artifacts; do not preserve the
paper's section order out of habit. Lead with the prescriptive headline; let the rest of
the talk earn it.

DO NOT produce Slidev slides yet. Outline only.

## Constraints to honor from SLIDES.md

- Time budget: 20 min talk + 10 min Q&A (default); ~16 min content + 2–3 min demo + buffer
- Max 5 bullets per slide; max 12 words per bullet
- Audience: ~80% PM/IS, ~20% adjacent; PMF NOT assumed known; "foundation model"
  defaults to LLM — must differentiate
- One-line definitions required for: zero-shot, LoRA, Entropic Relevance
- Critique-derived constraints in SLIDES.md are non-negotiable (slide 4 wording,
  closing hedge clause, ER memorized rebuttal)

## What good looks like

- Each beat is teachable in its allotted time (read slowly, leave breathing room).
- Each transition is one sentence the speaker can rehearse cold.
- The headline finding lives in spoken anchor lines, not just slide bullets.
- Visuals reuse paper figures where possible; new artwork is justified by what it teaches.
