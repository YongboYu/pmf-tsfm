# 04 · Visual feedback (Pass-3 slide critique)

Critique how a *rendered* slide looks and **route** each fix — do not edit anything yourself.

Scope: this prompt diagnoses an already-wired, already-rendered slide. It does NOT
produce figures, gifs, or the demo — that is out-of-band production (EPS conversion,
`scripts/make_figures.py`, the timeline SVG / DFG animation / screencast). Run this
*after* a beat's figure is wired, in the Pass-3 polish loop.

## Input
- A screenshot of slide [N], e.g. `pnpm export --format png --output screenshots/` → attach `screenshots/N.png`.
- One slide at a time.

## Read first (so a critique never violates a locked decision)
1. `SLIDES.md` → the slide's spoken anchor line(s) and the **critique-derived, non-negotiable
   constraints**, e.g.:
   - beat 4: "no event logs in pretraining, **to our knowledge**" (not "no process data")
   - beat 7a notes: "two strongest baselines from our prior benchmark"
   - beat 9: the **memorized ER rebuttal**, word-for-word in speaker notes
   - beat 10: the "four logs is not a paradigm" hedge
   - inline definitions required: **zero-shot**, **LoRA**, **Entropic Relevance**
2. The matching section of `talk_design/outline.md` (the beat's purpose + audience question).

**Never propose a fix that removes or weakens a locked constraint or anchor line.** If a
density fix would cut one, say so and stop.

## Critique (4 lenses)
1. **Density** — too much / too little / right for 30–60s of speaking?
2. **Visual hierarchy** — where does the eye land first? Is that the right place?
3. **One-question test** — what question does this slide answer? Can you tell from the
   slide alone, without the voiceover? (visual-IS-content beats 7b & 11 are exempt — thin text is fine)
4. **Bullets vs. visual weight** — is the slide carrying its weight visually, or a wall of text?

## Then: 2–3 specific fixes, each tagged by route (not a rewrite)
Classify every proposed fix and send it down the right path:
- **[FIGURE]** the chart/image content is wrong (wrong series, unreadable labels, bad scale)
  → change `scripts/make_figures.py` and regenerate; do not hand-edit the PNG.
- **[LAYOUT]** the slide composition is wrong (too many bullets, wrong columns, headline too small)
  → hand to `/03-content-pass <beat>` so the compliance gate (≤5 bullets / ≤12 words, anchors,
  constraints) re-runs on the change.
- **[TEXT-DEFER]** the wording itself is the problem
  → **flag as a post-rehearsal text note; do NOT rewrite inline** (Pass-2/3 FREEZE discipline —
  text revision after a visual is wired invalidates the visual work).

Examples: "[LAYOUT] Cut bullet 3, it duplicates the figure caption — via /03." ·
"[FIGURE] MAE bar labels overlap; widen xlim in make_figures.py." ·
"[TEXT-DEFER] 7b caption says 'the TSFM line' but two are revealed — note for later."

If the slide is fine, say so. Don't invent problems. This prompt edits nothing itself.
