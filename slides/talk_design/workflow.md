# Slide iteration workflow

How to refine the locked skeleton (`slides/template/slides.md`) into a finished deck.

## When to use which prompt

| Prompt | When |
|---|---|
| `/01-outline` | The outline itself needs to change — new beats, retired beats, reordering, time-budget shifts. Don't use after locking unless something major changes (paper revision, time-slot change). |
| `/02-critique` | After major outline changes, or before final rehearsal. Hostile-PC pass against the current outline + slides. |
| `/03-content-pass` | **Default loop.** Refine ONE beat at a time. Edits `slides.md` in place; reports compliance. |

## The per-beat loop

```
   pick a beat (1–11 main; or backup-<name>)
                  │
                  ▼
   /03-content-pass <beat-id> [: optional feedback]
                  │
                  ▼
   read the compliance report
     • all PASS / N/A  →  move to next beat
     • any FAIL / unclear  →  re-invoke same beat
                              with ": <feedback>"
                  │
                  ▼
              next beat
```

Reviewing each beat takes seconds when the compliance summary is tight. Don't
batch-review — feedback loses precision once two beats have shifted.

## Beat identifier conventions

- **Main beats**: `1` through `11` (per `outline.md`); or beat title
  (case-insensitive, e.g. `Why TSFMs`)
- **Sub-beats** when a beat spans 2 slides (beat 7 = results + drift callback):
  `7a`, `7b`
- **Backup slides**: prefix `backup-` (e.g. `backup-rmse`, `backup-multivariate`,
  `backup-sepsis`)
- **Slide range fallback** when nothing else disambiguates: `slides 12-13`

## Figure naming convention

All figures used by the deck live at
`slides/template/public/figures/<semantic-kebab-name>.{png,svg,webp,gif,mp4}`.
Slidev serves `public/` at root, so reference them as
`<img src="/figures/<name>" ... />`.

| Beat | Expected file(s) |
|---|---|
| 1 | (none — text-only) |
| 2 | `dfg-evolution.gif` (or `.mp4`) |
| 3 | `bpi2017-drift-xgb-only.png` |
| 4 | (none — text + comparison panel) |
| 5 | (none — text-only) |
| 6 | `tsfm-release-timeline.svg` |
| 7a | `results-mae-bars.png` |
| 7b | `bpi2017-drift-with-tsfm.png` |
| 8 | `ft-slope.png`, `ft-compute-bars.png` |
| 9 | `er-bars.png` |
| 10 | (none — text + artifact strip) |
| 11 | `demo-screencast.mp4` |
| backup | `baseline-ranking-full.png`, `rmse-full.png`, `df-complexity-radar.png`, `sepsis-variants.png` |

When `/03-content-pass` runs, it Bash-checks the expected file for the current beat:
- file present → replace placeholder `<div>` with `<img>`
- file absent → keep placeholder; report `PLACEHOLDER kept (<filename> not found)`

## What's OUT of scope for `/03-content-pass`

- **Generating figures** (matplotlib, Manim, screencasts) — separate concern; produce
  out of band, drop into `public/figures/`
- **Converting `slides/figures/*.eps` to PNG/SVG** — use `pdftocairo` or Inkscape
  headless, then move into `slides/template/public/figures/`
- **Theme styling** (CSS overrides, custom fonts, brand colors beyond the placeholder
  accent) — fold in once text content stabilizes
- **Vue component creation** — defer until a real reuse pattern emerges across slides
- **Structural outline rewrites** — use `/01-outline` instead

## Recommended iteration order

Iterate text first across all beats, then produce visuals as a separate pass.
Don't interleave — chart regeneration is ~100× the cost of bullet revision, and
the smart-detect mechanism in `/03-content-pass` was built for exactly this two-pass pattern.

### Pass 1 — Text (all 11 main beats + 7 backups)

For each beat, invoke `/03-content-pass <beat-id>` until compliance is clean.
Placeholders stay in place; refine their descriptive copy if it's unclear.
End-state: a rehearsable deck with all text settled.

### Rehearse once with placeholders

Between Pass 1 and Pass 2. The rehearsal exposes drag, weak transitions, and which
visuals are load-bearing — informs Pass 2 priorities and may demote some figures
from "must-build" to "fine as placeholder."

### Pass 2 — Visuals, in order of ascending risk

| Order | Type | Effort | Risk if redone |
|---|---|---|---|
| 1 | Data charts via `scripts/make_figures.py` (MAE bars, drift pair, FT slope, ER, RMSE, radar) | Low (matplotlib) | Low |
| 2 | Release timeline SVG (beat 6) | Medium (bespoke design) | Medium |
| 3 | DFG-evolution animation (beat 2) | High (Manim / matplotlib + ffmpeg) | High |
| 4 | Demo screencast (beat 11) | High (record + edit) | Very high |

**Data charts are produced by one committed script — `slides/scripts/make_figures.py`,
driven by `slides/scripts/figure_manifest.py`** (Step 0: pins every figure to its
authoritative source + the model→label map). After producing each figure it lands in
`slides/template/public/figures/`; wire with `/03-content-pass <beat>` (smart-detect
auto-wires the `<img>`).

Data-source notes (learned the hard way — see the visual-pass plan):
- **Authoritative results live at `/Users/yongboyu/Documents/PhD/pmf-tsfm/results/`** (sibling of
  `code/`, NOT under it). The tree is fragmented with duplicate/renamed copies — the manifest resolves them.
- **The drift pair (beats 3 / 7b) is regenerated from `.npy`, not EPS-converted** — beat 3 must
  omit the TSFM lines to preserve the reveal, which a baked EPS can't do. Both plots share axes so
  the callback overlays exactly. (Paper Fig 1 reveals *two* TSFM lines: Chronos-2 + MOIRAI-2.0.)
- **ER bars + DF-complexity radar are transcribed from paper Tables 7 / 3** (no clean machine source;
  ER must match the paper for the beat-9 Q&A defense).
- **The compute panel (beat 8 right) is deferred** — no wall-clock training time exists in the paper
  or `results/`; reconstruct from `wandb/`/`hydra/` as a separate data task.
- `results/` baselines + ER differ slightly from the camera-ready (re-runs); TSFM MAE matches exactly.

### Exception: visual-IS-the-content beats

**Beat 7b (drift callback)** and **beat 11 (demo)** carry meaning in the visual itself
with minimal text. For these, interleaving is fine — text is too thin for separation
to buy anything.

### Pass 3 — End-to-end rehearsal

Full deck with all figures wired. Polish remaining rough edges. Time the talk.

### Discipline required

Once Pass 2 starts, FREEZE the text. Text revision after a visual is wired
invalidates the visual work. If a Pass-2 visual exposes a text problem, note it
but DON'T fix it inline — finish the visual, then do a focused text-revision pass
after Pass 3 rehearsal exposes the real impact.

## Definition of "done" for the deck

- All 11 main beats: compliance reports all PASS or explicit N/A
- All backup slides have content (compliance-relaxed)
- All figures exist at expected paths — no `PLACEHOLDER` remaining
- Spoken anchor lines rehearsed cold — especially the ER memorized rebuttal
  (see `SLIDES.md` → "Spoken anchor lines")
- One end-to-end run via `cd slides/template && pnpm dev` to confirm rendering
- One PDF export via `pnpm export` as a final artifact
