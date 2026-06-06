# Verbatim manuscript — CAiSE 2026 talk

Spoken script for the 20-minute talk. **Prose is written in the content-revision phase** —
this file establishes the structure and the sync convention only.

## How this stays in sync with the deck

Every slide carries a stable anchor `[S-xx]` (two digits, in deck order). The anchor appears
in **both** places, so the manuscript and the slides never drift:

- **Here**, as the section heading for that slide's script: `## [S-03] From event log to DF time series`.
- **In `slides.md`**, as the first line of that slide's Slidev presenter-note comment:

  ```md
  <!--
  [S-03]
  Spoken cue / delivery note for the presenter view.
  -->
  ```

Renumber anchors only when the deck order changes; keep the `[S-xx]` ↔ slide mapping 1:1.
The short per-slide `<!-- -->` note is the **delivery cue** (transitions, timing, the one line
to land); the manuscript below is the **full verbatim**.

## Per-slide budget

20 min ≈ the v2 outline's beat budget. Keep each slide to its one key message; if a slide's
script runs long, the slide is doing too much (split it or cut).

---

## [S-01] Title
_(verbatim — content phase)_

**Opening anchor (rehearse cold):** "I'm going to show you something that took me a while to
believe: the best forecaster for your process model isn't one you trained — and probably isn't
one you should train."

## [S-02] PMF vs PPM
_(verbatim — content phase)_

## [S-03] From event log to DF time series
_(verbatim — content phase)_

---

_Add one `## [S-xx] <assertion>` section per slide as the content revision lands each slide.
Keep the order identical to `slides.md`._
