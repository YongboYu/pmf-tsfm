# SLIDES.md — pmf-tsfm @ CAiSE 2026

## Logistics
- **Total session: 30 min** (20 min presentation + 10 min Q&A)
- Presentation budget: ~16 min content + ~2–3 min demo + ~1 min buffer
- Format: in-person, single screen, 16:9
- Slidev (Vue + Markdown). Gifs, short video clips, and Manim animations encouraged.
- Audience: ~80% process mining / IS researchers, ~20% adjacent (data mining,
  forecasting). Mixed familiarity with TSFMs — assume they know
  "foundation model" but default to LLM mentally. Differentiate explicitly.

## The one thing
If an attendee remembers ONE sentence walking out:
> "For PMF, off-the-shelf TSFMs beat trained-from-scratch baselines zero-shot —
> and fine-tuning rarely helps on PMF-scale data, so don't bother."

## Three things, if they remember more
1. PMF reduces to forecasting DF time series — sparse, drift-heavy, heterogeneous.
2. Zero-shot TSFMs win on MAE/RMSE; LoRA & full-FT add little for much more compute.
3. ER reaches parity, not better — the bottleneck has moved from forecasting
   accuracy to process-aware representation.

## What we do NOT cover
- TSFM architectural details (tokenization, masking schemes, MoE routing)
- Hyperparameter sweep results — point to paper / backup slides
- HPC orchestration / SLURM setup
- Detailed PEFT theory beyond "LoRA = adapters on attention"

## Visual identity
- **Colour set is locked** — single source of truth in `palette.json` (mirrored to the deck in
  `template/style.css` `:root`, and read by `scripts/figure_manifest.py`). Keep the two in sync.
- Primary: KU Leuven blue `#00407A` (brand/headings/TSFM group). Accent: orange `#DD8A2E`,
  reserved for "our method"/attention — never a model colour; put **ink** text on it, not white.
- Figure strategy = **hybrid**: headline slides use two groups (baselines gray `#778496` vs TSFMs
  one blue); per-family hues only on family-comparison slides.
- Family trio (CVD-verified, Machado 2009 — 5-line worst-case 54): Chronos `#1B6FB0` (azure),
  MOIRAI `#57C0AE` (teal), TimesFM `#4C3A78` (deep indigo). Decoupled from brand so Chronos is its
  own colour. Multi-line plots also carry redundant cues (dash + markers); bars carry direct labels.
- Fonts: Slidev defaults (Inter + JetBrains Mono).
- Diagrams: clean, minimal, calm. No 3D, no shadows, no clipart.
- Animations: short (≤15s), purposeful, never decorative.
- Logo: `template/public/logos/kuleuven-liris.png` (official KU Leuven · LIRIS horizontal lockup).

## Hard constraints
- Max 5 bullets per slide; max 12 words per bullet
- Every equation appears beside a concrete example
- Every results claim points to a number visible on the slide
- Reuse paper figures verbatim where possible
- "Future work" is folded into the closing "Signals" slide — no dedicated future-work slide

## Locked outline

| # | Beat | Time | Visual |
|---|---|---|---|
| 1 | PMF vs PPM (concrete examples, calibrated 7-day horizon) | 60s | Side-by-side comparison |
| 2 | PMF workflow → DF time series | 60s | Workflow diagram + DFG-evolution animation (10–15s gif/Manim) |
| 3 | Where PMF stands (3 findings from Yu 2025) | 75s | BPI2017 drift plot, ground truth + XGBoost only |
| 4 | Why TSFMs (explicitly NOT LLMs) | 60s | LLM-vs-TSFM panel + "no event logs in pretraining, to our knowledge" |
| 5 | Three questions this talk answers (plain English) | 30s | Text scaffold |
| 6 | Method: 3 families + 3 settings | 45s | Release-timeline visual (2023→2026), settings strip, "univariate throughout" |
| 7 | Results bars + drift callback | 3 min | MAE bars (4 panels) → BPI2017 callback with TSFM revealed |
| 8 | Fine-tuning + timing | 3 min 15s | 3-pt slope chart (ZS→LoRA→FT, colored by dataset) + log-scale compute bars |
| 9 | ER: bottleneck has moved | 1 min 30s | Compact ER bars + reframe text |
| 10 | Signals + artifact strip | 2 min | 3 signals (hybrid synthesis) + GitHub/demo/HPC strip |
| 11 | Demo (pre-recorded screencast) | 2–3 min | Load log → pick TSFM → render forecasted DFG |
| | Buffer | ~1 min |  |
| | **Total** | **~20 min** |  |

Full beat-by-beat content lives in `talk_design/outline.md`.

## Spoken anchor lines (rehearse cold)

### Opening (slide 1)
> "I'm going to show you something that took me a while to believe:
> the best forecaster for your process model isn't one you trained —
> and probably isn't one you should train."

### Beat 3 bridge into beat 4
> "Three findings from our prior benchmark — Yu et al. 2025 — shape every
> design choice that follows. We use univariate. We need a model that doesn't
> overfit small heterogeneous data. We need one model that handles many
> patterns without retraining per log."

### Beat 4 bridge into beat 5
> "Specialized models overfit on small heterogeneous PMF data.
> Foundation models, pretrained on millions of diverse series, are designed to not."

### Results transition (into beat 7)
> "Three families. Eight model variants. No event logs in pretraining.
> Here's what happens when you point them at DF time series."

### ER memorized rebuttal — LOAD-BEARING, REHEARSE COLD (beat 9)
> "ER parity means we're not producing better DFG structures — we're producing
> DFGs with better edge weights. For tasks where the forecast itself is the
> deliverable — capacity planning, drift detection, anomaly baselines —
> edge-weight accuracy IS the contribution. For tasks where you need a different
> process model, ER says the bottleneck is now the DFG representation,
> not the forecaster."

### Closing (beat 10)
> "Zero-shot TSFMs are the new PMF default. Four logs is not a paradigm —
> but it is a strong enough signal that the right new default is to try a
> zero-shot TSFM first. The deeper signal is that time-series forecasting
> just became cheap enough that process mining can borrow from it without
> paying the training tax. What else can we borrow?"

## Critique-derived constraints (non-negotiable)

From the hostile-PC pass. Failure to honor invites Q&A pain.

- **Slide 4** says "no event logs in pretraining, **to our knowledge**" —
  NOT "no process data." Defensible vs strong.
- **Closing signal 1** contains the "four logs is not a paradigm" hedge clause.
- **ER slide** surfaces the memorized rebuttal *in speech*, not just text.
- **Results transition** names the two baselines as "strongest from our prior benchmark" —
  pre-empts the cherry-picking objection.
- One-line definitions required for: **zero-shot**, **LoRA**, **Entropic Relevance**.

## Required backup slides (Q&A defense)

1. Full baseline ranking from Yu 2025 benchmark
2. Full RMSE table (main slide shows MAE only)
3. Hyperparameter config (XGBoost Optuna trials, LoRA rank/alpha, full-FT settings)
4. DF complexity table (7 metrics × 4 logs vs 21 standard benchmarks)
5. Sepsis DFG sample showing behavioral heterogeneity (if available)
6. Horizon-sensitivity discussion (or explicit "7-day matches prior benchmark cadence")
7. Multivariate experiment notes ("we tried, no gains, consistent with benchmark")

## Demo / artifacts (beat 11)

- **Pre-recorded screencast preferred over live demo** — Wi-Fi / projector / submission
  risk too high in a single-screen room for live audience input.
- Screencast shows: load a small event log → pick a TSFM checkpoint → run zero-shot
  inference → render forecasted DFG.
- Artifact strip on closing slide: GitHub URL · demo URL · deployment matrix
  ("runs on laptop MPS / Linux CUDA / HPC").
- Optionally add a QR code for the demo URL so audience can try it during Q&A.

## Source of authority

This file is the brief. `talk_design/outline.md` is the canonical beat-by-beat content.
If they disagree, this file wins for *constraints*; outline wins for *content*.
