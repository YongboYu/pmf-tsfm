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
- Fonts: **Inter** (headings + body) + **JetBrains Mono** (the section-locator pill ONLY). Set via
  the `fonts:` headmatter in `slides.md`; `canvasWidth: 1280` so the px scale below renders as designed.
- Diagrams: clean, minimal, calm. No 3D, no shadows, no clipart.
- Animations: short (≤15s), purposeful, never decorative.
- Logo: `template/public/logos/kuleuven-liris.png` (official KU Leuven · LIRIS horizontal lockup).

### Slide template — Assertion-Evidence (locked)
Research-backed structure (Alley & Neeley, Penn State). Every content slide is three zones + one
chrome element, nothing else:

> **locator (navy pill) → assertion headline → accent rule + gap → evidence → page number**

- **Section locator** — generic act/section name (e.g. `Results`). Navy **pill**: `--brand` fill,
  white text, JetBrains Mono, rounded — same navy family as the headline (consistent), set apart by
  shape + inversion. Navigation only, never a slide summary. **No section-divider slides** — the
  persistent locator carries orientation.
- **Assertion headline** — ONE sentence stating the slide's claim; `--brand`. **Absorbs the old
  "takeaway"** — no separate takeaway strip or footnote.
- **Accent rule** — short `--accent` underline + gap between headline and evidence; *structural*, not
  attention. **Droppable**: delete the single `.assertion .rule` block in `style.css` (no layout impact).
- **Evidence** — figure / table / minimal text.
- **Chrome** — page number only (`$nav.currentPage / $nav.total`; never hardcode the slide count).
- **Callout** — `components/Callout.vue`, a clean accent box (ink text on `--accent`); the ONLY
  pointing accent element, so attention-orange stays meaningful.

**Guided-attention reveal (convention):** direct the eye with `<Callout v-click>` — figure and text
render at step 0, the Callout fades in on click. Default to ONE callout; use a second only when the
slide carries two distinct key messages, revealed **in sequence** (`v-click="1"`, `v-click="2"`) so
attention lands on one at a time (e.g. the ER slide: "Forecast ≫ reuse" then "TSFMs ≈ baselines").
The callout text should restate the slide's message — an aggregate stat matching the headline (MAE
slide: a plain pill, no pointer) or a pointer at the bar it describes. `@slidev/rough-notation`
(`v-mark`) stays rejected as too informal. Absolute-position over a figure by wrapping it in
`<div class="relative">` + a `<div class="absolute" style="...">`. **Text on the pill is always `--ink`,
never white** (ink-on-accent ≈ 5.4:1, passes WCAG AA; white-on-accent ≈ 2.7:1, fails) — see the palette
rule above.

Layouts in `template/layouts/`: `assertion-evidence`, `two-col-evidence`, `cover`.

### Type scale (px @ 1280 canvas → ≈ projected @ 1920)
| element | px | proj | notes |
|---|---|---|---|
| cover title | 56 | 84 | cover only |
| assertion headline | 37 | 55 | L — not XL |
| sub-head (h2) | 26 | 39 | |
| column lead | 23 | 34 | |
| **body / bullets** | **23** | **~34** | legibility-tuned default |
| locator pill (mono) | 17 | 25 | |
| caption | 14 | 21 | |
| page number | 14 | 21 | |

Line-height 1.6; page margin 52 / 92 / 56. `.dense` (18) / `.dense--xs` (16) escape hatches for tight
figure panels. Figures: matplotlib matched to Inter with graceful fallback (`scripts/make_figures.py`).
Speaker notes: per-slide Slidev `<!-- [S-xx] … -->` notes **plus** the verbatim
`talk_design/manuscript.md` (same `[S-xx]` anchors keep them in sync).

**Math (KaTeX):** equations render in `--ink` at the body scale (inline `$…$` inherits 23px); display
`$$…$$` is left-aligned with the evidence column (`.ae-body .katex-display` in `style.css` overrides
KaTeX's centre default). Every equation pairs with a worked example — the ER slide (`ER ≈ bits/trace`
beside Truth 1.86 → MOIRAI-2.0 2.52 → Training 5.83) is the reference exemplar. KaTeX ships with
Slidev; no plugin install needed.

## Hard constraints
- Max 5 bullets per slide; max 12 words per bullet
- Every equation appears beside a concrete example
- Every results claim points to a number visible on the slide
- Reuse paper figures verbatim where possible
- "Future work" is folded into the closing "Signals" slide — no dedicated future-work slide

## Locked outline

**v3 (2026-06-06)** — 19 content slides + backups, for the 20-min talk slot. Reopened from the
11-beat v2 via a `grill-with-docs` pass against `talk_design/revision_comments.md`. Timing is
indicative (~17 min content + ~2–3 min demo + buffer). One **key message** per slide.

| # | Slide | Time | Visual |
|---|---|---|---|
| 1 | Title (+ KU Leuven logo) | — | Cover |
| 2 | **[NEW]** Process discovery → static model, but processes drift | 60s | Two-period DFG snapshots (first vs last week) |
| 3 | PMF vs PPM (same loan-app; PPM = outcome + remaining time) | 60s | Visual side-by-side (case-trace vs system-DFG) |
| 4 | PMF = forecasting DF time series | 60s | Woven pipeline — DF series **stack** (middle) → DFG (right) |
| 5 | DF series are hard | 45s | Two truth-only series: drift + sparsity/intermittency |
| 6 | **[NEW]** Complexity + small data | 60s | 7-metric complexity (highlight transition/shifting/non-Gaussianity) + stats table; qualitative "vs 21 benchmarks (Li 2025)" annotation |
| 7 | What is a TSFM (vs LLM) | 45s | LLM↔TSFM panel; define zero-shot + fine-tuning |
| 8 | Why TSFMs for PMF | 45s | Overfitting → pretrained bet; "no event logs in pretraining, to our knowledge" |
| 9 | Three questions | 30s | Scaffold: zero-shot / fine-tuning / better forecasted DFG |
| 10 | The candidates | 45s | Release timeline (3 families) + settings strip (ZS · LoRA · full-FT) |
| 11 | **[NEW]** Experimental setup | 45s | Expanding-window schematic; stride = 1, 7-day horizon |
| 12 | **[KEY]** Zero-shot TSFMs beat both baselines on every log | 2 min | Grouped **2-colour** MAE bars + %Δ callout (pumps in on click) |
| 13 | Drift + sparsity, revealed | 75s | The two S5 plots, now with TSFM line |
| 14 | Fine-tuning isn't necessary | 90s | Slope ZS→LoRA→full-FT (**drop Chronos-2**); timing in text |
| 15 | ER: two findings | 90s | ER bars + two sequential callouts (forecast ≫ no-forecast; TSFMs ≈ baselines) |
| 16 | **[NEW]** Discussion: bottleneck moved + limitations/future | 60s | Process-aware representation; sharp limits |
| 17 | Takeaways | 75s | Practitioner pitch + scientific signals ("four logs is not a paradigm") |
| 18 | Artifacts (+ demo) | 2–3 min | GitHub · demo URL · MPS/CUDA/HPC matrix · recording (placeholder) · QR |
| 19 | Thank you | — | End |
| B1–B7 | Backups | — | Baseline ranking · RMSE · hyperparams · complexity · Sepsis · horizon · multivariate |

Full beat-by-beat content lives in `talk_design/outline.md` (v3).

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
