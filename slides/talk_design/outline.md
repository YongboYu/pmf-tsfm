# Talk outline — pmf-tsfm @ CAiSE 2026

Locked via grilling session (2026-05-22). Canonical beat-by-beat content.
For brief / hard constraints / spoken lines, see `../SLIDES.md`.

## Audience model
- ~80% process mining / IS researchers, ~20% adjacent (data mining, forecasting)
- Assume PPM known. **PMF NOT known** (system-level forecasting is a niche)
- Assume "foundation model" known but defaults to LLM mental model — must differentiate
- Time series forecasting itself: not universally known; explain via one analogy
  ("like website traffic per day")

## Time budget
- 30-min session = 20 min presentation + 10 min Q&A
- Presentation = ~16 min content + ~2–3 min demo + ~1 min buffer

## The mental order the audience needs
1. PMF (vs PPM) and its workflow → know what's being forecasted
2. PMF reduces to DF time series forecasting → know the sub-problem
3. Current ML/DL overfits on this data → know why prior approaches fail
4. TSFMs are designed for exactly this → know why we try them
5. Three questions + answers → have a scaffold to follow

---

## Beat 1 — PMF vs PPM (60s)
**Purpose:** Establish what PMF is by contrast with PPM, with calibrated horizon.
**Audience Q answered:** "What is PMF and why should I care?"
**Transition in:** opening hook line (see SLIDES.md).
**Slide content:**
- Side-by-side comparison:
  - **PPM (case-level)**: one ongoing case → next event / remaining time / outcome.
    Example: *"Will this loan application be cancelled?"*
  - **PMF (system-level)**: a window of the log → the process model for the next window.
    Example: *"Across all loan applications next week, how often does
    'offer sent → offer cancelled' fire?"*
- Footer line: *"Same event log. Different question."*
- Horizon calibration noted: PMF here is days-to-weeks, not months.

## Beat 2 — PMF workflow → DF time series (60s)
**Purpose:** Show how PMF reduces to a forecasting problem on DF time series.
**Audience Q answered:** "How does PMF become a forecasting problem?"
**Transition in:** *"Both questions need data. PPM uses case prefixes. PMF uses something different."*
**Slide content:**
- Workflow diagram: event log → daily DF counts (one univariate series per DF edge) →
  forecast 7 daily steps → Σ over the horizon → next week's DFG
  (each week's DFG is the Σ-aggregate of daily counts, not a weekly-extracted object)
- **Embedded DFG-evolution animation** (10–15s gif/Manim): DFG at t₁, t₂, t₃ →
  forecasted DFG at t₄ with predicted edges highlighted in accent color
- One sentence: *"Each DF edge becomes a univariate time series — like website
  traffic per day, but for one activity transition in your process."*
- Calibration: *"We aggregate daily, forecast 7 days ahead."*
- **Layout note:** rendered as one **full-width woven figure** (`DfgEvolution.vue`) — the
  workflow diagram and DFG-evolution animation are a single unified picture advanced by
  `v-click` (frames t₁→t₄, climaxing on the forecast frame). Departs from the original
  two-column split; pre-authorised by PRD #60 / issue #63.

## Beat 3 — Where current PMF stands (75s)
**Purpose:** Anchor the gap that motivates TSFMs.
**Audience Q answered:** "Why don't existing methods already solve this?"
**Transition in:** *"So we have a forecasting problem. Why isn't it already solved?"*
**Slide content:**
- Title: *"Where current PMF stands (Yu et al. 2025 benchmark)"*
- Three findings:
  1. ML/DL methods beat seasonal naive only marginally
  2. **Univariate forecasting outperforms multivariate** on DF data (motivates our univariate setting)
  3. No single method wins across event logs — DF characteristics vary too much
     (2–3× higher transition, shifting, and non-Gaussianity than 21 standard benchmarks)
- Visual: **BPI2017 drift plot, ground truth + XGBoost ONLY** (no TSFM yet — preserves callback)
- Spoken bridge into beat 4 (see SLIDES.md)

## Beat 4 — Why TSFMs (60s)
**Purpose:** Justify foundation models as the next paradigm AND prevent TSFM-as-LLM confusion.
**Audience Q answered:** "Why should a generic forecaster do better than a specialized one?"
**Transition in:** beat-3 bridge (see SLIDES.md).
**Slide content:**
- LLM-vs-TSFM comparison panel:

  | LLM | TSFM |
  |---|---|
  | Text in → text out | Numbers in → numbers out |
  | Web-scale text | Millions of diverse time series |
  | Generalizes across language tasks | Generalizes across forecasting tasks |

- Three lines below the panel:
  - Same paradigm as LLMs: pretrain at scale, generalize zero-shot
  - **No event logs in pretraining, to our knowledge**
  - Designed to handle: heterogeneity, small data, no per-task retraining
- Footer (the load-bearing sentence): *"Specialized models overfit on small
  heterogeneous PMF data. Foundation models, pretrained on millions of diverse
  series, are designed to not."*

## Beat 5 — Three questions this talk answers (30s)
**Purpose:** Give the audience a scaffold for the rest of the talk.
**Audience Q answered:** "What does this talk actually deliver?"
**Transition in:** *"We have a candidate. Here's what we ask of it."*
**Slide content:** three lines, plain English (no jargon):
  1. Can an **off-the-shelf forecaster** — trained on no process data — beat the best PMF models?
  2. If yes, does **adapting it to process data** help?
  3. Does a better forecast give us a **better process model**?

## Beat 6 — Method (45s)
**Purpose:** Name the candidates with enough vocabulary to read the results.
**Audience Q answered:** "Which models? Which settings?"
**Transition in:** *"Three questions. Here's what we point at them."*
**Slide content:**
- **Release-timeline visual**: horizontal axis 2023→2026, three lanes (Chronos / MOIRAI / TimesFM),
  dots for each version, latest dots highlighted in accent color
- One line per family:
  - **Chronos** (Amazon) — encoder-decoder, tokenizes time series like language
  - **MOIRAI** (Salesforce) — encoder, masked + any-variate attention
  - **TimesFM** (Google) — decoder-only, patch-based
- Settings strip: **zero-shot** · **LoRA** (small trainable adapters on attention) · **full fine-tuning**
- Footer: *"Univariate throughout (per Yu 2025 finding)"*

## Beat 7 — Results bars + drift callback (3 min, two slides)
**Purpose:** Deliver the headline finding visually.
**Audience Q answered:** "Does Q1 hold? Do TSFMs beat baselines?"
**Transition in:** results transition spoken line (see SLIDES.md).

### 7a — Headline bar chart (~2 min)
- 4 panels (one per dataset: BPI2017, BPI2019_1, Sepsis, Hospital Billing)
- 5 horizontal bars per panel: Seasonal-Naive, XGBoost (gray); Chronos-2, MOIRAI-2.0, TimesFM-2.5 (accent color)
- **MAE only**; RMSE confirmed verbally; full RMSE table on backup
- Spoken intro: *"We compare against the two strongest baselines from our prior benchmark; full ranking on backup."*

### 7b — Drift plot callback (~1 min)
- **Same BPI2017 plot from beat 3, now with the TSFM line revealed** (same EPS as paper Fig 1)
- One spoken line: *"Same data. Same prediction window. The TSFM tracks what XGBoost missed."*
- Briefly mention the recovery dynamic: TSFM misses the initial drift but catches up
  — visual proof of online adaptation

## Beat 8 — Fine-tuning + timing (3 min 15s, single slide, two panels)
**Purpose:** Land the "fine-tuning rarely pays" finding.
**Audience Q answered:** "Does adapting (Q2) help? Is it worth the compute?"
**Transition in:** *"TSFMs win zero-shot. Natural next question: can we make them better?"*
**Slide content:**
- **LEFT panel** — 3-point slope chart (ZS → LoRA → Full-FT), one line per (model × dataset),
  colored by dataset. Annotate 1–2 notable wins and 1–2 notable failures.
- **RIGHT panel** — log-scale wall-clock compute bars per setting (7 bars:
  3 ZS + 3 LoRA + 3 Full-FT + 1 XGBoost reference)
- Slide footer: *"Marginal accuracy. ~100× compute. Skip it at PMF data scale."*

## Beat 9 — ER: bottleneck has moved (1 min 30s)
**Purpose:** Honestly land the process-aware result and pivot to future work.
**Audience Q answered:** "Does the forecasting win translate to a better process model?"
**Transition in:** *"But forecasting accuracy isn't the only thing we care about in PM."*
**Slide content:**
- **LEFT** — compact ER bar chart (4 panels, 5 bars each)
- **RIGHT** — three lines:
  1. *"TSFMs match baselines on ER, not better."*
  2. *"Sepsis is the exception: high behavioral heterogeneity defeats all models."*
  3. *"The bottleneck has moved from forecasting accuracy to process-aware representation."*
- **Speaker must deliver the memorized rebuttal in speech** (see SLIDES.md) — this is the
  single most load-bearing rehearsed line of the talk.

## Beat 10 — Signals + artifacts (2 min)
**Purpose:** Three takeaways and an invitation to use/extend the work.
**Audience Q answered:** "What do I do with this? Where do I go next?"
**Transition in:** *"So what do we walk away with?"*
**Slide content:**
- Title: **Signals**
- Three signals (hybrid synthesis):
  1. **Zero-shot TSFMs are the new PMF default** — skip fine-tuning at PMF data scale.
     *Four logs is not a paradigm — but it is a strong enough signal to make zero-shot
     the right new starting baseline.*
  2. **The bottleneck has moved from forecasting accuracy to process-aware representation** —
     DFGs are a lossy target.
  3. **PM can borrow from adjacent fields cheaply** — a process-native FM is the next
     frontier, but needs a corpus of event logs we don't yet have.
- Artifact strip: GitHub URL · demo URL · *"runs on laptop (MPS) / GPU server (CUDA) / HPC"*
- Closing spoken line (see SLIDES.md).

## Beat 11 — Demo (2–3 min)
**Purpose:** Make the work tangible. Land the practitioner pitch.
**Slide content:**
- Pre-recorded screencast (live demo not recommended)
- Sequence: open small event log → pick TSFM checkpoint → run zero-shot inference →
  render forecasted DFG side-by-side with ground truth
- End with QR code / URL so audience can try during Q&A

---

## Two or three places audience attention is most at risk

1. **Beat 6 (method, 45s)** — three family names + three settings in <1 min is dense.
   Risk: audience zones out. Mitigation: release timeline visual carries the cognitive load,
   not the speaker's voice. Names appear on the screen, not in spoken intro.
2. **Beat 9 (ER)** — the slide deliberately concedes parity. If the speaker doesn't deliver
   the memorized rebuttal with energy, the audience hears "they didn't actually win." Risk
   of the talk's contribution claim collapsing here.
3. **Beat 8 → 9 transition** — going from "fine-tuning doesn't help" (a clean negative finding)
   to "ER is parity" (a defensive concession) is the talk's lowest-energy moment. Mitigation:
   open beat 9 with the question, not the result.

## Single slide that, if cut, most weakens the talk

**Beat 4 (Why TSFMs, not LLMs).** Without it, the audience spends the entire results section
quietly wondering whether you fine-tuned GPT to forecast. Every claim about TSFM
generalization is then weaker than it should be because half the room misunderstands the
input modality. Cutting any other single slide is recoverable; cutting beat 4 is not.

## Q&A defense

See `../SLIDES.md` for required backup slides and ranked hostile Q&A questions with
rehearsed answers.
