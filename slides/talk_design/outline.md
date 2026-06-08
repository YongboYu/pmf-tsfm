# Talk outline — pmf-tsfm @ CAiSE 2026 (v3, 20-slide)

**v3 (2026-06-06)** — reopened from v2 via a `grill-with-docs` session against
`revision_comments.md`. Expanded from 11 beats to **20 content slides + backups** for the official
**20-min talk + 10-min Q&A** slot. Canonical beat-by-beat content. For brief / hard constraints /
spoken lines, see `../SLIDES.md`. Supersedes the v1 11-beat outline.

**v3.1 (2026-06-07)** — during the JIT pass, split the old motivation off the truth-only S5: added
**S6 "Strongest prior method falls short"** (XGBoost-vs-truth + the ML≈naive benchmark finding) so
the "existing methods fail" beat lands *before* the TSFM bet; subsequent content beats shifted +1
(old S6–S19 → S7–S20).

> **Golden standard (from the revision brief):** decide the **one key message** for a slide first,
> *then* pick the single best visualization/material to convey it. Don't compress multiple messages
> onto one slide — it costs explanation time and overwhelms the audience. Prefer
> diagrams/animation/guided-attention over dry text throughout.

## Audience model
- ~80% process mining / IS researchers (CAiSE), ~20% adjacent (data mining, forecasting).
- Assume PPM known; **PMF NOT known**. Process discovery / DFGs **are** known — but the talk now
  opens from discovery because that's the audience's shared ground.
- "Foundation model" known but defaults to an LLM mental model — must differentiate explicitly.
- Time-series forecasting not universally known; explain via one analogy ("like website traffic
  per day").

## Time budget
- 30-min session = **20 min presentation + 10 min Q&A**.
- Presentation ≈ 16–17 min content + ~2–3 min demo + ~1 min buffer.

## Terminology discipline (revision brief)
Use established terms verbatim; introduce them briefly once, then rely on them.
- **zero-shot** (NOT "off-the-shelf"), **fine-tuning** (NOT "adapting"), **LoRA**, **full
  fine-tuning**, **DFG / DF**, **Entropic Relevance (ER)**, **expanding window / stride = 1**.

---

## Slide map (one key message each)

| # | Slide | New? | Key message |
|---|---|---|---|
| 1 | Title (+ KU Leuven logo) | | — |
| 2 | Process discovery → a static model, but processes drift | NEW | Discovery gives one static model; the real process changes over time |
| 3 | PMF vs PPM | revised | System-level forecast vs case-level prediction |
| 4 | PMF = forecasting DF time series | reorder | The pipeline: log → DF daily series → forecast → DFG |
| 5 | DF series are hard | revised | Three challenges: drift, intermittency, heterogeneity |
| 6 | Trained-from-scratch ML/DL don't win | NEW | XGBoost (top prior ML) misses drift + overfits intermittency |
| 7 | Complexity + small data | NEW | Why from-scratch ML/DL overfits here |
| 8 | What is a TSFM (vs LLM) | split | A foundation model for numbers — **not** an LLM |
| 9 | Why TSFMs for PMF | split | Pretrained on millions of series → resists overfitting on small/complex data |
| 10 | Three questions | reword | The scaffold for the results |
| 11 | The candidates | revised | We tried a lot: 3 families × 3 settings |
| 12 | Experimental setup | NEW | The evaluation is rigorous and fair |
| 13 | Zero-shot TSFMs beat both baselines on every log | **KEY** | The central win |
| 14 | Drift + sparsity, revealed | revised | TSFMs track what XGBoost missed; honest on sparsity |
| 15 | Fine-tuning isn't necessary | revised | Marginal/negative gain for big extra training cost |
| 16 | ER: two findings | revised | Forecast ≫ no-forecast; TSFMs ≈ baselines on process quality |
| 17 | Discussion / limitations + future work | NEW | The bottleneck has moved to process-aware representation |
| 18 | Takeaways | revised | Practitioner pitch + scientific signals |
| 19 | Artifacts | revised | Reproducible: code · demo · HPC matrix · recording |
| 20 | Thank you | | — |
| B1–B7 | Backups | revised | Q&A defense |

---

## Beat-by-beat

### Slide 1 — Title (cover)
- KU Leuven · LIRIS lockup logo; authors; arXiv:2512.07624.
- Opening anchor line (rehearse cold, in `SLIDES.md`): *"…the best forecaster for your process
  model isn't one you trained — and probably isn't one you should train."*

### Slide 2 — Process discovery → static model, but processes drift  [NEW]
**Key message:** process discovery extracts **one static** model from an event log, but the real
process **changes over time** — that gap is what PMF targets.
**Audience Q:** "Why isn't a discovered model enough?"
**Content:** discovery = event log (data) → process model (DFG). Then show **two DFGs from two
periods** (e.g. first vs last week) side-by-side — visibly different edges/weights. Land:
*discovery gives a snapshot; PMF aims to capture and forecast the change.*
**Visual:** two compact DFG snapshots + a "→ over time" arrow. Reuse/adapt `DfgSnapshot.vue`
(Codex worktree) or a static SVG; re-home onto the locked palette.
**Transition out:** "If the model drifts, the natural question is how — and that splits into two
very different prediction problems."

### Slide 3 — PMF vs PPM  (revised)
**Key message:** PPM predicts the future of **one case**; PMF forecasts the **whole system's**
process model for the next window.
**Audience Q:** "What is PMF, vs the PPM I already know?"
**Content:** side-by-side, **same loan-application context**. PPM = next event · **remaining time**
· outcome (*"Will this loan be cancelled? How long left?"*). PMF = how often each transition fires
next week (*"How often does offer-sent → cancelled fire across all cases?"*). Replace dry text with
a visual contrast (case-trace vs system-DFG).
**Constraint:** keep both questions in the same business scenario.
**Transition out:** "Both need data, but PMF needs something different — let me show you the
pipeline."

### Slide 4 — PMF = forecasting DF time series  (reordered pipeline)
**Key message:** PMF reduces to forecasting **DF time series**, reassembled into a DFG.
**Audience Q:** "How does PMF become a forecasting problem?"
**Content:** introduce **DFG** and **DF** abbreviations. Pipeline **left → right**: event log →
**(middle) a stack of daily DF time series** (we forecast *all* arcs, not one) → forecast →
**(right) the reassembled/forecasted DFG**. One line: *"each DF edge is a univariate series — like
website traffic per day, for one activity transition."* Keep the stride/window detail OFF this
slide (it lives on S12); just say "aggregate daily, forecast ahead".
**Visual:** the woven `DfgEvolution.vue` figure, reordered (TS-stack middle, DFGs right). Reconcile
wording with S12 so the two slides don't confuse.
**Transition out:** "So it's a forecasting problem. Why isn't it already solved?"

### Slide 5 — DF series are hard  (revised, truth-only, THREE challenges)
**Key message:** DF time series are **genuinely hard** on three fronts — **① drift, ② intermittency,
③ heterogeneity** (within and across logs) — not white noise, not trivially forecastable.
**Audience Q:** "What makes these series difficult?"
**Content:** **two ground-truth-only line panels** — ① drift = BPI2017 `Sent → Cancelled` (the S14
callback seed, drops ~46→~3); ② intermittent = **BPI2019-1 `Cancel Invoice Receipt → Record Invoice
Receipt`** (~80% zeros, spikes to 29). Bottom line states ③ heterogeneity: the two panels are
different logs with different shapes, and DF relations differ within one log. No model lines yet.
**Visual:** two truth-only **line** plots, consistent layout; **x = "Day"** (stride 1 ⇒ consecutive
windows = consecutive days), **y = "Value"** (7-day-ahead forecast's last day — not a 7-day sum).
**Transition out:** "These are hard. So — do the methods we already have handle them?"

### Slide 6 — Trained-from-scratch ML/DL don't win  [NEW]
**Key message:** **from-scratch ML/DL don't win on DF series in general** — even the prior benchmark's
top ML (tuned XGBoost) misses the drift AND overfits the intermittent series.
**Audience Q:** "Do the methods we already have handle these patterns?"
**Content:** the **same two series as S5**, now with the **XGBoost** line revealed (two panels, same
layout): drift = XGBoost stays high, misses the drop; intermittent = XGBoost **overfits**, hallucinating
40–71 where the truth is mostly zero. Bottom = the reason: small + complex + heterogeneous data →
**overfitting** (callback to S5's challenges); **XGBoost is the benchmark's recommended top ML
(Yu 2025) and still loses** → it becomes our ML baseline. Do **not** reveal the TSFM yet (S14).
One exhibit, three acts: hard (S5) → ML/DL overfit (S6) → FM wins (S14). Honest nuance (speech): the
TSFM wins by staying controlled, not by capturing the rare spikes.
**Visual:** two truth + XGBoost **line** panels (`s6-drift-xgb.png`, `s6-intermittent-xgb.png`); the
intermittent panel's y-axis grows past the truth range to fit XGBoost's overshoot (the overfit visual).
**Transition out:** "It's not bad luck — these series are statistically off the charts, and the logs
are tiny."

### Slide 7 — Complexity + small data  [NEW]
**Key message:** DF series are **statistically harder** than typical benchmarks **and** the logs are
**small** — so from-scratch ML/DL overfits.
**Audience Q:** "Why do trained-from-scratch models struggle here?"
**Content:** (a) the 7 complexity metrics (Table 3 / `stats_df.tex`) with **plain-language
explanations** — highlight **transition, shifting, non-Gaussianity** (the three the paper says are
higher than the 21 public benchmarks). State the comparison **qualitatively** as a labelled
annotation: *"higher transition / shifting / non-Gaussianity than the 21 public benchmarks
(Li et al. 2025)"* — **no fabricated benchmark bars** (those values aren't in our repo). (b)
small-data stats (`stats_log.tex`: variants/cases/events/DFs/days) — emphasize small + heterogeneous.
**Framing:** this is *proof* of the S3-era benchmark findings AND the motivation for S8–S9:
no-training + pretrained-at-scale could fix overfitting on small/complex data.
**Visual:** complexity chart (highlighted metrics) + a compact stats table. Calm, scientific.
**Transition out:** "Small, complex, heterogeneous data is exactly where a model you *don't* train
might win."

### Slide 8 — What is a TSFM (vs LLM)  (split from old S5)
**Key message:** a TSFM is a **foundation model for numbers** — same idea as an LLM, but we are
**not** using an LLM to forecast.
**Audience Q:** "What even is a TSFM?"
**Content:** LLM ↔ TSFM panel (text→text vs numbers→numbers; web text vs millions of diverse
series; generalizes across language vs forecasting tasks). Define **zero-shot** and **fine-tuning**
briefly (one line each) here so later slides can rely on them.
**Constraint:** one-line definitions for zero-shot (and later LoRA/ER).
**Transition out:** "Same recipe as LLMs. So why would that help *our* problem?"

### Slide 9 — Why TSFMs for PMF  (split from old S5)
**Key message:** pretrained on millions of diverse series, TSFMs are **designed not to overfit** —
the exact failure mode of from-scratch models on small/complex DF data (callback to S7).
**Audience Q:** "Why should a generic forecaster beat a specialized one?"
**Content:** the bet, tied explicitly back to S7's overfitting. Load-bearing line (deliver aloud):
*"Specialized models overfit on small heterogeneous PMF data. Foundation models, pretrained on
millions of diverse series, are designed to not."* Critique constraint: **"no event logs in
pretraining, to our knowledge."**
**Transition out:** "We have a candidate. Here's what we ask of it."

### Slide 10 — Three questions  (reworded, standard terminology)
**Key message:** the talk answers three precise questions.
1. Can **zero-shot** TSFMs give **better DF time-series forecasts** than the strongest **PMF
   baselines**? *(forecasting only — not process-model quality)*
2. Does **fine-tuning** improve the results?
3. Does a **better forecast** give us a **better forecasted process model**?
**Note:** Q1 is about forecasting accuracy vs PMF baselines (don't claim process-model quality
here — that's Q3 / ER). Maps to S13 (Q1), S15 (Q2), S16 (Q3).
**Transition out:** "Three questions. Here's what we point at them."

### Slide 11 — The candidates  (revised)
**Key message:** we evaluated a lot — **3 families × 3 settings**.
**Audience Q:** "Which models? Which settings?"
**Content:** release-timeline visual (Chronos / MOIRAI / TimesFM lanes, 2023→2026; latest
highlighted). One line per family (vendor + architecture). Settings strip: **zero-shot · LoRA ·
full fine-tuning**. Footer: "univariate throughout (per Yu 2025)." Names on screen, not in voice.
**Visual:** replace the `[PLACEHOLDER]` timeline with a real diagram (Vue/SVG or Python).

### Slide 12 — Experimental setup  [NEW]
**Key message:** the evaluation is **rigorous and fair** — here's the protocol.
**Audience Q:** "How exactly did you evaluate?"
**Content (brief, no clutter):** **expanding window, stride = 1 day**, **7-day horizon**, daily
aggregation; two strongest baselines from the prior benchmark (seasonal-naive, tuned XGBoost);
single H100, univariate inference. Frame as *assumption → design choice*. This is the slide that
"owns" the window/stride detail kept off S4.
**Visual:** a small expanding-window schematic + a compact setup strip.

### Slide 13 — Zero-shot TSFMs beat both baselines on every log  [KEY]
**Key message (headline):** **zero-shot TSFMs beat both baselines on every log.**
**Audience Q:** "Does Q1 hold?"
**Content:** the central result. **Group bars into TWO colours — baselines vs TSFMs** (NOT one
colour per TSFM; we are not comparing the 3 TSFMs to each other here). The **callout carries the
relative %Δ** (mean of the 3 latest TSFMs vs the best baseline per log, averaged across logs) and
**pumps in on click** (like the ER slide). Guided-attention arrow optional.
**Provenance:** MAE bars from **paper Table 4** (ADR-0006). Recompute the %Δ if any number changes.
**Constraint:** open with "two strongest baselines from our prior benchmark; full ranking on backup."
**Transition out:** "Same data, same window — here's what that win looks like."

### Slide 14 — Drift + sparsity, revealed  (revised)
**Key message:** on real **drift**, TSFMs track what XGBoost missed; on **sparsity**, be honest —
the pattern is just hard / low-signal.
**Audience Q:** "What does the win look like on real patterns?"
**Content:** the **two S5 plots, now with the TSFM line revealed** (callback). Drift = clear win
(TSFM misses initial drop, then adapts online, no retraining). Sparsity = a more modest/honest
message (TSFM hits the boundary / little signal). Retitle beyond "drift".
**Transition out:** "TSFMs win zero-shot. Can we make them better?"

### Slide 15 — Fine-tuning isn't necessary  (revised)
**Key message:** fine-tuning gives **marginal — sometimes negative** — gains for **large extra
training cost**: skip it at PMF scale.
**Audience Q:** "Does Q2 (fine-tuning) help? Is it worth it?"
**Content:** accuracy slope ZS→LoRA→Full-FT (**drop Chronos-2** — it has no LoRA). Most lines flat
or worse. **No compute bar chart** (we can't fairly time baselines); instead state the
**train-vs-inference cost in text** (zero-shot = inference only; LoRA/Full-FT add training; give an
approximate train:inference ratio). Footer: "Marginal accuracy. Large extra cost. Skip it at PMF
data scale."
**Provenance:** slope from paper Table 2 (`results_2.tex`).

### Slide 16 — ER: two findings  (revised — bottleneck MOVED OUT)
**Key message (two, sequential callouts):**
1. **Forecast ≫ no-forecast** — reusing the historical (Training) model is far worse than any
   forecast → PMF has value (callback to discovery's static model).
2. **TSFMs ≈ baselines on ER** — the 5 forecasts are close; better forecasting accuracy did **not**
   yield a better process model (answers Q3).
**Audience Q:** "Does the forecasting win translate to a better process model?"
**Content:** ER bars (lower = better) + the worked ER example (Truth → forecasts → Training).
**Two `<Callout v-click>` revealed in sequence.** Define **Entropic Relevance** in one line.
**Colour fix:** the shared message must NOT be in a single TSFM's colour (Chronos blue) — it
applies to all TSFMs; use neutral/ink.
**The "bottleneck has moved" line moves to S17** (do not put it here).
**Speech:** deliver the memorized ER rebuttal aloud (see `SLIDES.md`) — load-bearing.

### Slide 17 — Discussion / limitations + future work  [NEW]
**Key message:** the **bottleneck has moved from forecasting accuracy to process-aware
representation** — DFGs capture only the workflow aspect; richer relations are missing.
**Audience Q:** "So what's the real open problem?"
**Content:** the bottleneck framing (separated from S16) + sharp limitations / future work
(richer-than-DFG representation; drift-aware adaptive forecasting; more/bigger event-log corpora;
resources/decisions dimensions). Keep points sharp; can reference the paper.
**Note:** "future work" folds in here (no dedicated future-work slide, per `SLIDES.md`).

### Slide 18 — Takeaways  (revised)
**Key message:** practical recommendation + scientific signals.
**Content:**
- **Practitioner (condense to one point):** these TSFMs are **small vs LLMs**, deploy locally,
  infer fast on cheap GPUs / any laptop, little compute; fine-tuning doable but rarely needed.
- **Scientific signals:** zero-shot = the new PMF default; fine-tuning marginal at this data scale;
  the bottleneck is process-aware representation, not accuracy; richer representation is next.
**Constraint:** signal 1 contains the **"four logs is not a paradigm"** hedge.
**Closing anchor line** (rehearse cold; in `SLIDES.md`): "…What else can we borrow?"
**Note:** code/artifact links **move to S19** (not here).

### Slide 19 — Artifacts  (revised)
**Key message:** the work is **reproducible and usable**.
**Content:** GitHub URL · demo URL · deployment matrix ("runs on laptop MPS / Linux CUDA / HPC") ·
**demo recording (placeholder)** · optional QR for the demo.
**Note:** the demo screencast stays a `[PLACEHOLDER]` until recorded.

### Slide 20 — Thank you
- Contact + repo; Q&A.

---

## Backups (Q&A defense)
B1 Full baseline ranking (Yu 2025) — defends the 2-baseline choice. *(fill `[PLACEHOLDER]`)*
B2 Full RMSE table — results-sourced (no paper RMSE table); story unchanged.
B3 Hyperparameter config — XGBoost Optuna; LoRA r=2 α=4 Q/K/V/O, patch 16, batch 32, LR 1e-4,
   3 epochs; full-FT recipes.
B4 DF complexity (full 7-metric profile, Table 3).
B5 Sepsis — why it's hard (variant tail / sparse DF). *(fill `[PLACEHOLDER]`)*
B6 Horizon sensitivity — 7-day matches prior benchmark cadence; longer horizons = future work.
B7 Multivariate experiments — tried, no consistent gains, consistent with Yu 2025.

---

## Attention-risk points (carried from v1/v2)
1. **S11 (candidates)** — dense; let the timeline carry the load, not the voice.
2. **S16 (ER)** — concedes parity; the memorized rebuttal must be delivered with energy.
3. **S15 → S16 transition** — "fine-tuning doesn't help" (clean negative) → "ER parity" (a
   concession) is the lowest-energy moment; open S16 with the question, not the result.

## Single slide that, if cut, most weakens the talk
**S8 (What is a TSFM, vs LLM).** Without it, half the room spends the results section wondering
whether you fine-tuned GPT. Cutting it is not recoverable.
