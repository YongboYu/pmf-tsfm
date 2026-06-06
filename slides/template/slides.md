---
theme: default
title: Time Series Foundation Models for Process Model Forecasting
info: |
  CAiSE 2026 presentation — pmf-tsfm
  Yongbo Yu, Jari Peeperkorn, Johannes De Smedt, Jochen De Weerdt
  KU Leuven · LIRIS
transition: slide-left
duration: 20min
mdc: true
canvasWidth: 1280
fonts:
  sans: Inter
  mono: JetBrains Mono
  weights: '400,500,600,700'
layout: cover
eyebrow: CAiSE 2026
logo: /logos/kuleuven-liris.png
venue: |
  arXiv:2512.07624
  10 June · Verona, Italy
---

# Time Series Foundation Models<br/>for Process Model Forecasting

<div class="cover-bar"></div>

<div class="cover-authors">Yongbo Yu · Jari Peeperkorn · Johannes De Smedt · Jochen De Weerdt</div>

<style>
/* eyebrow lives in cover.vue (child component): a plain .cover-eyebrow selector won't reach it —
   pierce with :deep(). Defeats the global text-transform: uppercase so "CAiSE" keeps its lowercase i. */
:deep(.cover-eyebrow) { text-transform: none !important; }
/* nudge the kicker+title+author block up (cover layout is centred by default) */
.ae-cover { justify-content: flex-start; padding-top: 175px; }
</style>

<!--
[S-01]
OPENING ANCHOR LINE (rehearse cold):
"I'm going to show you something that took me a while to believe:
the best forecaster for your process model isn't one you trained —
and probably isn't one you should train."

Pause. Let it land. Then advance.
-->

---
layout: assertion-evidence
locator: Process mining today
assertion: Process discovery gives one static model, but processes drift
---

<div class="grid items-stretch mt-4" style="grid-template-columns: 30% 70%; gap: 20px; height: 380px">

  <!-- LEFT 30% — event log (data): two slices, row-aligned with the two DFGs they feed -->
  <div style="display: grid; grid-template-rows: 1fr 1fr; gap: 14px; min-height: 0; position: relative">
    <div class="s2log-ell" style="position: absolute; left: 48%; top: 50%; transform: translate(-50%, -50%)">⋮</div>
    <div style="display: flex; flex-direction: column; justify-content: center; min-height: 0">
      <div class="s2log-title">Event log (data)</div>
      <table class="s2log">
        <colgroup><col style="width: 30%" /><col style="width: 36%" /><col style="width: 34%" /></colgroup>
        <thead><tr><th>Case</th><th>Activity</th><th>Timestamp</th></tr></thead>
        <tbody>
          <tr><td>App 6528</td><td>Create Offer</td><td>Oct 02 09:12</td></tr>
          <tr><td>App 6528</td><td>Created</td><td>Oct 02 09:13</td></tr>
          <tr><td>App 6531</td><td>Sent</td><td>Oct 02 10:05</td></tr>
        </tbody>
      </table>
    </div>
    <div style="display: flex; flex-direction: column; justify-content: center; min-height: 0">
      <table class="s2log">
        <colgroup><col style="width: 30%" /><col style="width: 36%" /><col style="width: 34%" /></colgroup>
        <tbody>
          <tr><td>App 9043</td><td>Accepted</td><td>Oct 16 14:20</td></tr>
          <tr><td>App 9043</td><td>Cancelled</td><td>Oct 16 15:02</td></tr>
          <tr><td>App 9047</td><td>Sent</td><td>Oct 16 16:48</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- RIGHT 70% — two stacked weekly DFGs, each fed by a slice of the log -->
  <div style="display: grid; grid-template-columns: 44px 1fr; grid-template-rows: 1fr 1fr; gap: 14px; min-height: 0">
    <div class="s2conn">→</div>
    <div class="border border-gray-200 rounded-lg px-3 py-1 relative" style="min-height: 0">
      <DfgSnapshot :frame="0" title="early week's DFG" hero-edge="accepted__end" :revealed="$clicks > 0" />
      <div v-click="1" style="position: absolute; left: 81%; top: 72%; white-space: nowrap">
        <Callout dir="up">465</Callout>
      </div>
    </div>
    <div class="s2conn">→</div>
    <div class="border border-gray-200 rounded-lg px-3 py-1 relative" style="min-height: 0">
      <DfgSnapshot :frame="2" title="later week's DFG" hero-edge="accepted__end" :revealed="$clicks > 0" />
      <div v-click="1" style="position: absolute; left: 81%; top: 72%; white-space: nowrap">
        <Callout dir="up">306 (−34%)</Callout>
      </div>
    </div>
  </div>

</div>

<div class="mt-6" style="max-width: 1180px; margin-left: auto; margin-right: auto">
  <p style="font-size: 20px; color: #334155; line-height: 1.5; margin: 0 0 12px; white-space: nowrap">
    A <strong style="color: var(--brand)">Directly-Follows Graph (DFG)</strong>: nodes are activities, arrows are directly-follows relations with counts.
  </p>
  <p style="font-size: 20px; color: #334155; line-height: 1.5; margin: 0">
    <strong style="color: var(--brand)">Process discovery</strong> returns one static snapshot — <strong style="color: var(--brand)">Process Model Forecasting (PMF)</strong> predicts the next.
  </p>
</div>

<style>
.s2log { width: 100%; border-collapse: collapse; font-size: 16px; color: #0f172a; table-layout: fixed }
.s2log th { text-align: left; font-weight: 700; color: #64748b; border-bottom: 1.5px solid #cbd5e1; padding: 4px 6px; font-size: 13px; text-transform: uppercase; letter-spacing: .02em }
.s2log td { padding: 5px 6px; border-bottom: 1px solid #eef2f6; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; background: #fbedd8 }
.s2log-title { font-size: 16px; font-weight: 700; color: var(--brand); text-align: center; margin-bottom: 6px }
.s2log-ell { color: #475569; text-align: center; font-size: 32px; line-height: 0.6; font-weight: 800 }
.s2conn { display: flex; align-items: center; justify-content: center; font-size: 2rem; line-height: 1; color: var(--accent); font-weight: 800 }
</style>

<!--
[S-02]
Q: Why isn't a discovered model enough?

Process discovery extracts a process model from event-log data — here a Directly-Follows
Graph (DFG): nodes are activities, arrows count directly-follows relations.
Same BPI2017 offer process, two periods: same representation, but the counts move
(Accepted → End falls 465 → 306). The process drifted. PMF starts here — learn those
temporal changes and forecast the next DFG.

Transition out: "If the model drifts, the natural question is HOW — and that splits into
two very different prediction problems." → S3 (PMF vs PPM)
-->

---
layout: two-col-evidence
locator: Predictive process mining
assertion: PPM forecasts one case — PMF the whole process model
---

::left::

<div class="ae-collabel">Predictive Process Monitoring (PPM) · case-level</div>
<div class="ae-lead">One ongoing case → its future.</div>

<div class="s3-trace">
  <span class="s3-pill">Create</span>
  <span class="s3-arrow">→</span>
  <span class="s3-pill">Assess</span>
  <span class="s3-arrow">→</span>
  <span class="s3-pill s3-pill-q">?</span>
</div>

<div class="s3-tasks">
  <div class="s3-task">
    <span class="s3-task-label">Next event</span>
    <span class="s3-task-q">"What does <strong>this</strong> applicant do next?"</span>
  </div>
  <div class="s3-task">
    <span class="s3-task-label">Remaining time</span>
    <span class="s3-task-q">"How long until <strong>this</strong> case is decided?"</span>
  </div>
  <div class="s3-task">
    <span class="s3-task-label">Outcome</span>
    <span class="s3-task-q">"Will <strong>this</strong> loan application be cancelled?"</span>
  </div>
</div>

<div class="s3-horizon">Horizon — the rest of one case</div>

::right::

<div class="ae-collabel">Process Model Forecasting (PMF) · system-level</div>
<div class="ae-lead">A log window → the next process model.</div>

<div class="s3-flow">
  <span class="s3-pill">Log window</span>
  <span class="s3-arrow">→</span>
  <div class="s3-dfg">
    <svg viewBox="0 0 84 64" width="116" height="88">
      <line x1="12" y1="32" x2="40" y2="16" stroke="#475569" stroke-width="2" />
      <line x1="12" y1="32" x2="40" y2="48" stroke="#475569" stroke-width="2" />
      <line x1="40" y1="16" x2="70" y2="32" stroke="#475569" stroke-width="2" />
      <line x1="40" y1="48" x2="70" y2="32" stroke="#475569" stroke-width="2" />
      <circle cx="12" cy="32" r="5" fill="var(--brand)" />
      <circle cx="40" cy="16" r="5" fill="var(--brand)" />
      <circle cx="40" cy="48" r="5" fill="var(--brand)" />
      <circle cx="70" cy="32" r="5" fill="var(--brand)" />
    </svg>
    <div class="s3-dfg-cap">DFG now</div>
  </div>
  <span class="s3-arrow">→</span>
  <div class="s3-dfg">
    <svg viewBox="0 0 84 64" width="116" height="88">
      <line x1="12" y1="32" x2="40" y2="16" stroke="#475569" stroke-width="2" />
      <line x1="12" y1="32" x2="40" y2="48" stroke="#475569" stroke-width="2" />
      <line x1="40" y1="16" x2="70" y2="32" stroke="#475569" stroke-width="2" />
      <line x1="40" y1="48" x2="70" y2="32" stroke="var(--accent)" stroke-width="2" />
      <circle cx="12" cy="32" r="5" fill="var(--brand)" />
      <circle cx="40" cy="16" r="5" fill="var(--brand)" />
      <circle cx="40" cy="48" r="5" fill="var(--accent)" />
      <circle cx="70" cy="32" r="5" fill="var(--brand)" />
    </svg>
    <div class="s3-dfg-cap">next week's DFG</div>
  </div>
</div>

<div class="s3-ask">"How often does <strong>offer sent → cancelled</strong> occur?"</div>
<div class="s3-horizon">Horizon — the near-term system future</div>

<style>
.s3-trace, .s3-flow { display: flex; align-items: center; gap: 10px; margin: 20px 0 14px; flex-wrap: wrap; }
.s3-pill { display: inline-flex; align-items: center; justify-content: center; border: 1px solid var(--hairline); border-radius: 999px; padding: 7px 15px; font-size: 18px; font-weight: 600; color: var(--ink); background: #fff; white-space: nowrap; }
.s3-pill-q { border-color: var(--brand); color: var(--brand); font-weight: 800; min-width: 36px; }
.s3-arrow { color: var(--neutral); font-size: 18px; font-weight: 700; }
.s3-tasks { display: flex; flex-direction: column; gap: 16px; margin: 20px 0 24px; }
.s3-task { display: flex; flex-direction: column; gap: 2px; }
.s3-task-label { font-size: 16px; font-weight: 600; letter-spacing: 0.03em; text-transform: uppercase; color: var(--neutral); }
.s3-task-q { font-size: 23px; font-style: italic; color: var(--ink); }
.s3-task-q strong { color: var(--brand); font-weight: 800; }
.s3-ask { font-size: 23px; font-style: italic; color: var(--ink); margin: 40px 0 30px; }
.s3-ask strong { font-style: normal; color: var(--brand); }
.s3-horizon { font-size: 17px; font-weight: 600; letter-spacing: 0.02em; color: var(--neutral); margin-top: 12px; }
.s3-dfg { display: flex; flex-direction: column; align-items: center; flex: 0 0 auto; }
.s3-dfg-cap { font-size: 14px; font-weight: 600; text-align: center; color: var(--neutral); margin-top: 4px; }
</style>

<!--
[S-03]
Q: PMF vs the PPM I already know?

Same loan log, two different prediction problems. PPM is case-level: take ONE ongoing
application and predict its future — the next event, the remaining time, or the outcome
("will THIS loan be cancelled?"). Horizon = the rest of that one case.

PMF is system-level: take a WINDOW of the whole log, and forecast the next process model —
how often each transition fires next, e.g. how often "offer sent → cancelled" occurs across
ALL cases. Horizon = the near-term system future (conceptually weeks-to-months; the 7-day
experimental horizon is setup detail for S11 — don't say it here).

Same event log, different question. (Absorbed into the assertion — deliver in speech.)

60s. Transition out: "Both need data, but PMF needs something different — let me show you
the pipeline."
-->

---
layout: assertion-evidence
locator: Pipeline
assertion: Each directly-follows edge becomes a time series we forecast
clicks: 2
---

<script setup>
import { frameForClicks } from './components/frameForClicks.js'
</script>

<div style="font-size:21px; color:var(--ink); margin-top:8px">
directly-follows (DF) relations = one transition A→B — we aggregate it <strong>daily</strong> and forecast <strong>7 days</strong> ahead.
</div>

<div class="mt-3" style="height: 392px">
  <DfgPipeline :frame="frameForClicks($clicks, 3)" />
</div>

<!--
[S-04]
Q: How does PMF become a forecasting problem?

Transition IN (from S3): "Both need data, but PMF needs something different — let me show you the pipeline."

DF is introduced here (full term "directly-follows (DF) relations"); DFG is already known from S2 —
use the abbreviation. Let the THREE CLICKS carry it; don't talk over them:
  • click 0 — event log → one daily count series per DF edge (the navy "past" window; 3 arcs shown,
    the point being we forecast ALL arcs, not just the hero).
  • click 1 — forecast the next 7 days for every arc (amber tail; the "future" window).
  • click 2 — sum each arc's 7 daily forecasts → the reassembled, FORECASTED DFG (≈316 vs held-out 315).

Keep it to "aggregate daily, forecast 7 days ahead." Do NOT mention stride / expanding window /
60-20-20 here — that's S11. "past/future" are a conceptual history→horizon motif, not the
experimental window.

Transition OUT (to S5): "So it's a forecasting problem. Why isn't it already solved?"
-->

---
layout: two-cols-header
---

# Where current PMF stands

<div class="text-sm opacity-70">From our prior benchmark (Yu et al. 2025)</div>

::left::

**Three findings:**

1. **ML/DL doesn't reliably beat naive** — sparsity + small data defeat training
2. **Univariate beats multivariate** — heterogeneity within each log
3. **No model wins across logs** — heterogeneity also across logs

<div class="mt-3 text-xs opacity-70">
Benchmark recommends XGBoost as the strongest ML default — our baseline going forward.
</div>

<div class="mt-2 text-xs opacity-60">
Quantified in this paper: DF series score 2–3× higher on transition, shifting, non-Gaussianity than 21 standard benchmarks.
</div>

::right::

<img src="/figures/bpi2017-drift-xgb-only.png" class="block mx-auto rounded-lg w-full max-h-[300px] object-contain" alt="BPI2017 O_Sent→O_Cancelled — ground truth vs XGBoost (XGBoost misses the drift)" />

<div class="text-xs opacity-60 mt-2 text-center">
XGBoost stays flat. The actual line drops.
</div>

<!--
Q: Why don't existing methods already solve this?

Transition in: "So we have a forecasting problem. Why isn't it already solved?"

75s. Three DF properties — sparsity, heterogeneity (within and across logs), small scale — explain three benchmark findings. Land the WHY, not just the WHAT.

Also pre-loads the XGBoost baseline for beat 7. When the results chart appears, recall: "XGBoost was the strongest ML in the benchmark — that's why it's our ML baseline."

Transition out (BRIDGE INTO BEAT 4 — memorize):
"These three properties — sparsity, heterogeneity, small scale — defeat from-scratch ML/DL. That's exactly what foundation models are designed for."
-->

---

# Why a time series foundation model?

<div class="grid grid-cols-2 gap-8 mt-2">

<div class="border rounded-lg p-4">
<div class="font-bold text-center mb-3 opacity-80">LLM</div>

- Text in → text out
- Web-scale text corpus
- Generalizes across language tasks

</div>

<div class="border-2 rounded-lg p-4" style="border-color: #1d4ed8">
<div class="font-bold text-center mb-3" style="color: #1d4ed8">TSFM</div>

- **Numbers in → numbers out**
- Millions of diverse time series
- Generalizes across forecasting tasks

</div>

</div>

<div class="mt-6 space-y-2">

- Same paradigm as LLMs: pretrain at scale, generalize **zero-shot**
- **No event logs in pretraining, to our knowledge**
- Built for: heterogeneity · small data · no per-task retraining

</div>

<div class="mt-2 text-sm opacity-60">
zero-shot = run a pretrained model on a new series with no extra training
</div>

<div class="mt-6 text-center font-semibold opacity-90 italic">
"Specialized models overfit on small heterogeneous PMF data.<br/>
Foundation models, pretrained on millions of diverse series, are designed to not."
</div>

<!--
Q: Why should a generic forecaster do better than a specialized one?

Transition in (bridge from beat 3): "These three properties — sparsity, heterogeneity, small scale — defeat from-scratch ML/DL. That's exactly what foundation models are designed for."

60s. CRITIQUE CONSTRAINT: must say "no event logs in pretraining, TO OUR KNOWLEDGE." Not "no process data." The wording is non-negotiable.

Load-bearing line — deliver the footer aloud: "Specialized models overfit on small heterogeneous PMF data. Foundation models, pretrained on millions of diverse series, are designed to not."

Transition out (into beat 5): "We have a candidate. Here's what we ask of it."
-->

---
layout: center
---

# Three questions this talk answers

<div class="text-left max-w-3xl mt-8 space-y-6">

<div>
<span class="font-bold opacity-50 mr-2">1.</span>
Can an <strong>off-the-shelf forecaster</strong> — trained on no process data — beat the strongest PMF baselines?
</div>

<div>
<span class="font-bold opacity-50 mr-2">2.</span>
If yes, does <strong>adapting it to process data</strong> help?
</div>

<div>
<span class="font-bold opacity-50 mr-2">3.</span>
Does a better forecast give us a <strong>better process model</strong>?
</div>

</div>

<!--
Q: What does this talk actually deliver?

30s. Plain-English scaffold — names ("TSFM", "LoRA", "Entropic Relevance") land on later slides, not here.

The three questions map to the three findings:
- Q1 (zero-shot beats baselines)  → beat 7 (results bars + drift callback)
- Q2 (adapting it helps?)          → beat 8 (fine-tuning + timing): marginal, dataset-dependent
- Q3 (better forecast → better process model?) → beat 9 via Entropic Relevance: parity, not better; Sepsis the exception

Transition out (into beat 6): "Three questions. Here's what we point at them."
-->

---

# The candidates

<div class="border-2 border-dashed border-gray-400 rounded-lg p-4 mb-4 flex items-center justify-center min-h-[180px]">
<div class="text-center opacity-60 text-sm">
[PLACEHOLDER]<br/>
Release timeline — three lanes (Chronos / MOIRAI / TimesFM)<br/>
x-axis: release date (2024 → 2025)  ·  y-axis: model size (parameters, log scale)<br/>
one point per variant, labeled; latest of each highlighted in accent<br/>
Chronos-T5 · Bolt · 2  |  MOIRAI 1.0 · 1.1 · MoE · 2.0  |  TimesFM 1.0 · 2.0 · 2.5<br/>
(several variants span a size range — plot the representative/used size)
</div>
</div>

<div class="grid grid-cols-3 gap-4 text-sm">

<div>
<strong>Chronos</strong> <span class="opacity-60">— Amazon</span><br/>
<span class="opacity-80">encoder-decoder → encoder-only (Chronos-2)</span>
</div>

<div>
<strong>MOIRAI</strong> <span class="opacity-60">— Salesforce</span><br/>
<span class="opacity-80">masked encoder → decoder-only (MOIRAI-2.0)</span>
</div>

<div>
<strong>TimesFM</strong> <span class="opacity-60">— Google</span><br/>
<span class="opacity-80">decoder-only throughout (scales 1.0 → 2.5)</span>
</div>

</div>

<div class="mt-6 flex justify-center gap-3 text-sm">
<span class="px-3 py-1 rounded border" style="border-color: #1d4ed8; color: #1d4ed8">zero-shot</span>
<span class="px-3 py-1 rounded border opacity-80">LoRA <span class="text-xs opacity-70">(small trainable adapters on attention)</span></span>
<span class="px-3 py-1 rounded border opacity-80">full fine-tuning</span>
</div>

<div class="mt-4 text-center text-xs opacity-70">
Univariate throughout (per Yu 2025 finding)
</div>

<!--
Q: Which models? Which settings?

45s. Names appear on screen, not in voice. Let the timeline do the work.
Three evolving families; results use the latest of each (Chronos-2, MOIRAI-2.0, TimesFM-2.5).
Size axis tells a "newer = smaller, yet better" story — MOIRAI-2.0 is just 11.4M params.

Transition into results: "Three families. Eight model variants. No event logs in pretraining.
Here's what happens when you point them at DF time series."
-->

---
layout: assertion-evidence
locator: Results
assertion: Zero-shot TSFMs beat both baselines on every log
---

<img src="/figures/results-mae-bars.png" class="block mx-auto w-full max-h-[380px] object-contain rounded-lg" alt="Zero-shot MAE across 4 event logs — TSFMs vs Seasonal-Naive and XGBoost baselines" />

<div class="text-center mt-3" v-click>
  <Callout>−21% mean MAE vs best baseline · 3 models × 4 logs</Callout>
</div>

<div class="caption mt-2 text-center">Two strongest baselines from our prior benchmark (Yu 2025) — full ranking on backup.</div>

<!--
Q: Does Q1 hold — do zero-shot TSFMs beat the baselines?

Transition in: "Three families. Eight model variants. No event logs in pretraining.
Here's what happens when you point them at DF time series."

~2 min. CRITIQUE CONSTRAINT — open with the baseline framing verbatim:
"We compare against the two strongest baselines from our prior benchmark; full ranking
on backup." Pre-empts cherry-picking.

The three latest models (Chronos-2, MOIRAI-2.0, TimesFM-2.5) win MAE on all four logs —
including Sepsis (MOIRAI-2.0 ↓28%). Sepsis is NOT a MAE exception; it only becomes the
hard case later, on ER (beat 9). Don't undercut the MAE win here.

Q&A defense: some older/smaller variants don't beat the baselines on BPI2017 (paper
p.11) — that's why the bars show the latest of each family.

Transition out (into 7b): "Same data, same window — here's what that win looks like."
-->

---

# Drift, revealed

<div class="grid grid-cols-2 gap-6 items-center">

<div>

Same plot as before. Same prediction window.

This time with the TSFM line.

<div class="mt-6 text-sm space-y-2 opacity-90">

- XGBoost stays flat
- TSFM misses the initial drop, then **adapts**
- This is online adaptation from historical context — no retraining

</div>

</div>

<img src="/figures/bpi2017-drift-with-tsfm.png" class="block mx-auto rounded-lg w-full max-h-[340px] object-contain" alt="BPI2017 drift revealed — Chronos-2 and MOIRAI-2.0 track the drop XGBoost missed" />

</div>

<!--
Q: What does the zero-shot win look like on real drift?

~1 min. THE CALLBACK MOMENT — the audience remembers this plot from slide 3 (XGBoost
flat, actual line drops). Now reveal the TSFM line. Don't rush — let them see it appear.
Single spoken line: "Same data. Same prediction window. The TSFM tracks what XGBoost missed."

Mechanism (if asked): the TSFM adapts via an expanding historical-context window at
inference — no retraining (paper §4.1). Misses the initial drift, then catches up.

Transition out (into beat 8): "TSFMs win zero-shot. Natural next question: can we make
them better?"
-->

---

# Does fine-tuning help?

<div class="grid grid-cols-2 gap-4">

<div>
<div class="text-xs opacity-70 mb-1 text-center">Accuracy — MAE across settings</div>
<img src="/figures/ft-slope.png" class="block mx-auto rounded-lg w-full max-h-[330px] object-contain" alt="Fine-tuning vs zero-shot, normalized — most lines hug 1.0; a few overfit" />
</div>

<div>
<div class="text-xs opacity-70 mb-1 text-center">Compute — wall-clock (log scale)</div>
<div class="border-2 border-dashed border-gray-400 rounded-lg p-4 flex items-center justify-center min-h-[320px]">
<div class="text-center opacity-60 text-sm">
[PLACEHOLDER]<br/>
Log-scale bar chart<br/>
7 bars:<br/>
3 ZS · 3 LoRA · 3 Full-FT · 1 XGBoost reference
</div>
</div>
</div>

</div>

<div class="mt-4 text-center font-semibold opacity-90">
Marginal accuracy. ~100× compute. Skip it at PMF data scale.
</div>

<!--
~3 min 15s. Two findings: (a) fine-tuning rarely helps, (b) it costs a lot.
Drive the eye to the flat lines on the left, then the log-scale jump on the right.
Don't moralize — let the visual do it.
-->

---
layout: two-col-evidence
locator: Evaluation
assertion: TSFMs reach ER parity — not better
---

::left::

<div class="relative">
  <img src="/figures/er-hospital-billing.png" class="block w-full max-h-[340px] object-contain rounded-lg" alt="Hospital Billing ER — reusing the historical (Training) model is far worse than any forecast" />
  <div class="absolute" style="left: 34%; top: 26%" v-click="1">
    <Callout dir="left">Forecast ≫ reuse</Callout>
  </div>
  <div class="absolute" style="left: 52%; top: 40%" v-click="2">
    <Callout dir="down">TSFMs ≈ baselines</Callout>
  </div>
</div>

<div class="caption mt-3">ER = bits-per-trace · Hospital Billing · in-bar % = traces the forecasted model fits.</div>

::right::

$$\mathrm{ER}(L,M) \approx \text{bits to encode one trace}$$

<div class="dense">Hospital Billing — lower is better:</div>

- **Truth** — 1.86 *(ideal floor)*
- **Forecasts** — 2.1–2.5 *(baselines & TSFMs alike)*
- **Training** — 5.83 *(reuse the old model — worst)*

<div class="mt-5 font-semibold" style="color: var(--brand)">The bottleneck has moved from forecasting accuracy to process-aware representation.</div>

<!--
Q: Does the forecasting win translate to a better process model?

Transition in: "But forecasting accuracy isn't the only thing we care about in PM."
Open with the QUESTION, not the result — beat 8→9 is the talk's lowest-energy moment.

~1 min 30s. THE LOAD-BEARING SLIDE.
Memorized rebuttal — deliver in speech, not on slide (verbatim from SLIDES.md):

"ER parity means we're not producing better DFG structures — we're producing
DFGs with better edge weights. For tasks where the forecast itself is the
deliverable — capacity planning, drift detection, anomaly baselines —
edge-weight accuracy IS the contribution. For tasks where you need a different
process model, ER says the bottleneck is now the DFG representation,
not the forecaster."

Do not skip this line. Rehearse it cold.
-->

---

# Takeaways

<div class="space-y-5 mt-6">

<div class="border-l-4 pl-4" style="border-color: #1d4ed8">
<div class="font-semibold mb-1">For practitioners</div>
<div class="text-sm opacity-90">
Zero-shot TSFMs are the new PMF default — skip fine-tuning at PMF data scale.
<span class="opacity-70 italic">Four logs is not a paradigm — but it is a strong enough signal to make zero-shot the right new starting baseline.</span>
</div>
</div>

<div class="border-l-4 pl-4" style="border-color: #1d4ed8">
<div class="font-semibold mb-1">For PMF research</div>
<div class="text-sm opacity-90">
The bottleneck has moved from forecasting accuracy to process-aware representation.
DFGs are a lossy target.
</div>
</div>

<div class="border-l-4 pl-4" style="border-color: #1d4ed8">
<div class="font-semibold mb-1">For process mining</div>
<div class="text-sm opacity-90">
PM can borrow from adjacent fields cheaply — a process-native FM is the next frontier,
but needs a corpus of event logs we don't yet have.
</div>
</div>

</div>

<div class="mt-8 pt-4 border-t border-gray-300 flex justify-between items-center text-xs opacity-80">
<div>
<strong>github.com/YongboYu/pmf-tsfm</strong> · demo URL · paper
</div>
<div class="italic">
Runs on laptop (MPS) · GPU server (CUDA) · HPC
</div>
</div>

<!--
Q: What do I do with this? Where do I go next?

Transition in: "So what do we walk away with?"

~2 min. CRITIQUE CONSTRAINT: signal 1 must contain "four logs is not a paradigm" hedge.

CLOSING ANCHOR LINE (rehearse cold):
"Zero-shot TSFMs are the new PMF default. Four logs is not a paradigm — but it is
a strong enough signal that the right new default is to try a zero-shot TSFM first.
The deeper signal is that time-series forecasting just became cheap enough that
process mining can borrow from it without paying the training tax.
What else can we borrow?"
-->

---
layout: center
---

# Demo

<div class="border-2 border-dashed border-gray-400 rounded-lg p-6 mt-4 mb-4 min-h-[280px] flex items-center justify-center">
<div class="text-center opacity-60 text-sm">
[PLACEHOLDER]<br/>
Pre-recorded screencast (~2 min)<br/><br/>
1. Open small event log<br/>
2. Pick TSFM checkpoint<br/>
3. Run zero-shot inference<br/>
4. Render forecasted DFG vs ground truth
</div>
</div>

<div class="text-center text-sm opacity-80">
Try it yourself · <strong>[demo URL]</strong> · QR code below
</div>

<!--
~2-3 min. Live demo NOT recommended in single-screen room (Wi-Fi/projector risk).
Pre-recorded screencast is the safe play.
Audience can scan QR during Q&A.
-->

---
layout: end
---

# Thank you

<div class="text-lg opacity-80 mt-6">
Questions, please.
</div>

<div class="text-sm opacity-60 mt-8">
yongbo.yu@kuleuven.be · github.com/YongboYu/pmf-tsfm
</div>

<!--
10 min Q&A.
Backup slides follow. Have answers to the ranked hostile questions rehearsed cold
(see SLIDES.md "Required backup slides" and Q&A defense section).
-->

---
hideInToc: true
---

# Backup · Full baseline ranking (Yu 2025)

<div class="border-2 border-dashed border-gray-400 rounded-lg p-4 flex items-center justify-center min-h-[400px]">
<div class="text-center opacity-60 text-sm">
[PLACEHOLDER]<br/>
Full benchmark ranking from Yu et al. 2025<br/>
All methods × all datasets<br/>
Used to defend the 2-baseline choice on the main results slide
</div>
</div>

<!--
Q&A defense: "We compare against the two strongest baselines from our prior
benchmark — here's the full ranking."
-->

---
hideInToc: true
---

# Backup · Full RMSE table

<img src="/figures/rmse-full.png" class="block mx-auto rounded-lg w-full max-h-[420px] object-contain" alt="Full zero-shot RMSE table — all model variants × 4 logs; confirms the MAE ranking" />

<div class="caption mt-2 text-center">RMSE from our re-run (paper reports MAE, Table 4); baselines may differ slightly from camera-ready.</div>

<!--
Q&A defense: when asked "does RMSE tell the same story?" — yes, point at this.
Provenance: RMSE has no paper table (paper Table 4 is MAE); built from results/ CSVs, so its
baselines carry the re-run delta flagged in make_figures.py. Story (TSFMs win) unchanged.
-->

---
hideInToc: true
---

# Backup · ER across all four logs

<img src="/figures/er-bars.png" class="block mx-auto rounded-lg w-full max-h-[380px] object-contain" alt="Entropic Relevance across all 4 event logs — TSFMs match baselines except Sepsis" />

<div class="mt-3 text-sm opacity-80 text-center">
Parity on BPI2017 / BPI2019-1 / Hospital Billing. <strong>Sepsis is the exception</strong> — high behavioral heterogeneity defeats all models (TSFMs worst, tiny fitting ratios).
</div>

<!--
Q&A defense: the main ER slide shows Hospital Billing only. This is the full picture.
Sepsis: ER far above Truth AND lowest fitting ratios — the process is both heterogeneous
and hard to encode with any single forecasted model.
-->

---
hideInToc: true
---

# Backup · Hyperparameter configuration

<div class="grid grid-cols-2 gap-6 mt-4 text-sm">

<div>
<div class="font-semibold mb-2">XGBoost</div>
<div class="opacity-80">
- Optuna hyperparameter optimization<br/>
- N trials per dataset (specify exact count)<br/>
- Lagged features, daily aggregation
</div>
</div>

<div>
<div class="font-semibold mb-2">LoRA / Full FT</div>
<div class="opacity-80">
- LoRA: r=2, α=4, applied to Q/K/V/O<br/>
- Patch size 16, batch 32<br/>
- LR 1e-4, AdamW, 3 epochs<br/>
- Full FT follows original model recipes
</div>
</div>

</div>

<!--
Q&A defense: pre-empts "was your baseline actually tuned?" and "what's your LoRA setup?"
-->

---
hideInToc: true
---

# Backup · DF complexity vs standard benchmarks

<img src="/figures/df-complexity-radar.png" class="block mx-auto rounded-lg max-h-[400px] object-contain" alt="DF time-series complexity radar across 7 metrics for the 4 event logs (paper Table 3)" />

<!--
Q&A defense: backs the "DF time series are unusually hard" claim with the
full quantitative profile.
-->

---
hideInToc: true
---

# Backup · Sepsis — why it's hard

<div class="border-2 border-dashed border-gray-400 rounded-lg p-4 flex items-center justify-center min-h-[380px]">
<div class="text-center opacity-60 text-sm">
[PLACEHOLDER]<br/>
Sepsis DFG sample or trace variant distribution<br/>
Shows behavioral heterogeneity:<br/>
many infrequent variants, low fitting ratio<br/><br/>
Reinforces ER slide's Sepsis caveat
</div>
</div>

<!--
Q&A defense: "Why does Sepsis fail?" — show the variant tail and the sparse DF activity.
-->

---
hideInToc: true
---

# Backup · Horizon sensitivity

<div class="text-sm space-y-4 mt-4">

<div>
We forecast at a <strong>7-day horizon</strong>, matching:

- The cadence used in the Yu 2025 benchmark
- The operational reporting cadence of the source logs
- Comparable to the horizons TSFMs were pretrained on (weather, traffic series)
</div>

<div>
<strong>Open question:</strong> longer horizons (30+ days). Not tested here.
Future work — pair TSFM forecasts with drift detection for adaptive multi-horizon prediction.
</div>

</div>

<!--
Q&A defense: "What about month-long horizons?" — acknowledge openly,
flag as future work; don't pretend we tested it.
-->

---
hideInToc: true
---

# Backup · Multivariate experiments

<div class="text-sm space-y-4 mt-6">

<div>
MOIRAI, MOIRAI-MoE, and Chronos-2 support multivariate inference.
We ran initial multivariate experiments.
</div>

<div>
<strong>Result:</strong> no consistent gains over univariate, consistent with the
Yu 2025 finding that univariate outperforms multivariate on DF time series.
</div>

<div>
<strong>Why?</strong> Likely the high heterogeneity across DF series in one log makes
cross-series attention hurt more than it helps at this data scale.
</div>

<div>
<strong>Open question:</strong> dedicated multivariate fine-tuning for PMF — not
covered here. Future work.
</div>

</div>

<!--
Q&A defense: "Why didn't you use the multivariate capabilities of MOIRAI/Chronos-2?"
Pre-empted on slide 3 (finding #2), reinforced here with the actual experiment.
-->
