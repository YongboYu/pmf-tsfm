---
theme: default
colorSchema: light
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
          <tr><td>App 9043</td><td>Canceled</td><td>Oct 16 15:02</td></tr>
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
    A <strong style="color: var(--brand)">Directly-Follows Graph (DFG)</strong> is one kind of <strong style="color: var(--brand)">process model</strong>.
  </p>
  <p style="font-size: 20px; color: #334155; line-height: 1.5; margin: 0">
    <strong style="color: var(--brand)">Process discovery</strong> → one static snapshot · <strong style="color: var(--brand)">Process Model Forecasting (PMF)</strong> predicts the next.
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
assertion: PPM forecasts one case · PMF the whole process model
---

::left::

<div class="s3-collabel">Predictive Process Monitoring (PPM)</div>
<div class="s3-level">Scope: case-level</div>

<div class="s3-trace">
  <span class="s3-pill">Create</span>
  <span class="s3-arrow">→</span>
  <span class="s3-pill">Assess</span>
  <span class="s3-arrow">→</span>
  <span class="s3-pill s3-pill-q">?</span>
</div>

<div class="s3-tasks">
  <div class="s3-task">
    <span class="s3-task-q">"Will <strong>this</strong> loan be canceled?"</span>
  </div>
  <div class="s3-task">
    <span class="s3-task-q">"What does <strong>this</strong> applicant do next?"</span>
  </div>
</div>

<div class="s3-horizon">Horizon: the rest of one case</div>

::right::

<div class="s3-collabel">Process Model Forecasting (PMF)</div>
<div class="s3-level">Scope: system-level</div>

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

<div class="s3-ask">"How often does <strong>offer sent → canceled</strong> occur?"</div>
<div class="s3-horizon">Horizon: the near-term system future</div>

<style>
.s3-collabel { font-size: 23px; font-weight: 700; color: var(--brand); letter-spacing: 0.01em; margin-bottom: 6px; }
.s3-level { font-size: 20px; font-weight: 700; letter-spacing: 0.01em; color: var(--brand); }
.s3-trace, .s3-flow { display: flex; align-items: center; gap: 10px; margin: 28px 0 18px; flex-wrap: wrap; }
.s3-trace { justify-content: center; }
.s3-pill { display: inline-flex; align-items: center; justify-content: center; border: 1px solid var(--hairline); border-radius: 999px; padding: 7px 15px; font-size: 18px; font-weight: 600; color: var(--ink); background: #fff; white-space: nowrap; }
.s3-pill-q { border-color: var(--brand); color: var(--brand); font-weight: 800; min-width: 36px; }
.s3-arrow { color: var(--neutral); font-size: 18px; font-weight: 700; }
.s3-tasks { display: flex; flex-direction: column; gap: 24px; margin: 30px 0 36px; }
.s3-task { display: flex; flex-direction: column; gap: 2px; }
.s3-task-q { font-size: 20px; font-style: italic; color: var(--neutral); }
.s3-task-q strong { color: var(--brand); font-weight: 700; }
.s3-ask { font-size: 20px; font-style: italic; color: var(--neutral); margin: 30px 0 28px; }
.s3-ask strong { font-style: normal; color: var(--brand); }
.s3-horizon { font-size: 22px; font-weight: 700; letter-spacing: 0.01em; color: var(--brand); margin-top: 16px; }
.s3-dfg { display: flex; flex-direction: column; align-items: center; flex: 0 0 auto; }
.s3-dfg-cap { font-size: 14px; font-weight: 600; text-align: center; color: var(--neutral); margin-top: 4px; }
</style>

<!--
[S-03]
Q: PMF vs the PPM I already know?

Same loan log, two different prediction problems. PPM is case-level: take ONE ongoing
application and predict its future — the next event, the remaining time, or the outcome
("will THIS loan be canceled?"). Horizon = the rest of that one case.

PMF is system-level: take a WINDOW of the whole log, and forecast the next process model —
how often each transition fires next, e.g. how often "offer sent → canceled" occurs across
ALL cases. Horizon = the near-term system future (conceptually weeks-to-months; the 7-day
experimental horizon is setup detail for S12 — don't say it here).

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
directly-follows (DF) relation = one A→B transition <span style="margin:0 16px; color:var(--neutral-soft)">·</span> aggregated <strong>daily</strong> <span style="margin:0 16px; color:var(--neutral-soft)">·</span> forecast <strong>7 days</strong> ahead
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
60-20-20 here — that's S12. "past/future" are a conceptual history→horizon motif, not the
experimental window.

Transition OUT (to S5): "So it's a forecasting problem. Why isn't it already solved?"
-->

---
layout: assertion-evidence
locator: The challenge
assertion: "DF series are hard: drift, intermittency, and heterogeneity"
---

<div class="grid grid-cols-2 gap-8 mt-3" style="height: 350px">

  <!-- LEFT — drift (truth only) -->
  <div class="flex flex-col">
    <div class="s5-panel-label">① Drift: a real level shift</div>
    <img src="/figures/s5-drift-truth.png"
         class="block w-full object-contain rounded-lg" style="max-height: 280px"
         alt="BPI2017 Sent→Canceled daily values — drops from ~46 to ~3 over the test period" />
    <div class="s5-panel-cap">BPI2017 · Sent → Canceled</div>
  </div>

  <!-- RIGHT — intermittency (truth only) -->
  <div class="flex flex-col">
    <div class="s5-panel-label">② Intermittent: long zeros, sudden spikes</div>
    <img src="/figures/s5-intermittent-truth.png"
         class="block w-full object-contain rounded-lg" style="max-height: 280px"
         alt="BPI2019-1 Cancel→Record Invoice Receipt daily values — mostly zero with spikes to 29" />
    <div class="s5-panel-cap">BPI2019-1 · Cancel → Record Invoice Receipt</div>
  </div>

</div>

<div class="mt-4 text-center" style="font-size: 20px; color: #334155">
  ③ <strong style="color: var(--brand)">Heterogeneous</strong> · pattern differs <strong>within</strong> and <strong>across</strong> logs
</div>

<style>
.s5-panel-label { font-size: 18px; font-weight: 700; color: var(--brand); text-align: center; margin-bottom: 6px }
.s5-panel-cap { font-size: 14px; color: var(--neutral); text-align: center; margin-top: 6px }
</style>

<!--
[S-05]
Q: What makes these series difficult?

Transition IN (from S4): "So it's a forecasting problem. Why isn't it already solved?"

Three data challenges, shown with REAL ground-truth series (no model lines yet):
  ① Drift — BPI2017 Sent → Canceled: a genuine level shift, fires ~46/day early, decays to ~3.
  ② Intermittent — BPI2019-1 Cancel → Record Invoice Receipt: ~80% zeros, sudden spikes to ~29.
  ③ Heterogeneous — the two panels are different logs with completely different shapes; DF
    relations also differ within a single log. Neither is white noise, neither is a smooth trend.

Axes: x = day (stride = 1, so consecutive windows are consecutive days), y = value (the 7-day-ahead
forecast's last day). These two plots reappear on S14 with the model lines revealed — DO NOT show
any model line here.

45s. Transition OUT (to S6): "These are hard. So — do the methods we already have handle them?"
-->

---
layout: assertion-evidence
locator: Prior work
assertion: Machine Learning (ML) / Deep Learning (DL) underperform
---

<div class="grid grid-cols-2 gap-8 mt-3" style="height: 350px">

  <!-- LEFT — drift: XGBoost misses the drop -->
  <div class="flex flex-col">
    <div class="s6-panel-label">Drift: XGBoost stays high, misses the drop</div>
    <img src="/figures/s6-drift-xgb-box.png"
         class="block w-full object-contain rounded-lg" style="max-height: 280px"
         alt="BPI2017 drift — XGBoost stays elevated while the truth collapses late, amber box over the back-half" />
    <div class="s6-cap">BPI2017 · Sent → Canceled</div>
  </div>

  <!-- RIGHT — intermittent: XGBoost overshoots the zeros -->
  <div class="flex flex-col">
    <div class="s6-panel-label">Intermittent: XGBoost overshoots the zeros</div>
    <img src="/figures/s6-intermittent-xgb-arrow.png"
         class="block w-full object-contain rounded-lg" style="max-height: 280px"
         alt="BPI2019-1 intermittent — XGBoost hallucinates 40–71 where the truth is mostly zero, amber arrow at an overshoot peak" />
    <div class="s6-cap">BPI2019-1 · Cancel → Record Invoice Receipt</div>
  </div>

</div>

<div class="mt-4 text-center" style="font-size: 20px; color: #334155">
  <strong style="color: var(--brand)">XGBoost</strong>: the recommended top forecaster in the prior benchmark (Yu et al. 2025)
</div>

<style>
.s6-panel-label { font-size: 18px; font-weight: 700; color: var(--brand); text-align: center; margin-bottom: 6px }
.s6-cap { font-size: 14px; color: var(--neutral); text-align: center; margin-top: 6px }
</style>

<!--
[S-06]
Q: Do the methods we already have handle these patterns?

Transition IN (from S5): "These are hard. So — do existing methods handle them?"

ML/DL don't win in general. Here is the prior benchmark's TOP ML — tuned XGBoost (Yu et al. 2025) —
on the SAME two series from S5:
  • Drift: XGBoost stays high and misses the level-shift drop.
  • Intermittent: XGBoost overfits badly — it hallucinates large values (40–71) where the truth is
    mostly zero. A textbook overfit to a small, complex, heterogeneous signal.
Reason (callback to S5's challenges): small + complex + heterogeneous data makes from-scratch
ML/DL overfit. XGBoost is the benchmark's recommended top model and still loses — it becomes our ML
baseline going forward.

Honest nuance (speech only): the TSFM revealed on S14 wins by staying controlled (not overshooting),
not by capturing the rare spikes. Do NOT show any TSFM line here — that's S14.

45s. Transition OUT (to S7 complexity): "It's not bad luck — these series are statistically off the
charts, and the logs are tiny."
-->

---
layout: assertion-evidence
locator: Diagnosis
assertion: DF series are shorter and statistically harder
---

<div class="s7-wrap mt-6">
  <table class="s7-stats">
    <colgroup>
      <col style="width: 22%" /><col style="width: 11%" /><col style="width: 11%" />
      <col style="width: 18.67%" /><col style="width: 18.67%" /><col style="width: 18.67%" />
    </colgroup>
    <thead>
      <tr class="s7-grouprow">
        <th colspan="3" class="s7-nogroup"></th>
        <th colspan="3" class="s7-cxgroup">Complexity measurements</th>
      </tr>
      <tr>
        <th>Event log</th><th class="hot">Days</th><th>DFs</th>
        <th class="cx">Transition</th><th class="cx">Shifting</th><th class="cx">Non-Gaussianity</th>
      </tr>
    </thead>
    <tbody>
      <tr><td>BPI2017</td><td class="hot">319</td><td>21</td><td class="cx">.06</td><td class="cx">.27</td><td class="cx">.33</td></tr>
      <tr><td>BPI2019_1</td><td class="hot">307</td><td class="b">149</td><td class="cx">.15</td><td class="cx">.36</td><td class="cx">.47</td></tr>
      <tr><td>Sepsis</td><td class="hot">459</td><td class="b">135</td><td class="cx">.23</td><td class="cx">.28</td><td class="cx">.59</td></tr>
      <tr><td>Hospital Billing</td><td class="hot">726</td><td>73</td><td class="cx">.17</td><td class="cx">.48</td><td class="cx">.46</td></tr>
    </tbody>
    <tfoot>
      <tr class="s7-footrow">
        <td colspan="3" class="s7-foot-blank"></td>
        <td colspan="3" class="s7-foot-note">higher than 21 public time-series datasets (Li et al. 2025)</td>
      </tr>
    </tfoot>
  </table>

  <div class="s7-punch flex justify-center mt-8" v-click>
    <Callout>Complex signal + little data → <strong>trained-from-scratch</strong> ML/DL <strong>overfit</strong></Callout>
  </div>
</div>

<style>
.s7-wrap { max-width: 1000px; margin-left: auto; margin-right: auto }
.s7-stats { width: 100%; border-collapse: collapse; font-size: 19px; color: var(--ink); table-layout: fixed }
.s7-stats th { text-align: right; font-weight: 700; color: var(--neutral); border-bottom: 1.5px solid #cbd5e1; padding: 9px 12px; font-size: 15px; text-transform: uppercase; letter-spacing: .02em }
.s7-stats th:first-child, .s7-stats td:first-child { text-align: left }
.s7-stats td { padding: 9px 12px; border-bottom: 1px solid #eef2f6; text-align: right; font-variant-numeric: tabular-nums }
.s7-stats .hot { background: #fbedd8 }
.s7-stats .cx { background: #f4f8fc }
.s7-stats .b { font-weight: 700 }
.s7-grouprow th { border-bottom: none; padding-bottom: 6px }
.s7-nogroup { background: transparent }
.s7-stats th.s7-cxgroup { background: var(--brand); color: #fff; text-align: center; font-weight: 700; font-size: 15px; text-transform: uppercase; letter-spacing: .04em; padding: 8px 12px; border-bottom: none }
.s7-footrow td { border-bottom: none; padding-top: 12px }
.s7-foot-blank { background: transparent }
.s7-stats td.s7-foot-note { background: #e3edf6; color: var(--brand); font-weight: 700; font-size: 18px; text-align: center; padding: 12px 10px; border-radius: 0 0 8px 8px; white-space: nowrap }
.s7-punch .ae-callout { font-size: 20px !important; padding: 9px 16px !important }
.s7-punch .ae-callout strong { font-weight: 800 }
</style>

<!--
[S-07]
Q: Why do trained-from-scratch models struggle here?

Transition IN (from S6): "It's not bad luck — these series are statistically off the charts,
and the logs are tiny."

This is the quantitative WHY behind S6's overfitting. Two facts, one message:
  • COMPLEX — across 7 paper complexity metrics, three stand out: Transition, Shifting,
    Non-Gaussianity. The paper reports these as HIGHER than the 21 public forecasting
    benchmarks (Li et al. 2025). Say it qualitatively — we don't plot the benchmark
    numbers (not in our repo); the amber axes + badge carry it.
  • SMALL — every log is one short multivariate series: only 307–726 daily steps, split
    expanding-window, with up to 149 DF variables. Sepsis is the extreme (999 cases /
    16,009 events over 459 days) — the sparsity that drives its later ER failure (backup).

Together: complex signal + little data → from-scratch ML/DL overfit (exactly S6). Numbers are
paper-faithful (stats_log.tex / stats_df.tex). Don't re-explain S6's overfit — prove it.

45-60s. Transition OUT (to S8 TSFM): "Small, complex, heterogeneous data is exactly where a
model you *don't* train might win."
-->

---
layout: assertion-evidence
locator: Forecasting today
assertion: Foundation models are the new direction in forecasting
---

<div class="s8-evidence mt-4">
  <div class="grid grid-cols-2 items-stretch" style="gap: 186px">
    <!-- LEFT — LLM: foundation model for language (neutral gray) -->
    <div class="s8-card">
      <div class="s8-name">Large Language Model (LLM)</div>
      <div class="s8-ex">
        <span class="s8-ex-q">I am a math <span class="s8-blank">?</span></span>
        <span class="s8-ex-arrow">→</span>
        <span class="s8-ex-a">teacher</span>
      </div>
      <div class="s8-rows">
        <div class="s8-row"><span class="s8-rk">training data</span><span class="s8-rv">web-scale text</span></div>
        <div class="s8-row"><span class="s8-rk">tasks</span><span class="s8-rv">language</span></div>
      </div>
    </div>
    <!-- RIGHT — TSFM: foundation model for time series, what we use (brand-navy emphasis) -->
    <div class="s8-card s8-card--tsfm">
      <div class="s8-badge">what we use</div>
      <div class="s8-name s8-name--brand">Time Series Foundation Model (TSFM)</div>
      <div class="s8-ex">
        <span class="s8-ex-q">…12, 9, 14, 11, <span class="s8-blank">?</span></span>
        <span class="s8-ex-arrow">→</span>
        <span class="s8-ex-a s8-ex-a--brand">13</span>
      </div>
      <div class="s8-rows">
        <div class="s8-row"><span class="s8-rk">training data</span><span class="s8-rv s8-rv--brand">millions of diverse series</span></div>
        <div class="s8-row"><span class="s8-rk">tasks</span><span class="s8-rv s8-rv--brand">forecasting</span></div>
      </div>
    </div>
  </div>
  <!-- centered linking callouts (double arrows) bridging the two shared rows; reveal on click -->
  <div class="s8-link s8-link--data" v-click="1"><span class="s8-larr">←</span><Callout>pretrained at scale</Callout><span class="s8-rarr">→</span></div>
  <div class="s8-link s8-link--task" v-click="1"><span class="s8-larr">←</span><Callout>generalizable</Callout><span class="s8-rarr">→</span></div>
</div>

<div class="s8-defs">
  <div class="s8-def"><strong>zero-shot</strong>: frozen parameters, run as-is</div>
  <div class="s8-def"><strong>fine-tuning</strong>: tweak the parameters on your own data</div>
</div>

<style>
.s8-evidence { position: relative }
.s8-card { position: relative; border: 1.5px solid #cdd6e0; border-radius: 12px; padding: 22px 24px 26px; min-height: 218px; display: flex; flex-direction: column; gap: 16px }
.s8-card--tsfm { border: 2px solid var(--brand); background: #f4f8fc }
.s8-badge { position: absolute; top: -13px; right: 18px; background: var(--accent); color: var(--ink); font-size: 14px; font-weight: 700; border-radius: 6px; padding: 3px 10px; box-shadow: 0 2px 8px rgba(221, 138, 46, 0.3) }
.s8-name { font-size: 22px; font-weight: 700; color: var(--neutral); white-space: nowrap }
.s8-name--brand { color: var(--brand) }
.s8-ex { display: flex; align-items: center; align-self: center; gap: 10px; font-size: 19px; color: var(--ink); font-variant-numeric: tabular-nums }
.s8-ex-q { font-weight: 500 }
.s8-blank { display: inline-block; min-width: 24px; text-align: center; color: var(--neutral-soft); font-weight: 700; border-bottom: 2px dashed var(--neutral-soft); line-height: 1.1 }
.s8-ex-arrow { color: var(--neutral-soft); font-weight: 800 }
.s8-ex-a { font-weight: 700; color: var(--ink); background: #eef2f6; border-radius: 6px; padding: 1px 9px }
.s8-ex-a--brand { color: #fff; background: var(--brand) }
.s8-rows { margin-top: auto; display: flex; flex-direction: column; gap: 16px }
.s8-row { display: flex; align-items: baseline; gap: 14px }
.s8-rk { font-size: 17px; color: var(--neutral); font-weight: 600; width: 104px; flex: none }
.s8-rv { font-size: 21px; color: var(--ink) }
.s8-rv--brand { font-weight: 600 }
.s8-link { position: absolute; left: 50%; transform: translate(-50%, 50%); display: flex; align-items: center; gap: 6px; white-space: nowrap; z-index: 5 }
.s8-link--data { bottom: 88px }
.s8-link--task { bottom: 38px }
.s8-link .ae-callout { font-size: 22px !important; font-weight: 700 !important; padding: 8px 18px !important; box-shadow: 0 4px 14px rgba(221, 138, 46, 0.35) }
.s8-larr, .s8-rarr { color: var(--accent); font-size: 22px; font-weight: 800; line-height: 1 }
.s8-defs { margin: 46px auto 0; width: fit-content; text-align: left; display: flex; flex-direction: column; gap: 10px; font-size: 23px; color: var(--neutral) }
.s8-def strong { color: var(--ink) }
</style>

<!--
[S-08]
Q: What is the current direction in forecasting? (And are we forecasting with an LLM?)

Transition IN (from S7): "Small, complex, heterogeneous data is exactly where a model you
*don't* train might win."

THE load-bearing slide (outline: the single slide that, if cut, most weakens the talk — without
it half the room thinks we fine-tuned GPT). One message: the current direction in forecasting is
foundation models. Just as an LLM is a foundation model for language (text → text), a Time Series
Foundation Model (TSFM) is one for time series (time series → time series) — same recipe (pretrain
at scale, then generalize across tasks), different data. So we are NOT using an LLM.

Two terms defined here so later slides can lean on them:
  • zero-shot — run the pretrained model on a new log, no training.
  • fine-tuning — keep training it on task data. (LoRA / full variants defined later, S11/S15.)

45s. Transition OUT (to S9 — Why TSFMs for PMF): "Same recipe as LLMs. So why would that help
*our* problem?"
-->

---
layout: assertion-evidence
locator: The bet
assertion: Foundation models are built not to overfit
---

<div class="s9-flow">
  <!-- ROW 1 — our data: complex + short (number dropped; "small" sharpened to short series → few samples) -->
  <div class="s9-row">
    <div class="s9-tag">our data</div>
    <div class="s9-line">
      <div class="s9-sub">
        <span class="s9-lead">complex signal · <strong class="s9-em">short series (few samples)</strong></span>
      </div>
      <div class="s9-sub">
        <span class="s9-arr">→</span>
        <span class="s9-out s9-out--bad">from-scratch models <strong>overfit</strong></span>
      </div>
    </div>
  </div>

  <!-- pivot connector -->
  <div class="s9-so">so</div>

  <!-- THE BET — the focal claim: a foundation model brings a prior -->
  <div class="s9-row s9-row--bet">
    <div class="s9-tag s9-tag--brand">the bet</div>
    <div class="s9-line">
      <div class="s9-sub">
        <span class="s9-lead s9-lead--brand">pretrained on</span>
        <span class="s9-chip s9-chip--big">millions of diverse series</span>
      </div>
      <div class="s9-sub">
        <span class="s9-arr s9-arr--brand">→</span>
        <span class="s9-out s9-out--good">a <strong>broad prior</strong>, not a fresh fit</span>
      </div>
    </div>
  </div>
</div>

<div class="s9-caveat">No event logs in pretraining, <em>to our knowledge</em>.</div>

<style>
.s9-flow { margin-top: 40px; display: flex; flex-direction: column; align-items: center; gap: 10px }
.s9-row { display: grid; grid-template-columns: 110px 1fr; align-items: center; gap: 20px; width: 100%; max-width: 880px; border: 1.5px solid #cdd6e0; border-radius: 12px; padding: 18px 24px }
.s9-row--bet { border: 2px solid var(--brand); background: #f4f8fc }
.s9-tag { font-size: 16px; font-weight: 700; text-transform: uppercase; letter-spacing: .04em; color: var(--neutral) }
.s9-tag--brand { color: var(--brand) }
.s9-line { display: flex; flex-direction: column; align-items: flex-start; gap: 9px; font-size: 23px; color: var(--ink) }
.s9-sub { display: flex; align-items: center; gap: 14px; flex-wrap: wrap }
.s9-lead { font-weight: 600; color: var(--neutral) }
.s9-lead--brand { color: var(--brand) }
.s9-chip { font-size: 20px; font-weight: 700; border-radius: 7px; padding: 3px 12px; font-variant-numeric: tabular-nums }
.s9-chip--big { background: var(--brand); color: #fff; letter-spacing: .01em }
.s9-arr { color: var(--neutral); font-weight: 800 }
.s9-arr--brand { color: var(--brand) }
.s9-em { color: var(--ink); font-weight: 700 }
.s9-out--bad { color: var(--neutral) }
.s9-out--bad strong { color: var(--ink) }
.s9-out--good { color: var(--ink) }
.s9-out--good strong { color: var(--brand) }
.s9-so { font-size: 18px; font-style: italic; font-weight: 600; color: var(--neutral); letter-spacing: .04em }
.s9-caveat { margin-top: 26px; text-align: center; font-size: 21px; color: var(--neutral) }
.s9-caveat em { color: var(--ink); font-style: italic; font-weight: 600 }
</style>

<!--
[S-09]
Q: Why should a generic forecaster beat a specialized one?

Transition IN (from S8): "Same recipe as LLMs. So why would that help *our* problem?"

Callback to S7: our DF series are both COMPLEX and SMALL — only 307–726 daily points per log —
exactly where a model trained from scratch overfits.

LOAD-BEARING line (say aloud): "Specialized models overfit on small heterogeneous PMF data.
Foundation models, pretrained on millions of diverse series, are designed to not."

Caveat (critique constraint): "No event logs in pretraining, to our knowledge" — so this is
genuinely zero-shot transfer; we test whether generic forecasting knowledge carries to DF series.
(Say "no event logs in pretraining, to our knowledge" — NOT "no process data".)

~35s. Transition OUT (to S10 — Three questions): "We have a candidate. Here's what we ask of it."
-->

---
layout: center
---

<div class="s10">

<div class="s10-heading">Three questions this talk answers</div>
<div class="s10-rule"></div>

<div class="s10-list">

  <div class="s10-q">
    <span class="s10-num">1</span>
    <div class="s10-text">Can <strong>zero-shot</strong> TSFMs give better <strong>DF time-series forecasts</strong>?</div>
  </div>

  <div class="s10-q">
    <span class="s10-num">2</span>
    <div class="s10-text">Does <strong>fine-tuning</strong> improve the results further?</div>
  </div>

  <div class="s10-q s10-q--open">
    <span class="s10-num s10-num--amber">3</span>
    <div class="s10-text">Does a better forecast give us a <strong>better forecasted process model</strong>?</div>
  </div>

</div>

</div>

<style>
.s10 { max-width: 1040px; margin: 0 auto; }
.s10-heading { font-size: 40px; font-weight: 700; color: var(--brand); line-height: 1.12; letter-spacing: -0.01em; }
.s10-rule { width: 72px; height: 5px; background: var(--accent); border-radius: 3px; margin: 20px 0 44px; }
.s10-list { display: flex; flex-direction: column; gap: 32px; }
.s10-q { display: flex; align-items: flex-start; gap: 24px; }
.s10-num {
  flex: none; width: 48px; height: 48px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  background: var(--brand); color: #fff;
  font-weight: 700; font-size: 25px; line-height: 1; font-variant-numeric: tabular-nums;
}
.s10-num--amber { background: var(--accent); color: var(--ink); }
.s10-text { font-size: 30px; line-height: 1.4; color: var(--ink); padding-top: 7px; }
.s10-text strong { color: var(--brand); font-weight: 700; }
.s10-q--open .s10-text strong { color: #b06317; }
</style>

<!--
[S-10]
Q: What three things does this talk actually test?

Transition IN (from S9): "We have a candidate. Here's what we ask of it."

30s scaffold for the results. Names (Chronos / MOIRAI / TimesFM, LoRA, Entropic Relevance)
land on later slides, not here. Standard terminology only: zero-shot (NOT off-the-shelf),
fine-tuning (NOT adapting).

Q1 is forecasting accuracy vs the strongest PMF baselines ONLY (seasonal-naive + tuned
XGBoost) — the on-slide line is trimmed to "give better DF time-series forecasts?"; say the
"vs the strongest PMF baselines" comparator ALOUD. Do NOT claim process-model quality here —
that is Q3 / ER.
Q2 ("further") = does fine-tuning add on top of the zero-shot win.
Q3 (amber) = the open one: does a better forecast translate to a better forecasted process model?

Maps to the results (speaker-notes only): Q1 -> S13 (zero-shot beats both baselines),
Q2 -> S15 (fine-tuning marginal), Q3 -> S16 (ER parity).

Transition OUT (into S11 — The candidates): "Three questions. Here's what we point at them."
-->

---
layout: assertion-evidence
locator: Model coverage
assertion: 3 model families, 12 variants, 3 settings
---

<div class="s11-evidence">

<svg viewBox="-70 0 1010 430" width="100%" style="display:block">
<line x1="95" y1="40" x2="95" y2="345" stroke="var(--neutral)" stroke-width="1.5"/>
<line x1="95" y1="345" x2="760" y2="345" stroke="var(--neutral)" stroke-width="1.5"/>
<text transform="translate(28,192) rotate(-90)" text-anchor="middle" style="font-size:16px" fill="var(--neutral)" font-weight="600">model size (params, log)</text>
<text x="428" y="393" text-anchor="middle" style="font-size:16px" fill="var(--neutral)" font-weight="600">release date</text>
<line x1="95" y1="320.6" x2="760" y2="320.6" stroke="var(--neutral-soft)" stroke-width="1" stroke-dasharray="2 4" opacity="0.5"/>
<text x="85" y="324.6" text-anchor="end" style="font-size:16px" fill="var(--neutral)">10M</text>
<line x1="95" y1="162.8" x2="760" y2="162.8" stroke="var(--neutral-soft)" stroke-width="1" stroke-dasharray="2 4" opacity="0.5"/>
<text x="85" y="166.8" text-anchor="end" style="font-size:16px" fill="var(--neutral)">100M</text>
<line x1="95" y1="52.5" x2="760" y2="52.5" stroke="var(--neutral-soft)" stroke-width="1" stroke-dasharray="2 4" opacity="0.5"/>
<text x="85" y="56.5" text-anchor="end" style="font-size:16px" fill="var(--neutral)">500M</text>
<text x="95.0" y="371" text-anchor="middle" style="font-size:16px" fill="var(--neutral)" font-weight="600">2024</text>
<text x="427.5" y="371" text-anchor="middle" style="font-size:16px" fill="var(--neutral)" font-weight="600">2025</text>
<text x="760.0" y="371" text-anchor="middle" style="font-size:16px" fill="var(--neutral)" font-weight="600">2026</text>
<rect x="387.3" y="106.6" width="14" height="236.3" rx="7" fill="var(--family-chronos)" opacity="0.14"/>
<rect x="387.3" y="106.6" width="14" height="236.3" rx="7" fill="none" stroke="var(--family-chronos)" stroke-width="1" opacity="0.40"/>
<rect x="254.2" y="78.0" width="14" height="226.5" rx="7" fill="var(--family-moirai)" opacity="0.14"/>
<rect x="254.2" y="78.0" width="14" height="226.5" rx="7" fill="none" stroke="var(--family-moirai)" stroke-width="1" opacity="0.40"/>
<circle cx="394.3" cy="335.8" r="5" fill="var(--family-chronos)"/>
<circle cx="394.3" cy="269.7" r="5" fill="var(--family-chronos)"/>
<circle cx="394.3" cy="213.1" r="5" fill="var(--family-chronos)"/>
<circle cx="394.3" cy="113.6" r="5" fill="var(--family-chronos)"/>
<circle cx="693.5" cy="150.3" r="11" fill="none" stroke="var(--accent)" stroke-width="3.5"/>
<circle cx="693.5" cy="150.3" r="6.5" fill="var(--family-chronos)"/>
<circle cx="261.2" cy="297.5" r="5" fill="var(--family-moirai)"/>
<circle cx="261.2" cy="85.0" r="5" fill="var(--family-moirai)"/>
<circle cx="354.3" cy="173.1" r="5" fill="var(--family-moirai)"/>
<circle cx="627.0" cy="311.6" r="11" fill="none" stroke="var(--accent)" stroke-width="3.5"/>
<circle cx="627.0" cy="311.6" r="6.5" fill="var(--family-moirai)"/>
<circle cx="221.4" cy="115.3" r="5" fill="var(--family-timesfm)"/>
<circle cx="434.1" cy="52.5" r="5" fill="var(--family-timesfm)"/>
<circle cx="663.6" cy="115.3" r="11" fill="none" stroke="var(--accent)" stroke-width="3.5"/>
<circle cx="663.6" cy="115.3" r="6.5" fill="var(--family-timesfm)"/>
<text x="394.3" y="98.6" text-anchor="middle" style="font-size:18px" font-weight="700" fill="var(--family-chronos)">Chronos Bolt</text>
<text x="710.5" y="155.3" text-anchor="start" style="font-size:18px" font-weight="700" fill="var(--family-chronos)">Chronos-2</text>
<text x="261.2" y="70.0" text-anchor="middle" style="font-size:18px" font-weight="700" fill="var(--ink)">MOIRAI-1.1</text>
<text x="340.3" y="177.1" text-anchor="end" style="font-size:18px" font-weight="700" fill="var(--ink)">MOIRAI-MoE</text>
<text x="340.3" y="195.1" text-anchor="end" style="font-size:14px" fill="var(--neutral)">86M active · 935M total</text>
<text x="644.0" y="316.6" text-anchor="start" style="font-size:18px" font-weight="700" fill="var(--ink)">MOIRAI-2.0</text>
<text x="209.4" y="120.3" text-anchor="end" style="font-size:18px" font-weight="700" fill="var(--family-timesfm)">TimesFM-1.0</text>
<text x="448.1" y="56.5" text-anchor="start" style="font-size:18px" font-weight="700" fill="var(--family-timesfm)">TimesFM-2.0</text>
<text x="680.6" y="120.3" text-anchor="start" style="font-size:18px" font-weight="700" fill="var(--family-timesfm)">TimesFM-2.5</text>
</svg>

<div class="s11-strip">
  <span class="s11-pill s11-pill--zs">zero-shot <span class="ct">12</span></span>
  <span class="s11-pill">Low-Rank Adaptation (LoRA) <span class="ct">4</span></span>
  <span class="s11-pill">full fine-tuning <span class="ct">5</span></span>
</div>

</div>

<style>
.s11-evidence { margin-top:-16px }
.s11-evidence svg { display:block; max-width:940px; margin:0 auto }
.s11-strip { display:flex; justify-content:center; gap:14px; margin-top:12px }
.s11-pill { font-size:19px; border:1.5px solid var(--neutral-soft); border-radius:999px; padding:6px 18px; color:var(--neutral); font-weight:600 }
.s11-pill .ct { color:var(--ink); font-weight:800 }
.s11-pill--zs { border-color:var(--brand); color:var(--brand); background:#f4f8fc }
.s11-pill--zs .ct { color:var(--brand) }
</style>

<!--
[S-11]
Q: Which models, and what settings?

Key message: we tried a lot — 3 model families across 3 settings (12 zero-shot variants).

Names stay ON SCREEN, not in the voice (attention-risk #1 — let the timeline carry the load).
Say the shape, not the roster: each family has shipped several generations; the latest of each
(amber) is what the results use, and the trend is "newer = smaller, yet better" — MOIRAI-2.0 is just
~11.4M params. univariate throughout (Yu 2025). LoRA = Low-Rank Adaptation, small trainable adapters
on attention (full form is on the slide). 45s.

Transition IN (from S10 three questions): "Three questions. Here's what we point at them."
Transition OUT (to S12 experimental setup): "That's the coverage — now exactly how we evaluated it."
(The "...no event logs in pretraining ... point them at DF time series" line belongs at the results
entry, S13, not here.)
-->

---
layout: assertion-evidence
locator: Experimental setup
assertion: Each step adds one day, then re-forecasts the next seven
---

<div class="s12-wrap mt-0">
  <div class="s12-schem">
    <div class="s12-schem-title">Expanding window · stride = 1 day · 7-day horizon</div>
    <svg viewBox="0 0 600 224" class="s12-svg" role="img" aria-label="Expanding-window evaluation: history grows one day per step, each step forecasts 7 days">
      <rect x="72" y="26" width="188" height="26" rx="4" fill="#00407a"/>
      <rect x="260" y="26" width="84" height="26" rx="4" fill="#dd8a2e"/>
      <rect x="72" y="66" width="214" height="26" rx="4" fill="#00407a"/>
      <rect x="286" y="66" width="84" height="26" rx="4" fill="#dd8a2e"/>
      <rect x="72" y="106" width="240" height="26" rx="4" fill="#00407a"/>
      <rect x="312" y="106" width="84" height="26" rx="4" fill="#dd8a2e"/>
      <rect x="72" y="146" width="266" height="26" rx="4" fill="#00407a" opacity="0.4"/>
      <rect x="338" y="146" width="84" height="26" rx="4" fill="#dd8a2e" opacity="0.4"/>
      <text x="26" y="99" class="s12-side" text-anchor="middle" transform="rotate(-90 26 99)">+1 day per step</text>
      <line x1="72" y1="188" x2="560" y2="188" class="s12-axis"/>
      <text x="558" y="206" class="s12-axis-lbl" text-anchor="end">Day →</text>
    </svg>
    <div class="s12-legend">
      <span><i class="s12-sw s12-sw--navy"></i>history</span>
      <span><i class="s12-sw s12-sw--amber"></i>7-day forecast</span>
    </div>
  </div>
  <div class="s12-strip">
    <span class="s12-pill">daily DF aggregation</span>
    <span class="s12-pill">baselines · seasonal-naive + XGBoost</span>
    <span class="s12-pill">univariate inference</span>
  </div>
</div>

<style>
.s12-wrap { position: relative }
.s12-schem { display: flex; flex-direction: column; align-items: center }
.s12-schem-title { font-size: 18px; font-weight: 700; color: var(--brand); margin-bottom: 20px }
.s12-svg { width: 690px; max-width: 100%; height: auto }
.s12-side { font-size: 16px; fill: var(--neutral); font-weight: 700 }
.s12-axis { stroke: var(--neutral-soft); stroke-width: 1.5 }
.s12-axis-lbl { font-size: 16px; fill: var(--neutral) }
.s12-legend { display: flex; gap: 28px; font-size: 16px; color: var(--neutral); margin-top: 6px }
.s12-sw { display: inline-block; width: 14px; height: 14px; border-radius: 3px; margin-right: 7px; vertical-align: -2px }
.s12-sw--navy { background: #00407a }
.s12-sw--amber { background: #dd8a2e }
.s12-strip { display: flex; flex-wrap: wrap; gap: 12px; justify-content: center; margin-top: 32px }
.s12-pill { font-size: 16px; color: var(--ink); background: #eef2f6; border: 1px solid #cdd6e0; border-radius: 8px; padding: 6px 14px; font-weight: 500 }
</style>

<!--
[S-12]
Q: How exactly did you evaluate? (And was it fair?)

Transition IN (from S11 candidates): "Before the results, one slide on exactly how we tested."

This slide OWNS the window/stride detail kept off S4. Reconciliation: S4 says only "aggregate
daily, forecast 7 days ahead"; S12 adds the protocol — expanding window, stride = 1 day. Nothing
on S4 contradicts this.

Deliver the assumption -> design-choice mapping ALOUD (it is intentionally NOT on the slide):
  - you re-plan as new data arrives each day  -> stride = 1 day
  - you care about the coming week            -> 7-day horizon
  - you never throw history away              -> expanding (not sliding) window
Fairness (the Callout / key message): the SAME expanding window and 7-day horizon apply to every
model — baselines and TSFMs alike — and the two baselines are the STRONGEST from our prior
benchmark (seasonal-naive + XGBoost), not weak strawmen. Univariate inference throughout.

45s. Transition OUT (to S13 — zero-shot results): "With that setup, here are the zero-shot results."
-->

---
layout: assertion-evidence
locator: Results
assertion: The latest zero-shot TSFMs beat both baselines on every log
---

<div class="text-center mb-2" style="font-size:15px;color:var(--neutral)">
  Mean Absolute Error (MAE), lower is better ·
  <span style="color:var(--baseline);font-weight:600">gray = baselines</span> ·
  color = latest TSFM per family
</div>

<div class="relative">
  <img src="/figures/results-mae-bars.png" class="block mx-auto w-full max-h-[400px] object-contain rounded-lg" alt="Zero-shot MAE across 4 event logs — gray baselines (Seasonal-Naive, XGBoost) vs the latest TSFM of each family (Chronos-2, MOIRAI-2.0, TimesFM-2.5); colored bars lower on every log" />
  <div class="absolute" style="left:0;top:50%;transform:translateY(-50%)" v-click="1">
    <Callout style="text-align:center;line-height:1.5">−21% mean MAE<br/>vs best baseline</Callout>
  </div>
  <div class="absolute" style="right:0;top:50%;transform:translateY(-50%)" v-click="2">
    <Callout style="text-align:center;line-height:1.5">All 12 variants:<br/>92% beat baseline<br/>−15% mean MAE</Callout>
  </div>
</div>

<!--
[S-13]
Q: Does Q1 hold — do zero-shot TSFMs beat the baselines?

Transition in: "Three families. Twelve variants. No event logs in pretraining.
Here's what happens when you point them at DF time series."

~2 min. Say "Mean Absolute Error" on first use (kicker carries MAE). CRITIQUE CONSTRAINT —
open with the baseline framing verbatim (the on-slide caption was dropped; deliver it aloud):
"We compare against the two strongest baselines from our prior benchmark; full ranking
on backup." Pre-empts cherry-picking.

Callouts (click 1 then 2): (1) the 3 latest TSFMs average −21% MAE vs the best baseline.
(2) across ALL twelve zero-shot variants × 4 logs, 92% of results beat the best baseline
(−15% mean); only four older/smaller models miss, all on BPI2017.

The three latest models (Chronos-2, MOIRAI-2.0, TimesFM-2.5) win MAE on all four logs —
including Sepsis (MOIRAI-2.0 ↓28%). Sepsis is NOT a MAE exception; it only becomes the
hard case later, on ER (beat 9). Don't undercut the MAE win here.

Q&A defense: some older/smaller variants don't beat the baselines on BPI2017 (paper
p.11) — that's why the bars show the latest of each family.

Transition out (into 7b): "Same data, same window — here's what that win looks like."
-->

---
layout: assertion-evidence
locator: Results
assertion: Zero-shot tracks the drift, stays controlled on sparsity
---

<div class="grid grid-cols-2 gap-8 mt-3" style="height: 330px">

  <!-- LEFT — drift: reveal MOIRAI-2.0 + amber adaptation box -->
  <div class="flex flex-col">
    <div class="s14-panel-label">Drift: tracked</div>
    <div class="s14-stack">
      <img src="/figures/s6-drift-xgb.png" v-click.hide="1" class="s14-img"
           alt="BPI2017 drift — XGBoost stays high while truth collapses late" />
      <img src="/figures/s14-drift-tsfm.png" v-click="1" class="s14-img s14-img--over"
           alt="BPI2017 drift revealed — MOIRAI-2.0 dips toward the truth in the boxed back-half where XGBoost stays elevated" />
    </div>
    <div class="s14-cap">BPI2017 · Sent → Canceled</div>
  </div>

  <!-- RIGHT — sparsity: reveal MOIRAI-2.0 + amber arrow to the zero line -->
  <div class="flex flex-col">
    <div class="s14-panel-label">Sparsity: controlled</div>
    <div class="s14-stack">
      <img src="/figures/s6-intermittent-xgb.png" v-click.hide="1" class="s14-img"
           alt="BPI2019-1 intermittent — XGBoost hallucinates 40–71 where truth is mostly zero" />
      <img src="/figures/s14-intermittent-tsfm.png" v-click="1" class="s14-img s14-img--over"
           alt="BPI2019-1 intermittent revealed — MOIRAI-2.0 holds near zero (amber arrow), no false bursts, no spikes" />
    </div>
    <div class="s14-cap">BPI2019-1 · Cancel → Record Invoice Receipt</div>
  </div>

</div>

<div class="mt-4 text-center" style="font-size: 20px; color: #334155; line-height: 1.5">
  <div>Drift: misses the drop, then <strong style="color: var(--brand)">adapts</strong>. Online context, no retraining.</div>
  <div>Sparsity: holds near zero. No false bursts, but <strong>no spikes either</strong>.</div>
</div>

<style>
.s14-panel-label { font-size: 18px; font-weight: 700; color: var(--brand); text-align: center; margin-bottom: 6px }
.s14-cap { font-size: 14px; color: var(--neutral); text-align: center; margin-top: 6px }
.s14-stack { position: relative; width: 100%; max-height: 270px }
.s14-img { display: block; width: 100%; object-fit: contain; max-height: 270px; border-radius: 8px }
.s14-img--over { position: absolute; inset: 0 }
</style>

<!--
[S-14]
Q: What does the zero-shot win look like on real DF patterns?

Transition IN (from S13 bars): "Same data, same window — here's what that win looks like."

THE CALLBACK MOMENT — the audience remembers these two plots from S5/S6 (truth only, then XGBoost
failing). One click reveals the MOIRAI-2.0 line on BOTH, with the amber marks. Don't rush the click.
  • Drift (left, amber box): "Same plot, same window. The TSFM tracks the drop XGBoost missed."
    MOIRAI-2.0 misses the initial drop, then catches up at the troughs — online adaptation, NO
    retraining. The box frames where it comes down while XGBoost stays stuck high.
  • Sparsity (right, amber arrow) — the HONEST beat: MOIRAI-2.0 holds near zero. It avoids
    XGBoost's false bursts (XGBoost hallucinated 40–71 where truth is mostly 0; MOIRAI-2.0 MAE 2.0,
    −89%), but it does NOT catch the rare spikes. The arrow marks it: down to ~0, still present.
    "Not a miracle spike detector — on a near-empty signal, the honest answer is to stay near zero."

Mechanism (if asked): the TSFM adapts via an EXPANDING historical-context window at inference —
no retraining (paper §4.1).

~1 min. Transition OUT (to S15 fine-tuning): "TSFMs win zero-shot. Natural next question: can we
make them better?"
-->

---
layout: assertion-evidence
locator: Fine-tuning
assertion: Fine-tuning barely helps
---

<div class="flex items-center gap-5 mt-2">

<div style="flex: 1">
<img src="/figures/ft-slope.png" class="block w-full object-contain rounded-lg" style="max-height: 500px" alt="MAE relative to zero-shot across LoRA and full fine-tuning — most lines barely move; full fine-tuning sometimes overfits (amber), worst +87%" />
</div>

<div style="flex: 0 0 23%" v-click class="space-y-8">
<Callout dir="left" style="line-height:1.5">53% of fine-tuning runs got worse</Callout>
<div class="font-semibold text-xl" style="color: var(--brand)">Skip fine-tuning at PMF data scale.</div>
</div>

</div>

<!--
[S-15] Q: Does fine-tuning (Q2) help? Is it worth it?

~1 min 30s. The clean-negative beat before S16's low-energy ER concession.
Transition in: "TSFMs win zero-shot. Natural next question — can we make them better?"

Panel order (logs are small on screen — name them aloud): top-left BPI2017, top-right BPI2019-1,
bottom-left Sepsis, bottom-right Hospital Billing.

Read the slope: most lines hug the 1.0 zero-shot baseline (gray bundle — fine-tuning barely moves
accuracy), while a few full-FT lines shoot up in amber (overfitting on small logs):
MOIRAI-1.1-R-large +87% on BPI2019-1, +31% on Sepsis; Chronos-2 +16% on Sepsis. Click reveals the
tally: across all 36 LoRA + full fine-tuning runs, 19 (53%) landed worse than zero-shot. The few real
gains (Sepsis full-FT −10 to −13%) are small and dataset-dependent.

COST — speak it, do NOT put on slide (no wall-clock logged, no fabricated ratio):
"And it never comes for free. Zero-shot is one forward pass per window. LoRA and full fine-tuning each
add a whole training stage — repeated for every model and every log — and in the worst case full
fine-tuning nearly doubled the error. The accuracy payoff is a coin-flip and sometimes catastrophic.
So at PMF data scale, skip it."

Transition out: "Fine-tuning isn't the win. So does the forecasting win even translate into a better
process model?"
-->

---
layout: assertion-evidence
locator: Evaluation
assertion: Forecasting beats reuse, but better forecasts don't make better models
---

<div class="text-center mb-3" style="font-size:19px;max-width:1040px;margin-left:auto;margin-right:auto">
  <b style="color:var(--ink)">Entropic Relevance (ER):</b> <span style="color:var(--neutral)">average bits to encode one log trace with the forecasted process model ·</span> <b style="color:var(--brand)">lower = better</b>
</div>

<div class="relative">
  <img src="/figures/er-hospital-billing.png" class="block mx-auto w-full max-h-[380px] object-contain rounded-lg" alt="Hospital Billing ER — reusing the historical (Training) model is far worse than any forecast; the five forecasts cluster close together" />
  <div class="absolute" style="left:40%;top:14%" v-click="1">
    <Callout dir="left">Forecast ≫ reuse</Callout>
  </div>
  <div class="absolute" style="left:58%;top:40%" v-click="2">
    <Callout dir="down">TSFMs ≈ baselines</Callout>
  </div>
</div>

<div class="caption text-center mt-2" style="color:var(--neutral)">Hospital Billing · Truth = ideal floor · Training = reuse the historical model</div>

<!--
[S-16]
Q: Does the forecasting win translate into a better process model?

Transition in: "But forecasting accuracy isn't the only thing we care about in PM."
OPEN WITH THE QUESTION, not the result — S14→S16 is the talk's lowest-energy moment.

Introduce the term: say "Entropic Relevance" in full on first use (the on-slide line carries it
full-form-first); use "ER" thereafter. One line: ER = expected bits to encode a log trace under the
forecasted model; lower = better; Truth is the floor.

Two findings (revealed in sequence with the two callouts):
  1. v-click 1 — Forecast ≫ reuse: the Training bar (reuse the historical/discovered model) towers at
     5.83 while EVERY forecast lands ~2.1–2.5 → forecasting has value (callback to discovery's static
     model in S2).
  2. v-click 2 — TSFMs ≈ baselines: the three FMs (2.43 / 2.52 / 2.39) are no better than the two
     baseline forecasts (2.17 / 2.12); the differences are small → a better forecast did NOT yield a
     better process model (answers Q3).

~1 min 30s. THE LOAD-BEARING SLIDE.
Memorized rebuttal — deliver in SPEECH, not on slide (verbatim from SLIDES.md):

"ER parity means we're not producing better DFG structures — we're producing
DFGs with better edge weights. For tasks where the forecast itself is the
deliverable — capacity planning, drift detection, anomaly baselines —
edge-weight accuracy IS the contribution. For tasks where you need a different
process model, ER says the bottleneck is now the DFG representation,
not the forecaster."

Do not skip this line. Rehearse it cold.

Transition out (hands the "bottleneck has moved" framing to S17): "So the forecasting is solved —
which means the open problem has moved somewhere else."
-->

---
layout: assertion-evidence
locator: Discussion
assertion: The bottleneck has moved from forecasting accuracy to process-aware representation
---

<div class="s17">
<div class="s17-rows">
<div class="s17-row" v-click="1">
<span class="s17-t">Beyond control flow</span> <span class="s17-sep">—</span> <span class="s17-d">add resources, decisions, richer relations</span>
</div>
<div class="s17-row" v-click="1">
<span class="s17-t">Smarter fine-tuning</span> <span class="s17-sep">—</span> <span class="s17-d">lightweight, incremental tuning for PMF</span>
</div>
<div class="s17-row" v-click="1">
<span class="s17-t">Larger log corpora</span> <span class="s17-sep">—</span> <span class="s17-d">bigger, higher-quality logs for a process-native model</span>
</div>
</div>
</div>

<style>
.s17 { margin-top: 70px }
.s17-rows { display: flex; flex-direction: column; gap: 48px }
.s17-row { border-left: 4px solid var(--brand); padding: 4px 0 4px 20px; font-size: 23px; line-height: 1.35 }
.s17-t { font-weight: 700; color: var(--ink) }
.s17-sep { color: var(--neutral); font-weight: 700; margin: 0 4px }
.s17-d { color: var(--brand); font-weight: 600 }
</style>

<!--
[S-17]
Q: Where does process model forecasting go from here?

Transition in (from S16): "So the forecasting is solved — the open problem has moved somewhere else."

LEAD WITH THE HEADER (say it first): the bottleneck has moved from forecasting accuracy
(solved — zero-shot TSFMs beat the baselines) to PROCESS-AWARE REPRESENTATION. DFGs capture
only the control-flow / workflow aspect — a lossy target (callback to S16's ER parity). This
slide OWNS the "bottleneck has moved" line — S18 must not repeat it.

Then the three future directions (paper §Discussion):
1. Beyond control flow — directly-follows is a control-flow-only view; enrich to other process
   dimensions (resources, bottlenecks, decisions) and richer relations (loops, long-range deps).
2. Smarter adaptation — fine-tuning gains were marginal at this data scale; resource/method
   constraints limited what we tried; better adaptation methods are open.
3. Larger log corpora — only four logs; larger, higher-quality collections would generalize the
   findings and could support a process-native foundation model.

Also mention aloud (kept off-slide): concept drift is a key challenge — combine drift detection
with incremental / lightweight fine-tuning for adaptive retraining (paper §Discussion).

~1 min. Transition out (to S18 Takeaways): "So what do we actually walk away with?"
-->

---
layout: assertion-evidence
locator: Takeaways
assertion: Off-the-shelf forecasting is strong enough to build on
---

<div class="s18">

  <!-- For Process Model Forecasting — the two PMF-specific signals, boxed (four-logs hedge stays spoken) -->
  <div class="s18-prac s18-pmf">
    <div class="s18-prac-label">For process mining</div>
    <div class="s18-pmf-rows">
      <div class="s18-pmf-row"><strong>Zero-shot</strong> is the new default of PMF&ensp;·&ensp;<strong>Fine-tuning is marginal</strong> at this data scale</div>
    </div>
  </div>

  <!-- For practitioners — applied guidance (brand-tinted banner) -->
  <div class="s18-prac">
    <div class="s18-prac-label">For practitioners</div>
    <div class="s18-prac-body">
      These TSFMs are <strong>tiny beside LLMs</strong>: deploy locally, forecast fast on a
      laptop, with little compute.
    </div>
  </div>

  <!-- General CTA — the broadening close, the one amber Callout, on click -->
  <div class="flex justify-center mt-3 s18-cta" v-click>
    <Callout>Try TSFMs for your forecasts</Callout>
  </div>

</div>

<style>
.s18 { display: flex; flex-direction: column; gap: 22px; margin-top: 6px }
.s18-prac { border: 2px solid var(--brand); background: #f4f8fc; border-radius: 12px; padding: 18px 22px }
.s18-prac-label { font-size: 14px; font-weight: 700; color: var(--neutral); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 9px }
.s18-prac-body { font-size: 23px; color: var(--ink); line-height: 1.4 }
.s18-prac-body strong { color: var(--brand) }
.s18-pmf-rows { display: flex; flex-direction: column; gap: 8px }
.s18-pmf-row { font-size: 23px; color: var(--ink); line-height: 1.35 }
.s18-pmf-row strong { color: var(--brand); font-weight: 700 }
.s18-cta .ae-callout { font-size: 27px !important; font-weight: 700 !important; padding: 11px 24px !important }
</style>

<!--
[S-18]
Q: What do I walk away with — and what do I do next?

Transition in: "So what do we walk away with?"  ~75s. Two audiences: practitioners (what to do)
and the field (what we learned).

Practitioner — ONE point: these models are small. Run them on a laptop or a cheap GPU, get a
forecast fast, little compute. You CAN fine-tune, but at PMF data scale it's rarely worth it.

Two signals:
  1. Zero-shot is the new PMF default. CRITIQUE CONSTRAINT — say the hedge verbatim:
     "four logs is not a paradigm" — but a strong enough signal that the right new default is to
     TRY a zero-shot TSFM first.
  2. Fine-tuning is marginal at this data scale (callback to S15's flat slope).

Do NOT say the "bottleneck has moved" line here (that's S17). No artifact/code links here (S19).

CLOSING ANCHOR — rehearse COLD, deliver in SPEECH (not on slide):
"Zero-shot TSFMs are the new PMF default. Four logs is not a paradigm — but it is a strong enough
signal that the right new default is to try a zero-shot TSFM first. The deeper signal is that
time-series forecasting just became cheap enough that process mining can borrow from it without
paying the training tax. What else can we borrow?"

Transition out (to S19 artifacts): "And all of this is yours to run."
-->

---
layout: two-col-evidence
locator: Artifacts
assertion: Public, reproducible — and live to try right now
---

::left::

<!-- Live deck: poster frame → ONE click swaps in the screencast, which plays once (no loop).
     PDF/PPTX export fallback (e.g. the 5-shot carousel) to be designed later. -->
<div class="demo-vstack">
<img src="/figures/demo-screencast-poster.png" v-click.hide="1" class="demo-vlayer" alt="Demo app — click to play the ~29s live walkthrough" />
<SlidevVideo v-click="1" autoplay="once" class="demo-vlayer demo-vlayer--over" poster="/figures/demo-screencast-poster.png" print-poster="/figures/demo-screencast-poster.png">
<source src="/figures/demo-screencast.mp4" type="video/mp4" />
</SlidevVideo>
</div>

<div class="caption demo-cap-one">Live app · ~29 s walkthrough — click to play (once)</div>

<style>
.demo-vstack { position: relative; width: 100%; aspect-ratio: 16 / 10; border: 1px solid var(--hairline); border-radius: 8px; overflow: hidden; background: #fff }
.demo-vlayer { display: block; width: 100%; height: 100%; object-fit: contain }
.demo-vlayer--over { position: absolute; inset: 0 }
.demo-cap-one { text-align: center; margin-top: 8px }
</style>

::right::

<!-- live demo = QR block (QR + readable URL in one): top of column, the single amber emphasis -->
<div class="border-2 rounded-lg flex items-center gap-4 px-4 py-3" style="border-color: var(--accent); background: var(--surface-alt)">
<img src="/figures/demo-qr.png" class="w-[120px] h-[120px]" alt="QR to the live demo on Hugging Face Spaces" />
<div>
<div class="font-semibold" style="color: var(--brand)">Live demo — scan or visit</div>
<div class="dense--xs">huggingface.co/spaces/YongboYu/pmf-tsfm-demo</div>
<div class="dense--xs" style="color: var(--neutral)">Chronos-2 on HF ZeroGPU</div>
</div>
</div>

<div class="space-y-3 mt-4">

<div class="border rounded-lg px-4 py-2" style="border-color: var(--hairline)">
<div class="font-semibold" style="color: var(--brand)">Code</div>
<div class="dense--xs">github.com/YongboYu/pmf-tsfm</div>
</div>

</div>

<div class="flex gap-2 mt-4 dense--xs" style="color: var(--neutral)">
<span class="border rounded px-2 py-1" style="border-color: var(--hairline)">laptop · MPS</span>
<span class="border rounded px-2 py-1" style="border-color: var(--hairline)">Linux · CUDA</span>
<span class="border rounded px-2 py-1" style="border-color: var(--hairline)">HPC · H100 / Slurm</span>
</div>

<div class="mt-3 caption">yongbo.yu@kuleuven.be · arXiv:2512.07624</div>

<!--
[S-19]
~1.5–2 min. THE DEMO BEAT — the work is reproducible and usable.

Transition in: "Everything I've shown is public and runs out of the box."

ONE click plays the ~29s screencast (once, no loop) — a real recording of the live app.
What it walks through:
  • Forecast vs actual future — the bundled backtest, forecast DFG beside the held-out truth.
  • Accuracy — ER / MAE / RMSE against that real future (ER truth = the floor to beat).
  • Diff — where the forecast matched, missed (amber), or over-called (red): absolute, then relative %.
  • Upload your own log — the live ZeroGPU path, forecast vs the last-known window.
Talk over it as it plays (~29s); let it finish before the next click advances the slide.

Point at the QR: "scan it now — the live Space, Chronos-2 on HF ZeroGPU, is in your
pocket for the Q&A; upload your own log."

Reproducibility: the same code path runs on a laptop (MPS), a CUDA box, or the HPC
cluster (H100 via Slurm) — nothing in the paper needs special hardware.
-->

---
layout: end
class: ty-light
---

<style>
.slidev-layout.ty-light {
  background:
    radial-gradient(1100px 500px at 78% -8%, #eaf1f8 0%, rgba(234,241,248,0) 60%),
    var(--surface) !important;
  color: var(--ink) !important;
  text-align: center;
}
.slidev-layout.ty-light h1 { color: var(--brand) !important; }
</style>

# Thank you

<div class="text-2xl mt-4" style="color: var(--neutral)">
Questions, please.
</div>

<div class="mt-10 text-lg">
<a href="mailto:yongbo.yu@kuleuven.be" style="color: var(--brand); font-weight: 600">yongbo.yu@kuleuven.be</a>
<span style="color: var(--neutral-soft)">&nbsp;·&nbsp;</span>
<a href="https://github.com/YongboYu/pmf-tsfm" style="color: var(--brand); font-weight: 600">github.com/YongboYu/pmf-tsfm</a>
</div>

<img src="/logos/kuleuven-liris.png" class="h-12 mx-auto mt-12" alt="KU Leuven · LIRIS" />

<!--
[S-20]
10 min Q&A. Backup slides follow, hidden from ToC (each backup carries `hideInToc: true`).
Hostile-Q answers rehearsed cold — see SLIDES.md "Required backup slides" + the Q&A defense section.
-->

---
layout: assertion-evidence
class: backup
hideInToc: true
locator: Backup
assertion: Both baselines are the strongest from our prior benchmark
---

<div class="ae-cols">
  <div class="ae-col">
    <div class="ae-collabel">Seasonal-Naive</div>
    <div class="ae-lead">Strongest non-trained baseline</div>
    <ul>
      <li>7-day lag matches the weekly cadence</li>
    </ul>
  </div>
  <div class="ae-col">
    <div class="ae-collabel">XGBoost</div>
    <div class="ae-lead">Strongest trained ML baseline</div>
    <ul>
      <li>Recommended by the Yu 2025 benchmark</li>
    </ul>
  </div>
</div>

<div class="text-center mt-10" style="font-size:20px;color:var(--neutral)">Not cherry-picked — the full ranking lives in the Yu 2025 benchmark.</div>

<!-- speaker note: Honest answer to "why only two baselines?" We did not cherry-pick — these are the top non-trained and top trained methods from the external Yu 2025 benchmark. The full ranking table lives in that benchmark, not in this deck; happy to point to it offline. -->

---
layout: assertion-evidence
class: backup
hideInToc: true
locator: Backup
assertion: Latest zero-shot TSFMs lead MAE across all variants
---

<img src="/figures/mae-full.png" class="block mx-auto w-full max-h-[430px] object-contain rounded-lg" alt="Full MAE table across all four event logs (paper Table 4) — amber = best per log, gray = baseline rows; the latest zero-shot TSFM per family is best or competitive on every log" />

<div class="flex justify-center mt-3">
  <Callout>Best or competitive on every log — all 14 variants</Callout>
</div>

<!-- speaker note: The full MAE table behind the S13 headline bars — paper Table 4, all 14 variants across the four logs. The headline compared the latest TSFM per family against the two baselines; this is the complete field. The story holds variant by variant: the latest Chronos / MOIRAI / TimesFM are best or statistically competitive on every log, and the occasional baseline-level number sits with the small/older variants. Amber marks the best (lowest) per log; gray rows are the two baselines. -->

---
layout: assertion-evidence
class: backup
hideInToc: true
locator: Backup
assertion: RMSE confirms the MAE ranking on every log
---

<img src="/figures/rmse-full.png" class="block mx-auto w-full max-h-[430px] object-contain rounded-lg" alt="Full RMSE table across all four event logs (paper Table 5) — amber = best per log, gray = baseline rows; latest zero-shot TSFMs are best or competitive on every log" />

<div class="flex justify-center mt-3">
  <Callout>Latest zero-shot TSFMs best or competitive on all four logs</Callout>
</div>

<!-- speaker note: Same field, RMSE instead of MAE — sourced from the paper's dedicated RMSE table (Table 5), so it is camera-ready faithful. Switching the error metric does not change the story: the latest zero-shot TSFM per family is best or statistically competitive on every one of the four logs, the same ranking as MAE. -->

---
layout: assertion-evidence
class: backup
hideInToc: true
locator: Backup
assertion: LoRA and full fine-tuning rarely beat zero-shot
---

<img src="/figures/ft-table.png" class="block mx-auto w-full max-h-[430px] object-contain rounded-lg" alt="Fine-tuning results (paper Table 6) — MAE and RMSE for zero-shot, LoRA and full fine-tuning on five TSFMs across the four logs; gains over zero-shot are small and inconsistent, and MOIRAI-1.1-R-large full fine-tuning degrades sharply on BPI2019-1 (23.06 vs 12.30)" />

<div class="flex justify-center mt-3">
  <Callout>Gains are marginal and inconsistent — full fine-tuning can degrade</Callout>
</div>

<!-- speaker note: [S-backup-FT] Paper Table 6 — the detail behind the S15 slope. Five fine-tunable models (Chronos-2 has no LoRA, hence two rows); MAE and RMSE per log for zero-shot vs LoRA vs full tune, grouped by model. Gains over zero-shot are small and dataset-dependent, and several rows get worse — the extreme is MOIRAI-1.1-R-large full fine-tuning on BPI2019-1 (23.06 vs 12.30 zero-shot). This is the per-number evidence behind "skip fine-tuning at PMF data scale." -->

---
layout: assertion-evidence
class: backup
hideInToc: true
locator: Backup
assertion: ER reaches parity on three logs — Sepsis is the exception
---

<img src="/figures/er-bars.png" class="block mx-auto w-full max-h-[400px] object-contain rounded-lg" alt="Entropic Relevance across four logs — TSFMs match baselines on BPI2017, BPI2019-1 and Hospital Billing, but trail far behind on Sepsis (bottom-left panel)" />

<div class="flex justify-center mt-3">
  <Callout>Sepsis (bottom-left): ER far above the rest — TSFMs worst</Callout>
</div>

<!-- speaker note: The main ER slide showed only Hospital Billing; this generalizes it. On three of four logs TSFMs sit at parity with the baselines. Sepsis is the lone exception — it is both heterogeneous (many rare variants) and intrinsically hard to encode, so ER stays high. This sets up the next Sepsis-specific frame. -->

---
layout: assertion-evidence
class: backup
hideInToc: true
locator: Backup
assertion: Baselines were tuned; the fine-tuning recipe is fully specified
---

<div class="ae-cols">
  <div class="ae-col">
    <div class="ae-collabel">XGBoost</div>
    <div class="ae-lead">Tuned per dataset, not left at defaults</div>
    <ul>
      <li>Optuna hyperparameter optimization, per dataset</li>
      <li>Lagged features, daily aggregation</li>
    </ul>
  </div>
  <div class="ae-col">
    <div class="ae-collabel">LoRA / full fine-tuning</div>
    <div class="ae-lead">Fully specified, reproducible recipe</div>
    <ul>
      <li>LoRA r=2, α=4 on Q/K/V/O attention</li>
      <li>Patch 16 · batch 32 · AdamW · LR 1e-4 · 3 epochs</li>
      <li>Full fine-tuning uses each model's original recipe</li>
    </ul>
  </div>
</div>

<!-- speaker note: Pre-empts "was the baseline even tuned?" — yes, XGBoost gets per-dataset Optuna search over lagged daily-aggregated features, so the comparison is fair. And "what was the LoRA setup?" — rank 2, alpha 4 on the attention projections (Q/K/V/O), patch 16, batch 32, AdamW at 1e-4 for 3 epochs; full fine-tuning instead follows each model's own published recipe. Deliberately not quoting an Optuna trial count: the exact N isn't pinned in the repo, so I won't invent one — happy to confirm offline. -->

---
layout: assertion-evidence
class: backup
hideInToc: true
locator: Backup
assertion: DF series are hard on transition, shifting, non-Gaussianity
---

<div class="relative">
  <img src="/figures/df-complexity-radar.png" class="block mx-auto w-full max-h-[400px] object-contain rounded-lg" alt="Radar of seven complexity metrics for DF series — seasonality, trend, stationarity, transition, shifting, correlation, non-Gaussianity, with transition, shifting and non-Gaussianity markedly elevated" />
  <div class="absolute" style="right:0;top:50%;transform:translateY(-50%)">
    <Callout style="text-align:center">Elevated:<br/>Transition · Shifting<br/>· Non-Gaussianity</Callout>
  </div>
</div>

<div class="text-center mt-2" style="font-size:18px;color:var(--neutral)">Higher than typical public benchmarks (Li et al. 2025)</div>

<!-- speaker note: The full seven-metric profile — seasonality, trend, stationarity, transition, shifting, correlation, non-Gaussianity — backs the "DF series are unusually hard" claim. Three metrics stand out: frequent regime transitions, distribution shifting over time, and strong non-Gaussianity. The comparison to public forecasting benchmarks stays qualitative: we have no benchmark bars in the repo, so claim only that these three metrics run higher than typical public benchmarks (Li et al. 2025). -->

---
layout: assertion-evidence
class: backup
hideInToc: true
locator: Backup
assertion: Sepsis is an ER exception — extreme behavioral heterogeneity
---

<div class="dense" style="max-width:1000px;margin-left:auto;margin-right:auto">

<div class="ae-lead text-center">790 trace variants over 999 cases — huge variant tail, sparse DF activity</div>

<div class="text-center mt-4" style="font-size:16px;color:var(--neutral)">
  ER fitting ratio — higher = easier to encode · lower = harder
</div>

<div class="text-center mt-1" style="font-size:19px;color:var(--ink)">
  Truth 100% &nbsp;·&nbsp; Training 77.9% &nbsp;·&nbsp; Naive 39.4% &nbsp;·&nbsp; XGBoost 74.6%
</div>
<div class="text-center mt-1" style="font-size:19px;color:var(--brand);font-weight:700">
  Chronos-2 13.2% &nbsp;·&nbsp; MOIRAI-2.0 4.1% &nbsp;·&nbsp; TimesFM-2.5 17.5%
</div>

<div class="text-center mt-3" style="color:var(--neutral)">All models struggle; the TSFMs struggle most.</div>

</div>

<div class="flex justify-center mt-4">
  <Callout>ER exception only — Sepsis still wins on MAE (MOIRAI-2.0 ↓28%)</Callout>
</div>

<!-- speaker note: Sepsis is the ER outlier. 790 variants over 999 cases means an enormous long tail and very sparse DF activity per relation — so the forecasted model encodes traces poorly and fitting ratios collapse (TSFMs 4–18%). High heterogeneity (variant tail + sparse DF) drives the tiny ratios. Crucially this is ER only: on MAE, Sepsis is still a win — MOIRAI-2.0 cuts error 28%. Don't conflate the two metrics. Sources: stats_log.tex, results_ER.tex, results_1_mae.tex. -->

---
layout: assertion-evidence
class: backup
hideInToc: true
locator: Backup
assertion: We forecast a 7-day horizon; longer horizons are future work
---

<div class="ae-cols">
  <div class="ae-col">
    <div class="ae-collabel">What we tested — 7 days</div>
    <div class="ae-lead">Chosen to match every reference point</div>
    <ul>
      <li>Matches the Yu 2025 benchmark cadence</li>
      <li>Matches the source logs' reporting cadence</li>
      <li>Comparable to TSFM pretraining horizons</li>
    </ul>
  </div>
  <div class="ae-col">
    <div class="ae-collabel">What remains open — 30+ days</div>
    <div class="ae-lead">Longer horizons stay future work</div>
    <ul>
      <li>Longer horizons untested here</li>
      <li>Future work: pair forecasts with drift detection</li>
    </ul>
  </div>
</div>

<!-- speaker note: 7 days is a deliberate choice, not a limitation we hide — it lines up with the Yu 2025 benchmark, the logs' own reporting cadence, and typical TSFM pretraining horizons, so comparisons are clean. We did NOT test month-long horizons; I'd flag that openly as future work, ideally paired with drift detection so a forecast is refreshed when the process shifts rather than extrapolated blindly. -->

---
layout: assertion-evidence
class: backup
hideInToc: true
locator: Backup
assertion: Multivariate gave no consistent gains over univariate
---

<div class="ae-cols">
  <div class="ae-col">
    <div class="ae-collabel">What we tried</div>
    <div class="ae-lead">Multivariate inference, where the model supports it</div>
    <ul>
      <li>MOIRAI, MOIRAI-MoE, Chronos-2 all support it</li>
      <li>No consistent gains over univariate</li>
      <li>Consistent with Yu 2025</li>
    </ul>
  </div>
  <div class="ae-col">
    <div class="ae-collabel">Why &amp; what next</div>
    <div class="ae-lead">Cross-DF heterogeneity makes shared attention hurt</div>
    <ul>
      <li>Cross-series attention hurts at this scale</li>
      <li>Future work: dedicated multivariate fine-tuning for PMF</li>
    </ul>
  </div>
</div>

<div class="mt-6 text-center">
  <Callout>Univariate stays our default</Callout>
</div>

<!-- speaker note: This was pre-empted on the DF-complexity slide — DF series are highly heterogeneous, so the cross-series attention that helps homogeneous panels does not transfer. Here it's reinforced with the actual experiment: we ran true multivariate inference on MOIRAI, MOIRAI-MoE, and Chronos-2 (all three support it natively), and saw no consistent improvement over univariate, matching Yu 2025. We don't claim multivariate is hopeless — only that off-the-shelf shared attention doesn't help at this scale; dedicated multivariate fine-tuning for PMF is the open direction. -->
