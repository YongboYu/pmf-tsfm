# pmf-tsfm

Zero-shot forecasting of process behaviour: time-series foundation models predict future
directly-follows frequencies from an event log, which are reconstructed into a forecast
Directly-Follows Graph. This glossary fixes the shared language; it is not a spec.

## Language

### Process-mining terms

**Event log**:
A set of recorded process executions, read from XES. The unit of input.
_Avoid_: dataset (reserve "dataset" for a *named, bundled* log).

**Trace** (a.k.a. **Case**):
One end-to-end process execution — an ordered sequence of events for one `case:concept:name`.
_Avoid_: instance, run.

**Activity**:
A process step label (`concept:name`). Nodes of a DFG are activities.
_Avoid_: task, action, event-type.

**Directly-Follows relation** (**DF**):
An ordered pair of activities "A -> B" meaning B occurred immediately after A within a trace.
Each DF is one **feature** (one column) of the time series.
_Avoid_: edge, transition, arc (those name the DFG's rendering, not the relation).

**Directly-Follows Graph** (**DFG**):
A directed graph whose nodes are activities and whose arcs are DF relations weighted by frequency.
Includes artificial **▶** (start) and **■** (end) markers, cleaned to "Start"/"End" internally.
_Avoid_: process model, workflow net (a DFG is neither).

**Entropic Relevance** (**ER**):
The paper's metric scoring how well a DFG explains a log's traces; used to compare a forecast DFG
against the ground-truth DFG.

### Forecasting terms

**TSFM**:
A Time-Series Foundation Model — Chronos, Moirai, or TimesFM. Forecasts without per-log training.
_Avoid_: "the model" unqualified when the variant matters.

**Zero-shot forecast**:
Predicting future daily DF frequencies with a pretrained TSFM and **no fine-tuning** on the log.
_Avoid_: prediction (too generic), inference (reserve for the runtime act of calling the model).

**Horizon**:
The number of future **daily** steps forecast. Default 7.
_Avoid_: window (reserve "window" for the rolling context/evaluation windows in ER).

**Forecast DFG**:
The DFG reconstructed from forecast DF frequencies.

**Comparison DFG**:
The DFG shown beside the forecast DFG. For a **bundled log** it is the *actual-future* DFG (real
ground truth exists); for a **custom upload** it is the *last-known-window* DFG (no future truth
exists). The "compared to what?" reference.

### Demo vocabulary

**Bundled log**:
One of the four named logs shipped with the demo (bpi2017, bpi2019_1, sepsis, hospital_billing),
for which forecasts are precomputed.
_Avoid_: example, sample.

**Custom upload**:
A user's own XES log, forecast live at request time.

**Precompute path**:
Serving a result computed ahead of time (offline, on HPC) — instant, no GPU at request time.
Used for every bundled log × model × horizon.

**Live path**:
Computing a forecast on demand for a custom upload, on serverless GPU, under size/rate caps.

## Flagged ambiguities

- **"Dataset"** — only a *bundled, named* log. A user's uploaded file is a **custom upload**, never
  a "dataset".
- **"Window"** — the ER rolling evaluation window, *not* the forecast horizon. Keep them distinct.
- **"Model"** — always qualify the TSFM variant (e.g. Chronos-2) when it affects behaviour.

## Example dialogue

> **Dev:** When someone uploads a log, what do we put next to the forecast DFG?
> **Domain expert:** The comparison DFG. For a custom upload that's the last-known-window DFG —
> there's no actual future to compare against, unlike a bundled log.
> **Dev:** And each "A -> B" column is one DF relation, so the TSFM is forecasting a multivariate
> series, one feature per DF?
> **Domain expert:** Right. Sum the forecast over the horizon, round, and those frequencies weight
> the arcs of the forecast DFG — plus the artificial ▶/■ arcs.
> **Dev:** Horizon 7 means seven daily steps?
> **Domain expert:** Yes. Don't call that a window — "window" is the rolling span we use for ER.
