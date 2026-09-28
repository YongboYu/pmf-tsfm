---
title: Process Model Forecasting Explorer
emoji: 🔮
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 6.16.0
app_file: app.py
pinned: true
license: mit
short_description: Explore TSFM forecasts of directly-follows process behaviour
suggested_hardware: zero-a10g
models:
  - amazon/chronos-2
  - Salesforce/moirai-2.0-R-small
  - google/timesfm-2.5-200m-pytorch
---

# Process Model Forecasting Explorer

An explorer for the CAiSE 2026 paper
[*Time Series Foundation Models for Process Model Forecasting*](https://arxiv.org/abs/2512.07624)
(code: [github.com/YongboYu/pmf-tsfm](https://github.com/YongboYu/pmf-tsfm)). It makes the paper's
result tangible: off-the-shelf Time-Series Foundation Models (Chronos-2, Moirai-2.0, TimesFM-2.5) can
forecast how the **directly-follows (DF) relations** of a process evolve over time.

## What you can do

**Bundled explorer.** Pick one of the four bundled event logs (`bpi2017`, `bpi2019_1`, `sepsis`,
`hospital_billing`) and a model. You see:

- the **forecast DFG** (predicted next week of process behaviour) beside the **actual-future DFG**
  (what really happened) — the bundled path is a *holdout backtest*: the real last week is held out
  and forecast from the rest, so genuine ground truth exists;
- a **Side-by-side / Diff** toggle, with the Diff view offering **Absolute** (forecast | actual
  counts) and **Relative** (signed change %) labelling;
- an **ER / MAE / RMSE** accuracy strip, with the truth-DFG ER baseline for context.

**Live upload (your log).** Upload a custom **XES** log and forecast its genuine next week (forecast
origin = the log end) on **ZeroGPU**. Because an upload has no future ground truth, this tab reports
**drift** — the DF relations the forecast adds or drops vs the **last-known window** — and **never**
an accuracy metric. Live forecasts run with **Chronos-2** (Moirai-2 and TimesFM-2.5 are
available in the Bundled explorer tab); oversize logs are rejected so a call stays under the GPU
time limit.

## How it runs

The **bundled** path is served from **precomputed, committed assets** (incl. pre-rendered SVGs) — so
it is instant, infinitely concurrent, needs **no GPU**, and cannot be DoS'd into a bill. The **live**
path runs preprocessing + a Chronos-2 forecast under `@spaces.GPU` on **ZeroGPU** and renders its
DFGs at request time (so the Space carries graphviz + the model libs; see `packages.txt` /
`requirements.txt`).

Run it locally:

```bash
uv run --with gradio python app.py
```

## Self-host & agent API

This Space is the hosted GUI only; its source is
[`demo/`](https://github.com/YongboYu/pmf-tsfm/tree/main/demo) in the GitHub repository, deployed
automatically on push. To run the core forecasting pipeline on your own data without the Space's
limits, use the self-host [Docker image](https://github.com/YongboYu/pmf-tsfm/blob/main/docker/README.md)
(GUI + Hydra CLIs) or the [MCP server](https://github.com/YongboYu/pmf-tsfm/blob/main/mcp/README.md)
for agents; both wrap `pmf_tsfm.api` directly.
