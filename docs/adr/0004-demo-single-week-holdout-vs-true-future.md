# The demo forecasts a single week; bundled is a holdout backtest, upload is true-future

The paper evaluates forecasts over **many rolling windows** — faithful, but hard to read at a glance
and asymmetric with what a user uploading their own log would expect. For the demo we collapse this
to **one forecast of one week** (horizon 7), and we split *where the forecast origin sits* by path:

- **Bundled log → holdout backtest.** Forecast origin is `log_end − 7`: we hold out the real last
  week, forecast it from the rest, and compare against the **actual-future** DFG. Real ground truth
  exists, so **ER / MAE / RMSE are genuine accuracy**. This reuses the *final window already in
  `predictions.npy`* — no re-inference.
- **Custom upload → true-future.** Forecast origin is the **log end**: we forecast the genuine,
  unseen next week and compare against the **last-known-window** DFG. There is no future truth, so we
  report **drift** (directly-follows relations added/removed), never accuracy.

Both paths read identically — "forecast a week" — and the *only* difference is the origin, which is
exactly why one yields accuracy and the other yields drift.

## Considered Options
- **Rolling-window slider (paper-faithful).** A scrubber over every test window, DFG + ER updating
  per step. Richest and most faithful, but asymmetric with the upload path and the heaviest UI; the
  rolling window was flagged as hard to illustrate. Kept as a possible bundled-only enhancement later
  (the windows are already on disk, so it is purely additive).
- **Both paths true-future (pure visualisation).** No holdout anywhere → no ground truth → no
  accuracy metric on either path. Rejected: it throws away the paper's headline evidence that TSFMs
  forecast process behaviour *well*, and would need a fresh forecast past data-end for bundled.
- **Both paths holdout.** Upload also hides its last week to gain a truth comparison. Rejected: an
  odd ask of an uploader, and it prevents the tool from forecasting the user's actual unseen future.

## Consequences
- The bundled metric panel (ER/MAE/RMSE) is **accuracy**; the upload panel is **drift** and must
  never be labelled or read as accuracy (see `CONTEXT.md` — Drift, Comparison DFG).
- Bundled precompute = the **final** forecast window + the truth DFG of the held-out last week; no
  model is re-run for the demo.
- Selectable horizons / the rolling-window slider remain documented future work, not v1.
