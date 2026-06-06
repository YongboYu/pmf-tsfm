# The headline MAE figure is plotted from paper Table 4, not the results re-run

The deck's main results figure (`results-mae-bars.png`, beat 7a) plots zero-shot MAE for the two
baselines and the three latest TSFMs across four logs. There were two sources: the **results/**
`comprehensive_evaluation` CSVs (machine-precise, what `make_figures.py` read) or the **camera-ready
paper Table 4** values (transcribed in `figure_manifest.py`). They agree on the TSFM rows but the
**baselines differ materially** — most starkly XGBoost on Sepsis: paper `.169` vs the re-run's
`~.094`, plus smaller XGBoost deltas on every log and Naive on Hospital Billing. The original
`fig_mae_bars` read results/, so the figure understated the baselines — distorting the central claim
("zero-shot TSFMs beat both baselines on every log") and, on the bad XGBoost-Sepsis value, even
flipping TimesFM-2.5 from a win to an apparent loss.

We now **plot the paper Table 4 values** (`TABLE4_MAE`, extended to include the baseline rows) and
keep the results CSVs as a **cross-check** that prints the expected baseline deltas at build time.
The decisive reason is the `SLIDES.md` hard rule: *slide result numbers must be paper-faithful*. The
talk defends the camera-ready paper, so the camera-ready table — not a later re-run — is authoritative
for what appears on screen. This realigns with the original intent already recorded in the throwaway
`preview_mae_main.py` ("sources values directly from paper Table 4, camera-ready, defensible"); the
results-sourced `fig_mae_bars` was a regression from it.

## Consequences
- `results-mae-bars.png` is regenerated; baseline bars rise to camera-ready values (XGBoost Sepsis
  `0.17`, Hospital `2.67`, …). All three TSFM bars now sit below both baselines on every panel, and
  the headline holds cleanly. The "−21% mean MAE vs best baseline" callout matches the bars.
- **MAE only.** The paper reports MAE (Table 4); it has no RMSE table. So the **RMSE backup**
  (`rmse-full.png`) and the **drift** plot stay results-sourced — there is no paper machine source for
  them. Their baselines carry the same re-run delta, flagged in `make_figures.py`'s header and in the
  RMSE slide caption. The drift line matches paper Fig 1 qualitatively.
- This is a figure-data fix only. Per-slide verification that every number/bar matches the paper is
  the explicit job of the **content-revision phase** (the slide migration that follows this template
  work) — not claimed complete here.
