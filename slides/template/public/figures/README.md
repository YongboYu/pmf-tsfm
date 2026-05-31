# Slidev figure assets

Served by Slidev at root: reference in `slides.md` as `<img src="/figures/<name>" ... />`.

Semantic kebab-case naming. Full per-beat mapping in
`slides/talk_design/workflow.md` (Figure naming convention).

## Currently expected (drop files here as you produce them)

| File | Used by | Notes |
|---|---|---|
| `dfg-evolution.gif` (or `.mp4`) | beat 2 | DFG t₁→t₂→t₃→forecasted t₄ animation, 10–15s |
| `bpi2017-drift-xgb-only.png` | beat 3 | Ground truth + XGBoost only — TSFM line stripped for callback |
| `tsfm-release-timeline.svg` | beat 6 | 2023→2026 horizontal timeline, 3 family lanes, dots per version |
| `results-mae-bars.png` | beat 7a | 4 panels, 5 horizontal bars per panel, baselines gray + TSFMs accent |
| `bpi2017-drift-with-tsfm.png` | beat 7b | Full reveal — actual + XGBoost + TSFM (paper Fig 1 BPI2017 panel) |
| `ft-slope.png` | beat 8 | 3-point slope ZS→LoRA→Full-FT, one line per (model × dataset), colored by dataset |
| `ft-compute-bars.png` | beat 8 | Log-scale wall-clock bars, 3 ZS + 3 LoRA + 3 Full-FT + 1 XGBoost |
| `er-bars.png` | beat 9 | Compact 4-panel ER bars, 3 TSFMs + 2 baselines, Truth/Training reference lines |
| `demo-screencast.mp4` | beat 11 | ~2 min pre-recorded screencast: load log → pick TSFM → render forecasted DFG |
| `baseline-ranking-full.png` | backup | Full benchmark ranking from Yu 2025 |
| `rmse-full.png` | backup | Full RMSE table |
| `df-complexity-radar.png` | backup | 7-metric DF complexity vs 21 standard benchmarks |
| `sepsis-variants.png` | backup | Sepsis trace variant distribution / DFG showing heterogeneity |

When a file lands here, the next `/03-content-pass <beat>` invocation auto-wires it
into the slide (replaces the placeholder `<div>` with `<img>`).

## Source pipeline

- For the 4 plots derived from existing `.eps` in `slides/figures/`:
  convert with `pdftocairo -png -r 300 source.eps target` and rename per the table above.
- For new charts (results bars, slope, compute, ER): generate from your experiment outputs
  with matplotlib; save as PNG at 2× resolution.
- For animations: Manim or matplotlib + ffmpeg.
- For the screencast: any screen-recording tool; export at 720p+, ≤2 min.
