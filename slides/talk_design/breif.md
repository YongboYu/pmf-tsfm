Audience: CAiSE 2026 — Information Systems Engineering, BPM, process mining, AI for IS.
  ~80% PM/IS researchers; ~20% adjacent (data mining, forecasting).
  Assume PPM is known. Assume PMF is NOT known. Assume "foundation model" → LLM by default.

Session length: 30 min (20 min talk + 10 min Q&A).

Goal — what the audience must remember:
1. The problem matters because: process model forecasting (PMF) tells you where the whole
   process is heading, which compliance, capacity planning, and anomaly detection need —
   case-level predictive monitoring cannot answer this.
2. Existing methods fail because: ML/DL forecasters overfit on small, sparse, heterogeneous
   DF time series; no single trained-from-scratch model wins across event logs.
3. Our idea is: apply time series foundation models (TSFMs) — pretrained on millions of
   non-process time series — directly, zero-shot, to PMF.
4. The key evidence is: zero-shot TSFMs beat the two strongest baselines from our prior
   benchmark on MAE/RMSE across 4 event logs; fine-tuning (LoRA, full) adds little for
   ~100× the compute; ER (process-aware) reaches parity, not better.
5. The take-away is: zero-shot TSFMs are the new PMF default; the bottleneck has moved from
   forecasting accuracy to process-aware representation; PM can borrow from adjacent fields
   cheaply.

My paper:
- Title: Time Series Foundation Models for Process Model Forecasting
- Abstract: see manuscript/main.tex
- Research question: Can time series foundation models, applied zero-shot and via fine-tuning,
  outperform specialized PMF methods on directly-follows time series derived from real-life
  event logs?
- Main contribution: First systematic cross-family evaluation of TSFMs for PMF; adaptation
  guidance (zero-shot vs LoRA vs full fine-tuning); process-aware analysis tying TSFM
  performance to DF time series characteristics.
- Method: Convert event logs to DF time series via time-windowed DFGs; evaluate 8+ TSFM
  variants from 3 families (Chronos, MOIRAI, TimesFM) in zero-shot, LoRA, and full
  fine-tuning settings; univariate throughout.
- Dataset: BPI Challenge 2017, BPI Challenge 2019 (sub-flow 1), Sepsis, Hospital Billing.
  Daily aggregation, 7-day forecast horizon, 60/20/20 train/val/test for fine-tuning.
- Baselines: 7-day-lag seasonal naive; hyperparameter-optimized XGBoost (strongest two
  from Yu et al. 2025 benchmark).
- Main result: TSFMs generally outperform baselines zero-shot on MAE/RMSE; LoRA and full
  fine-tuning give only modest, dataset-dependent gains; ER reaches parity with baselines
  (except Sepsis, where all models struggle).
- Limitation: 4 event logs; univariate-only evaluation; 7-day horizon; single-seed
  fine-tuning runs.
- What I want people to ask after the talk: "Can we try this on our log?" / "What would a
  process-native foundation model look like?" / "How do you combine TSFM forecasts with
  drift detection?"

Style:
- Academic but clear
- Minimal text — max 5 bullets per slide, max 12 words per bullet
- One message per slide
- Diagrams and animations where they earn their time
- Spoken anchor lines rehearsed cold (see ../SLIDES.md)
