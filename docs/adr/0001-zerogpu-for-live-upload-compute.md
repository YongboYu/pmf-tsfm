# Live custom-upload forecasts run on HF ZeroGPU, not Modal or an always-on GPU

The demo lets users forecast on their own uploaded logs using GPU-hungry TSFMs
(Chronos-2 / Moirai-2.0 / TimesFM-2.5), with bursty conference traffic and a tight, fixed budget.
We chose **HF ZeroGPU** (PRO, ~$9/mo): it is a *flat fee with no usage-based overage* — heavy demand
degrades to per-visitor throttling and queueing rather than a bill, so we cannot be surprise-billed,
and it sits natively in the same HF Space as the app.

## Considered Options
- **Modal** — no per-call duration cap (handles big/slow uploads) and $30/mo free credits, but
  *genuinely metered*: a busy day can exceed credits and bill real money. Kept as a fallback if the
  ZeroGPU cap proves too limiting.
- **Always-on rented GPU** — predictable latency but continuous idle cost; rejected on budget.

## Consequences
ZeroGPU imposes a **~120s per-call wall-time cap**. Moirai/TimesFM on a large uploaded log can exceed
it and *fail* (not cost money). We therefore enforce tight upload size caps, default uploads to the
fast Chronos-2, and gate Moirai/TimesFM to small logs. Bundled logs never touch the GPU (see ADR-0002).
