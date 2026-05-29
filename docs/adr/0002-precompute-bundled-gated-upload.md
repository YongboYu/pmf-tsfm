# Serving = precomputed bundled results + a gated live-upload path

GPU-hungry models, high concurrency, and a low budget are in direct tension. We resolve it by
**splitting the serving path in two**: forecasts for the four bundled logs (× all paper model
variants × a few horizons) are **precomputed offline on VSC HPC** and served as static artifacts —
instant, infinitely concurrent, ~$0, and impossible to DoS. Only **custom uploads** run live, on a
**rate-limited, size-capped** serverless GPU (see ADR-0001).

## Consequences
- A precompute job (`scripts/precompute_demo.py`, run on HPC) must produce and commit/bundle the
  forecast DFG, comparison DFG, rendered images, and time-series data for every bundled
  combination. Adding a model/dataset/horizon means re-running it.
- The bundled explorer can offer **all paper variants** cheaply (it only serves files), while the
  live upload path is deliberately restricted (3 latest models) to bound GPU cost and UI clutter.
- A live conference burst hits the precompute path, not the GPU — the demo stays responsive.
