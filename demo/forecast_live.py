"""Assemble a live-path forecast for a custom-uploaded log.

The sibling of :func:`forecast.forecast_bundled`, for the *live* path (ADR-0004):
the forecast origin is the **log end**, so it forecasts the genuine, unseen next
window and compares it against the **last-known-window** DFG. There is no future
truth for an upload, so this path reports **drift** (DF relations the forecast
adds/removes vs the recent past) and **never** an accuracy metric (ER/MAE/RMSE).

The public :func:`forecast_live` is agent-clean (typed args + docstring, no Gradio
objects) — its signature is the future MCP/REST tool schema. The actual
preprocessing + model inference sits behind :func:`_live_windows`, which is wired
to ZeroGPU in a follow-up slice (#115); the assembly, guard, and drift logic here
are GPU-free and tested in isolation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import upload_guard
from dfg_build import frequencies_to_dfg_json
from dfg_diff import dfg_diff
from log_to_series import log_to_df_series
from render import dfg_json_to_svg, diff_svg

# All three TSFMs are wired on the hosted live path (model parity with the bundled tab).
# Each model has its own lazy loader + ``(context, horizon) -> forecast`` adapter below,
# dispatched by ``_live_windows``. The heavy two (moirai2 / timesfm2.5) are gated to small
# logs by ``upload_guard`` so a single GPU call stays under the ZeroGPU wall-time cap.
#
# Model ids match the paper configs: configs/model/chronos/chronos2.yaml,
# configs/model/moirai/2_0_small.yaml, configs/model/timesfm/2_5_200m.yaml.
# Loading the Chronos-2 ``s3://`` checkpoint needs boto3.
_CHRONOS2_MODEL_ID = "s3://autogluon/chronos-2/"
_MOIRAI2_MODEL_ID = "Salesforce/moirai-2.0-R-small"
_TIMESFM_MODEL_ID = "google/timesfm-2.5-200m-pytorch"
# Per-model caches, each loaded once per (forked) worker process. See the ZeroGPU note on
# _load_chronos2: every loader runs lazily INSIDE the @spaces.GPU worker, never at import.
_chronos2_pipeline: object | None = None
_moirai2_module: object | None = None
_timesfm_model: object | None = None


def _load_chronos2() -> object:
    """Load (and cache) the Chronos-2 pipeline. Imports torch/chronos lazily so the
    live-core module stays importable — and testable — without the model libs.

    **Called only from inside the ``@spaces.GPU`` worker** (via :func:`_chronos2_forecast`),
    never at module import: ZeroGPU forks a worker per GPU call, and initialising CUDA in
    the parent process (e.g. an eager ``device_map="cuda"`` load at import) breaks that fork
    with "process PID not found (pid=0)". So the model is loaded on the real GPU that exists
    inside the worker; ``torch.cuda.is_available()`` is ``True`` there. The weights cache to
    disk on first download, so subsequent workers load fast.
    """
    global _chronos2_pipeline
    if _chronos2_pipeline is None:
        import torch
        from chronos import BaseChronosPipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _chronos2_pipeline = BaseChronosPipeline.from_pretrained(
            _CHRONOS2_MODEL_ID, device_map=device
        )
    return _chronos2_pipeline


def _chronos2_forecast(context: np.ndarray, horizon: int) -> np.ndarray:
    """Forecast the next ``horizon`` days of every DF relation with Chronos-2.

    A minimal port of ``pmf_tsfm.models.chronos.ChronosAdapter._predict_chronos2_batched``
    (batch every feature into one ``predict_df`` call via a per-feature ``item_id``, take
    the median) — kept here so the live path needs only chronos-forecasting + torch, not
    the full ``pmf_tsfm`` dep tree. **This is the GPU work** (wrapped by ``@spaces.GPU`` in
    the GUI) and is what tests monkeypatch.

    Args:
        context: ``(n_days, n_features)`` history — the whole log's daily DF series.
        horizon: number of future days to forecast.

    Returns:
        ``(horizon, n_features)`` forecast frequencies (median).
    """
    import pandas as pd

    pipeline = _load_chronos2()
    n_features = context.shape[1]
    rows = [
        {"item_id": feat_idx, "timestamp": t, "target": float(value)}
        for feat_idx in range(n_features)
        for t, value in enumerate(context[:, feat_idx])
    ]
    pred_df = pipeline.predict_df(  # type: ignore[attr-defined]
        pd.DataFrame(rows), prediction_length=horizon, quantile_levels=[0.1, 0.5, 0.9]
    )
    forecast = np.zeros((horizon, n_features))
    for feat_idx in range(n_features):
        feat = pred_df[pred_df["item_id"] == feat_idx]
        forecast[:, feat_idx] = feat["predictions"].to_numpy()[:horizon]
    return forecast


def _load_moirai2() -> object:
    """Load (and cache) the Moirai-2 module. Imports ``uni2ts`` lazily so the live-core
    module stays importable — and testable — without the model libs.

    Loaded only from inside the ``@spaces.GPU`` worker (via :func:`_moirai2_forecast`),
    never at module import: ZeroGPU forks a worker per GPU call, and initialising CUDA in
    the parent process breaks that fork with "process PID not found (pid=0)". The forecast
    object built per call moves to the GPU that exists inside the worker. The weights cache
    to disk on first download (~43 MB), so subsequent workers load fast.
    """
    global _moirai2_module
    if _moirai2_module is None:
        from uni2ts.model.moirai2 import Moirai2Module

        _moirai2_module = Moirai2Module.from_pretrained(_MOIRAI2_MODEL_ID)
    return _moirai2_module


def _moirai2_forecast(context: np.ndarray, horizon: int) -> np.ndarray:
    """Forecast the next ``horizon`` days of every DF relation with Moirai-2.

    A minimal port of ``pmf_tsfm.models.moirai.MoiraiAdapter`` (the 2.0 path): batch every
    feature into one GluonTS ``predict`` over a ``ListDataset``, take the median quantile —
    kept here so the live path needs only ``uni2ts``/``gluonts``, not the full ``pmf_tsfm``
    dep tree. **This is the GPU work** (wrapped by ``@spaces.GPU`` in the GUI) and is what
    tests monkeypatch.

    Args:
        context: ``(n_days, n_features)`` history — the whole log's daily DF series.
        horizon: number of future days to forecast.

    Returns:
        ``(horizon, n_features)`` forecast frequencies (median).
    """
    from gluonts.dataset.common import ListDataset
    from uni2ts.model.moirai2 import Moirai2Forecast

    module = _load_moirai2()
    n_features = context.shape[1]
    model = Moirai2Forecast(
        module=module,
        prediction_length=horizon,
        context_length=context.shape[0],
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )
    predictor = model.create_predictor(batch_size=16)
    dataset = ListDataset(
        [{"target": context[:, i].tolist(), "start": "2020-01-01"} for i in range(n_features)],
        freq="D",
    )
    forecasts = list(predictor.predict(dataset))
    forecast = np.zeros((horizon, n_features))
    for feat_idx, fc in enumerate(forecasts):
        forecast[:, feat_idx] = np.asarray(fc.quantile(0.5))[:horizon]
    return forecast


def _load_timesfm() -> object:
    """Load (and cache) the TimesFM-2.5 model. Imports ``timesfm`` lazily so the live-core
    module stays importable — and testable — without the model libs.

    Loaded only from inside the ``@spaces.GPU`` worker (via :func:`_timesfm_forecast`),
    never at module import — same ZeroGPU fork rule as :func:`_load_chronos2`. The forecast
    config mirrors configs/model/timesfm/2_5_200m.yaml. The weights cache to disk on first
    download (~882 MB), so subsequent workers load fast.
    """
    global _timesfm_model
    if _timesfm_model is None:
        import timesfm

        model_cls = timesfm.TimesFM_2p5_200M_torch
        # huggingface_hub >= 0.24 passes transport kwargs (proxies, resume_download) to
        # ``_from_pretrained``, which timesfm forwards into ``__init__`` and chokes on. Strip
        # them for the duration of the load (mirrors pmf_tsfm.models.timesfm's workaround).
        _orig = model_cls.__dict__.get("_from_pretrained")
        if _orig is not None:
            _orig_fn = _orig.__func__

            @classmethod  # type: ignore[misc]
            def _patched(cls, *, proxies=None, resume_download=None, **kwargs):
                return _orig_fn(cls, **kwargs)

            model_cls._from_pretrained = _patched
        try:
            model = model_cls.from_pretrained(_TIMESFM_MODEL_ID)
        finally:
            if _orig is not None:
                model_cls._from_pretrained = _orig
        model.compile(
            timesfm.ForecastConfig(
                max_context=1024,
                max_horizon=64,
                normalize_inputs=True,
                per_core_batch_size=1,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )
        _timesfm_model = model
    return _timesfm_model


def _timesfm_forecast(context: np.ndarray, horizon: int) -> np.ndarray:
    """Forecast the next ``horizon`` days of every DF relation with TimesFM-2.5.

    A minimal port of ``pmf_tsfm.models.timesfm.TimesFMAdapter`` (the 2.5 path): forecast
    every feature as a univariate series in one ``model.forecast`` call, take the point
    forecast — kept here so the live path needs only ``timesfm``, not the full ``pmf_tsfm``
    dep tree. **This is the GPU work** (wrapped by ``@spaces.GPU`` in the GUI) and is what
    tests monkeypatch.

    Args:
        context: ``(n_days, n_features)`` history — the whole log's daily DF series.
        horizon: number of future days to forecast.

    Returns:
        ``(horizon, n_features)`` forecast frequencies (point forecast).
    """
    model = _load_timesfm()
    n_features = context.shape[1]
    inputs = [context[:, i].astype(np.float32) for i in range(n_features)]
    point_forecast, _ = model.forecast(horizon=horizon, inputs=inputs)  # type: ignore[attr-defined]
    # point_forecast is (n_features, horizon) — transpose to (horizon, n_features).
    return np.asarray(point_forecast).T[:horizon]


def _live_windows(
    log: str | Path, model: str, horizon: int
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Preprocess + forecast one upload — the GPU seam (ADR-0004, ZeroGPU in #115).

    Returns ``(forecast_window, last_known_window, feature_names)`` where both windows
    are ``(horizon, n_features)`` DF-relation frequency arrays in the **same** feature
    space (so the two DFGs key consistently): the model forecast for the next ``horizon``
    days *from the log end* (the genuine unseen future), and the actual last ``horizon``
    days of the log (the recent past the drift is measured against).

    The whole daily series is the forecast context; its tail is the last-known window.
    Tests monkeypatch ``log_to_df_series`` / the per-model ``_*_forecast`` adapters to keep
    real model inference out of CI (as slice 3a kept this whole seam mocked). The dispatch
    map is built per call (not at module level) so those monkeypatches take effect.

    Raises:
        UploadRejected: the model is not wired on the hosted live path, or the log spans
            too few days to forecast a ``horizon``-day window from history.
    """
    adapters = {
        "chronos2": _chronos2_forecast,
        "moirai2": _moirai2_forecast,
        "timesfm2.5": _timesfm_forecast,
    }
    if model not in adapters:
        raise upload_guard.UploadRejected(
            f"{model} is not available on the hosted live demo; choose one of {sorted(adapters)}."
        )

    series = log_to_df_series(log)
    feature_names = list(series.columns)
    values = series.to_numpy(dtype=float)
    if len(values) < horizon + 1:
        raise upload_guard.UploadRejected(
            f"the log spans only {len(values)} day(s); at least {horizon + 1} are needed to "
            f"forecast a {horizon}-day window from prior history."
        )

    last_known_window = values[-horizon:]
    forecast_window = adapters[model](values, horizon)
    return forecast_window, last_known_window, feature_names


def drift_report(
    forecast_dfg: dict[str, Any], comparison_dfg: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    """Partition DF relations as **drift** of the forecast vs the last-known window.

    Reuses :func:`dfg_diff.dfg_diff` but reframes it relative to the recent past
    (not an accuracy comparison): ``dfg_diff(forecast, comparison)`` calls a relation
    present only in the forecast ``removed`` and one present only in the comparison
    ``added`` — the *opposite* of the drift reading — so the two are swapped here.

    Args:
        forecast_dfg:   The forecast DFG (``{"nodes": [...], "arcs": [...]}``).
        comparison_dfg: The last-known-window DFG, same shape.

    Returns:
        ``{"added": [...], "removed": [...], "stable": [...]}`` where ``added`` are
        relations the forecast introduces vs the recent past, ``removed`` are ones it
        drops, and ``stable`` are common to both. Each entry is
        ``{"from": label, "to": label, "forecast_freq": int, "recent_freq": int}`` —
        the ``recent_freq`` (not ``actual_freq``) naming keeps this off the accuracy
        framing: there is no future truth to score against (ADR-0004).
    """
    diff = dfg_diff(forecast_dfg, comparison_dfg)

    def entry(e: dict[str, Any]) -> dict[str, Any]:
        return {
            "from": e["from"],
            "to": e["to"],
            "forecast_freq": e["forecast_freq"],
            "recent_freq": e["actual_freq"],
        }

    return {
        "added": [entry(e) for e in diff["removed"]],  # in forecast, not in recent past
        "removed": [entry(e) for e in diff["added"]],  # in recent past, not in forecast
        "stable": [entry(e) for e in diff["matched"]],
    }


def forecast_live(
    log: str | Path,
    model: str = "chronos2",
    horizon: int = 7,
) -> dict[str, Any]:
    """Forecast a custom-uploaded log's true future and report its drift.

    The live path (ADR-0004): the forecast origin is the log end, so the forecast is
    the genuine unseen next ``horizon`` days, compared against the **last-known-window**
    DFG. There is no future truth for an upload, so the bundle carries **drift** (DF
    relations the forecast adds/removes vs the recent past) and **never** an accuracy
    metric (no ER/MAE/RMSE). This is the function the MCP/REST tool exposes.

    Args:
        log:     Path to the uploaded XES log.
        model:   TSFM id (default ``"chronos2"``; ``moirai2``/``timesfm2.5`` are gated to
                 small logs by the upload guard).
        horizon: Forecast horizon in days. Fixed at 7 in v1.

    Returns:
        ``{"forecast_dfg": <dfg json>, "comparison_dfg": <dfg json>,
           "drift": {"added": [...], "removed": [...], "stable": [...]},
           "forecast_svg": str, "comparison_svg": str,
           "diff_absolute_svg": str, "diff_relative_svg": str}`` — the JSON DFGs (the
        regenerable source of truth) alongside pre-rendered figures. The ``comparison_*``
        is the last-known-window DFG (not actual future), and the bundle reports drift,
        never accuracy.

    Raises:
        UploadRejected: the log is too large or the model is gated for its size.

    Note:
        The diff SVGs are rendered with ``framing="drift"`` so their legend reads
        "forecast | last-known window" / "% change from recent past" (never "actual" or
        accuracy) — the relabel the bundle carries, so the same SVGs are honest whether
        shown in the GUI or returned by the future MCP tool.
    """
    model = upload_guard.check_upload(log, model)
    forecast_window, last_known_window, feature_names = _live_windows(log, model, horizon)

    forecast_dfg = frequencies_to_dfg_json(forecast_window, feature_names)
    comparison_dfg = frequencies_to_dfg_json(last_known_window, feature_names)
    return {
        "forecast_dfg": forecast_dfg,
        "comparison_dfg": comparison_dfg,
        "drift": drift_report(forecast_dfg, comparison_dfg),
        "forecast_svg": dfg_json_to_svg(forecast_dfg),
        "comparison_svg": dfg_json_to_svg(comparison_dfg),
        "diff_absolute_svg": diff_svg(
            forecast_dfg, comparison_dfg, mode="absolute", framing="drift"
        ),
        "diff_relative_svg": diff_svg(
            forecast_dfg, comparison_dfg, mode="relative", framing="drift"
        ),
    }
