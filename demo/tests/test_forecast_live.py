"""Tests for the live-path forecast core (slice 3a, #114).

The live path forecasts a custom upload's true future (forecast origin = log end)
and compares it against the **last-known-window** DFG, so it reports **drift**
(DF relations added/removed) and **never** accuracy (ADR-0004). These deep modules
are gradio-free and tested through their public interfaces:

upload_guard.py
  - check_upload — size caps + model gating + default Chronos-2, clear rejections
forecast_live.py
  - drift_report  — DF relations partitioned as drift vs the recent past
  - forecast_live — assembles the live bundle (drift, never accuracy) from a log
"""

from __future__ import annotations

import numpy as np
import pytest
from upload_guard import UploadRejected, check_upload


def _dfg(*relations):
    """Build a DFG JSON document from ``(from_label, to_label, freq)`` triples."""
    labels = sorted({end for f, t, _ in relations for end in (f, t)})
    ids = {label: i for i, label in enumerate(labels)}
    return {
        "nodes": [{"label": label, "id": i} for label, i in ids.items()],
        "arcs": [{"from": ids[f], "to": ids[t], "freq": freq} for f, t, freq in relations],
    }


def _log_of_size(tmp_path, n_bytes: int):
    """A throwaway file standing in for an uploaded XES log of a given byte size."""
    path = tmp_path / "upload.xes"
    path.write_bytes(b"x" * n_bytes)
    return path


def test_oversize_log_is_rejected(tmp_path, monkeypatch):
    """A log past the hard byte cap is rejected with a clear message, not forecast."""
    import upload_guard

    monkeypatch.setattr(upload_guard, "MAX_UPLOAD_BYTES", 100)
    log = _log_of_size(tmp_path, 200)
    with pytest.raises(UploadRejected, match="cap"):
        check_upload(log, "chronos2")


def test_default_model_is_chronos2(tmp_path):
    """With no model named, the guard selects the GPU-cheap default (Chronos-2)."""
    log = _log_of_size(tmp_path, 10)
    assert check_upload(log) == "chronos2"


def test_unknown_model_is_rejected(tmp_path):
    """An unrecognised model id is refused rather than passed through to the GPU."""
    log = _log_of_size(tmp_path, 10)
    with pytest.raises(UploadRejected, match="unknown model"):
        check_upload(log, "gpt5")


@pytest.mark.parametrize("gated", ["moirai2", "timesfm2.5"])
def test_gated_model_rejected_on_large_log(tmp_path, monkeypatch, gated):
    """Moirai/TimesFM are refused on a log over the small-log threshold (under the hard cap)."""
    import upload_guard

    monkeypatch.setattr(upload_guard, "SMALL_LOG_BYTES", 100)
    log = _log_of_size(tmp_path, 200)
    with pytest.raises(UploadRejected, match="small logs"):
        check_upload(log, gated)


def test_gated_model_allowed_on_small_log(tmp_path, monkeypatch):
    """The same gated model is allowed when the log is under the small-log threshold."""
    import upload_guard

    monkeypatch.setattr(upload_guard, "SMALL_LOG_BYTES", 100)
    log = _log_of_size(tmp_path, 50)
    assert check_upload(log, "moirai2") == "moirai2"


def test_default_model_unaffected_by_small_log_gate(tmp_path, monkeypatch):
    """Chronos-2 stays available on a large (under-cap) log — only the heavy models are gated."""
    import upload_guard

    monkeypatch.setattr(upload_guard, "SMALL_LOG_BYTES", 100)
    log = _log_of_size(tmp_path, 200)
    assert check_upload(log, "chronos2") == "chronos2"


# --- drift_report -----------------------------------------------------------


def _rels(entries):
    return {(e["from"], e["to"]) for e in entries}


def test_drift_report_partitions_relative_to_recent_past():
    """Drift is framed against the recent past: a relation only in the forecast is *added*
    (the forecast introduces it); one only in the recent window is *removed* (the forecast
    drops it); one in both is *stable*."""
    from forecast_live import drift_report

    forecast = _dfg(("A", "B", 5), ("A", "C", 3))  # A->C is new vs recent past
    comparison = _dfg(("A", "B", 4), ("A", "D", 2))  # A->D existed recently, forecast drops it

    drift = drift_report(forecast, comparison)

    assert _rels(drift["added"]) == {("A", "C")}
    assert _rels(drift["removed"]) == {("A", "D")}
    assert _rels(drift["stable"]) == {("A", "B")}


def test_drift_report_uses_recent_not_actual_vocabulary():
    """Entries carry forecast vs *recent* weights — never the accuracy framing (actual/er/mae)."""
    from forecast_live import drift_report

    drift = drift_report(_dfg(("A", "B", 5)), _dfg(("A", "B", 4)))

    stable = drift["stable"][0]
    assert stable == {"from": "A", "to": "B", "forecast_freq": 5, "recent_freq": 4}
    blob = repr(drift).lower()
    assert (
        "actual" not in blob
        and "er" not in stable
        and not (blob.count("mae") or blob.count("rmse"))
    )


# --- forecast_live ----------------------------------------------------------

_LIVE_BUNDLE_KEYS = {
    "forecast_dfg",
    "comparison_dfg",
    "drift",
    "forecast_svg",
    "comparison_svg",
    "diff_absolute_svg",
    "diff_relative_svg",
}


@pytest.fixture
def stub_windows(monkeypatch):
    """Replace the GPU seam with a fixed forecast / last-known window pair.

    Forecast predicts A->B and A->C; the recent window held A->B and A->D — so the
    drift is: A->C added, A->D removed, A->B stable.
    """
    import forecast_live

    feature_names = ["A -> B", "A -> C", "A -> D"]
    forecast_window = np.array([[3.0, 2.0, 0.0], [2.0, 1.0, 0.0]])  # A->B=5, A->C=3, A->D=0
    last_known_window = np.array([[2.0, 0.0, 1.0], [2.0, 0.0, 1.0]])  # A->B=4, A->C=0, A->D=2
    monkeypatch.setattr(
        forecast_live,
        "_live_windows",
        lambda log, model, horizon: (forecast_window, last_known_window, feature_names),
    )


def test_forecast_live_returns_drift_bundle(tmp_path, stub_windows):
    """forecast_live assembles the live bundle: twin DFGs + drift + pre-rendered SVGs."""
    from forecast_live import forecast_live

    log = _log_of_size(tmp_path, 10)
    bundle = forecast_live(log, "chronos2", horizon=2)

    assert set(bundle) == _LIVE_BUNDLE_KEYS
    assert {"nodes", "arcs"} <= set(bundle["forecast_dfg"])
    assert {"nodes", "arcs"} <= set(bundle["comparison_dfg"])
    for key in ("forecast_svg", "comparison_svg", "diff_absolute_svg", "diff_relative_svg"):
        assert "<svg" in bundle[key]


def test_forecast_live_reports_drift_never_accuracy(tmp_path, stub_windows):
    """The live bundle carries drift (added/removed) and **no** accuracy metric (ADR-0004)."""
    from forecast_live import forecast_live

    bundle = forecast_live(_log_of_size(tmp_path, 10), "chronos2", horizon=2)

    assert "metrics" not in bundle
    drift = bundle["drift"]
    assert _rels(drift["added"]) == {("A", "C")}
    assert _rels(drift["removed"]) == {("A", "D")}
    blob = repr(bundle["drift"]).lower()
    assert "er" not in bundle and "mae" not in blob and "rmse" not in blob


def test_forecast_live_rejects_oversize_before_inference(tmp_path, monkeypatch):
    """A guard rejection short-circuits: the GPU seam is never reached for a bad upload."""
    import forecast_live

    monkeypatch.setattr(forecast_live.upload_guard, "MAX_UPLOAD_BYTES", 100)

    def _boom(*a, **k):  # the seam must not run when the guard rejects
        raise AssertionError("_live_windows reached despite guard rejection")

    monkeypatch.setattr(forecast_live, "_live_windows", _boom)
    with pytest.raises(UploadRejected):
        forecast_live.forecast_live(_log_of_size(tmp_path, 200), "chronos2")


def test_forecast_live_signature_is_typed_and_documented():
    """forecast_live's signature is the future MCP schema: every arg + return typed, documented."""
    import inspect

    from forecast_live import forecast_live

    sig = inspect.signature(forecast_live)
    assert all(p.annotation is not inspect.Parameter.empty for p in sig.parameters.values())
    assert sig.return_annotation is not inspect.Signature.empty
    assert (forecast_live.__doc__ or "").strip()


def test_forecast_live_imports_no_gradio():
    """The live core stays gradio-free, so flipping mcp_server=True derives a clean tool schema.

    Checked in a fresh interpreter (like the serve-import test) so an in-process import of
    app (which pulls gradio) can't pollute the result.
    """
    import subprocess
    import sys
    from pathlib import Path

    demo = Path(__file__).resolve().parent.parent
    code = (
        f"import sys; sys.path.insert(0, {str(demo)!r}); "
        "import forecast_live; "
        "assert 'gradio' not in sys.modules, 'forecast_live must not import gradio'"
    )
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell, trusted constant code
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
