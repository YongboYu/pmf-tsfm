"""Tests for the live-path forecast core (slice 3a, #114).

The live path forecasts a custom upload's true future (forecast origin = log end)
and compares it against the **last-known-window** DFG, so it reports **drift**
(DF relations added/removed) and **never** accuracy (ADR-0004). These deep modules
are gradio-free and tested through their public interfaces:

upload_guard.py
  - check_upload — size caps + model gating + default Chronos-2, clear rejections
forecast_live.py
  - drift_report   — DF relations partitioned as drift vs the recent past
  - forecast_live  — assembles the live bundle (drift, never accuracy) from a log
  - resolve_source — maps an agent-supplied URL / "example" keyword to a local XES path
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
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


# --- the committed example log ----------------------------------------------


def test_example_log_is_a_usable_live_input():
    """The live tab's one-click example must be a valid live input: under the small-log
    gate and spanning enough days that log_to_df_series yields >= horizon+1 rows (so a
    7-day window can be forecast from prior history, per _live_windows)."""
    from pathlib import Path

    from log_to_series import log_to_df_series

    example = Path(__file__).resolve().parent.parent / "examples" / "sepsis_sample.xes"
    assert example.exists(), "the committed example log is missing"

    from upload_guard import SMALL_LOG_BYTES

    assert example.stat().st_size < SMALL_LOG_BYTES

    series = log_to_df_series(example)
    horizon = 7
    assert len(series) >= horizon + 1, "example log spans too few days to forecast from"
    # An active tail (not a sparse trail-off) so the drift view is meaningful.
    assert series.to_numpy()[-horizon:].sum() > 0


# --- _live_windows (the seam: preprocess + forecast) ------------------------


@pytest.fixture
def stub_live_seam(monkeypatch):
    """Stand in for the two halves of the seam: the log→series converter and the
    Chronos-2 call. A 5-day series over 3 relations; the forecast is a constant 9.0."""
    import forecast_live

    cols = ["▶ -> A", "A -> B", "B -> ■"]
    frame = pd.DataFrame(
        [[1, 2, 1], [1, 1, 1], [0, 3, 0], [2, 2, 2], [1, 0, 1]],
        columns=cols,
        index=pd.date_range("2020-01-01", periods=5),
    )
    monkeypatch.setattr(forecast_live, "log_to_df_series", lambda log, **k: frame)
    monkeypatch.setattr(
        forecast_live,
        "_chronos2_forecast",
        lambda context, horizon: np.full((horizon, context.shape[1]), 9.0),
    )
    return frame


def test_live_windows_forecasts_from_end_and_tails_the_recent_window(tmp_path, stub_live_seam):
    """Both windows are ``(horizon, F)`` in one feature space: the forecast from the log
    end, and the actual last ``horizon`` days (the recent past)."""
    from forecast_live import _live_windows

    forecast_window, last_known_window, feature_names = _live_windows(
        tmp_path / "x.xes", "chronos2", 2
    )

    assert feature_names == list(stub_live_seam.columns)
    assert forecast_window.shape == (2, 3) and last_known_window.shape == (2, 3)
    # last-known = the recent tail of the series; forecast = the model's true-future call.
    np.testing.assert_array_equal(last_known_window, stub_live_seam.to_numpy()[-2:])
    np.testing.assert_array_equal(forecast_window, np.full((2, 3), 9.0))


def test_live_windows_rejects_log_with_too_little_history(tmp_path, monkeypatch):
    """A log spanning fewer than ``horizon + 1`` days can't anchor the forecast — rejected."""
    import forecast_live

    frame = pd.DataFrame(
        [[1, 1]], columns=["A -> B", "B -> ■"], index=pd.date_range("2020-01-01", periods=1)
    )
    monkeypatch.setattr(forecast_live, "log_to_df_series", lambda log, **k: frame)
    with pytest.raises(UploadRejected, match="day"):
        forecast_live._live_windows(tmp_path / "x.xes", "chronos2", 7)


def test_live_windows_rejects_unwired_model_before_preprocessing(tmp_path, monkeypatch):
    """Only chronos2 is wired this slice; a gated model is rejected before any work runs."""
    import forecast_live

    def _boom(*a, **k):
        raise AssertionError("preprocessing ran for an unwired model")

    monkeypatch.setattr(forecast_live, "log_to_df_series", _boom)
    with pytest.raises(UploadRejected, match="chronos2"):
        forecast_live._live_windows(tmp_path / "x.xes", "moirai2", 7)


def test_forecast_live_runs_through_the_real_seam(tmp_path, stub_live_seam):
    """End-to-end through the *real* _live_windows (only the converter + model mocked):
    the bundle assembles and still carries drift, never accuracy."""
    from forecast_live import forecast_live

    bundle = forecast_live(_log_of_size(tmp_path, 10), "chronos2", horizon=2)

    assert set(bundle) == _LIVE_BUNDLE_KEYS
    assert "metrics" not in bundle
    assert "<svg" in bundle["forecast_svg"]


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


# --- resolve_source (the agent-tool input resolver, #116) -------------------
#
# The MCP/REST tool can't drive a browser file picker, so it takes a *source string*:
# an http(s) URL to an XES log, or the keyword "example". resolve_source maps that to a
# local XES path forecast_live can read. It stays gradio/spaces-free (stdlib only).


class _FakeResp:
    """A stand-in for a urlopen() response: a readable, context-manager byte stream."""

    def __init__(self, data: bytes):
        self._buf = io.BytesIO(data)

    def read(self, n: int = -1) -> bytes:
        return self._buf.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_resolve_source_example_keyword_returns_committed_log():
    """The "example" (and "sepsis_sample") keyword resolves to the committed sample log,
    so an agent has a zero-setup input — no URL needed to see the tool work."""
    from forecast_live import EXAMPLE_LOG, resolve_source

    assert resolve_source("example") == str(EXAMPLE_LOG)
    assert resolve_source("sepsis_sample") == str(EXAMPLE_LOG)
    assert Path(resolve_source("example")).exists()


def test_resolve_source_rejects_bundled_name_pointing_to_gui():
    """A bundled dataset name is *not* a live input: its full log isn't on the live surface
    (accuracy lives in the GUI), so it's rejected with a message pointing there."""
    from forecast_live import resolve_source

    with pytest.raises(UploadRejected, match="http"):
        resolve_source("sepsis")


@pytest.mark.parametrize("bad", ["file:///etc/passwd", "ftp://host/log.xes", "/local/path.xes"])
def test_resolve_source_rejects_non_http_scheme(bad):
    """Only http(s) is fetched — no local-file or other-scheme reads (path/SSRF safety)."""
    from forecast_live import resolve_source

    with pytest.raises(UploadRejected):
        resolve_source(bad)


def test_resolve_source_downloads_http_url(monkeypatch):
    """An http(s) URL is streamed to a local .xes temp file resolve_source returns."""
    import forecast_live

    payload = b"<log><trace/></log>"
    monkeypatch.setattr(forecast_live, "urlopen", lambda url, *a, **k: _FakeResp(payload))

    path = forecast_live.resolve_source("https://example.com/log.xes")

    assert path.endswith(".xes")
    assert Path(path).read_bytes() == payload


def test_resolve_source_rejects_oversize_download(monkeypatch):
    """A download past the upload byte cap is rejected (and not left on disk)."""
    import forecast_live

    monkeypatch.setattr(forecast_live.upload_guard, "MAX_UPLOAD_BYTES", 4)
    monkeypatch.setattr(forecast_live, "urlopen", lambda url, *a, **k: _FakeResp(b"toolong"))

    with pytest.raises(UploadRejected, match="cap"):
        forecast_live.resolve_source("https://example.com/big.xes")
