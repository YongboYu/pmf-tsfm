"""Gradio app — the bundled explorer (slice 1) + the live upload path (slice 3b).

Two tabs over the two agent-clean deep modules:

* **Bundled explorer** — ``forecast.forecast_bundled``: a dataset/model picker drives
  twin panes (forecast DFG | actual-future DFG) with a ``[Side-by-side | Diff]`` toggle
  and an ER/MAE/RMSE strip. A holdout backtest (ADR-0004), so the metrics are genuine
  accuracy. Served entirely from precomputed assets (pre-rendered SVGs) — GPU-free.
* **Live upload** — ``forecast_live.forecast_live``: upload a custom XES log, forecast
  its genuine next week *on ZeroGPU*, and read the **drift** of that forecast vs the
  **last-known window** (DF relations added/removed). No future truth for an upload, so
  this path shows **drift, never accuracy** (ADR-0004). The live forecast renders DFGs
  at request time, so this path needs the ``dot`` binary + the model libs (unlike the
  GPU-free bundled path) — see ``requirements.txt`` / ``packages.txt``.

``mcp_server`` stays off in this slice (flipped on in #116, ADR-0003).

    uv run --with gradio python demo/app.py
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import gradio as gr
from forecast import forecast_bundled
from forecast_live import forecast_live
from upload_guard import MAX_UPLOAD_BYTES, UploadRejected

# Only the precomputed bundled pairs are offered as choices: the full 4-by-3 matrix.
DATASETS: list[str] = ["bpi2017", "bpi2019_1", "sepsis", "hospital_billing"]
MODELS: list[str] = ["chronos2", "moirai2", "timesfm2.5"]

# The live path runs on ZeroGPU and offers Chronos-2 only — by design, not a stopgap.
# Moirai-2 / TimesFM-2.5 stay in the Bundled explorer tab: uni2ts pins numpy~=1.26, which
# breaks the numpy-2.x Chronos/torch ZeroGPU stack on the shared Space (live parity was
# attempted in #122, reverted in #123). The upload guard still recognises them as gated.
LIVE_MODELS: list[str] = ["chronos2"]

# A tiny committed sample (a dense six-week slice of the bundled sepsis log) so the live
# tab has a one-click "Try an example" — no one needs their own XES to see the path work.
EXAMPLE_LOG = Path(__file__).resolve().parent / "examples" / "sepsis_sample.xes"

# ZeroGPU boundary: ``@spaces.GPU`` requests a GPU slice for the ~120s of the live call
# (ADR-0001). ``spaces`` only exists on the HF Space, so off-Space (local / CI) we fall
# back to a no-op decorator and the same code runs on CPU for the visual check.
#
# IMPORTANT — do NOT load the model on cuda at module import. ZeroGPU forks a worker per
# @spaces.GPU call; if CUDA is initialised in *this* (parent) process first, the fork's
# process tracking fails with "process PID not found (pid=0)". HF's "load on cuda at
# module level" guidance relies on the emulation intercepting `.to('cuda')`, which it does
# for standard transformers/diffusers — but NOT for the autogluon Chronos-2 loader
# (`BaseChronosPipeline.from_pretrained(device_map="cuda")`), which inits real CUDA in the
# parent and breaks the fork. So Chronos-2 is loaded lazily INSIDE the @spaces.GPU call
# (see forecast_live._chronos2_forecast), where a real GPU exists. The weights cache to disk
# on first download, so later calls are fast.
try:
    import spaces

    _gpu = spaces.GPU(duration=120)
except ImportError:  # pragma: no cover - exercised only off-Space

    def _gpu(fn):  # type: ignore[no-redef]
        return fn


@_gpu
def _run_live(log_path: str, model: str) -> dict[str, Any]:
    """Run the live forecast under the ZeroGPU slice — the GPU entry point.

    A thin wrapper so the GPU boundary sits at the GUI layer; ``forecast_live`` stays
    agent-clean (no ``spaces`` dependency) for the future MCP/REST tool (#116).
    """
    return forecast_live(log_path, model)


# Vendored (not CDN) so the Hugging Face Space works fully offline / behind
# restrictive networks. svg-pan-zoom v3.6.2 — https://github.com/bumbu/svg-pan-zoom
_VENDOR_DIR = Path(__file__).resolve().parent / "static"
_SVG_PAN_ZOOM_JS = _VENDOR_DIR / "svg-pan-zoom.min.js"

# Re-apply pan/zoom whenever Gradio swaps the innerHTML of a `.dfg-pane`
# (initial load, dropdown change, Side-by-side↔Diff toggle). A MutationObserver
# watches for fresh `<svg>` elements; each gets sized to fill its pane and gets
# its own svg-pan-zoom instance, destroying any stale one first.
_PAN_ZOOM_INIT = """
<script>
(function () {
  function attach(svg) {
    if (!svg || svg.dataset.panzoom === "ready") return;
    svg.dataset.panzoom = "ready";
    svg.style.width = "100%";
    svg.style.height = "70vh";
    svg.style.maxHeight = "70vh";
    svg.style.margin = "0";
    try {
      if (svg._panzoom) { svg._panzoom.destroy(); }
      svg._panzoom = svgPanZoom(svg, {
        controlIconsEnabled: true,
        fit: true,
        center: true,
      });
    } catch (e) {
      svg.dataset.panzoom = "";  // let a later mutation retry
    }
  }
  function scan() {
    document.querySelectorAll(".dfg-pane svg").forEach(attach);
  }
  function boot() {
    scan();
    new MutationObserver(scan).observe(document.body, {
      childList: true,
      subtree: true,
    });
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
</script>
"""


def _head() -> str:
    """Assemble the ``<head>`` payload: inlined svg-pan-zoom + the pane initializer.

    Inlining (rather than a static-file URL or CDN) keeps the Space self-contained
    and works offline. Read at app-build time so the bytes ship with the page.
    """
    vendor = _SVG_PAN_ZOOM_JS.read_text(encoding="utf-8")
    return f"<script>{vendor}</script>\n{_PAN_ZOOM_INIT}"


def _scrollable(svg: str) -> str:
    """Wrap a rendered SVG so it scrolls within its pane, centred horizontally.

    Gradio renders ``<svg>`` as ``display:block``, so it ignores ``text-align``;
    ``margin:0 auto`` on the element itself centres it when it is narrower than the
    pane, while ``overflow:auto`` keeps a wider graph fully scrollable.
    """
    centred = svg.replace("<svg ", '<svg style="display:block;margin:0 auto" ', 1)
    return f'<div style="overflow:auto; max-height:70vh">{centred}</div>'


def _fmt(value: float) -> str:
    """Format a metric to 3 decimals, showing ``n/a`` when it is not finite.

    Some upstream ER sweeps yield ``nan`` for a window where ER cannot be computed
    (e.g. sepsis × moirai2): we keep the value faithful and only present the gap as
    ``n/a`` rather than a raw ``nan``.
    """
    return f"{value:.3f}" if math.isfinite(value) else "n/a"


def _render_panes(result: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    """Assemble the side-by-side outputs from a resolved forecast result.

    The SVGs are pre-rendered at precompute time (no ``dot`` at runtime); the app
    only wraps them in a scrollable pane. The strip pairs the forecast ER with the
    truth-DFG ER baseline so the gap between them reads as the forecast's ER error.
    """
    metrics = result["metrics"]
    return (
        _scrollable(result["forecast_svg"]),
        _scrollable(result["actual_svg"]),
        _fmt(metrics["er"]),
        _fmt(metrics["truth_er"]),
        _fmt(metrics["mae"]),
        _fmt(metrics["rmse"]),
    )


def _render_diff(result: dict[str, Any]) -> tuple[str, str]:
    """Wrap the two pre-rendered diff overlays (absolute, relative) as scrollable panes.

    Both keep the same grey/amber/red line styling; only the arc text differs — the
    raw ``forecast | actual`` pair (absolute) vs the signed change % (relative).
    """
    return (
        _scrollable(result["diff_absolute_svg"]),
        _scrollable(result["diff_relative_svg"]),
    )


def load(dataset: str, model: str) -> tuple[str, str, str, str, str, str]:
    """Resolve one bundled forecast into the six side-by-side UI outputs.

    Returns:
        ``(forecast_pane_html, actual_pane_html, er, truth_er, mae, rmse)``.
    """
    return _render_panes(forecast_bundled(dataset, model))


def load_diff(dataset: str, model: str) -> tuple[str, str]:
    """Resolve one bundled forecast into the two colour-coded diff overlay panes.

    Returns:
        ``(absolute_html, relative_html)`` — both wrapped scrollable panes with the
        same line coding (grey = matched, amber dashed = it happened but the forecast
        missed it, red dashed = the forecast predicted it but it did not happen); the
        absolute pane labels arcs ``forecast | actual``, the relative pane the signed
        change %.
    """
    return _render_diff(forecast_bundled(dataset, model))


def _drift_summary(drift: dict[str, list[dict[str, Any]]]) -> str:
    """One-line drift tally vs the last-known window — counts only, never accuracy."""
    return (
        "**Drift vs the last-known window** — "
        f"{len(drift['added'])} relation(s) the forecast **adds**, "
        f"{len(drift['removed'])} it **drops**, "
        f"{len(drift['stable'])} unchanged. "
        "_This is change from the recent past, not accuracy — an upload has no future "
        "truth to score against (ADR-0004)._"
    )


# Blank panes + cleared summary, reused when an upload is rejected or absent.
_LIVE_BLANK: tuple[str, str, str, str, str] = ("", "", "", "", "")

# Shown the instant Forecast is pressed, before the (slow, cold-loading) GPU call returns,
# so the live tab never just sits frozen for ~120s. Yielded ahead of the result below.
_LIVE_WORKING = (
    "⏳ Forecasting on ZeroGPU — the first call cold-loads Chronos-2 (this can take ~60s); "
    "later calls are fast. Hold on…"
)


def run_live(log_path: str | None, model: str):
    """Forecast an uploaded log and stream the live-tab outputs.

    A generator: it **yields an interim "working" state immediately** (so the UI shows
    progress, not a frozen pane, while the ~120s ZeroGPU call runs), then yields the final
    ``(status_md, forecast_html, comparison_html, diff_abs_html, diff_rel_html,
    drift_summary_md)``. A guard rejection (oversize / gated model / too-short log) or any
    other failure short-circuits to a clear status message with blank panes — the GPU is
    only reached for an accepted upload.
    """
    if not log_path:
        yield (
            "⬆️ Upload an XES event log (or pick the example), then press **Forecast**.",
            *_LIVE_BLANK,
        )
        return
    yield (_LIVE_WORKING, *_LIVE_BLANK)
    try:
        result = _run_live(log_path, model)
    except UploadRejected as rejected:
        yield (f"⚠️ Upload rejected: {rejected}", *_LIVE_BLANK)
        return
    except Exception as err:
        yield (f"⚠️ Could not forecast this log: {err}", *_LIVE_BLANK)
        return
    yield (
        "✅ Forecast complete — showing the genuine next week vs the last-known window.",
        _scrollable(result["forecast_svg"]),
        _scrollable(result["comparison_svg"]),
        _scrollable(result["diff_absolute_svg"]),
        _scrollable(result["diff_relative_svg"]),
        _drift_summary(result["drift"]),
    )


def _example_note() -> str:
    """Frame the one-click example so it doesn't read as a duplicate of the bundled sepsis
    entry: it is a ready-made **stand-in for your own upload**, and the live path forecasts
    *this slice's* genuine next week — distinct from the Bundled tab's backtest of the full log.
    """
    return (
        "The example is a six-week slice of the **sepsis** log (one of the bundled datasets), a "
        "ready-made stand-in for your own upload — the live path forecasts *this slice's* genuine "
        "next week and its drift. The **Bundled explorer** tab instead backtests the *full* sepsis "
        "log against known truth."
    )


def _live_caps_note() -> str:
    """The upload limits shown on the live tab, **derived** from the guard constants so the
    displayed caps can never drift from what ``upload_guard.check_upload`` actually enforces.
    """
    return (
        f"**Limits** — max upload **{MAX_UPLOAD_BYTES / 1e6:.1f} MB**; "
        f"Chronos-2 forecasts any log up to that cap."
    )


def _build_bundled_tab() -> tuple[Any, list[Any], list[Any]]:
    """The bundled-explorer tab (slice 1), unchanged but nested under a gr.Tab.

    Returns the ``(fn, inputs, outputs)`` the caller wires to ``demo.load`` (so the
    initial render happens once the Blocks context is open).
    """
    gr.Markdown(
        "Holdout backtest (ADR-0004): the real last week is **held out** and forecast "
        "from the rest, then compared against what actually happened — so ER / MAE / RMSE "
        "are genuine accuracy."
    )
    with gr.Accordion("What am I looking at?", open=False):
        gr.Markdown(
            "A **directly-follows graph (DFG)** summarises a process: each arc *a → b* is a "
            "**directly-follows (DF) relation** (b directly follows a), the number on it the "
            "count over the week.\n\n"
            "- **Forecast** — the held-out last week, as the model predicted it from prior weeks.\n"
            "- **Actual future** — what really happened that week.\n"
            "- **ER / MAE / RMSE** — how accurate the forecast was against that actual future. "
            "**Entropic Relevance (ER)** scores how well a DFG explains the real behaviour "
            "(lower is better; the *truth (baseline)* box is the ER of the actual DFG itself — the "
            "floor to beat). **MAE / RMSE** measure the DF-frequency error.\n\n"
            "Accuracy is meaningful here because the actual future is known. The **Live upload** "
            "tab forecasts a genuinely unseen future instead, so it reports **drift, never accuracy**."
        )
    with gr.Row():
        dataset = gr.Dropdown(
            DATASETS,
            value=DATASETS[0],
            label="Dataset",
            info="One of the paper's four event logs",
        )
        model = gr.Dropdown(
            MODELS,
            value=MODELS[0],
            label="Model",
            info="The time-series foundation model that produced the forecast",
        )
    view = gr.Radio(["Side-by-side", "Diff"], value="Side-by-side", label="View")
    with gr.Row(visible=True) as twin_panes:
        with gr.Column():
            gr.Markdown("### Forecast — the held-out last week, predicted")
            forecast_pane = gr.HTML(elem_classes=["dfg-pane"])
        with gr.Column():
            gr.Markdown("### Actual future — what really happened that week")
            actual_pane = gr.HTML(elem_classes=["dfg-pane"])
    with gr.Column(visible=False) as diff_overlay:
        gr.Markdown("### Diff — forecast vs actual future")
        diff_mode = gr.Radio(
            ["Absolute", "Relative"],
            value="Absolute",
            label="Diff mode",
            info="Absolute: forecast | actual. Relative: signed change %.",
        )
        with gr.Row(visible=True) as diff_abs_row:
            diff_abs_pane = gr.HTML(elem_classes=["dfg-pane"])
        with gr.Row(visible=False) as diff_rel_row:
            diff_rel_pane = gr.HTML(elem_classes=["dfg-pane"])
    with gr.Row():
        er = gr.Textbox(label="ER — forecast", interactive=False)
        truth_er = gr.Textbox(label="ER — truth (baseline)", interactive=False)
        mae = gr.Textbox(label="MAE", interactive=False)
        rmse = gr.Textbox(label="RMSE", interactive=False)

    outputs = [forecast_pane, actual_pane, er, truth_er, mae, rmse]
    inputs = [dataset, model]

    def refresh(dataset: str, model: str) -> tuple[Any, ...]:
        """Recompute every pane so every view stays in sync with the pickers."""
        result = forecast_bundled(dataset, model)
        return (*_render_panes(result), *_render_diff(result))

    all_outputs = [*outputs, diff_abs_pane, diff_rel_pane]
    for picker in inputs:
        picker.change(refresh, inputs, all_outputs)

    def set_view(view: str) -> tuple[Any, Any]:
        is_diff = view == "Diff"
        return gr.update(visible=not is_diff), gr.update(visible=is_diff)

    view.change(set_view, view, [twin_panes, diff_overlay])

    def set_diff_mode(mode: str) -> tuple[Any, Any]:
        is_rel = mode == "Relative"
        return gr.update(visible=not is_rel), gr.update(visible=is_rel)

    diff_mode.change(set_diff_mode, diff_mode, [diff_abs_row, diff_rel_row])
    return refresh, inputs, all_outputs


def _build_live_tab() -> None:
    """The live upload tab (slice 3b): upload → ZeroGPU forecast → drift view."""
    gr.Markdown(
        "Upload a custom **XES** log to forecast its genuine next week (forecast origin = "
        "the log end) on ZeroGPU, then read the **drift** of that forecast vs the "
        "**last-known window** — DF relations the forecast adds or drops. There is no "
        "future truth for an upload, so this path shows **drift, never accuracy** (ADR-0004)."
    )
    with gr.Accordion("What am I looking at?", open=False):
        gr.Markdown(
            "Each arc *a → b* in a **directly-follows graph (DFG)** is a **directly-follows "
            "(DF) relation** (b directly follows a). The live path forecasts the **genuine next "
            "week** from your log's end and compares it to the **last-known window** (its recent "
            "past):\n\n"
            "- **Forecast** — the predicted next week's DFG.\n"
            "- **Last-known window** — the most recent week already in your log.\n"
            "- **Drift** — the DF relations the forecast **adds** or **drops** vs that recent past.\n\n"
            "No accuracy metric (ER / MAE / RMSE) is shown: an upload has no future ground truth to "
            "score against (ADR-0004). For scored accuracy, see the **Bundled explorer** tab."
        )
    with gr.Row():
        upload = gr.File(label="XES event log", file_types=[".xes"], type="filepath")
        live_model = gr.Dropdown(
            LIVE_MODELS,
            value=LIVE_MODELS[0],
            label="Model",
            info="Forecasts run with Chronos-2 on the hosted GPU; Moirai-2 / TimesFM-2.5 live in the Bundled explorer tab",
        )
        run = gr.Button("Forecast", variant="primary")
    gr.Examples(
        examples=[[str(EXAMPLE_LOG)]],
        inputs=[upload],
        label="Try an example — a six-week slice of the sepsis log",
        cache_examples=False,
    )
    gr.Markdown(_example_note())
    gr.Markdown(_live_caps_note())
    status = gr.Markdown(
        "⬆️ Upload an XES event log (or pick the example), then press **Forecast**."
    )
    view = gr.Radio(
        ["Side-by-side", "Drift"],
        value="Side-by-side",
        label="View",
        info="Side-by-side: forecast vs last-known window. Drift: their overlay.",
    )
    with gr.Row(visible=True) as twin_panes:
        with gr.Column():
            gr.Markdown("### Forecast — the genuine next week, predicted")
            forecast_pane = gr.HTML(elem_classes=["dfg-pane"])
        with gr.Column():
            gr.Markdown("### Last-known window — the recent past it is compared against")
            comparison_pane = gr.HTML(elem_classes=["dfg-pane"])
    with gr.Column(visible=False) as drift_overlay:
        gr.Markdown("### Drift — forecast vs the last-known window")
        drift_mode = gr.Radio(
            ["Absolute", "Relative"],
            value="Absolute",
            label="Drift view",
            info="Absolute: forecast | last-known window. Relative: % change from recent past.",
        )
        with gr.Row(visible=True) as diff_abs_row:
            diff_abs_pane = gr.HTML(elem_classes=["dfg-pane"])
        with gr.Row(visible=False) as diff_rel_row:
            diff_rel_pane = gr.HTML(elem_classes=["dfg-pane"])
    drift_summary = gr.Markdown()

    run.click(
        run_live,
        [upload, live_model],
        [status, forecast_pane, comparison_pane, diff_abs_pane, diff_rel_pane, drift_summary],
    )

    def set_view(view: str) -> tuple[Any, Any]:
        is_drift = view == "Drift"
        return gr.update(visible=not is_drift), gr.update(visible=is_drift)

    view.change(set_view, view, [twin_panes, drift_overlay])

    def set_drift_mode(mode: str) -> tuple[Any, Any]:
        is_rel = mode == "Relative"
        return gr.update(visible=not is_rel), gr.update(visible=is_rel)

    drift_mode.change(set_drift_mode, drift_mode, [diff_abs_row, diff_rel_row])


def build() -> gr.Blocks:
    """Build the Gradio Blocks app (without launching it)."""
    with gr.Blocks(title="Process Model Forecasting explorer") as demo:
        gr.Markdown(
            "# Process Model Forecasting explorer\n"
            "Can off-the-shelf time-series foundation models forecast how a process's "
            "directly-follows behaviour evolves? Explore the paper's **bundled** backtests, "
            "or **upload your own log** for a live forecast of its next week."
        )
        with gr.Tabs():
            with gr.Tab("Bundled explorer"):
                load_target = _build_bundled_tab()
            with gr.Tab("Live upload (your log)"):
                _build_live_tab()

        # Render the bundled tab once on load (the live tab waits for an upload).
        refresh, inputs, all_outputs = load_target
        demo.load(refresh, inputs, all_outputs)

    # Gradio 6 takes `head` at launch() (not the Blocks constructor); stash it on
    # the Blocks so callers (main, tests, drivers) pass the same inlined payload.
    demo.demo_head = _head()
    return demo


def main() -> None:
    demo = build()
    demo.launch(head=demo.demo_head)


if __name__ == "__main__":
    main()
