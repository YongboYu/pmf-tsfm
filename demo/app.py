"""Gradio app — the bundled forecast explorer (demo slice 1).

Thin glue over the deep module ``forecast.forecast_bundled``: a dataset/model
picker drives twin panes (forecast DFG | actual-future DFG), with a
``[Side-by-side | Diff]`` toggle that swaps them for one colour-coded diff
overlay. An ER/MAE/RMSE metrics strip stays below in both views. The bundled
path is a holdout backtest (ADR-0004), so the metrics are genuine accuracy.

Served entirely from precomputed assets — including the pre-rendered SVG figures,
so the app needs no ``dot`` binary at runtime — GPU-free. Launch plainly;
``mcp_server`` stays off in this slice (ADR-0003).

    uv run --with gradio python demo/app.py
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import gradio as gr
from forecast import forecast_bundled

# Only the precomputed bundled pairs are offered as choices: the full 4-by-3 matrix.
DATASETS: list[str] = ["bpi2017", "bpi2019_1", "sepsis", "hospital_billing"]
MODELS: list[str] = ["chronos2", "moirai2", "timesfm2.5"]

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


def _render_diff(result: dict[str, Any]) -> str:
    """Wrap the pre-rendered colour-coded diff overlay SVG in a scrollable pane."""
    return _scrollable(result["diff_svg"])


def load(dataset: str, model: str) -> tuple[str, str, str, str, str, str]:
    """Resolve one bundled forecast into the six side-by-side UI outputs.

    Returns:
        ``(forecast_pane_html, actual_pane_html, er, truth_er, mae, rmse)``.
    """
    return _render_panes(forecast_bundled(dataset, model))


def load_diff(dataset: str, model: str) -> str:
    """Resolve one bundled forecast into the colour-coded diff overlay pane.

    Returns:
        The diff overlay SVG wrapped in a scrollable pane (grey = matched,
        amber dashed = it happened but the forecast missed it, red dashed = the
        forecast predicted it but it did not happen).
    """
    return _render_diff(forecast_bundled(dataset, model))


def build() -> gr.Blocks:
    """Build the Gradio Blocks app (without launching it)."""
    with gr.Blocks(title="Bundled forecast explorer") as demo:
        gr.Markdown(
            "# Bundled forecast explorer\n"
            "Holdout backtest (ADR-0004): the real last week is **held out** and forecast "
            "from the rest, then compared against what actually happened — so ER / MAE / RMSE "
            "are genuine accuracy."
        )
        with gr.Row():
            dataset = gr.Dropdown(DATASETS, value=DATASETS[0], label="Dataset")
            model = gr.Dropdown(MODELS, value=MODELS[0], label="Model")
        view = gr.Radio(
            ["Side-by-side", "Diff"],
            value="Side-by-side",
            label="View",
        )
        with gr.Row(visible=True) as twin_panes:
            with gr.Column():
                gr.Markdown("### Forecast — the held-out last week, predicted")
                forecast_pane = gr.HTML(elem_classes=["dfg-pane"])
            with gr.Column():
                gr.Markdown("### Actual future — what really happened that week")
                actual_pane = gr.HTML(elem_classes=["dfg-pane"])
        with gr.Row(visible=False) as diff_overlay, gr.Column():
            gr.Markdown("### Diff — forecast vs actual future")
            diff_pane = gr.HTML(elem_classes=["dfg-pane"])
        with gr.Row():
            er = gr.Textbox(label="ER — forecast", interactive=False)
            truth_er = gr.Textbox(label="ER — truth (baseline)", interactive=False)
            mae = gr.Textbox(label="MAE", interactive=False)
            rmse = gr.Textbox(label="RMSE", interactive=False)

        outputs = [forecast_pane, actual_pane, er, truth_er, mae, rmse]
        inputs = [dataset, model]

        def refresh(dataset: str, model: str) -> tuple[Any, ...]:
            """Recompute every pane so both views stay in sync with the pickers.

            Resolves the forecast once and fans it into both views, so the diff
            overlay and the side-by-side panes always show the same triple.
            """
            result = forecast_bundled(dataset, model)
            return (*_render_panes(result), _render_diff(result))

        all_outputs = [*outputs, diff_pane]
        for picker in inputs:
            picker.change(refresh, inputs, all_outputs)
        demo.load(refresh, inputs, all_outputs)

        def set_view(view: str) -> tuple[Any, Any]:
            """Swap the twin panes for the diff overlay; metrics strip stays put."""
            is_diff = view == "Diff"
            return gr.update(visible=not is_diff), gr.update(visible=is_diff)

        view.change(set_view, view, [twin_panes, diff_overlay])

    # Gradio 6 takes `head` at launch() (not the Blocks constructor); stash it on
    # the Blocks so callers (main, tests, drivers) pass the same inlined payload.
    demo.demo_head = _head()
    return demo


def main() -> None:
    demo = build()
    demo.launch(head=demo.demo_head)


if __name__ == "__main__":
    main()
