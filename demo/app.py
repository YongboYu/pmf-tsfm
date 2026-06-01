"""Gradio app — the bundled forecast explorer (demo slice 1).

Thin glue over the deep modules ``forecast.forecast_bundled``,
``render.dfg_json_to_svg`` / ``render.diff_svg`` and ``dfg_diff``: a
dataset/model picker drives twin panes (forecast DFG | actual-future DFG), with
a ``[Side-by-side | Diff]`` toggle that swaps them for one colour-coded diff
overlay. An ER/MAE/RMSE metrics strip stays below in both views. The bundled
path is a holdout backtest (ADR-0004), so the metrics are genuine accuracy.

Served entirely from precomputed assets — GPU-free. Launch plainly; ``mcp_server``
stays off in this slice (ADR-0003).

    uv run --with gradio python demo/app.py
"""

from __future__ import annotations

from typing import Any

import gradio as gr
from forecast import forecast_bundled
from render import dfg_json_to_svg, diff_svg

# Slice-1 scope: only the precomputed bundled pairs are offered as choices.
DATASETS: list[str] = ["bpi2017"]
MODELS: list[str] = ["chronos2"]


def _scrollable(svg: str) -> str:
    """Wrap a rendered SVG so it scrolls within its pane, centred horizontally.

    Gradio renders ``<svg>`` as ``display:block``, so it ignores ``text-align``;
    ``margin:0 auto`` on the element itself centres it when it is narrower than the
    pane, while ``overflow:auto`` keeps a wider graph fully scrollable.
    """
    centred = svg.replace("<svg ", '<svg style="display:block;margin:0 auto" ', 1)
    return f'<div style="overflow:auto; max-height:70vh">{centred}</div>'


def _pane(dfg: dict[str, Any]) -> str:
    """Render a DFG to a scrollable SVG pane."""
    return _scrollable(dfg_json_to_svg(dfg))


def _render_panes(result: dict[str, Any]) -> tuple[str, str, str, str, str]:
    """Render the side-by-side outputs from a resolved forecast triple."""
    metrics = result["metrics"]
    return (
        _pane(result["forecast_dfg"]),
        _pane(result["actual_dfg"]),
        f"{metrics['er']:.3f}",
        f"{metrics['mae']:.3f}",
        f"{metrics['rmse']:.3f}",
    )


def _render_diff(result: dict[str, Any]) -> str:
    """Render the colour-coded diff overlay pane from a resolved forecast triple."""
    return _scrollable(diff_svg(result["forecast_dfg"], result["actual_dfg"]))


def load(dataset: str, model: str) -> tuple[str, str, str, str, str]:
    """Resolve one bundled forecast into the five side-by-side UI outputs.

    Returns:
        ``(forecast_pane_html, actual_pane_html, er, mae, rmse)``.
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
            forecast_pane = gr.HTML(label="Forecast DFG")
            actual_pane = gr.HTML(label="Actual-future DFG")
        with gr.Row(visible=False) as diff_overlay:
            diff_pane = gr.HTML(label="Diff (forecast vs actual)")
        with gr.Row():
            er = gr.Textbox(label="ER", interactive=False)
            mae = gr.Textbox(label="MAE", interactive=False)
            rmse = gr.Textbox(label="RMSE", interactive=False)

        outputs = [forecast_pane, actual_pane, er, mae, rmse]
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
    return demo


def main() -> None:
    build().launch()


if __name__ == "__main__":
    main()
