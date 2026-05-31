"""Gradio app — the bundled forecast explorer (demo slice 1).

Thin glue over the deep modules ``forecast.forecast_bundled`` and
``render.dfg_json_to_svg``: a dataset/model picker drives twin panes (forecast
DFG | actual-future DFG) with an ER/MAE/RMSE metrics strip below. The bundled
path is a holdout backtest (ADR-0004), so the metrics are genuine accuracy.

Served entirely from precomputed assets — GPU-free. Launch plainly; ``mcp_server``
stays off in this slice (ADR-0003).

    uv run --with gradio python demo/app.py
"""

from __future__ import annotations

from typing import Any

import gradio as gr
from forecast import forecast_bundled
from render import dfg_json_to_svg

# Slice-1 scope: only the precomputed bundled pairs are offered as choices.
DATASETS: list[str] = ["bpi2017"]
MODELS: list[str] = ["chronos2"]


def _pane(dfg: dict[str, Any]) -> str:
    """Wrap a rendered DFG SVG so it scrolls within its pane."""
    return f'<div style="overflow:auto; max-height:70vh">{dfg_json_to_svg(dfg)}</div>'


def load(dataset: str, model: str) -> tuple[str, str, str, str, str]:
    """Resolve one bundled forecast into the five UI outputs.

    Returns:
        ``(forecast_pane_html, actual_pane_html, er, mae, rmse)``.
    """
    result = forecast_bundled(dataset, model)
    metrics = result["metrics"]
    return (
        _pane(result["forecast_dfg"]),
        _pane(result["actual_dfg"]),
        f"{metrics['er']:.3f}",
        f"{metrics['mae']:.3f}",
        f"{metrics['rmse']:.3f}",
    )


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
        with gr.Row():
            forecast_pane = gr.HTML(label="Forecast DFG")
            actual_pane = gr.HTML(label="Actual-future DFG")
        with gr.Row():
            er = gr.Textbox(label="ER", interactive=False)
            mae = gr.Textbox(label="MAE", interactive=False)
            rmse = gr.Textbox(label="RMSE", interactive=False)

        outputs = [forecast_pane, actual_pane, er, mae, rmse]
        inputs = [dataset, model]
        for picker in inputs:
            picker.change(load, inputs, outputs)
        demo.load(load, inputs, outputs)
    return demo


def main() -> None:
    build().launch()


if __name__ == "__main__":
    main()
