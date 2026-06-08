#!/usr/bin/env bash
# Dispatch table for the pmf-tsfm self-host image (issue #134).
# The first argument selects the surface; everything after it passes through.
set -euo pipefail

cmd="${1:-list-models}"
shift || true

case "$cmd" in
  backtest)
    # Thin argparse CLI over pmf_tsfm.api.forecast_backtest (BYO-data happy path).
    exec python /app/docker/backtest_cli.py "$@"
    ;;
  inference)
    exec python -m pmf_tsfm.inference "$@"
    ;;
  train)
    exec python -m pmf_tsfm.train "$@"
    ;;
  evaluate)
    exec python -m pmf_tsfm.evaluate "$@"
    ;;
  evaluate_er)
    exec python -m pmf_tsfm.er.evaluate_er "$@"
    ;;
  preprocess)
    # Module lives under data/ — note the `data.` prefix.
    exec python -m pmf_tsfm.data.preprocess "$@"
    ;;
  mcp)
    # The MCP server ships with the #133 track (mcp/server.py); guard until it lands.
    if python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('mcp.server') else 1)"; then
      exec python -m mcp.server "$@"
    fi
    echo "error: the 'mcp' server is not bundled in this image." >&2
    echo "       It ships with the MCP track (#133, mcp/server.py); rebuild once that merges." >&2
    exit 2
    ;;
  list-models)
    exec python -c "from pmf_tsfm.api import list_models; print('\n'.join(list_models()))"
    ;;
  -h | --help | help)
    cat >&2 <<'EOF'
pmf-tsfm — self-host image for the core forecasting pipeline.

Usage: docker run [docker opts] pmf-tsfm <command> [args]

Commands:
  backtest --input <log.xes|.parquet> [--model chronos/chronos2] [--horizon 7] [--device cpu]
                        Zero-shot holdout backtest on your own log; prints forecast + MAE/RMSE/ER.
  inference    <hydra overrides>   Run the Hydra inference CLI (python -m pmf_tsfm.inference).
  evaluate     <hydra overrides>   Run the evaluation CLI.
  evaluate_er  <hydra overrides>   Run the Entropic Relevance CLI.
  preprocess   <hydra overrides>   Run the preprocessing CLI.
  train        <hydra overrides>   Run the training CLI (LoRA / full fine-tune).
  list-models                      List available model config groups.
  mcp                              Launch the FastMCP server (requires the #133 mcp/ track).

Examples:
  docker run -v "$PWD/data:/data" -v pmf-hf-cache:/cache/huggingface pmf-tsfm \
    backtest --input /data/processed_logs/sepsis.xes --model chronos/chronos2
  docker run pmf-tsfm inference data=bpi2017 model=chronos/chronos2 device=cpu
EOF
    exit 2
    ;;
  *)
    echo "error: unknown command '$cmd'" >&2
    echo "       try: backtest, inference, evaluate, evaluate_er, preprocess, train, list-models, mcp, help" >&2
    exit 2
    ;;
esac
