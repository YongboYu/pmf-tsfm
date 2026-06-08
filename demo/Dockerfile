# Self-host Docker image for the Process Model Forecasting explorer (ADR-0003 amendment,
# "one codebase, two packagings"). The hosted runtime stays a Gradio-SDK HF Space; this is
# the SECOND packaging — a self-host image (GUI by default) for talk-recording and for
# others running the demo on their own logs/hardware with no HF caps and no ZeroGPU
# cold-start. Built from the SAME demo/ code.
#
#   Build:  docker build -t pmf-tsfm-demo demo/
#   Run:    docker run -p 7860:7860 pmf-tsfm-demo
#   Open:   http://localhost:7860
#
# The live tab runs on plain CPU here: there is no `spaces` package in the image, so the
# @spaces.GPU decorator in app.py no-ops (app.py:59-66) and the same code runs without
# ZeroGPU. The first live forecast downloads the Chronos-2 weights from
# s3://autogluon/chronos-2/ (one-off, a few minutes on a cold cache). To persist that cache
# across container runs, mount a volume at HF_HOME:
#
#   docker run -p 7860:7860 -v pmf-hf-cache:/app/.cache/huggingface pmf-tsfm-demo
#
# This is the MVP GUI-only target (issue #129): the Hydra research CLI (src/pmf_tsfm/) is
# NOT installed — that is the future full-image target (build context = repo root).

FROM python:3.11-slim

# graphviz provides the `dot` binary the live path needs to render DFGs (mirrors
# demo/packages.txt). The bundled tab serves pre-rendered SVGs and needs none of this.
RUN apt-get update \
    && apt-get install -y --no-install-recommends graphviz \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the serve-time deps, but NOT `spaces`: ZeroGPU is HF-only and its absence makes
# the live forecast fall back to CPU (see header). Filtering keeps requirements.txt the
# single source of truth rather than forking a docker-specific copy.
COPY requirements.txt ./requirements.txt
RUN grep -v '^spaces' requirements.txt > /tmp/requirements.docker.txt \
    && pip install --no-cache-dir -r /tmp/requirements.docker.txt

# app.py uses flat imports (`from forecast import ...`), so the demo/ files sit at WORKDIR.
COPY . /app

# Gradio binds to 127.0.0.1 by default, unreachable from outside the container. These env
# vars make app.py's `demo.launch()` serve on all interfaces without any code change (keeps
# main() un-forked so the future MCP slice's mcp_server=True flows straight through).
ENV GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    HF_HOME=/app/.cache/huggingface

EXPOSE 7860

CMD ["python", "app.py"]
