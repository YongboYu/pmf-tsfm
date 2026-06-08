# syntax=docker/dockerfile:1

# =============================================================================
# pmf-tsfm — self-host image for the core forecasting pipeline (issue #134)
#
# Build context = REPO ROOT:
#     docker build -f docker/Dockerfile -t pmf-tsfm .
#
# Default image = Chronos + Moirai (both are regular deps). TimesFM is opt-in
# (fragile/large GitHub-pinned zip, needs python >= 3.11):
#     docker build -f docker/Dockerfile --build-arg INSTALL_TIMESFM=1 -t pmf-tsfm:timesfm .
#
# Built with uv (NOT pip): the [tool.uv] override-dependencies = ["torch>=2.8.0"]
# in pyproject.toml resolves the uni2ts<->chronos numpy/torch clash, and that
# resolution only holds under uv + uv.lock. Plain pip silently breaks it.
# =============================================================================

# ---- Builder: resolve + install the locked env into /app/.venv via uv --------
FROM python:3.12-slim AS builder

# uv binary from its official distroless image.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# TimesFM opt-in: 0 (default) = Chronos + Moirai only; 1 = add the timesfm_v25 extra.
ARG INSTALL_TIMESFM=0

# Step 1 — dependencies only (no project). Cached unless pyproject/uv.lock change.
# .python-version (3.12) pins the interpreter uv resolves against to the system one.
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project \
      $( [ "$INSTALL_TIMESFM" = "1" ] && echo "--extra timesfm_v25" )

# Step 2 — bring in source + configs, then editable-install the project.
# Editable install is REQUIRED: api.py and the Hydra @hydra.main decorators resolve
# configs/ RELATIVE TO THE MODULE FILE (parents[2] / "configs", ../../configs). So
# src/ must live at /app/src and configs/ at /app/configs at runtime — a non-editable
# copy into site-packages would break config resolution. README.md is needed because
# pyproject sets readme = "README.md" (hatchling reads it when building the project).
COPY README.md ./README.md
COPY src/ ./src/
COPY configs/ ./configs/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev \
      $( [ "$INSTALL_TIMESFM" = "1" ] && echo "--extra timesfm_v25" )

# ---- Runtime: slim image with graphviz + the prebuilt venv -------------------
FROM python:3.12-slim

# graphviz: DFG / Entropic-Relevance rendering (mirrors demo/packages.txt).
# tini: a real PID-1 init. Without it python runs as PID 1, so os.getppid()==0 and
# pm4py's import-time psutil.Process(getppid()) raises NoSuchProcess. tini gives the
# python process a valid parent (PID 1 = tini), so pm4py imports cleanly.
RUN apt-get update \
    && apt-get install -y --no-install-recommends graphviz ca-certificates tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the whole /app (venv + src + configs + pyproject) at the SAME path so the
# editable install's .pth -> /app/src keeps resolving.
COPY --from=builder /app /app

# Docker-only launcher + thin backtest CLI (kept out of the package).
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
COPY docker/backtest_cli.py /app/docker/backtest_cli.py
RUN chmod +x /usr/local/bin/entrypoint.sh

# Weight caches, unified under /cache so ONE volume persists every model family:
#   HF_HOME          -> HF-hub models (chronos bolt_*, moirai/*, timesfm/*)
#   XGD_CACHE_HOME    -> Chronos-2, which loads from s3://autogluon/chronos-2/ and
#                        caches to $XGD_CACHE_HOME/chronos/ (note: the chronos lib
#                        literally reads the misspelled "XGD_CACHE_HOME", not XDG).
ENV PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/cache/huggingface \
    XGD_CACHE_HOME=/cache \
    PYTHONUNBUFFERED=1 \
    PMF_DEVICE=cpu \
    PMF_OUTPUT=/work/outputs

# Mount a volume over /cache to persist weights across runs; /work/outputs for artifacts.
VOLUME ["/cache", "/work/outputs"]

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/entrypoint.sh"]
CMD ["list-models"]
