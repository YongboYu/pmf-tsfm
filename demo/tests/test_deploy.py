"""Deployment contract for the explorer's HF Space (slice 1 #113, slice 3b #115).

These tests pin the *serve-time* contract the Hugging Face Space relies on. ADR-0005 is
**amended** for slice 3b: the bundled path is still served from precomputed assets and
stays import-pure, but the live tab forecasts uploads on ZeroGPU and renders their DFGs
at request time, so the Space now legitimately carries graphviz + the model libs. The
one hard line that remains: the Space never installs **pmf_tsfm** (its heavy model/uni2ts
tree) — the live path reuses only the lean, vendored demo modules.

  - ``requirements.txt`` pins gradio (matching the README) and carries the live-path deps,
    but never ``pmf_tsfm``.
  - ``README.md`` carries valid HF Space YAML frontmatter incl. ``suggested_hardware`` for ZeroGPU.
  - ``import forecast`` (the bundled core) stays pure; ``import app`` may pull the live deps
    (render/graphviz) but never ``pmf_tsfm`` or the precompute module.
  - every offered bundled dataset x model combo has its committed assets (ADR-0002).
  - the deploy workflow syncs the serve-time subset of ``demo/`` (incl. the live modules), but
    not ``precompute_demo.py`` (it imports pmf_tsfm).
"""

from __future__ import annotations

import ast
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

DEMO = Path(__file__).resolve().parent.parent
REPO = DEMO.parent
WORKFLOW = REPO / ".github" / "workflows" / "deploy-demo.yml"

# The one dep that must never reach the Space: pmf_tsfm (the heavy paper package with its
# wandb/training tree). The live path installs the model libs directly and reuses the
# vendored demo modules instead (ADR-0005 amended, #115).
_FORBIDDEN_SERVE_DEPS = ("pmf-tsfm", "pmf_tsfm")
# The live tab needs these at serve time (ZeroGPU forecast + request-time DFG rendering):
# the three model libs (chronos-forecasting / uni2ts / timesfm) + render/GPU shims.
_REQUIRED_LIVE_DEPS = (
    "spaces",
    "graphviz",
    "torch",
    "chronos-forecasting",
    "uni2ts",
    "timesfm",
    "pm4py",
)


def _requirement_names(text: str) -> set[str]:
    """The bare package names in a requirements.txt, lower-cased, stripped of version specifiers."""
    names = set()
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        name = re.split(r"[<>=!~ \[]", line, maxsplit=1)[0].strip().lower()
        if name:
            names.add(name)
    return names


def _frontmatter(text: str) -> dict[str, str]:
    """Parse a leading ``---``-delimited YAML block of flat ``key: value`` pairs."""
    m = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    if not m:
        return {}
    out = {}
    for line in m.group(1).splitlines():
        if ":" in line and not line.lstrip().startswith("#"):
            key, _, val = line.partition(":")
            out[key.strip()] = val.strip()
    return out


def test_requirements_serve_both_paths_but_never_pmf_tsfm():
    names = _requirement_names((DEMO / "requirements.txt").read_text())
    assert "gradio" in names
    # The live tab's serve deps are present...
    missing = set(_REQUIRED_LIVE_DEPS) - names
    assert not missing, f"requirements.txt is missing live-path serve deps: {missing}"
    # ...but pmf_tsfm (the heavy tree) is never installed on the Space (ADR-0005 amended).
    assert not (names & set(_FORBIDDEN_SERVE_DEPS)), (
        f"requirements.txt must never carry pmf_tsfm; found {names & set(_FORBIDDEN_SERVE_DEPS)}"
    )


def test_readme_has_valid_hf_space_frontmatter():
    fm = _frontmatter((DEMO / "README.md").read_text())
    assert fm.get("sdk") == "gradio", "HF Space must be a Gradio-SDK Space (ADR-0003 amended)"
    assert fm.get("app_file") == "app.py", "app_file must point at the Space-root app.py"
    assert "sdk_version" in fm, "HF needs a pinned sdk_version to build the Space"
    # The live tab runs on ZeroGPU — the Space must request that hardware (#115).
    assert fm.get("suggested_hardware") == "zero-a10g", (
        "README must request ZeroGPU hardware (suggested_hardware: zero-a10g) for the live tab"
    )
    # The pinned Space SDK version must agree with the serve-time requirements pin.
    req = (DEMO / "requirements.txt").read_text()
    pin = next(
        (
            line.split("==", 1)[1].strip()
            for line in req.splitlines()
            if line.strip().lower().startswith("gradio==")
        ),
        None,
    )
    assert fm["sdk_version"] == pin, (
        f"README sdk_version ({fm['sdk_version']}) must match requirements.txt gradio pin ({pin})"
    )


def _assert_serve_import_excludes(import_stmt: str, forbidden: tuple[str, ...]) -> None:
    # In a fresh interpreter with only demo/ on the path (no src/), importing the serve path must
    # succeed and must NOT have pulled in any ``forbidden`` module. Run in a subprocess so the
    # in-process suite (which imports render/precompute_demo) can't pollute the module table.
    code = (
        f"import sys; sys.path.insert(0, {str(DEMO)!r}); "
        f"{import_stmt}; "
        f"bad = [m for m in {forbidden!r} if m in sys.modules]; "
        "assert not bad, bad"
    )
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell, trusted constant code
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_serve_path_imports_stay_pmf_tsfm_free():
    # The bundled core stays import-pure: `import forecast` must drag in none of the precompute /
    # render / graphviz / pmf_tsfm machinery (it reads precomputed assets only). Checkable in CI.
    _assert_serve_import_excludes(
        "import forecast", ("render", "precompute_demo", "graphviz", "pmf_tsfm")
    )
    # app.py imports gradio (and, for the live tab, render/graphviz) — only checkable where gradio
    # is installed. The amended invariant: the whole serve path still never pulls pmf_tsfm or the
    # precompute module (the live path reuses the vendored demo modules instead).
    if importlib.util.find_spec("gradio") is not None:
        _assert_serve_import_excludes("import app, forecast", ("precompute_demo", "pmf_tsfm"))


# The serve artifacts every bundled dataset x model combo must ship (ADR-0002/0005).
_ASSET_FILES = (
    "forecast_dfg.json",
    "actual_dfg.json",
    "metrics.json",
    "forecast.svg",
    "actual.svg",
    "diff_absolute.svg",
    "diff_relative.svg",
)


def _app_string_list(name: str) -> list[str]:
    """Read a module-level ``name: list[str] = [...]`` literal from app.py without importing it.

    Importing app.py would pull in gradio, which is absent in CI; the dropdown choices are plain
    string literals, so we lift them straight from the source instead.
    """
    tree = ast.parse((DEMO / "app.py").read_text())
    for node in tree.body:
        target = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target = node.target.id
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            target = node.targets[0].id
        if target == name and isinstance(node.value, (ast.List, ast.Tuple)):
            return [el.value for el in node.value.elts if isinstance(el, ast.Constant)]
    raise AssertionError(f"{name} list literal not found in app.py")


def test_every_offered_combo_has_committed_assets():
    # The dropdown choices are the source of truth for what must be served.
    datasets = _app_string_list("DATASETS")
    models = _app_string_list("MODELS")
    assert datasets and models, "could not read DATASETS/MODELS from app.py"

    missing = []
    for dataset in datasets:
        for model in models:
            for fname in _ASSET_FILES:
                path = DEMO / "assets" / dataset / model / fname
                if not path.is_file():
                    missing.append(str(path.relative_to(DEMO)))
    assert not missing, f"offered combos missing committed assets: {missing}"


def test_deploy_workflow_syncs_the_serve_time_subset():
    text = WORKFLOW.read_text()
    # Triggered by demo/ changes on main, authenticated with the HF token secret.
    assert "branches:" in text and "main" in text
    assert "demo/" in text, "workflow should be path-filtered to demo/ changes"
    assert "HF_TOKEN" in text, "workflow must auth to the Space with the HF_TOKEN secret"
    # The live tab's modules + apt packages must be staged for the Space (#115).
    for required in (
        "forecast_live.py",
        "upload_guard.py",
        "log_to_series.py",
        "dfg_build.py",
        "dfg_diff.py",
        "render.py",
        "packages.txt",
    ):
        assert required in text, f"deploy workflow must sync {required} for the live tab"
    # The live tab's one-click example log must reach the Space (gr.Examples points at it).
    assert "demo/examples" in text, "deploy workflow must sync demo/examples for the live example"
    # precompute_demo.py imports pmf_tsfm, which the Space never installs — keep it out.
    assert "precompute_demo.py" not in text, (
        "deploy workflow must not sync precompute_demo.py (it imports pmf_tsfm)"
    )
