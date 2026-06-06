"""Deployment contract for the bundled forecast explorer's HF Space (slice 1, #113).

These tests pin the *serve-time* contract the Hugging Face Space relies on:

  - ``requirements.txt`` lists **gradio only** — no graphviz / pmf_tsfm / numpy / pandas
    (ADR-0005: serve-time deps stay gradio-only so the Space needs no ``dot`` binary and no GPU).
  - ``README.md`` carries valid HF Space YAML frontmatter (so HF builds the Space at all).
  - importing the serve path (``forecast`` / ``app``) never pulls the precompute-time modules.
  - every offered dataset x model combo has its committed assets (the Space serves files, ADR-0002).
  - the deploy workflow syncs only the serve-time subset of ``demo/`` to the Space.
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

# Heavy / precompute-time deps that must never reach the gradio-only serve runtime (ADR-0005).
_FORBIDDEN_SERVE_DEPS = ("graphviz", "pmf-tsfm", "pmf_tsfm", "numpy", "pandas", "torch")


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


def test_requirements_pins_gradio_and_nothing_heavy():
    names = _requirement_names((DEMO / "requirements.txt").read_text())
    assert "gradio" in names
    assert not (names & set(_FORBIDDEN_SERVE_DEPS)), (
        f"requirements.txt must stay gradio-only (ADR-0005); found {names & set(_FORBIDDEN_SERVE_DEPS)}"
    )


def test_readme_has_valid_hf_space_frontmatter():
    fm = _frontmatter((DEMO / "README.md").read_text())
    assert fm.get("sdk") == "gradio", "HF Space must be a Gradio-SDK Space (ADR-0003 amended)"
    assert fm.get("app_file") == "app.py", "app_file must point at the Space-root app.py"
    assert "sdk_version" in fm, "HF needs a pinned sdk_version to build the Space"
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


def _assert_clean_serve_import(import_stmt: str) -> None:
    # In a fresh interpreter with only demo/ on the path (no src/), importing the serve path must
    # succeed and must NOT have pulled in the precompute-time modules. Run in a subprocess so the
    # in-process suite (which imports render/precompute_demo) can't pollute the module table.
    code = (
        f"import sys; sys.path.insert(0, {str(DEMO)!r}); "
        f"{import_stmt}; "
        "bad = [m for m in ('render', 'precompute_demo', 'graphviz', 'pmf_tsfm') "
        "if m in sys.modules]; "
        "assert not bad, bad"
    )
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell, trusted constant code
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_serve_path_imports_without_precompute_deps():
    # forecast.py is the gradio-free serve core — always checkable, incl. in CI (no gradio there).
    # This guards ADR-0005 against a stray `import render` (which would drag in graphviz).
    _assert_clean_serve_import("import forecast")
    # app.py is the other half of the serve path but imports gradio; only check it where gradio
    # is installed (locally / on the Space), matching the demo suite's gradio-optional convention.
    if importlib.util.find_spec("gradio") is not None:
        _assert_clean_serve_import("import app, forecast")


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


def test_deploy_workflow_syncs_only_the_serve_time_subset():
    text = WORKFLOW.read_text()
    # Triggered by demo/ changes on main, authenticated with the HF token secret.
    assert "branches:" in text and "main" in text
    assert "demo/" in text, "workflow should be path-filtered to demo/ changes"
    assert "HF_TOKEN" in text, "workflow must auth to the Space with the HF_TOKEN secret"
    # The gradio-only Space must never receive precompute-time modules (ADR-0005).
    for forbidden in ("precompute_demo.py", "render.py"):
        assert forbidden not in text, (
            f"deploy workflow must not sync {forbidden} to the Space (keep it gradio-only)"
        )
