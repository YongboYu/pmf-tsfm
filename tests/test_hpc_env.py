"""Regression tests for HPC environment helpers."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HPC_ENV_PATH = REPO_ROOT / "scripts" / "hpc" / "hpc_env.sh"
BASH_PATH = "/bin/bash"


def _run_hpc_env_helper(project_root: Path, extra_env: dict[str, str] | None = None) -> str:
    env = os.environ.copy()
    env.pop("SLURM_MAIL_USER", None)
    env.update(
        {
            "VSC_DATA": str(project_root / "vsc_data"),
            "VSC_SCRATCH": str(project_root / "vsc_scratch"),
            "PROJECT_ROOT": str(project_root),
        }
    )
    if extra_env:
        env.update(extra_env)

    result = subprocess.run(  # noqa: S603 - exercises a repo-controlled shell helper
        [
            BASH_PATH,
            "-lc",
            f'set -eu; source "{HPC_ENV_PATH}"; slurm_mail_directives "END,FAIL"',
        ],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return result.stdout


class TestSlurmMailDirectives:
    def test_mail_directives_present_when_mail_user_is_set(self, tmp_path: Path) -> None:
        output = _run_hpc_env_helper(
            tmp_path,
            {"SLURM_MAIL_USER": "user@example.com"},
        )

        assert "#SBATCH --mail-type=END,FAIL" in output
        assert "#SBATCH --mail-user=user@example.com" in output

    def test_mail_directives_omitted_when_mail_user_is_unset(self, tmp_path: Path) -> None:
        output = _run_hpc_env_helper(tmp_path)

        assert output == ""

    def test_project_env_file_propagates_mail_user(self, tmp_path: Path) -> None:
        (tmp_path / ".env").write_text("SLURM_MAIL_USER=env_user@example.com\n")

        output = _run_hpc_env_helper(tmp_path)

        assert "#SBATCH --mail-user=env_user@example.com" in output
