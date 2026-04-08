"""Helpers for resolving dataset asset paths across naming variants.

The repo's canonical dataset IDs remain:
    BPI2017, BPI2019_1, Sepsis, Hospital_Billing

Published Zenodo bundles may use lowercase filenames instead:
    bpi2017.parquet, bpi2019_1.parquet, sepsis.xes, hospital_billing.parquet

These helpers keep the canonical IDs stable while allowing the pipeline to
accept both the repo-native and Zenodo-published asset names.
"""

from __future__ import annotations

from pathlib import Path


def _ordered_unique(paths: list[Path]) -> list[Path]:
    """Preserve order while removing duplicate candidate paths."""
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def _existing_path_with_actual_case(path: Path) -> Path | None:
    """Return the actual on-disk path, preserving filename casing when possible."""
    if not path.exists():
        return None

    parent = path.parent
    if not parent.exists():
        return path

    exact_match: Path | None = None
    case_insensitive_match: Path | None = None
    for child in parent.iterdir():
        if child.name == path.name:
            exact_match = child
            break
        if child.name.lower() == path.name.lower() and case_insensitive_match is None:
            case_insensitive_match = child

    return exact_match or case_insensitive_match or path


def candidate_dataset_asset_paths(
    preferred_path: str | Path,
    dataset_name: str | None = None,
) -> list[Path]:
    """Build candidate paths for a dataset asset using canonical and lowercase names."""
    preferred = Path(preferred_path)
    suffix = preferred.suffix

    candidates = [preferred]
    if preferred.name.lower() != preferred.name:
        candidates.append(preferred.with_name(preferred.name.lower()))

    if dataset_name is not None:
        candidates.append(preferred.with_name(f"{dataset_name}{suffix}"))
        candidates.append(preferred.with_name(f"{dataset_name.lower()}{suffix}"))

    return _ordered_unique(candidates)


def resolve_dataset_asset_path(
    preferred_path: str | Path,
    *,
    dataset_name: str | None = None,
    asset_label: str = "dataset asset",
) -> Path:
    """Resolve an existing path for a dataset asset, accepting lowercase fallbacks."""
    candidates = candidate_dataset_asset_paths(preferred_path, dataset_name)
    for candidate in candidates:
        existing = _existing_path_with_actual_case(candidate)
        if existing is not None:
            return existing

    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"{asset_label.capitalize()} not found. Tried: {tried}")
