"""Path helpers for scripts run from the repository root."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the repository/workspace root inferred from this file."""

    return Path(__file__).resolve().parents[3]


def default_train_data() -> Path:
    return project_root() / "数据集" / "dataset_train_unified.json"


def default_frr_data() -> Path:
    return project_root() / "数据集" / "dataset_frr_calibration.json"
