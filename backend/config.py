"""Configuration for the reliability-aware chest X-ray MVP.

The service deliberately keeps all data local.  Paths can be supplied with
environment variables so a MIMIC download does not need to be copied into
the project directory.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
]

DEFAULT_THRESHOLDS = {
    label: {
        "positive_threshold": 0.65,
        "negative_threshold": 0.35,
        "min_reliability": 0.30,
    }
    for label in LABELS
}


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


@dataclass(frozen=True)
class Settings:
    image_root: Path = field(default_factory=lambda: _env_path("MIMIC_IMAGE_ROOT", r"D:\MIMIC\jpg"))
    data_root: Path = field(
        default_factory=lambda: _env_path(
            "MIMIC_DATA_ROOT", r"C:\Users\Krishna Karthik\Downloads"
        )
    )
    checkpoint_path: Path = field(
        default_factory=lambda: _env_path(
            "MIMIC_CHECKPOINT", str(Path(__file__).resolve().parent / "artifacts" / "baseline.joblib")
        )
    )
    deep_checkpoint_path: Path = field(
        default_factory=lambda: _env_path(
            "MIMIC_DENSENET_CHECKPOINT", str(Path(__file__).resolve().parent / "artifacts" / "densenet121.pt")
        )
    )
    clinical_model_path: Path = field(
        default_factory=lambda: _env_path(
            "MIMIC_CLINICAL_MODEL", str(Path(__file__).resolve().parent / "artifacts" / "clinical_model.joblib")
        )
    )
    threshold_path: Path = field(
        default_factory=lambda: _env_path(
            "MIMIC_THRESHOLDS", str(Path(__file__).resolve().parent / "artifacts" / "thresholds.json")
        )
    )
    scan_ttl_seconds: int = field(default_factory=lambda: int(os.environ.get("MIMIC_SCAN_TTL_SECONDS", "300")))
    max_upload_bytes: int = field(
        default_factory=lambda: int(os.environ.get("MIMIC_MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
    )


def load_thresholds(path: Path) -> dict[str, dict[str, float]]:
    """Load per-label threshold settings, falling back safely per label."""

    thresholds: dict[str, dict[str, float]] = {
        label: dict(values) for label, values in DEFAULT_THRESHOLDS.items()
    }
    if not path.is_file():
        return thresholds
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return thresholds
    if not isinstance(raw, dict):
        return thresholds
    for label in LABELS:
        values = raw.get(label)
        if not isinstance(values, dict):
            continue
        for key in ("positive_threshold", "negative_threshold", "min_reliability"):
            try:
                value = float(values[key])
            except (KeyError, TypeError, ValueError):
                continue
            thresholds[label][key] = max(0.0, min(1.0, value))
    return thresholds
