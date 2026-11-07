"""Structured metadata features and the selected clinical model interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


NUMERIC_FEATURES = ["rows", "columns", "study_hour", "metadata_completeness"]
CATEGORICAL_FEATURES = [
    "view_position",
    "procedure_description",
    "procedure_code",
    "view_code",
    "orientation",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _first(row: dict[str, Any], aliases: tuple[str, ...]) -> Any:
    lowered = {str(key).lower(): value for key, value in row.items()}
    for alias in aliases:
        value = lowered.get(alias.lower())
        if value is not None and str(value).strip():
            return value
    return None


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _hour(value: Any) -> float | None:
    text = str(value or "").strip()
    if len(text) >= 2 and text[:2].isdigit():
        return float(int(text[:2]))
    return _number(value)


def clinical_feature_row(row: dict[str, Any] | None) -> dict[str, Any]:
    row = row or {}
    values = {
        "view_position": _first(row, ("ViewPosition", "view_position", "view")),
        "procedure_description": _first(row, ("PerformedProcedureStepDescription", "procedure_description")),
        "procedure_code": _first(row, ("ProcedureCodeSequence_CodeMeaning", "procedure_code")),
        "view_code": _first(row, ("ViewCodeSequence_CodeMeaning", "view_code")),
        "orientation": _first(row, ("PatientOrientationCodeSequence_CodeMeaning", "orientation")),
        "rows": _number(_first(row, ("Rows", "rows"))),
        "columns": _number(_first(row, ("Columns", "columns"))),
        "study_hour": _hour(_first(row, ("StudyTime", "study_time"))),
    }
    known = sum(1 for value in values.values() if value is not None and str(value).strip())
    values["metadata_completeness"] = float(known / max(len(values), 1))
    for name in CATEGORICAL_FEATURES:
        if values[name] is None:
            values[name] = "MISSING"
    return values


def clinical_frame(rows: list[dict[str, Any]] | dict[str, Any]) -> pd.DataFrame:
    if isinstance(rows, dict):
        rows = [rows]
    return pd.DataFrame([clinical_feature_row(row) for row in rows], columns=FEATURE_COLUMNS)


def predict_selected_clinical(payload: dict[str, Any] | None, row: dict[str, Any] | None, labels: list[str]) -> dict[str, float] | None:
    """Predict with the saved per-label pipelines, if a comparison was run."""
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), dict):
        return None
    frame = clinical_frame(row or {})
    values: dict[str, float] = {}
    for label in labels:
        entry = payload["models"].get(label)
        if not isinstance(entry, dict):
            continue
        if entry.get("constant") is not None:
            values[label] = float(entry["constant"])
            continue
        model = entry.get("model")
        if model is None:
            continue
        try:
            values[label] = float(model.predict_proba(frame)[0, 1])
        except Exception:
            continue
    return values or None
