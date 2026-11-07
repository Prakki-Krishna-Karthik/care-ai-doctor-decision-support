"""Create a conservative model-readiness and label-bias audit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ARTIFACTS = Path(__file__).resolve().parent / "artifacts"


def _gap(payload: dict[str, Any], metric: str) -> float | None:
    validation = payload.get("aggregate", {}).get("validation", {}).get(metric)
    test = payload.get("aggregate", {}).get("test", {}).get(metric)
    if validation is None or test is None:
        return None
    return float(validation) - float(test)


def audit() -> dict[str, Any]:
    metrics_path = ARTIFACTS / "densenet_metrics.json"
    if not metrics_path.is_file():
        result = {"available": False, "status": "not_evaluated", "message": "Training metrics are unavailable."}
        (ARTIFACTS / "model_readiness.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    gaps = {metric: _gap(payload, metric) for metric in ("auprc", "auroc", "f1", "balanced_accuracy", "brier_score", "ece_10bin")}
    sparse_labels: list[dict[str, Any]] = []
    for label, values in payload.get("metrics", {}).items():
        test = values.get("test", {})
        positives = int(test.get("positive_count", 0))
        negatives = int(test.get("negative_count", 0))
        if positives < 20 or negatives < 20:
            sparse_labels.append({"label": label, "positive_count": positives, "negative_count": negatives})
    warnings: list[str] = []
    if gaps.get("auprc") is not None and gaps["auprc"] > 0.05:
        warnings.append("Validation AUPRC is more than five points above test AUPRC.")
    if gaps.get("f1") is not None and gaps["f1"] > 0.10:
        warnings.append("Validation F1 is more than ten points above test F1.")
    if gaps.get("ece_10bin") is not None and gaps["ece_10bin"] < -0.03:
        warnings.append("Test calibration error is materially worse than validation calibration error.")
    if sparse_labels:
        warnings.append("Some labels have too little test support for stable per-label conclusions.")
    result = {
        "available": True,
        "status": "research_only_review_required" if warnings else "provisional",
        "deployment_allowed": False if warnings else True,
        "validation_minus_test_gap": gaps,
        "sparse_test_labels": sparse_labels,
        "warnings": warnings,
        "recommendation": "Use repeated patient-level cross-validation and an untouched external test set before clinical claims.",
    }
    (ARTIFACTS / "model_readiness.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
