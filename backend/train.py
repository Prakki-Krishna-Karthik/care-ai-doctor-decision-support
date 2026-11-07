"""Train a lightweight patient-level baseline from local MIMIC files.

This is deliberately a small, auditable baseline: image statistics are used
as features and a one-vs-rest logistic regression is fitted for each label
with enough known examples.  It is not a clinical-grade model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from config import LABELS, Settings
from data import DatasetIndex
from inference import FEATURE_NAMES, feature_vector

try:
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:  # pragma: no cover - dependency installation issue
    raise RuntimeError("Training requires scikit-learn and joblib; install backend/requirements.txt") from exc


def _read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _split_indices(groups: list[str], seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(groups))
    if len(set(groups)) < 3 or len(indices) < 5:
        cutoff = max(1, int(len(indices) * 0.8))
        return indices[:cutoff], indices[cutoff:], indices[cutoff:]
    first = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_val, test = next(first.split(indices, groups=groups))
    train_val_groups = [groups[index] for index in train_val]
    if len(set(train_val_groups)) < 2:
        return train_val, test, test
    second = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed + 1)
    relative_train, relative_val = next(second.split(train_val, groups=train_val_groups))
    return train_val[relative_train], train_val[relative_val], test


def _threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.65
    best_threshold, best_f1 = 0.65, -1.0
    for threshold in np.linspace(0.30, 0.80, 26):
        score = f1_score(y_true, probabilities >= threshold, zero_division=0)
        if score > best_f1:
            best_threshold, best_f1 = float(threshold), float(score)
    return best_threshold


def _expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float | None:
    if not len(y_true):
        return None
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= lower) & (probabilities <= upper if upper == 1 else probabilities < upper)
        if not mask.any():
            continue
        confidence = float(probabilities[mask].mean())
        accuracy = float(y_true[mask].mean())
        error += float(mask.mean()) * abs(confidence - accuracy)
    return float(error)


def _metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    if not len(y_true):
        return {
            "n": 0,
            "positive_count": 0,
            "negative_count": 0,
            "positive_rate": None,
            "threshold": threshold,
            "accuracy": None,
            "precision": None,
            "recall_sensitivity": None,
            "specificity": None,
            "f1": None,
            "balanced_accuracy": None,
            "auroc": None,
            "auprc": None,
            "brier_score": None,
            "ece_10bin": None,
            "confusion_matrix": [[0, 0], [0, 0]],
        }
    predictions = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if tn + fp else None
    values: dict[str, Any] = {
        "n": int(len(y_true)),
        "positive_count": int(y_true.sum()),
        "negative_count": int(len(y_true) - y_true.sum()),
        "positive_rate": float(y_true.mean()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, predictions, zero_division=0)),
        "specificity": specificity,
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "auroc": float(roc_auc_score(y_true, probabilities)) if len(np.unique(y_true)) > 1 else None,
        "auprc": float(average_precision_score(y_true, probabilities)) if len(np.unique(y_true)) > 1 else None,
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "ece_10bin": _expected_calibration_error(y_true, probabilities),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
    return values


def _aggregate(metrics: dict[str, Any], split: str) -> dict[str, Any]:
    names = ["accuracy", "precision", "recall_sensitivity", "specificity", "f1", "balanced_accuracy", "auroc", "auprc", "brier_score", "ece_10bin"]
    summary: dict[str, Any] = {"labels_evaluated": 0}
    for name in names:
        values = [entry[split][name] for entry in metrics.values() if entry.get(split, {}).get(name) is not None]
        summary[name] = float(np.mean(values)) if values else None
    summary["labels_evaluated"] = sum(1 for entry in metrics.values() if entry.get(split, {}).get("n", 0) > 0)
    return summary


def _write_analysis_report(path: Path, payload: dict[str, Any]) -> None:
    test = payload["aggregate"]["test"]
    lines = [
        "# MIMIC-CXR Baseline ML Analysis",
        "",
        "This report describes the locally trained research baseline. It is not a clinical validation report.",
        "",
        "## Dataset and split",
        "",
        f"- Images discovered: {payload['dataset']['images_discovered']}",
        f"- Label-matched images used: {payload['dataset']['labeled_images_used']}",
        f"- Unique patient groups: {payload['dataset']['unique_patient_groups']}",
        f"- Patient-level split: {payload['split']['train']} train / {payload['split']['validation']} validation / {payload['split']['test']} test",
        "",
        "## Held-out test macro averages",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for name in ("accuracy", "precision", "recall_sensitivity", "specificity", "f1", "balanced_accuracy", "auroc", "auprc", "brier_score", "ece_10bin"):
        value = test.get(name)
        lines.append(f"| {name} | {'—' if value is None else f'{value:.4f}'} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "Thresholds were selected on the validation split by maximizing F1 over a fixed candidate grid, then applied unchanged to the held-out test split. AUROC and AUPRC use continuous probabilities. Specificity is TN/(TN+FP), while balanced accuracy is the mean of sensitivity and specificity. Brier score and ECE summarize probability quality, not only class decisions.",
        "",
        "The baseline uses image-quality and structured clinical features with one-vs-rest logistic regression. It is intended for pipeline verification and comparison against the future DenseNet-121 image branch; it must not be interpreted as a clinical performance claim.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def train(output_path: Path, max_images: int | None = None, seed: int = 17) -> dict[str, Any]:
    settings = Settings()
    index = DatasetIndex(settings)
    index.refresh()
    records = index.images[:max_images] if max_images else index.images
    features: list[np.ndarray] = []
    targets: dict[str, list[float | None]] = {label: [] for label in LABELS}
    groups: list[str] = []
    used_case_ids: list[str] = []
    for record in records:
        labels = index.labels_for_image(record)
        if not any(value is not None for value in labels.values()):
            continue
        try:
            vector, _, _ = feature_vector(
                _read_bytes(record.path), index.clinical_for_image(record)
            )
        except (OSError, ValueError):
            continue
        features.append(vector)
        for label in LABELS:
            targets[label].append(labels.get(label))
        groups.append(record.subject_id or record.study_id or record.image_id)
        used_case_ids.append(record.image_id)
    if not features:
        raise RuntimeError(
            "No image/label matches found. Check MIMIC_IMAGE_ROOT and MIMIC_DATA_ROOT, "
            "and ensure a CheXpert/NegBio CSV is present."
        )
    X = np.vstack(features)
    train_idx, val_idx, test_idx = _split_indices(groups, seed)
    models: dict[str, dict[str, Any]] = {}
    thresholds: dict[str, dict[str, float]] = {}
    metrics: dict[str, Any] = {}
    for label in LABELS:
        values = np.asarray(targets[label], dtype=object)
        known_train = np.asarray([value is not None for value in values[train_idx]])
        if known_train.sum() == 0:
            continue
        train_values = np.asarray([value for value in values[train_idx][known_train]], dtype=np.int64)
        if len(np.unique(train_values)) < 2:
            models[label] = {"constant": float(train_values.mean())}
            model = None
        else:
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=250, class_weight="balanced", random_state=seed),
            )
            model.fit(X[train_idx][known_train], train_values)
            models[label] = {"model": model}
        known_val = np.asarray([value is not None for value in values[val_idx]])
        if known_val.sum():
            val_values = np.asarray([value for value in values[val_idx][known_val]], dtype=np.int64)
            if model is None:
                val_probabilities = np.full(len(val_values), models[label]["constant"], dtype=float)
            else:
                val_probabilities = model.predict_proba(X[val_idx][known_val])[:, 1]
            threshold = _threshold(val_values, val_probabilities)
        else:
            threshold = 0.65
            val_values = np.asarray([], dtype=np.int64)
            val_probabilities = np.asarray([], dtype=float)
        known_test = np.asarray([value is not None for value in values[test_idx]])
        if known_test.sum():
            test_values = np.asarray([value for value in values[test_idx][known_test]], dtype=np.int64)
            if model is None:
                test_probabilities = np.full(len(test_values), models[label]["constant"], dtype=float)
            else:
                test_probabilities = model.predict_proba(X[test_idx][known_test])[:, 1]
        else:
            test_values = np.asarray([], dtype=np.int64)
            test_probabilities = np.asarray([], dtype=float)
        metrics[label] = {
            "validation": _metrics(val_values, val_probabilities, threshold),
            "test": _metrics(test_values, test_probabilities, threshold),
        }
        thresholds[label] = {
            "positive_threshold": threshold,
            "negative_threshold": round(1.0 - threshold, 5),
            "min_reliability": 0.30,
        }

    if not models:
        raise RuntimeError("No label has enough known training values to fit a baseline.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_summary = {
        "images_discovered": len(index.images),
        "labeled_images_used": len(used_case_ids),
        "unique_patient_groups": len(set(groups)),
        "label_known_counts": {
            label: int(sum(value is not None for value in targets[label])) for label in LABELS
        },
        "label_positive_counts": {
            label: int(sum(value == 1.0 for value in targets[label])) for label in LABELS
        },
    }
    split_summary = {"patient_level": True, "seed": seed, "train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)}
    payload = {
        "version": 1,
        "mode": "trained_baseline",
        "models": models,
        "thresholds": thresholds,
        "metrics": metrics,
        "aggregate": {"validation": _aggregate(metrics, "validation"), "test": _aggregate(metrics, "test")},
        "dataset": dataset_summary,
        "feature_names": FEATURE_NAMES,
        "split": split_summary,
        "used_case_ids": used_case_ids,
    }
    analysis_payload = {
        "version": payload["version"],
        "mode": payload["mode"],
        "metrics": metrics,
        "aggregate": payload["aggregate"],
        "dataset": dataset_summary,
        "split": split_summary,
        "feature_names": FEATURE_NAMES,
    }
    joblib.dump(payload, output_path)
    thresholds_path = output_path.with_name("thresholds.json")
    thresholds_path.write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    metrics_path = output_path.with_name("metrics.json")
    metrics_path.write_text(json.dumps(analysis_payload, indent=2), encoding="utf-8")
    report_path = output_path.with_name("analysis_report.md")
    _write_analysis_report(report_path, analysis_payload)
    return {"checkpoint": str(output_path), "thresholds": str(thresholds_path), "metrics": str(metrics_path), "analysis_report": str(report_path), "labels_trained": sorted(models), "split": payload["split"], "test_macro": payload["aggregate"]["test"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "artifacts" / "baseline.joblib")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    print(json.dumps(train(args.output, args.max_images, args.seed), indent=2))


if __name__ == "__main__":
    main()
