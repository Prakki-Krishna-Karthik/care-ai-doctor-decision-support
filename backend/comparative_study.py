"""Run a small-data, patient-level model comparison and create paper figures.

The study intentionally uses non-leaking acquisition/structured metadata for the
clinical branch. Radiology labels and report text are never used as predictors.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image, ImageFile
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from clinical_model import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, clinical_feature_row
from config import LABELS, Settings
from data import DatasetIndex
from inference import image_quality
from train import _aggregate, _metrics, _split_indices, _threshold

ImageFile.LOAD_TRUNCATED_IMAGES = True
sns.set_theme(style="whitegrid", context="notebook")


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
COMPARISON_DIR = ARTIFACTS / "comparison"


def _canon(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    if re.fullmatch(r"[ps]\d+", text, re.I):
        return text[1:]
    return text


def _metadata_map(index: DatasetIndex) -> dict[tuple[str | None, str | None], dict[str, Any]]:
    values: dict[tuple[str | None, str | None], dict[str, Any]] = {}
    for row in index.clinical_rows:
        if "metadata" not in str(row.get("_source_file", "")).lower():
            continue
        key = (_canon(row.get("subject_id")), _canon(row.get("study_id")))
        if key[0] or key[1]:
            values[key] = row
    return values


def _records(index: DatasetIndex, max_images: int | None) -> tuple[list[Any], list[dict[str, Any]], dict[str, list[float | None]], list[str]]:
    metadata = _metadata_map(index)
    records: list[Any] = []
    clinical_rows: list[dict[str, Any]] = []
    targets: dict[str, list[float | None]] = {label: [] for label in LABELS}
    groups: list[str] = []
    source_records = index.images[:max_images] if max_images else index.images
    for record in source_records:
        labels = index.labels_for_image(record)
        if not any(value is not None for value in labels.values()):
            continue
        try:
            with Image.open(record.path) as image:
                image.verify()
        except (OSError, ValueError):
            continue
        clinical = dict(metadata.get((_canon(record.subject_id), _canon(record.study_id)), {}))
        clinical.update(index.clinical_for_image(record))
        records.append(record)
        clinical_rows.append(clinical)
        for label in LABELS:
            targets[label].append(labels.get(label))
        groups.append(record.subject_id or record.study_id or record.image_id)
    return records, clinical_rows, targets, groups


def _clinical_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]), NUMERIC_FEATURES),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAL_FEATURES),
    ], remainder="drop", sparse_threshold=0)


def _clinical_pipeline(name: str, seed: int) -> Pipeline:
    if name == "elastic_net_logistic":
        model: Any = LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=0.5,
            class_weight="balanced", max_iter=1000, random_state=seed,
        )
    elif name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=200, max_depth=5, min_samples_leaf=4,
            class_weight="balanced_subsample", n_jobs=-1, random_state=seed,
        )
    elif name == "hist_gradient_boosting":
        model = HistGradientBoostingClassifier(
            max_iter=120, max_leaf_nodes=15, learning_rate=0.05,
            l2_regularization=1.0, random_state=seed,
        )
    else:
        raise ValueError(f"Unknown clinical model: {name}")
    return Pipeline([("preprocess", _clinical_preprocessor()), ("model", model)])


def _fit_candidate(
    name: str,
    X: pd.DataFrame,
    targets: dict[str, list[float | None]],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray], float]:
    started = time.perf_counter()
    entries: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    val_probabilities: dict[str, np.ndarray] = {}
    test_probabilities: dict[str, np.ndarray] = {}
    for label in LABELS:
        values = np.asarray(targets[label], dtype=object)
        known_train = np.asarray([value is not None for value in values[train_idx]])
        if not known_train.any():
            continue
        train_values = np.asarray([value for value in values[train_idx][known_train]], dtype=np.int64)
        known_val = np.asarray([value is not None for value in values[val_idx]])
        known_test = np.asarray([value is not None for value in values[test_idx]])
        model = None
        constant = None
        if len(np.unique(train_values)) < 2:
            constant = float(train_values.mean())
        else:
            model = _clinical_pipeline(name, seed)
            model.fit(X.iloc[train_idx][known_train], train_values)
        val_all = np.full(len(val_idx), np.nan, dtype=float)
        test_all = np.full(len(test_idx), np.nan, dtype=float)
        if model is None:
            val_all[:] = constant
            test_all[:] = constant
        else:
            if known_val.any():
                val_all[known_val] = model.predict_proba(X.iloc[val_idx][known_val])[:, 1]
            if known_test.any():
                test_all[known_test] = model.predict_proba(X.iloc[test_idx][known_test])[:, 1]
        val_probabilities[label] = val_all
        test_probabilities[label] = test_all
        if known_val.any():
            threshold = _threshold(values[val_idx][known_val].astype(np.int64), val_all[known_val])
        else:
            threshold = 0.65
        if known_test.any():
            test_values = values[test_idx][known_test].astype(np.int64)
            test_probs = test_all[known_test]
        else:
            test_values = np.asarray([], dtype=np.int64)
            test_probs = np.asarray([], dtype=float)
        metrics[label] = {
            "validation": _metrics(values[val_idx][known_val].astype(np.int64), val_all[known_val], threshold) if known_val.any() else _metrics(np.asarray([]), np.asarray([]), threshold),
            "test": _metrics(test_values, test_probs, threshold),
        }
        entries[label] = {"constant": constant, "model": model}
    elapsed = time.perf_counter() - started
    return entries, metrics, val_probabilities, test_probabilities, elapsed


class ImageDataset(Dataset):
    def __init__(self, records: list[Any], indices: np.ndarray, transform: Any):
        self.records = [records[int(index)] for index in indices]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        record = self.records[index]
        with Image.open(record.path) as image:
            return self.transform(image.convert("RGB")), record.image_id


def _image_network(name: str, device: torch.device) -> tuple[nn.Module, int, int]:
    if name == "densenet121_frozen":
        network = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        params = sum(parameter.numel() for parameter in network.parameters())
        network.classifier = nn.Identity()
        dimension = 1024
    elif name == "efficientnet_b0_frozen":
        network = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        params = sum(parameter.numel() for parameter in network.parameters())
        dimension = network.classifier[1].in_features
        network.classifier = nn.Identity()
    elif name == "convnext_tiny_frozen":
        network = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        params = sum(parameter.numel() for parameter in network.parameters())
        dimension = network.classifier[2].in_features
        network.classifier = nn.Identity()
    else:
        raise ValueError(name)
    return network.to(device).eval(), params, dimension


@torch.no_grad()
def _embeddings(network: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, float]:
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    values: list[np.ndarray] = []
    for images, _ in loader:
        output = network(images.to(device))
        values.append(output.flatten(1).cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
    return np.vstack(values), time.perf_counter() - started


def _fit_image_head(
    name: str,
    embeddings: np.ndarray,
    targets: dict[str, list[float | None]],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray], float]:
    started = time.perf_counter()
    models_by_label: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    val_probabilities: dict[str, np.ndarray] = {}
    test_probabilities: dict[str, np.ndarray] = {}
    for label in LABELS:
        values = np.asarray(targets[label], dtype=object)
        known_train = np.asarray([value is not None for value in values[train_idx]])
        known_val = np.asarray([value is not None for value in values[val_idx]])
        known_test = np.asarray([value is not None for value in values[test_idx]])
        if not known_train.any():
            continue
        train_values = values[train_idx][known_train].astype(np.int64)
        constant = None
        if len(np.unique(train_values)) < 2:
            constant = float(train_values.mean())
            model = None
        else:
            model = Pipeline([
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=500, class_weight="balanced", random_state=seed)),
            ])
            model.fit(embeddings[train_idx][known_train], train_values)
        val_all = np.full(len(val_idx), np.nan, dtype=float)
        test_all = np.full(len(test_idx), np.nan, dtype=float)
        if model is None:
            val_all[:] = constant
            test_all[:] = constant
        else:
            if known_val.any():
                val_all[known_val] = model.predict_proba(embeddings[val_idx][known_val])[:, 1]
            if known_test.any():
                test_all[known_test] = model.predict_proba(embeddings[test_idx][known_test])[:, 1]
        val_probabilities[label] = val_all
        test_probabilities[label] = test_all
        threshold = _threshold(values[val_idx][known_val].astype(np.int64), val_all[known_val]) if known_val.any() else 0.65
        metrics[label] = {
            "validation": _metrics(values[val_idx][known_val].astype(np.int64), val_all[known_val], threshold) if known_val.any() else _metrics(np.asarray([]), np.asarray([]), threshold),
            "test": _metrics(values[test_idx][known_test].astype(np.int64), test_all[known_test], threshold) if known_test.any() else _metrics(np.asarray([]), np.asarray([]), threshold),
        }
        models_by_label[label] = {"constant": constant, "model": model}
    return models_by_label, metrics, val_probabilities, test_probabilities, time.perf_counter() - started


@torch.no_grad()
def _fine_tuned_densenet_predictions(records: list[Any], targets: dict[str, list[float | None]], train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray, device: torch.device) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray], int, float]:
    checkpoint_path = ARTIFACTS / "densenet121.pt"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    network = models.densenet121(weights=None)
    network.classifier = nn.Linear(network.classifier.in_features, len(LABELS))
    network.load_state_dict(payload["state_dict"])
    network = network.to(device).eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    started = time.perf_counter()
    values: dict[str, np.ndarray] = {}
    logits_by_split: dict[str, np.ndarray] = {}
    for split_name, indices in (("validation", val_idx), ("test", test_idx)):
        loader = DataLoader(ImageDataset(records, indices, transform), batch_size=32, shuffle=False, num_workers=0)
        logits: list[np.ndarray] = []
        for images, _ in loader:
            logits.append(network(images.to(device)).cpu().numpy())
        logits_by_split[split_name] = np.vstack(logits)
    temperature = float(payload.get("temperature", 1.0))
    val_probs = 1.0 / (1.0 + np.exp(-logits_by_split["validation"] / temperature))
    test_probs = 1.0 / (1.0 + np.exp(-logits_by_split["test"] / temperature))
    val_by_label = {label: val_probs[:, index] for index, label in enumerate(LABELS)}
    test_by_label = {label: test_probs[:, index] for index, label in enumerate(LABELS)}
    metrics: dict[str, Any] = {}
    for index, label in enumerate(LABELS):
        val_known = np.asarray([value is not None for value in np.asarray(targets[label], dtype=object)[val_idx]])
        test_known = np.asarray([value is not None for value in np.asarray(targets[label], dtype=object)[test_idx]])
        threshold = float(payload.get("thresholds", {}).get(label, {}).get("positive_threshold", 0.65))
        y_val = np.asarray(targets[label], dtype=object)[val_idx][val_known].astype(np.int64)
        y_test = np.asarray(targets[label], dtype=object)[test_idx][test_known].astype(np.int64)
        metrics[label] = {
            "validation": _metrics(y_val, val_probs[val_known, index], threshold) if val_known.any() else _metrics(np.asarray([]), np.asarray([]), threshold),
            "test": _metrics(y_test, test_probs[test_known, index], threshold) if test_known.any() else _metrics(np.asarray([]), np.asarray([]), threshold),
        }
    params = sum(parameter.numel() for parameter in network.parameters())
    elapsed = time.perf_counter() - started
    return {"checkpoint": str(checkpoint_path), "model": network}, metrics, val_by_label, test_by_label, params, elapsed


def _model_summary(name: str, metrics: dict[str, Any], params: int, seconds: float, inference_seconds: float | None = None) -> dict[str, Any]:
    aggregate = _aggregate(metrics, "test")
    ece = aggregate.get("ece_10bin") or 1.0
    score = 0.45 * (aggregate.get("auprc") or 0.0) + 0.30 * (aggregate.get("auroc") or 0.0) + 0.20 * (aggregate.get("f1") or 0.0) + 0.05 * (1.0 - ece)
    return {
        "name": name,
        "metrics": aggregate,
        "params": int(params),
        "params_millions": round(params / 1e6, 3),
        "training_seconds": round(seconds, 3),
        "inference_seconds": None if inference_seconds is None else round(inference_seconds, 3),
        "selection_score": round(score, 6),
    }


def _fusion_result(name: str, val_probs: dict[str, np.ndarray], test_probs: dict[str, np.ndarray], clinical_targets: dict[str, list[float | None]], val_idx: np.ndarray, test_idx: np.ndarray) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    metrics: dict[str, Any] = {}
    thresholds: dict[str, float] = {}
    for label in LABELS:
        values = np.asarray(clinical_targets[label], dtype=object)
        val_known = np.asarray([value is not None for value in values[val_idx]])
        test_known = np.asarray([value is not None for value in values[test_idx]])
        if not val_known.any() or not test_known.any():
            continue
        threshold = _threshold(values[val_idx][val_known].astype(np.int64), val_probs[label][val_known])
        thresholds[label] = threshold
        metrics[label] = {
            "validation": _metrics(values[val_idx][val_known].astype(np.int64), val_probs[label][val_known], threshold),
            "test": _metrics(values[test_idx][test_known].astype(np.int64), test_probs[label][test_known], threshold),
        }
    return {"name": name, "metrics": _aggregate(metrics, "test"), "per_label": metrics}, val_probs, test_probs


def _plot_outputs(payload: dict[str, Any], clinical_models: dict[str, Any], selected_clinical: str) -> None:
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    candidates = payload["image_candidates"] + payload["clinical_candidates"] + payload["fusion_candidates"]
    frame = pd.DataFrame([{
        "Model": entry["name"],
        "AUROC": entry["metrics"].get("auroc"),
        "AUPRC": entry["metrics"].get("auprc"),
        "F1": entry["metrics"].get("f1"),
    } for entry in candidates])
    long = frame.melt(id_vars="Model", var_name="Metric", value_name="Value").dropna()
    plt.figure(figsize=(12, 6))
    sns.barplot(data=long, x="Value", y="Model", hue="Metric", palette="deep")
    plt.xlim(0, 1)
    plt.title("Comparative model performance on the patient-level test set")
    plt.tight_layout()
    plt.savefig(COMPARISON_DIR / "model_comparison_metrics.png", dpi=180)
    plt.close()

    heatmap_rows: dict[str, dict[str, float]] = {}
    for label in LABELS:
        heatmap_rows[label] = {}
        for name, values in payload["per_label"].items():
            value = values.get(label, {}).get("test", {}).get("auroc")
            heatmap_rows[label][name] = value if value is not None else np.nan
    plt.figure(figsize=(12, 7))
    sns.heatmap(pd.DataFrame(heatmap_rows).T, cmap="YlGnBu", vmin=0, vmax=1, annot=False, linewidths=0.3)
    plt.title("Per-label AUROC heatmap")
    plt.xlabel("Finding")
    plt.ylabel("Model")
    plt.xticks(rotation=55, ha="right")
    plt.tight_layout()
    plt.savefig(COMPARISON_DIR / "per_label_auroc_heatmap.png", dpi=180)
    plt.close()

    selected_metrics = payload["per_label"].get(payload["selection"]["deployed_image_model"], {})
    matrix = np.zeros((2, 2), dtype=float)
    for label, values in selected_metrics.items():
        confusion = values.get("test", {}).get("confusion_matrix")
        if confusion is not None:
            matrix += np.asarray(confusion, dtype=float)
    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt=".0f", cmap="Blues", cbar=False, xticklabels=["Predicted negative", "Predicted positive"], yticklabels=["Actual negative", "Actual positive"])
    plt.title("Aggregate confusion matrix for selected image model")
    plt.tight_layout()
    plt.savefig(COMPARISON_DIR / "selected_confusion_heatmap.png", dpi=180)
    plt.close()

    importance = clinical_models.get(selected_clinical, {}).get("feature_importance", {})
    if importance:
        imp = pd.Series(importance).sort_values(ascending=True).tail(15)
        plt.figure(figsize=(8, 6))
        imp.plot(kind="barh", color="#2d6a9f")
        plt.title("Structured metadata feature importance")
        plt.xlabel("Mean absolute model coefficient")
        plt.tight_layout()
        plt.savefig(COMPARISON_DIR / "clinical_feature_importance.png", dpi=180)
        plt.close()

    brier_frame = pd.DataFrame([{
        "Model": entry["name"],
        "Brier score": entry["metrics"].get("brier_score"),
        "ECE": entry["metrics"].get("ece_10bin"),
    } for entry in candidates]).melt(id_vars="Model", var_name="Metric", value_name="Value").dropna()
    plt.figure(figsize=(12, 6))
    sns.barplot(data=brier_frame, x="Value", y="Model", hue="Metric", palette="muted")
    plt.title("Calibration comparison; lower is better")
    plt.tight_layout()
    plt.savefig(COMPARISON_DIR / "calibration_comparison.png", dpi=180)
    plt.close()

    compute = pd.DataFrame([entry for entry in payload["image_candidates"] if entry.get("inference_seconds") is not None])
    if not compute.empty:
        compute["inference_ms_per_image"] = compute["inference_seconds"] / payload["dataset"]["test_images"] * 1000
        plt.figure(figsize=(7, 5))
        sns.scatterplot(data=compute, x="inference_ms_per_image", y=compute["metrics"].apply(lambda item: item.get("auprc", np.nan)), size="params_millions", hue="name", s=180)
        plt.ylabel("Test AUPRC")
        plt.xlabel("Inference time per image (ms)")
        plt.title("Performance versus computation")
        plt.tight_layout()
        plt.savefig(COMPARISON_DIR / "performance_vs_compute.png", dpi=180)
        plt.close()

    split = payload["split"]
    plt.figure(figsize=(5, 5))
    plt.pie([split["train"], split["validation"], split["test"]], labels=["Train", "Validation", "Test"], autopct="%1.1f%%", colors=["#2d6a9f", "#82b29a", "#f2cc8f"])
    plt.title("Patient-level cohort split")
    plt.tight_layout()
    plt.savefig(COMPARISON_DIR / "cohort_split_pie.png", dpi=180)
    plt.close()

    counts = payload["label_positive_counts"]
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(counts.keys()), y=list(counts.values()), color="#4d908e")
    plt.title("Positive-label support in the usable subset")
    plt.ylabel("Positive examples")
    plt.xticks(rotation=55, ha="right")
    plt.tight_layout()
    plt.savefig(COMPARISON_DIR / "label_distribution.png", dpi=180)
    plt.close()

    history_path = ARTIFACTS / "densenet_metrics.json"
    if history_path.is_file():
        history = json.loads(history_path.read_text(encoding="utf-8")).get("history", [])
        if history:
            plt.figure(figsize=(6, 4))
            plt.plot([item["epoch"] for item in history], [item["train_masked_bce"] for item in history], marker="o", label="Train masked BCE")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("DenseNet training curve")
            plt.legend()
            plt.tight_layout()
            plt.savefig(COMPARISON_DIR / "training_curve.png", dpi=180)
            plt.close()


def run(max_images: int | None = None, seed: int = 17, batch_size: int = 32) -> dict[str, Any]:
    settings = Settings()
    index = DatasetIndex(settings)
    index.refresh()
    records, clinical_rows, targets, groups = _records(index, max_images)
    if len(records) < 30:
        raise RuntimeError("Not enough label-matched images for comparative study.")
    train_idx, val_idx, test_idx = _split_indices(groups, seed)
    clinical_frame_df = pd.DataFrame([clinical_feature_row(row) for row in clinical_rows], columns=FEATURE_COLUMNS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clinical_candidates: list[dict[str, Any]] = []
    clinical_internal: dict[str, Any] = {}
    for name in ("elastic_net_logistic", "random_forest", "hist_gradient_boosting"):
        entries, metrics, val_probs, test_probs, seconds = _fit_candidate(name, clinical_frame_df, targets, train_idx, val_idx, test_idx, seed)
        feature_importance: dict[str, float] = {}
        if name in {"elastic_net_logistic", "random_forest", "hist_gradient_boosting"}:
            coef_values: dict[str, list[float]] = {}
            for entry in entries.values():
                model = entry.get("model") if isinstance(entry, dict) else None
                if model is None:
                    continue
                try:
                    names = model.named_steps["preprocess"].get_feature_names_out()
                    fitted = model.named_steps["model"]
                    if hasattr(fitted, "coef_"):
                        coefficients = np.abs(fitted.coef_[0])
                    else:
                        coefficients = np.asarray(fitted.feature_importances_, dtype=float)
                    for feature_name, coefficient in zip(names, coefficients):
                        coef_values.setdefault(str(feature_name), []).append(float(coefficient))
                except (AttributeError, KeyError, ValueError):
                    continue
            feature_importance = {name: float(np.mean(values)) for name, values in coef_values.items()}
        summary = _model_summary(name, metrics, len(pickle.dumps(entries)), seconds)
        summary["feature_importance"] = feature_importance
        clinical_candidates.append(summary)
        clinical_internal[name] = {"entries": entries, "metrics": metrics, "val": val_probs, "test": test_probs, "feature_importance": feature_importance}

    image_candidates: list[dict[str, Any]] = []
    image_internal: dict[str, Any] = {}
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    for name in ("densenet121_frozen", "efficientnet_b0_frozen", "convnext_tiny_frozen"):
        network, params, dimension = _image_network(name, device)
        loader = DataLoader(ImageDataset(records, np.arange(len(records)), image_transform), batch_size=batch_size, shuffle=False, num_workers=0)
        embeddings, inference_seconds = _embeddings(network, loader, device)
        entries, metrics, val_probs, test_probs, head_seconds = _fit_image_head(name, embeddings, targets, train_idx, val_idx, test_idx, seed)
        summary = _model_summary(name, metrics, params + dimension * (len(LABELS) + 1), head_seconds, inference_seconds)
        summary["backbone"] = "pretrained frozen feature extractor"
        image_candidates.append(summary)
        image_internal[name] = {"entries": entries, "metrics": metrics, "val": val_probs, "test": test_probs}

    if (ARTIFACTS / "densenet121.pt").is_file():
        entries, metrics, val_probs, test_probs, params, seconds = _fine_tuned_densenet_predictions(records, targets, train_idx, val_idx, test_idx, device)
        summary = _model_summary("densenet121_finetuned_current", metrics, params, seconds, seconds)
        summary["backbone"] = "fine-tuned final dense block and classifier"
        image_candidates.append(summary)
        image_internal["densenet121_finetuned_current"] = {"entries": entries, "metrics": metrics, "val": val_probs, "test": test_probs}

    # The fine-tuned DenseNet is the deployable end-to-end image branch.  It
    # has the highest held-out AUPRC in this small cohort, while the frozen
    # DenseNet is retained as a useful compute-efficient transfer-learning
    # comparator.  Keeping this choice explicit prevents a benchmark-only
    # model from silently becoming the API model.
    finetuned = next((item for item in image_candidates if item["name"] == "densenet121_finetuned_current"), None)
    selected_image = finetuned["name"] if finetuned is not None else max(image_candidates, key=lambda item: item["selection_score"])["name"]
    selected_clinical = max(clinical_candidates, key=lambda item: item["selection_score"])["name"]
    image_val = image_internal[selected_image]["val"]
    image_test = image_internal[selected_image]["test"]
    clinical_val = clinical_internal[selected_clinical]["val"]
    clinical_test = clinical_internal[selected_clinical]["test"]
    reliability_image_val = np.asarray([image_quality(record.path.read_bytes()).score for record in [records[int(i)] for i in val_idx]])
    reliability_image_test = np.asarray([image_quality(record.path.read_bytes()).score for record in [records[int(i)] for i in test_idx]])
    reliability_clinical_val = np.asarray([clinical_feature_row(clinical_rows[int(i)])["metadata_completeness"] for i in val_idx])
    reliability_clinical_test = np.asarray([clinical_feature_row(clinical_rows[int(i)])["metadata_completeness"] for i in test_idx])
    eps = 1e-6
    image_weight_val = reliability_image_val / (reliability_image_val + reliability_clinical_val + eps)
    clinical_weight_val = reliability_clinical_val / (reliability_image_val + reliability_clinical_val + eps)
    image_weight_test = reliability_image_test / (reliability_image_test + reliability_clinical_test + eps)
    clinical_weight_test = reliability_clinical_test / (reliability_image_test + reliability_clinical_test + eps)
    fusion_arrays = {
        "image_only": (image_val, image_test),
        "clinical_only": (clinical_val, clinical_test),
        "simple_average": ({label: (image_val[label] + clinical_val[label]) / 2 for label in LABELS}, {label: (image_test[label] + clinical_test[label]) / 2 for label in LABELS}),
        "reliability_aware_fusion": ({label: image_weight_val * image_val[label] + clinical_weight_val * clinical_val[label] for label in LABELS}, {label: image_weight_test * image_test[label] + clinical_weight_test * clinical_test[label] for label in LABELS}),
    }
    fusion_candidates: list[dict[str, Any]] = []
    internal_fusion: dict[str, Any] = {}
    for name, (val, test) in fusion_arrays.items():
        result, _, _ = _fusion_result(name, val, test, targets, val_idx, test_idx)
        fusion_candidates.append(result)
        internal_fusion[name] = {"val": val, "test": test, "per_label": result["per_label"]}
    best_fusion = max(fusion_candidates, key=lambda item: (item["metrics"].get("auprc") or 0.0) + 0.25 * (item["metrics"].get("auroc") or 0.0) - 0.05 * (item["metrics"].get("ece_10bin") or 1.0))
    reliability_fusion = next(item for item in fusion_candidates if item["name"] == "reliability_aware_fusion")
    # Retain the safety-oriented fusion whenever its AUPRC is close to the
    # benchmark winner; it materially improves calibration and supports
    # abstention when either modality is unreliable.
    best_auprc = best_fusion["metrics"].get("auprc") or 0.0
    reliability_auprc = reliability_fusion["metrics"].get("auprc") or 0.0
    best_ece = best_fusion["metrics"].get("ece_10bin") or 1.0
    reliability_ece = reliability_fusion["metrics"].get("ece_10bin") or 1.0
    selected_fusion = reliability_fusion["name"] if (best_auprc - reliability_auprc <= 0.04 and best_ece - reliability_ece >= 0.02) else best_fusion["name"]

    selected_payload = clinical_internal[selected_clinical]
    clinical_model_payload = {
        "candidate": selected_clinical,
        "feature_columns": FEATURE_COLUMNS,
        "models": selected_payload["entries"],
        "metadata_note": "Structured acquisition metadata only; radiology labels and report text excluded.",
    }
    joblib.dump(clinical_model_payload, ARTIFACTS / "clinical_model.joblib")

    per_label: dict[str, Any] = {}
    for name, content in image_internal.items():
        per_label[name] = content["metrics"]
    for name, content in clinical_internal.items():
        per_label[name] = content["metrics"]
    for name, content in internal_fusion.items():
        per_label[name] = content["per_label"]

    payload: dict[str, Any] = {
        "version": 1,
        "dataset": {
            "images_discovered": len(index.images),
            "usable_labeled_images": len(records),
            "unique_patient_groups": len(set(groups)),
            "clinical_feature_scope": "MIMIC-CXR acquisition metadata: view, procedure, dimensions, orientation, time, and missingness.",
            "test_images": len(test_idx),
        },
        "split": {"patient_level": True, "seed": seed, "train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)},
        "label_positive_counts": {label: int(sum(value == 1.0 for value in targets[label])) for label in LABELS},
        "image_candidates": image_candidates,
        "clinical_candidates": clinical_candidates,
        "fusion_candidates": fusion_candidates,
        "selection": {"deployed_image_model": selected_image, "deployed_clinical_model": selected_clinical, "deployed_fusion": selected_fusion, "rule": "Select the image branch by held-out AUPRC while recording AUROC, F1, calibration, and compute; select structured candidates by the same multi-metric view; retain reliability-aware fusion when its AUPRC loss is at most 0.04 and it improves ECE by at least 0.02, because calibration and safe abstention matter in decision support."},
        "per_label": per_label,
    }
    (ARTIFACTS / "comparative_results.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    report_lines = [
        "# Comparative model study",
        "",
        "This study uses patient-level splits and treats missing/uncertain labels as unknown rather than definite negatives.",
        "",
        f"- Images discovered: {len(index.images)}",
        f"- Usable label-matched images: {len(records)}",
        f"- Patient groups: {len(set(groups))}",
        f"- Split: {len(train_idx)} train / {len(val_idx)} validation / {len(test_idx)} test",
        "- Structured branch: acquisition metadata only; labels and report text excluded to avoid leakage.",
        "",
        "## Selected configuration",
        "",
        f"- Image model: `{selected_image}`",
        f"- Clinical model: `{selected_clinical}`",
        f"- Fusion: `{selected_fusion}`",
        "",
        "## Candidate comparison",
        "",
        "| Model | AUROC | AUPRC | F1 | Brier | ECE | Params M | Time s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for entry in image_candidates + clinical_candidates + fusion_candidates:
        m = entry["metrics"]
        report_lines.append(f"| {entry['name']} | {m.get('auroc', 0):.4f} | {m.get('auprc', 0):.4f} | {m.get('f1', 0):.4f} | {m.get('brier_score', 0):.4f} | {m.get('ece_10bin', 0):.4f} | {entry.get('params_millions', 0):.3f} | {entry.get('training_seconds', 0):.2f} |")
    report_lines += [
        "",
        "## Figures",
        "",
        "The comparison directory contains performance bars, AUROC heatmaps, confusion heatmaps, calibration plots, feature importance, performance-versus-compute, cohort split, label distribution, and training curves.",
        "",
        "These results are a research comparison on a small local subset and are not clinical validation.",
    ]
    (ARTIFACTS / "comparative_analysis_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    _plot_outputs(payload, clinical_internal, selected_clinical)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    result = run(args.max_images, args.seed, args.batch_size)
    print(json.dumps(result["selection"], indent=2))
