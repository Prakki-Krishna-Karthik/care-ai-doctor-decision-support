"""Train the DenseNet-121 image branch on the local MIMIC subset.

The training/evaluation split is patient-level. CheXpert uncertain and missing
labels are masked out of the loss and metrics. The saved checkpoint is used by
the API when present; the lightweight baseline remains available for
comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageFile
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

from config import LABELS, Settings
from data import DatasetIndex
from train import _aggregate, _metrics, _split_indices, _threshold


class CXRDataset(Dataset):
    def __init__(self, records: list[Any], targets: dict[str, list[float | None]], indices: np.ndarray, transform: Any):
        self.records = [records[int(index)] for index in indices]
        self.targets = targets
        self.indices = [int(index) for index in indices]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, position: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        record = self.records[position]
        with Image.open(record.path) as source:
            image = self.transform(source.convert("RGB"))
        target = torch.tensor([
            0.0 if self.targets[label][self.indices[position]] is None else float(self.targets[label][self.indices[position]])
            for label in LABELS
        ], dtype=torch.float32)
        known = torch.tensor([
            self.targets[label][self.indices[position]] is not None for label in LABELS
        ], dtype=torch.float32)
        return image, target, known, record.image_id


def _model(device: torch.device, unfreeze_final_block: bool = True, regularized: bool = False) -> nn.Module:
    weights = models.DenseNet121_Weights.DEFAULT
    network = models.densenet121(weights=weights)
    classifier = nn.Linear(network.classifier.in_features, len(LABELS))
    network.classifier = nn.Sequential(nn.Dropout(p=0.35), classifier) if regularized else classifier
    for parameter in network.features.parameters():
        parameter.requires_grad = False
    if unfreeze_final_block:
        for name, parameter in network.features.named_parameters():
            if name.startswith("denseblock4") or name.startswith("norm5"):
                parameter.requires_grad = True
    return network.to(device)


def _temperature(logits: np.ndarray, targets: np.ndarray, known: np.ndarray) -> float:
    if not known.any():
        return 1.0
    best_temperature, best_loss = 1.0, float("inf")
    for candidate in np.linspace(0.5, 3.0, 26):
        values = logits[known] / candidate
        probabilities = 1.0 / (1.0 + np.exp(-values))
        loss = float(-(targets[known] * np.log(np.clip(probabilities, 1e-6, 1 - 1e-6)) + (1 - targets[known]) * np.log(np.clip(1 - probabilities, 1e-6, 1 - 1e-6))).mean())
        if loss < best_loss:
            best_loss, best_temperature = loss, float(candidate)
    return best_temperature


@torch.no_grad()
def _predict(network: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    network.eval()
    logits, targets, known, ids = [], [], [], []
    for images, batch_targets, batch_known, batch_ids in loader:
        logits.append(network(images.to(device)).detach().cpu().numpy())
        targets.append(batch_targets.numpy())
        known.append(batch_known.numpy().astype(bool))
        ids.extend(batch_ids)
    return np.vstack(logits), np.vstack(targets), np.vstack(known), ids


def train(output_path: Path, max_images: int | None = None, epochs: int = 8, batch_size: int = 32, seed: int = 17, regularized: bool = False) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    index = DatasetIndex(Settings())
    index.refresh()
    all_records = []
    targets: dict[str, list[float | None]] = {label: [] for label in LABELS}
    groups: list[str] = []
    for record in index.images[:max_images] if max_images else index.images:
        labels = index.labels_for_image(record)
        if not any(value is not None for value in labels.values()):
            continue
        try:
            with Image.open(record.path) as image:
                image.verify()
        except (OSError, ValueError):
            continue
        all_records.append(record)
        for label in LABELS:
            targets[label].append(labels.get(label))
        groups.append(record.subject_id or record.study_id or record.image_id)
    if len(all_records) < 20:
        raise RuntimeError("Not enough label-matched images for DenseNet training.")

    train_idx, val_idx, test_idx = _split_indices(groups, seed)
    train_steps: list[Any] = [transforms.Resize((224, 224))]
    if regularized:
        # Preserve laterality and use conservative radiograph-safe changes.
        train_steps.extend([
            transforms.RandomAffine(degrees=5, translate=(0.03, 0.03), scale=(0.97, 1.03)),
            transforms.ColorJitter(brightness=0.08, contrast=0.08),
        ])
    else:
        train_steps.append(transforms.RandomHorizontalFlip(p=0.5))
    train_steps.extend([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform = transforms.Compose(train_steps)
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_loader = DataLoader(CXRDataset(all_records, targets, train_idx, transform), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(CXRDataset(all_records, targets, val_idx, eval_transform), batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(CXRDataset(all_records, targets, test_idx, eval_transform), batch_size=batch_size, shuffle=False, num_workers=0)

    network = _model(device, unfreeze_final_block=not regularized, regularized=regularized)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in network.parameters() if parameter.requires_grad],
        lr=1e-3 if regularized else 2e-4,
        weight_decay=5e-4 if regularized else 1e-4,
    )
    train_known = np.asarray([[targets[label][int(index)] is not None for label in LABELS] for index in train_idx], dtype=bool)
    train_values = np.asarray([[0.0 if targets[label][int(index)] is None else float(targets[label][int(index)]) for label in LABELS] for index in train_idx], dtype=np.float32)
    pos_weight_values: list[float] = []
    for label_index in range(len(LABELS)):
        mask = train_known[:, label_index]
        positives = float(train_values[mask, label_index].sum()) if mask.any() else 0.0
        negatives = float(mask.sum() - positives)
        pos_weight_values.append(float(np.clip(negatives / max(positives, 1.0), 1.0, 4.0)))
    pos_weight = torch.tensor(pos_weight_values, dtype=torch.float32, device=device) if regularized else None
    criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = float("inf")
    stale_epochs = 0
    patience = 3
    for epoch in range(epochs):
        network.train()
        total_loss, seen = 0.0, 0
        for images, batch_targets, batch_known, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            output = network(images.to(device))
            effective_targets = batch_targets.to(device)
            if regularized:
                effective_targets = effective_targets * 0.96 + 0.02
            loss_values = criterion(output, effective_targets)
            mask = batch_known.to(device)
            denominator = mask.sum().clamp_min(1.0)
            loss = (loss_values * mask).sum() / denominator
            loss.backward()
            if regularized:
                torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(images)
            seen += len(images)
        val_logits_epoch, val_targets_epoch, val_known_epoch, _ = _predict(network, val_loader, device)
        val_probabilities_epoch = 1.0 / (1.0 + np.exp(-val_logits_epoch))
        val_loss_values = -(val_targets_epoch * np.log(np.clip(val_probabilities_epoch, 1e-6, 1 - 1e-6)) + (1.0 - val_targets_epoch) * np.log(np.clip(1.0 - val_probabilities_epoch, 1e-6, 1 - 1e-6)))
        val_loss = float(val_loss_values[val_known_epoch].mean()) if val_known_epoch.any() else float("inf")
        history.append({"epoch": float(epoch + 1), "train_masked_bce": total_loss / max(seen, 1), "validation_masked_bce": val_loss})
        print(f"epoch {epoch + 1}/{epochs}: train_bce={history[-1]['train_masked_bce']:.4f} val_bce={val_loss:.4f}", flush=True)
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in network.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if regularized and stale_epochs >= patience:
                break

    if best_state is not None:
        network.load_state_dict(best_state)

    val_logits, val_targets, val_known, _ = _predict(network, val_loader, device)
    test_logits, test_targets, test_known, _ = _predict(network, test_loader, device)
    temperature = _temperature(val_logits, val_targets, val_known)
    val_probabilities = 1.0 / (1.0 + np.exp(-val_logits / temperature))
    test_probabilities = 1.0 / (1.0 + np.exp(-test_logits / temperature))
    metrics: dict[str, Any] = {}
    thresholds: dict[str, dict[str, float]] = {}
    for label_index, label in enumerate(LABELS):
        val_mask = val_known[:, label_index]
        test_mask = test_known[:, label_index]
        threshold = _threshold(val_targets[val_mask, label_index], val_probabilities[val_mask, label_index]) if val_mask.any() else 0.65
        thresholds[label] = {"positive_threshold": threshold, "negative_threshold": round(1.0 - threshold, 5), "min_reliability": 0.30}
        metrics[label] = {
            "validation": _metrics(val_targets[val_mask, label_index], val_probabilities[val_mask, label_index], threshold),
            "test": _metrics(test_targets[test_mask, label_index], test_probabilities[test_mask, label_index], threshold),
        }
    dataset_summary = {
        "images_discovered": len(index.images),
        "labeled_images_used": len(all_records),
        "unique_patient_groups": len(set(groups)),
        "device": str(device),
        "label_known_counts": {label: int(sum(value is not None for value in targets[label])) for label in LABELS},
        "label_positive_counts": {label: int(sum(value == 1.0 for value in targets[label])) for label in LABELS},
    }
    split_summary = {"patient_level": True, "seed": seed, "train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)}
    analysis_payload = {
        "version": 1,
        "mode": "densenet121_regularized" if regularized else "densenet121_trained",
        "model": "DenseNet-121 with ImageNet initialization; frozen-backbone regularized transfer learning" if regularized else "DenseNet-121 with ImageNet initialization; fine-tuned classifier and final dense block",
        "temperature": temperature,
        "history": history,
        "metrics": metrics,
        "aggregate": {"validation": _aggregate(metrics, "validation"), "test": _aggregate(metrics, "test")},
        "dataset": dataset_summary,
        "split": split_summary,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": network.state_dict(), "temperature": temperature, "thresholds": thresholds, "labels": LABELS, "mode": analysis_payload["mode"]}, output_path)
    metrics_path = output_path.with_name("densenet_regularized_metrics.json" if regularized else "densenet_metrics.json")
    metrics_path.write_text(json.dumps(analysis_payload, indent=2), encoding="utf-8")
    output_path.with_name("densenet_regularized_thresholds.json" if regularized else "densenet_thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    return {"checkpoint": str(output_path), "metrics": str(metrics_path), "mode": analysis_payload["mode"], "device": str(device), "split": split_summary, "test_macro": analysis_payload["aggregate"]["test"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "artifacts" / "densenet121.pt")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--regularized", action="store_true", help="Use frozen-backbone transfer learning, conservative augmentation, class weighting, label smoothing, and early stopping.")
    args = parser.parse_args()
    print(json.dumps(train(args.output, args.max_images, args.epochs, args.batch_size, args.seed, args.regularized), indent=2))
