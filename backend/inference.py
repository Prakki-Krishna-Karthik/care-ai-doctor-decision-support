"""Quality-gated multimodal inference.

The default path is deterministic demo inference.  It intentionally abstains
from clinical conclusions until a locally trained checkpoint is available.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFile, ImageFilter

ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    import torch
    from torchvision import models, transforms
except ImportError:  # pragma: no cover
    torch = None
    models = None
    transforms = None

from config import LABELS, Settings, load_thresholds
from clinical_model import predict_selected_clinical
from clinical_guidance import build_clinical_guidance
from rag import LocalEvidenceRetriever, compare_report_to_labels

try:  # Optional at runtime; the API still works in demo mode without joblib.
    import joblib
except ImportError:  # pragma: no cover - exercised only in minimal installs
    joblib = None


FEATURE_NAMES = [
    "mean_intensity",
    "contrast_std",
    "edge_std",
    "width_norm",
    "height_norm",
    "aspect_ratio",
    "contrast_quality",
    "sharpness_quality",
    "size_quality",
    "image_quality",
    "clinical_completeness",
    "age_norm",
]

EVIDENCE_CARDS = [
    {"title": "MIMIC-CXR: a large publicly available database of labeled chest radiographs", "source": "Johnson et al. · arXiv:1901.07042", "citation": "MIMIC-CXR", "keywords": {"general"}},
    {"title": "CheXpert: a large chest radiograph dataset with uncertainty labels", "source": "Irvin et al. · AAAI 2019", "citation": "CheXpert", "keywords": {"general", "uncertainty"}},
    {"title": "A ConvNet for the 2020s", "source": "Liu et al. · CVPR 2022", "citation": "ConvNeXt", "keywords": {"general", "model"}},
    {"title": "Densely Connected Convolutional Networks", "source": "Huang et al. · CVPR 2017", "citation": "DenseNet", "keywords": {"general", "model"}},
    {"title": "Hidden stratification causes clinically meaningful errors in medical image classification", "source": "Oakden-Rayner et al. · Nature Machine Intelligence", "citation": "Hidden stratification", "keywords": {"general", "reliability"}},
    {"title": "Measuring calibration in modern neural networks", "source": "Guo et al. · ICML 2017", "citation": "Calibration", "keywords": {"general", "uncertainty"}},
]


@dataclass(frozen=True)
class ImageQuality:
    contrast: float
    sharpness: float
    size: float
    score: float
    width: int
    height: int


def _clip(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _image_stats(image_bytes: bytes) -> tuple[ImageQuality, np.ndarray]:
    with Image.open(io.BytesIO(image_bytes)) as source:
        image = source.convert("L")
        width, height = image.size
        if max(width, height) > 768:
            resampling = getattr(Image, "Resampling", Image).LANCZOS
            image.thumbnail((768, 768), resampling)
        array = np.asarray(image, dtype=np.float32)
        contrast_std = float(array.std())
        edge_array = np.asarray(image.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
        edge_std = float(edge_array.std())
        contrast = _clip(contrast_std / 64.0)
        sharpness = _clip(edge_std / 48.0)
        size = _clip(min(width, height) / 512.0)
        score = _clip(0.40 * contrast + 0.40 * sharpness + 0.20 * size)
        quality = ImageQuality(contrast, sharpness, size, score, width, height)
        return quality, array


def image_quality(image_bytes: bytes) -> ImageQuality:
    return _image_stats(image_bytes)[0]


def clinical_completeness(clinical: dict[str, Any]) -> float:
    if not clinical:
        return 0.0
    aliases = {
        "age": ("age", "patient_age"),
        "sex": ("sex", "gender"),
        "view_position": ("view_position", "view", "projection"),
        "indication": ("indication", "clinical_indication"),
        "history": ("history", "clinical_history", "report", "symptoms", "notes"),
    }
    present = 0
    for names in aliases.values():
        if any(str(clinical.get(name, "")).strip() for name in names):
            present += 1
    # When a caller provides a structured row with other useful fields, count
    # those too, without allowing arbitrary metadata to produce full trust.
    if present == 0:
        present = sum(1 for value in clinical.values() if str(value).strip())
        return _clip(present / 5.0)
    return _clip(present / len(aliases))


def clinical_context(clinical: dict[str, Any]) -> dict[str, Any]:
    """Remove identifiers, provenance, and target labels before feature use."""
    excluded = {label.lower() for label in LABELS}
    excluded.update({"study_id", "subject_id", "patient_id", "dicom_id", "image_id"})
    return {
        str(key): value
        for key, value in clinical.items()
        if not str(key).startswith("_") and str(key).lower() not in excluded
    }


def _age_norm(clinical: dict[str, Any]) -> float:
    for key in ("age", "patient_age"):
        try:
            return _clip(float(clinical.get(key, 0)) / 100.0)
        except (TypeError, ValueError):
            pass
    return 0.0


def feature_vector(image_bytes: bytes, clinical: dict[str, Any] | None = None) -> tuple[np.ndarray, ImageQuality, float]:
    clinical = clinical or {}
    quality, array = _image_stats(image_bytes)
    values = [
        float(array.mean() / 255.0),
        float(array.std() / 128.0),
        float(np.asarray(Image.fromarray(array.astype(np.uint8)).filter(ImageFilter.FIND_EDGES)).std() / 128.0),
        min(quality.width / 2048.0, 1.0),
        min(quality.height / 2048.0, 1.0),
        _clip(quality.width / max(quality.height, 1)),
        quality.contrast,
        quality.sharpness,
        quality.size,
        quality.score,
        clinical_completeness(clinical),
        _age_norm(clinical),
    ]
    return np.asarray(values, dtype=np.float32), quality, clinical_completeness(clinical)


def _stable_signal(label: str, quality: ImageQuality, array: np.ndarray) -> float:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    phase = digest[0] / 255.0
    mean_signal = float(array.mean() / 255.0)
    texture_signal = _clip(float(array.std()) / 96.0)
    return _clip(0.10 + 0.20 * phase + 0.25 * quality.score + 0.20 * mean_signal + 0.15 * texture_signal)


def _clinical_signal(label: str, clinical: dict[str, Any]) -> float:
    text = " ".join(str(value).lower() for value in clinical.values())
    if not text:
        return 0.10
    label_tokens = [token for token in label.lower().split() if len(token) > 3]
    hits = sum(token in text for token in label_tokens)
    return _clip(0.10 + 0.20 * hits + 0.15 * clinical_completeness(clinical))


def _read_model(path: Any) -> dict[str, Any] | None:
    if joblib is None or not path.is_file():
        return None
    try:
        payload = joblib.load(path)
    except Exception:
        return None
    return payload if isinstance(payload, dict) and payload.get("models") else None


def _read_clinical_model(path: Any) -> dict[str, Any] | None:
    if joblib is None or not path.is_file():
        return None
    try:
        payload = joblib.load(path)
    except Exception:
        return None
    return payload if isinstance(payload, dict) and isinstance(payload.get("models"), dict) else None


def _model_probability(model_entry: Any, vector: np.ndarray) -> float | None:
    if not isinstance(model_entry, dict):
        return None
    if model_entry.get("constant") is not None:
        return _clip(float(model_entry["constant"]))
    model = model_entry.get("model")
    if model is None:
        return None
    try:
        return _clip(float(model.predict_proba(vector.reshape(1, -1))[0, 1]))
    except Exception:
        return None


class InferenceEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.rag = LocalEvidenceRetriever(settings.data_root)
        self.checkpoint = _read_model(settings.checkpoint_path)
        self.clinical_model = _read_clinical_model(settings.clinical_model_path)
        self.deployment_selection: dict[str, Any] = {}
        self.readiness: dict[str, Any] = {}
        comparison_path = settings.deep_checkpoint_path.with_name("comparative_results.json")
        if comparison_path.is_file():
            try:
                comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
                if isinstance(comparison.get("selection"), dict):
                    self.deployment_selection = comparison["selection"]
            except (OSError, ValueError, TypeError):
                self.deployment_selection = {}
        readiness_path = settings.deep_checkpoint_path.with_name("model_readiness.json")
        if readiness_path.is_file():
            try:
                value = json.loads(readiness_path.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    self.readiness = value
            except (OSError, ValueError, TypeError):
                self.readiness = {}
        self.thresholds = load_thresholds(settings.threshold_path)
        self.deep_model = None
        self.deep_temperature = 1.0
        self.deep_device = None
        self.deep_transform = None
        # Keep the checkpoint self-contained when a custom output path is used.
        if self.checkpoint and isinstance(self.checkpoint.get("thresholds"), dict):
            for label, values in self.checkpoint["thresholds"].items():
                if label not in self.thresholds or not isinstance(values, dict):
                    continue
                for key in ("positive_threshold", "negative_threshold", "min_reliability"):
                    if key in values:
                        try:
                            self.thresholds[label][key] = _clip(float(values[key]))
                        except (TypeError, ValueError):
                            pass
        if torch is not None and models is not None and settings.deep_checkpoint_path.is_file():
            try:
                payload = torch.load(settings.deep_checkpoint_path, map_location="cpu", weights_only=False)
                network = models.densenet121(weights=None)
                state_dict = payload.get("state_dict", {})
                # Support both the original checkpoint (Linear head) and the
                # regularized checkpoint (Dropout + Linear head).
                if any(str(key).startswith("classifier.1.") for key in state_dict):
                    network.classifier = torch.nn.Sequential(
                        torch.nn.Dropout(p=0.35),
                        torch.nn.Linear(network.classifier.in_features, len(LABELS)),
                    )
                else:
                    network.classifier = torch.nn.Linear(network.classifier.in_features, len(LABELS))
                network.load_state_dict(state_dict)
                self.deep_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.deep_model = network.to(self.deep_device).eval()
                self.deep_temperature = float(payload.get("temperature", 1.0))
                saved_thresholds = payload.get("thresholds")
                if isinstance(saved_thresholds, dict):
                    for label, values in saved_thresholds.items():
                        if isinstance(values, dict):
                            self.thresholds[label] = values
                self.deep_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])
            except Exception:
                self.deep_model = None

    @property
    def demo_mode(self) -> bool:
        return self.deep_model is None and self.checkpoint is None

    def status(self) -> dict[str, Any]:
        mode = "densenet121_trained" if self.deep_model is not None else "trained_baseline" if self.checkpoint is not None else "demo"
        return {
            "mode": mode,
            "demo_mode": self.demo_mode,
            "checkpoint_path": str(self.settings.checkpoint_path),
            "checkpoint_available": self.checkpoint is not None,
            "deep_checkpoint_path": str(self.settings.deep_checkpoint_path),
            "deep_checkpoint_available": self.deep_model is not None,
            "device": str(self.deep_device) if self.deep_device is not None else "cpu",
            "decision_policy": "abstain in demo mode; thresholded and reliability-gated with a checkpoint",
            "clinical_model_path": str(self.settings.clinical_model_path),
            "clinical_model_available": self.clinical_model is not None,
            "clinical_model_candidate": self.clinical_model.get("candidate") if self.clinical_model else None,
            "fusion_mode": self.deployment_selection.get("deployed_fusion", "reliability_aware_late_fusion"),
            "deployed_image_model": self.deployment_selection.get("deployed_image_model", "densenet121_trained"),
            "readiness_status": self.readiness.get("status", "not_evaluated"),
            "deployment_allowed": self.readiness.get("deployment_allowed", False),
            "evidence_retrieval": "local_tfidf_with_optional_mimic_report",
            "uncertainty_method": "three-view image augmentation spread",
        }

    def _deep_predict(self, image_bytes: bytes) -> dict[str, float]:
        if self.deep_model is None or self.deep_transform is None or self.deep_device is None:
            return {}
        with Image.open(io.BytesIO(image_bytes)) as source:
            tensor = self.deep_transform(source.convert("RGB")).unsqueeze(0).to(self.deep_device)
        with torch.no_grad():
            logits = self.deep_model(tensor) / max(self.deep_temperature, 1e-6)
            probabilities = torch.sigmoid(logits)[0].detach().cpu().numpy()
        return {label: _clip(float(probabilities[index])) for index, label in enumerate(LABELS)}

    def _deep_predict_with_uncertainty(self, image_bytes: bytes) -> tuple[dict[str, float], dict[str, float], float]:
        """Estimate prediction spread under small, anatomy-preserving changes.

        This is an uncertainty indicator, not a formal Bayesian posterior.  It
        deliberately avoids horizontal flips because laterality is clinically
        meaningful in chest radiographs.
        """

        if self.deep_model is None:
            return {}, {}, 1.0
        variants = [image_bytes]
        try:
            with Image.open(io.BytesIO(image_bytes)) as source:
                base = source.convert("RGB")
                for image in (
                    ImageEnhance.Contrast(base).enhance(0.90),
                    ImageEnhance.Brightness(base).enhance(1.05),
                ):
                    buffer = io.BytesIO()
                    image.save(buffer, format="PNG")
                    variants.append(buffer.getvalue())
        except (OSError, ValueError):
            pass
        predictions = [self._deep_predict(item) for item in variants]
        matrix = np.asarray([[prediction.get(label, 0.5) for label in LABELS] for prediction in predictions])
        means = matrix.mean(axis=0)
        spreads = matrix.std(axis=0)
        return (
            {label: _clip(float(means[index])) for index, label in enumerate(LABELS)},
            {label: _clip(float(spreads[index])) for index, label in enumerate(LABELS)},
            _clip(float(spreads.mean() * 5.0)),
        )

    def analyze(self, image_bytes: bytes, clinical: dict[str, Any], image_filename: str | None = None) -> dict[str, Any]:
        raw_clinical = dict(clinical or {})
        study_id = raw_clinical.get("study_id")
        report_text = raw_clinical.get("radiology_report")
        clinical = clinical_context(raw_clinical)
        vector, quality, clinical_score = feature_vector(image_bytes, clinical)
        image_reliability = quality.score
        clinical_reliability = clinical_score
        eps = 1e-8
        wc = clinical_reliability / (clinical_reliability + image_reliability + eps)
        wi = image_reliability / (clinical_reliability + image_reliability + eps)
        selected_fusion = self.deployment_selection.get("deployed_fusion", "reliability_aware_late_fusion")
        if selected_fusion == "image_only":
            wi, wc = 1.0, 0.0
        elif selected_fusion == "clinical_only":
            wi, wc = 0.0, 1.0

        image_spread: dict[str, float] = {label: 0.0 for label in LABELS}
        ensemble_uncertainty = 0.0
        if self.deep_model is not None:
            image_probs, image_spread, ensemble_uncertainty = self._deep_predict_with_uncertainty(image_bytes)
            clinical_probs = predict_selected_clinical(self.clinical_model, clinical, LABELS) or {label: _clinical_signal(label, clinical) for label in LABELS}
        elif self.demo_mode:
            quality_obj, array = _image_stats(image_bytes)
            image_probs = {label: _stable_signal(label, quality_obj, array) for label in LABELS}
            clinical_probs = {label: _clinical_signal(label, clinical) for label in LABELS}
        else:
            image_probs = {}
            # Prefer the fitted structured-data branch selected by the
            # comparative study; use the transparent fallback only when the
            # artifact is unavailable.
            clinical_probs = predict_selected_clinical(self.clinical_model, clinical, LABELS) or {
                item: _clinical_signal(item, clinical) for item in LABELS
            }
            for label in LABELS:
                probability = _model_probability(self.checkpoint.get("models", {}).get(label), vector)
                image_probs[label] = probability if probability is not None else 0.5

        # Confidence shrinkage is deliberately conservative: a probability
        # is pulled toward 0.5 when the supporting input is weak. This limits
        # overconfident predictions on low-quality images or sparse context.
        support_score = _clip(wi * image_reliability + wc * clinical_reliability)
        support_factor = support_score / (support_score + 0.30)
        labels: list[dict[str, Any]] = []
        for label in LABELS:
            raw_fused = _clip(wi * image_probs[label] + wc * clinical_probs[label])
            fused = _clip(0.5 + support_factor * (raw_fused - 0.5))
            threshold = self.thresholds.get(label, {})
            positive_threshold = float(threshold.get("positive_threshold", 0.65))
            negative_threshold = float(threshold.get("negative_threshold", 0.35))
            minimum_reliability = float(threshold.get("min_reliability", 0.30))
            reliable_enough = support_score >= minimum_reliability
            if self.demo_mode or not reliable_enough:
                decision = "abstain"
            elif fused >= positive_threshold:
                decision = "positive_signal_for_review"
            elif fused <= negative_threshold:
                decision = "negative_signal_for_review"
            else:
                decision = "abstain"
            labels.append(
                {
                    "label": label,
                    "probability": round(fused, 5),
                    "raw_probability": round(raw_fused, 5),
                    "support_score": round(support_score, 5),
                    "support_factor": round(support_factor, 5),
                    "image_probability": round(image_probs[label], 5),
                    "clinical_probability": round(clinical_probs[label], 5),
                    "ensemble_spread": round(image_spread.get(label, 0.0), 5),
                    "positive_threshold": positive_threshold,
                    "negative_threshold": negative_threshold,
                    "decision": decision,
                    "abstained": decision == "abstain",
                }
            )

        top_index = max(range(len(LABELS)), key=lambda index: image_probs.get(LABELS[index], 0.0)) if image_probs else 0
        overlay_metadata = self._gradcam_metadata(image_bytes, top_index, quality)
        predictive_uncertainty = _clip(0.70 * ensemble_uncertainty + 0.30 * (1.0 - support_score))
        positive_count = sum(item["decision"] == "positive_signal_for_review" for item in labels)
        if support_score < 0.45 or predictive_uncertainty >= 0.20:
            review_priority = "high" if positive_count else "moderate"
        elif positive_count:
            review_priority = "moderate"
        else:
            review_priority = "routine"
        evidence, retrieved_report, retrieval_meta = self.rag.retrieve(
            labels,
            clinical,
            report_text=report_text,
            study_id=str(study_id) if study_id else None,
        )
        report_comparison = compare_report_to_labels(retrieved_report, labels)
        clinical_guidance = build_clinical_guidance(
            labels=labels,
            clinical=raw_clinical,
            quality_score=quality.score,
            clinical_score=clinical_score,
            support_score=support_score,
            predictive_uncertainty=predictive_uncertainty,
            review_priority=review_priority,
            demo_mode=self.demo_mode,
        )
        return {
            "model": self.status(),
            "disclaimer": "Demo-safe decision-support output; this is not a diagnosis and requires clinician review.",
            "quality": {
                "score": round(quality.score, 5),
                "contrast": round(quality.contrast, 5),
                "sharpness": round(quality.sharpness, 5),
                "size": round(quality.size, 5),
                "width": quality.width,
                "height": quality.height,
            },
            "clinical": {
                "completeness_score": round(clinical_score, 5),
                "fields_received": sorted(str(key) for key in clinical),
            },
            "fusion": {
                "clinical_reliability": round(clinical_reliability, 5),
                "image_reliability": round(image_reliability, 5),
                "clinical_weight_wc": round(wc, 5),
                "image_weight_wi": round(wi, 5),
                "epsilon": eps,
                "formula": "wc=Rc/(Rc+Ri+eps), wi=Ri/(Rc+Ri+eps)",
                "support_score": round(support_score, 5),
                "support_factor": round(support_factor, 5),
                "confidence_policy": "probabilities shrink toward 0.5 when modality support is weak",
                "note": f"Support-aware confidence factor is {support_factor:.2f}; weak support is pulled toward uncertainty and routed for review.",
            },
            "overall_decision": "abstain_and_review" if self.demo_mode else self._overall_decision(labels),
            "predictive_uncertainty": round(predictive_uncertainty, 5),
            "review_priority": review_priority,
            "uncertainty": {
                "method": "spread across original, contrast-adjusted, and brightness-adjusted image views",
                "ensemble_spread": round(ensemble_uncertainty, 5),
                "interpretation": "Higher spread means the model is less stable under small image-quality changes.",
            },
            "labels": labels,
            "evidence": evidence,
            "retrieval": retrieval_meta,
            "report_comparison": report_comparison,
            "clinical_guidance": clinical_guidance,
            "audit": [
                {"time": "NOW", "text": f"{len(labels)} label signals generated with {self.status()['mode']}"},
                {"time": "NOW", "text": f"Selected fusion applied ({selected_fusion}): image {wi:.2f}, clinical {wc:.2f}"},
                {"time": "NOW", "text": f"Support gate applied: {support_score:.2f} support, {support_factor:.2f} confidence factor"},
                {"time": "NOW", "text": f"Structured branch: {self.clinical_model.get('candidate') if self.clinical_model else 'transparent fallback'}"},
                {"time": "NOW", "text": f"Local evidence retrieval completed: {len(evidence)} sources; report available: {bool(retrieved_report)}"},
                {"time": "NOW", "text": f"Review priority assigned: {review_priority}; augmentation spread: {ensemble_uncertainty:.3f}"},
                {"time": "NOW", "text": "Threshold policy applied; uncertain signals are routed for clinician review"},
            ],
            "explainability": overlay_metadata,
        }

    @staticmethod
    def _retrieve_evidence(labels: list[dict[str, Any]]) -> list[dict[str, str]]:
        # Lightweight local retrieval keeps the prototype reproducible and
        # avoids sending protected clinical data to an external service.
        positive = {item["label"] for item in labels if item["decision"] == "positive_signal_for_review"}
        terms = {"general"}
        if positive:
            terms.update({"uncertainty", "reliability"})
        selected = [card for card in EVIDENCE_CARDS if card["keywords"] & terms]
        return [
            {key: card[key] for key in ("title", "source", "citation")}
            for card in selected[:4]
        ]

    @staticmethod
    def _overall_decision(labels: list[dict[str, Any]]) -> str:
        if any(item["decision"] == "positive_signal_for_review" for item in labels):
            return "review_positive_signals"
        if all(item["decision"] == "negative_signal_for_review" for item in labels):
            return "review_negative_signals"
        return "abstain_and_review"

    def _gradcam_metadata(self, image_bytes: bytes, target_index: int, quality: ImageQuality) -> dict[str, Any]:
        """Return a compact Grad-CAM grid for the selected image model.

        The API returns numbers rather than a rendered clinical overlay so the
        UI can remain local and lightweight.  The grid is an explanation aid,
        not a lesion segmentation or diagnostic proof.
        """
        if self.deep_model is not None and self.deep_transform is not None and self.deep_device is not None and torch is not None:
            activations: dict[str, Any] = {}
            gradients: dict[str, Any] = {}
            target_layer = self.deep_model.features.denseblock4

            def save_activation(_module: Any, _inputs: Any, output: Any) -> None:
                activations["value"] = output

            def save_gradient(_module: Any, _grad_inputs: Any, grad_outputs: Any) -> None:
                if grad_outputs:
                    gradients["value"] = grad_outputs[0]

            forward_handle = target_layer.register_forward_hook(save_activation)
            backward_handle = target_layer.register_full_backward_hook(save_gradient)
            try:
                with Image.open(io.BytesIO(image_bytes)) as source:
                    tensor = self.deep_transform(source.convert("RGB")).unsqueeze(0).to(self.deep_device)
                self.deep_model.zero_grad(set_to_none=True)
                with torch.enable_grad():
                    logits = self.deep_model(tensor) / max(self.deep_temperature, 1e-6)
                    logits[0, target_index].backward()
                activation = activations.get("value")
                gradient = gradients.get("value")
                if activation is not None and gradient is not None:
                    weights = gradient.mean(dim=(2, 3), keepdim=True)
                    cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
                    cam = torch.nn.functional.interpolate(cam, size=(8, 8), mode="bilinear", align_corners=False)[0, 0]
                    cam = cam.detach().cpu().numpy().astype(np.float32)
                    cam -= float(cam.min())
                    maximum = float(cam.max())
                    if maximum > 1e-8:
                        cam /= maximum
                    return {
                        "method": "grad_cam",
                        "grad_cam_available": True,
                        "overlay_available": False,
                        "target_label": LABELS[target_index],
                        "heatmap_grid": [[round(float(value), 5) for value in row] for row in cam],
                        "message": "Grad-CAM highlights image regions that most influenced the selected model signal; it is not a segmentation or diagnosis.",
                    }
            except (OSError, RuntimeError, ValueError, IndexError):
                pass
            finally:
                forward_handle.remove()
                backward_handle.remove()

        return self._saliency_metadata(quality)

    @staticmethod
    def _saliency_metadata(quality: ImageQuality) -> dict[str, Any]:
        """Safe fallback when a trained convolutional model is unavailable."""
        return {
            "method": "placeholder_quality_guided_overlay",
            "grad_cam_available": False,
            "overlay_available": False,
            "message": "No trained localization head is available; regions are placeholders and must not be interpreted as evidence.",
            "regions": [
                {
                    "x": 0.15,
                    "y": 0.15,
                    "width": 0.70,
                    "height": 0.70,
                    "weight": round(quality.score, 5),
                }
            ],
        }


def tiny_preview_base64(image_bytes: bytes) -> str:
    """Return a small JPEG data URL for optional local UI previews."""
    with Image.open(io.BytesIO(image_bytes)) as source:
        image = source.convert("L")
        image.thumbnail((256, 256))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=70)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
