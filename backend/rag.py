"""Local retrieval and report-comparison utilities.

This module deliberately stays offline.  When the credentialed MIMIC report
archive is available, it retrieves the report for a supplied study id and
matches relevant report chunks with TF-IDF.  No patient data or report text
is sent to a remote model or service.
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


LOCAL_CITATIONS = [
    {
        "title": "MIMIC-CXR: A large publicly available database of labeled chest radiographs",
        "source": "Johnson et al. · arXiv:1901.07042",
        "citation": "MIMIC-CXR",
        "text": "Chest radiograph reports and labels support reproducible evaluation of automated findings.",
        "keywords": {"general", "dataset"},
    },
    {
        "title": "CheXpert: A large chest radiograph dataset with uncertainty labels",
        "source": "Irvin et al. · AAAI 2019",
        "citation": "CheXpert",
        "text": "Uncertain labels should be treated explicitly rather than silently converted into reliable positive or negative labels.",
        "keywords": {"general", "uncertainty", "labels"},
    },
    {
        "title": "Densely Connected Convolutional Networks",
        "source": "Huang et al. · CVPR 2017",
        "citation": "DenseNet",
        "text": "Dense connectivity encourages feature reuse and is a practical transfer-learning backbone for image classification.",
        "keywords": {"general", "model", "image"},
    },
    {
        "title": "Hidden stratification causes clinically meaningful errors in medical image classification",
        "source": "Oakden-Rayner et al. · Nature Machine Intelligence",
        "citation": "Hidden stratification",
        "text": "Aggregate performance can hide subgroup and acquisition differences, so error analysis should examine clinically relevant cohorts.",
        "keywords": {"general", "reliability", "fairness"},
    },
    {
        "title": "Measuring calibration in deep learning systems",
        "source": "Guo et al. · ICML 2017",
        "citation": "Calibration",
        "text": "A probability should reflect the observed frequency of the event; calibration is distinct from discrimination.",
        "keywords": {"general", "uncertainty", "calibration"},
    },
]


LABEL_ALIASES: dict[str, tuple[str, ...]] = {
    "Atelectasis": ("atelectasis", "collapse"),
    "Cardiomegaly": ("cardiomegaly", "enlarged heart"),
    "Consolidation": ("consolidation",),
    "Edema": ("edema", "pulmonary edema"),
    "Enlarged Cardiomediastinum": ("enlarged cardiomediastinum",),
    "Fracture": ("fracture",),
    "Lung Lesion": ("lung lesion",),
    "Lung Opacity": ("lung opacity", "opacity"),
    "No Finding": ("no acute cardiopulmonary", "no finding", "normal"),
    "Pleural Effusion": ("pleural effusion", "effusion"),
    "Pleural Other": ("pleural",),
    "Pneumonia": ("pneumonia",),
    "Pneumothorax": ("pneumothorax",),
    "Support Devices": ("support device", "support devices", "tube", "catheter"),
}


def _chunks(text: str, size: int = 900) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    result: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > size:
            result.append(current)
            current = ""
        current = f"{current} {sentence}".strip()
    if current:
        result.append(current)
    return result


def _is_negated(text: str, start: int) -> bool:
    window = text[max(0, start - 45) : start].lower()
    return bool(re.search(r"\b(no|not|without|absent|negative|denies|denied)\b", window))


def compare_report_to_labels(report_text: str | None, labels: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare model signals with explicitly mentioned report findings."""

    if not report_text or not report_text.strip():
        return {"available": False, "message": "No radiology report was supplied or found."}
    text = re.sub(r"\s+", " ", report_text).strip()
    comparisons: list[dict[str, Any]] = []
    for item in labels:
        label = str(item.get("label", ""))
        aliases = LABEL_ALIASES.get(label, (label.lower(),))
        mentions: list[str] = []
        negated: list[str] = []
        for alias in aliases:
            for match in re.finditer(re.escape(alias), text, flags=re.IGNORECASE):
                (negated if _is_negated(text, match.start()) else mentions).append(alias)
        if mentions or negated:
            report_state = "negative_in_report" if not mentions else "mentioned_in_report"
        else:
            report_state = "not_mentioned"
        comparisons.append(
            {
                "label": label,
                "model_probability": item.get("probability"),
                "model_decision": item.get("decision"),
                "report_state": report_state,
                "matched_terms": sorted(set(mentions + negated)),
            }
        )
    return {
        "available": True,
        "method": "keyword-aware report comparison; negation is reported separately",
        "comparison": comparisons,
        "report_excerpt": text[:1200],
    }


class LocalEvidenceRetriever:
    def __init__(self, data_root: Path):
        self.data_root = data_root
        self.report_zip_path = data_root / "mimic-cxr-reports.zip"
        self._zip: zipfile.ZipFile | None = None
        self._report_names: dict[str, str] | None = None

    def _ensure_index(self) -> None:
        if self._report_names is not None or not self.report_zip_path.is_file():
            return
        try:
            archive = zipfile.ZipFile(self.report_zip_path)
            self._zip = archive
            self._report_names = {
                Path(name).stem.lower(): name
                for name in archive.namelist()
                if name.lower().endswith(".txt")
            }
        except (OSError, zipfile.BadZipFile):
            self._zip = None
            self._report_names = {}

    def report_for_study(self, study_id: str | None) -> str | None:
        self._ensure_index()
        if not study_id or not self._zip or not self._report_names:
            return None
        name = self._report_names.get(str(study_id).strip().lower())
        if not name:
            return None
        try:
            return self._zip.read(name).decode("utf-8", errors="replace")
        except (KeyError, OSError, UnicodeError):
            return None

    def retrieve(
        self,
        labels: list[dict[str, Any]],
        clinical: dict[str, Any],
        report_text: str | None = None,
        study_id: str | None = None,
    ) -> tuple[list[dict[str, Any]], str | None, dict[str, Any]]:
        report = report_text or self.report_for_study(study_id)
        positive = [str(item.get("label", "")) for item in labels if item.get("decision") == "positive_signal_for_review"]
        query = " ".join(positive + [str(clinical.get("symptoms", "")), str(clinical.get("notes", ""))]).strip()
        if not query:
            query = "chest radiograph uncertainty reliability calibration"

        documents: list[dict[str, Any]] = []
        for card in LOCAL_CITATIONS:
            documents.append({**card, "source_type": "local_citation"})
        if report:
            for index, chunk in enumerate(_chunks(report)):
                documents.append(
                    {
                        "title": f"MIMIC-CXR report evidence · section {index + 1}",
                        "source": f"Local report{f' · study {study_id}' if study_id else ''}",
                        "citation": "MIMIC-CXR report",
                        "text": chunk,
                        "source_type": "mimic_report",
                    }
                )

        texts = [str(document["text"]) for document in documents]
        try:
            matrix = TfidfVectorizer(stop_words="english").fit_transform(texts + [query])
            scores = (matrix[:-1] @ matrix[-1].T).toarray().ravel()
        except ValueError:
            scores = np.zeros(len(documents), dtype=float)
        ranked = sorted(zip(scores, documents), key=lambda item: float(item[0]), reverse=True)
        selected: list[dict[str, Any]] = []
        for score, document in ranked[:5]:
            selected.append(
                {
                    "title": document["title"],
                    "source": document["source"],
                    "citation": document["citation"],
                    "score": round(float(score), 5),
                    "source_type": document["source_type"],
                    "excerpt": document["text"][:500],
                }
            )
        retrieval_meta = {
            "method": "local_tfidf_retrieval",
            "report_available": bool(report),
            "study_id": study_id,
            "documents_considered": len(documents),
            "patient_data_sent_external": False,
        }
        return selected, report, retrieval_meta
