"""Local MIMIC image and CSV discovery with tolerant joins."""

from __future__ import annotations

import csv
import gzip
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from config import LABELS, Settings


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    path: Path
    subject_id: str | None
    study_id: str | None
    dicom_id: str


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _canonical_id(value: str | None) -> str | None:
    """Make CSV numeric IDs join with directory IDs such as ``p10``/``s20``."""
    if value is None:
        return None
    text = value.strip()
    if re.fullmatch(r"[ps]\d+", text, re.I):
        return text[1:]
    return text


def _first(row: dict[str, Any], names: Iterable[str]) -> str | None:
    lowered = {str(k).strip().lower(): v for k, v in row.items()}
    for name in names:
        value = _clean(lowered.get(name.lower()))
        if value:
            return value
    return None


def _parse_image_path(path: Path, identity_path: Path | None = None) -> ImageRecord:
    # A flat download has the cleanest storage path but does not contain the
    # patient/study folders. When a nested copy exists, use its folders only
    # as identity metadata while keeping the shallow image path selected.
    parts = list(path.parts)
    if identity_path is not None:
        parts.extend(identity_path.parts)
    subject_id = next((part for part in reversed(parts) if re.fullmatch(r"p\d+", part, re.I)), None)
    study_id = next((part for part in reversed(parts) if re.fullmatch(r"s\d+", part, re.I)), None)
    dicom_id = path.stem
    return ImageRecord(
        image_id=dicom_id,
        path=path,
        subject_id=subject_id,
        study_id=study_id,
        dicom_id=dicom_id,
    )


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            opener = gzip.open if path.suffix.lower() == ".gz" else open
            with opener(path, "rt", encoding=encoding, newline="") as handle:
                return list(csv.DictReader(handle))
        except (UnicodeDecodeError, csv.Error, OSError):
            continue
    return []


def _normalise_label_value(value: Any) -> float | None:
    text = _clean(value)
    if text is None:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    # CheXpert uses -1 for uncertain.  Treat it as unknown for the baseline.
    if number < 0:
        return None
    return 1.0 if number >= 0.5 else 0.0


def _label_count(row: dict[str, Any]) -> int:
    return sum(1 for label in LABELS if _normalise_label_value(row.get(label)) is not None)


@dataclass
class DatasetIndex:
    settings: Settings
    images: list[ImageRecord] = field(default_factory=list)
    clinical_files: list[Path] = field(default_factory=list)
    clinical_rows: list[dict[str, str]] = field(default_factory=list)
    rows_by_dicom: dict[str, dict[str, str]] = field(default_factory=dict)
    rows_by_study: dict[tuple[str | None, str | None], dict[str, str]] = field(default_factory=dict)
    raw_image_count: int = 0
    duplicate_image_count: int = 0
    duplicate_image_ids: list[str] = field(default_factory=list)
    last_scan: float = 0.0
    scan_error: str | None = None

    def refresh_if_needed(self, force: bool = False) -> None:
        if not force and self.last_scan and time.time() - self.last_scan < self.settings.scan_ttl_seconds:
            return
        self.refresh()

    def refresh(self) -> None:
        self.scan_error = None
        try:
            image_paths = self._image_paths()
            self.raw_image_count = len(image_paths)
            selected_paths: dict[str, Path] = {}
            identity_paths: dict[str, Path] = {}
            duplicate_ids: set[str] = set()
            # Prefer the shallowest path. This makes a direct image such as
            # D:\\MIMIC\\jpg\\<id>.jpg win over an older nested copy while
            # retaining a deterministic policy for any other duplicates.
            for path in sorted(
                image_paths,
                key=lambda item: (len(item.relative_to(self.settings.image_root).parts), str(item).lower()),
            ):
                image_id = path.stem.lower()
                if image_id in selected_paths:
                    duplicate_ids.add(image_id)
                else:
                    selected_paths[image_id] = path
                # Keep the richest path as identity metadata, even when the
                # direct flat file is preferred for the actual image bytes.
                if image_id not in identity_paths or len(path.parts) > len(identity_paths[image_id].parts):
                    identity_paths[image_id] = path
            self.duplicate_image_ids = sorted(duplicate_ids)
            self.duplicate_image_count = self.raw_image_count - len(selected_paths)
            self.images = sorted(
                (_parse_image_path(path, identity_paths.get(image_id)) for image_id, path in selected_paths.items()),
                key=lambda item: item.image_id,
            )
            self.clinical_files = sorted(self._csv_paths(), key=lambda path: str(path).lower())
            self.clinical_rows = []
            self.rows_by_dicom = {}
            self.rows_by_study = {}
            for path in self.clinical_files:
                rows = _read_csv_rows(path)
                for row in rows:
                    row["_source_file"] = str(path)
                    self.clinical_rows.append(row)
                    dicom_id = _first(row, ("dicom_id", "dicom_uid", "image_id"))
                    subject_id = _canonical_id(_first(row, ("subject_id", "patient_id", "subject")))
                    study_id = _canonical_id(_first(row, ("study_id", "study")))
                    if dicom_id:
                        previous = self.rows_by_dicom.get(dicom_id)
                        if previous is None or _label_count(row) >= _label_count(previous):
                            self.rows_by_dicom[dicom_id] = row
                    if subject_id or study_id:
                        key = (subject_id, study_id)
                        previous = self.rows_by_study.get(key)
                        if previous is None or _label_count(row) >= _label_count(previous):
                            self.rows_by_study[key] = row
        except (OSError, PermissionError) as exc:
            self.scan_error = str(exc)
        self.last_scan = time.time()

    def _image_paths(self) -> list[Path]:
        if not self.settings.image_root.is_dir():
            return []
        try:
            return [
                path
                for path in self.settings.image_root.rglob("*")
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
            ]
        except OSError as exc:
            self.scan_error = str(exc)
            return []

    def _csv_paths(self) -> list[Path]:
        if not self.settings.data_root.is_dir():
            return []
        try:
            return [
                path
                for path in self.settings.data_root.rglob("*")
                if path.is_file() and path.name.lower().endswith((".csv", ".csv.gz"))
            ]
        except OSError as exc:
            self.scan_error = str(exc)
            return []

    def clinical_for_image(self, record: ImageRecord) -> dict[str, str]:
        # A record-list CSV often wins the direct DICOM join but has no labels;
        # merge it with the richest study-level row (for example CheXpert).
        merged: dict[str, str] = {}
        dicom_row = self.rows_by_dicom.get(record.dicom_id)
        subject_id = _canonical_id(record.subject_id)
        study_id = _canonical_id(record.study_id)
        if dicom_row:
            subject_id = subject_id or _canonical_id(_first(dicom_row, ("subject_id", "patient_id", "subject")))
            study_id = study_id or _canonical_id(_first(dicom_row, ("study_id", "study")))
        study_row = self.rows_by_study.get((subject_id, study_id))
        if study_row:
            merged.update(study_row)
        if dicom_row:
            merged.update(dicom_row)
        return merged

    def labels_for_image(self, record: ImageRecord) -> dict[str, float | None]:
        row = self.clinical_for_image(record)
        return {label: _normalise_label_value(row.get(label)) for label in LABELS if label in row}

    def case_dict(self, record: ImageRecord) -> dict[str, Any]:
        labels = self.labels_for_image(record)
        clinical = self.clinical_for_image(record)
        subject_id = record.subject_id or _first(clinical, ("subject_id", "patient_id", "subject"))
        study_id = record.study_id or _first(clinical, ("study_id", "study"))
        try:
            relative_path = str(record.path.relative_to(self.settings.image_root))
        except ValueError:
            relative_path = record.path.name
        return {
            "case_id": record.image_id,
            "image_id": record.image_id,
            "filename": record.path.name,
            "relative_path": relative_path,
            "subject_id": subject_id,
            "study_id": study_id,
            "dicom_id": record.dicom_id,
            "has_clinical_row": bool(clinical),
            "labels": labels,
        }

    def cases(self, offset: int = 0, limit: int = 50, search: str | None = None) -> tuple[list[dict[str, Any]], int]:
        filtered = self.images
        if search:
            term = search.lower()
            filtered = [
                record
                for record in filtered
                if term in record.image_id.lower()
                or term in (record.subject_id or "").lower()
                or term in (record.study_id or "").lower()
            ]
        total = len(filtered)
        return [self.case_dict(record) for record in filtered[offset : offset + limit]], total

    def label_field_counts(self) -> dict[str, int]:
        return {
            label: sum(1 for row in self.clinical_rows if _normalise_label_value(row.get(label)) is not None)
            for label in LABELS
        }
