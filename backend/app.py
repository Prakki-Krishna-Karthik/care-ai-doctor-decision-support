"""FastAPI entry point for the local chest X-ray decision-support MVP."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from config import Settings
from data import DatasetIndex
from inference import InferenceEngine


settings = Settings()
dataset = DatasetIndex(settings)
engine = InferenceEngine(settings)

app = FastAPI(
    title="Reliability-aware multimodal chest X-ray decision support",
    version="0.1.0",
    description="Local research MVP. Outputs are demo-safe decision-support signals, never a diagnosis.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _parse_clinical_json(raw: str | None) -> dict[str, Any]:
    if raw is None or not raw.strip():
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"clinical_data must be valid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise HTTPException(status_code=422, detail="clinical_data must be a JSON object")
    return value


@app.get("/api/health")
def health() -> dict[str, Any]:
    dataset.refresh_if_needed()
    return {
        "status": "ok",
        "service": "doctor-decision-support-backend",
        "model": engine.status(),
        "data": {
            "image_root": str(settings.image_root),
            "image_root_exists": settings.image_root.is_dir(),
            "data_root": str(settings.data_root),
            "data_root_exists": settings.data_root.is_dir(),
            "images_discovered": len(dataset.images),
            "raw_image_files": dataset.raw_image_count,
            "duplicate_image_files_excluded": dataset.duplicate_image_count,
            "duplicate_image_ids": dataset.duplicate_image_ids,
            "clinical_files_discovered": len(dataset.clinical_files),
            "last_scan": dataset.last_scan,
            "scan_error": dataset.scan_error,
        },
    }


@app.get("/api/stats")
def stats() -> dict[str, Any]:
    dataset.refresh_if_needed()
    matched = sum(1 for record in dataset.images if dataset.clinical_for_image(record))
    return {
        "images": len(dataset.images),
        "raw_image_files": dataset.raw_image_count,
        "duplicate_image_files_excluded": dataset.duplicate_image_count,
        "duplicate_image_ids": dataset.duplicate_image_ids,
        "image_integrity": {
            "policy": "one record per image filename; shallowest path wins",
            "unique_images_used": len(dataset.images),
            "raw_files_seen": dataset.raw_image_count,
        },
        "clinical_files": len(dataset.clinical_files),
        "clinical_rows": len(dataset.clinical_rows),
        "matched_image_cases": matched,
        "label_fields": dataset.label_field_counts(),
        "roots": {"images": str(settings.image_root), "clinical": str(settings.data_root)},
        "model": engine.status(),
    }


@app.get("/api/fairness")
def fairness() -> dict[str, Any]:
    """Return a lightweight cohort audit from available local metadata.

    This endpoint reports cohort sizes and label coverage.  It intentionally
    does not claim subgroup model performance without enough positive and
    negative examples in each cohort.
    """

    dataset.refresh_if_needed()
    # MIMIC-CXR metadata provides acquisition/view fields but does not expose
    # patient sex or age in the downloaded metadata file.  Do not manufacture
    # demographic groups from missing values.
    groups: dict[str, dict[str, int]] = {"view_position": {}}
    for row in dataset.clinical_rows:
        view = str(row.get("ViewPosition", row.get("view_position", "Unknown")) or "Unknown").strip() or "Unknown"
        groups["view_position"][view] = groups["view_position"].get(view, 0) + 1
    return {
        "available": True,
        "method": "metadata cohort audit",
        "groups": groups,
        "unavailable_dimensions": ["sex", "age"],
        "unique_images": len(dataset.images),
        "caution": "Use subgroup AUROC/F1 only when each cohort has adequate positive and negative support; counts alone are not fairness proof.",
    }


@app.post("/api/feedback")
def feedback(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Store optional clinician-review feedback locally for later analysis."""

    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="feedback must be a JSON object")
    path = settings.deep_checkpoint_path.parent / "feedback.jsonl"
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "feedback": payload,
    }
    try:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"feedback could not be saved: {exc}") from exc
    return {"saved": True, "path": str(path), "message": "Feedback stored locally for later review."}


@app.get("/api/metrics")
def metrics() -> dict[str, Any]:
    """Return the saved training/evaluation analysis when available."""
    deep_path = settings.deep_checkpoint_path.with_name("densenet_metrics.json")
    baseline_path = settings.checkpoint_path.with_name("metrics.json")
    path = deep_path if deep_path.is_file() else baseline_path
    if not path.is_file():
        return {"available": False, "message": "Run backend/train.py to create ML metrics."}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=f"metrics file could not be read: {exc}") from exc
    comparison_path = settings.deep_checkpoint_path.with_name("comparative_results.json")
    comparison = None
    if comparison_path.is_file():
        try:
            comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            comparison = None
    response = {"available": True, "active_model": "densenet121" if path == deep_path else "lightweight_baseline", "baseline_metrics_available": baseline_path.is_file(), **payload}
    readiness_path = settings.deep_checkpoint_path.with_name("model_readiness.json")
    if readiness_path.is_file():
        try:
            response["readiness"] = json.loads(readiness_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            response["readiness"] = {"available": False}
    if comparison is not None:
        response["comparison"] = {
            "dataset": comparison.get("dataset", {}),
            "split": comparison.get("split", {}),
            "selection": comparison.get("selection", {}),
            "image_candidates": comparison.get("image_candidates", []),
            "clinical_candidates": comparison.get("clinical_candidates", []),
            "fusion_candidates": comparison.get("fusion_candidates", []),
        }
    return response


@app.get("/api/comparison")
def comparison() -> dict[str, Any]:
    """Return the reproducible image/clinical/fusion model comparison."""
    path = settings.deep_checkpoint_path.with_name("comparative_results.json")
    if not path.is_file():
        return {"available": False, "message": "Run backend/comparative_study.py first."}
    try:
        return {"available": True, **json.loads(path.read_text(encoding="utf-8"))}
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=f"comparison file could not be read: {exc}") from exc


@app.get("/api/cases")
def cases(
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    search: str | None = Query(None, max_length=200),
) -> dict[str, Any]:
    dataset.refresh_if_needed()
    items, total = dataset.cases(offset=offset, limit=limit, search=search)
    return {"items": items, "total": total, "offset": offset, "limit": limit}


@app.post("/api/analyze")
async def analyze(
    image: UploadFile | None = File(None, description="JPEG/PNG chest X-ray image"),
    clinical_data: str | None = Form(
        None,
        description="Optional JSON object, for example {\"age\": 65, \"sex\": \"F\", \"view_position\": \"AP\"}",
    ),
    clinical_json: str | None = Form(None, description="Alias for clinical_data"),
) -> dict[str, Any]:
    if image is None:
        dataset.refresh_if_needed()
        if not dataset.images:
            raise HTTPException(status_code=422, detail="Upload an image or configure a local MIMIC image directory.")
        try:
            image_bytes = dataset.images[0].path.read_bytes()
        except OSError as exc:
            raise HTTPException(status_code=415, detail=f"sample image could not be read: {exc}") from exc
    else:
        image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=422, detail="image is empty")
    if len(image_bytes) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail=f"image exceeds {settings.max_upload_bytes} bytes")
    clinical = _parse_clinical_json(clinical_data if clinical_data is not None else clinical_json)
    try:
        result = engine.analyze(
            image_bytes,
            clinical,
            image_filename=image.filename if image else dataset.images[0].path.name,
        )
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=415, detail=f"image could not be decoded: {exc}") from exc
    result["input"] = {
        "filename": image.filename if image else dataset.images[0].path.name,
        "content_type": image.content_type if image else "image/jpeg",
        "bytes": len(image_bytes),
    }
    return result


FRONTEND_ROOT = Path(__file__).resolve().parents[1] / "frontend"
ARTIFACT_ROOT = settings.deep_checkpoint_path.parent
if ARTIFACT_ROOT.is_dir():
    app.mount("/artifacts", StaticFiles(directory=ARTIFACT_ROOT), name="artifacts")
if FRONTEND_ROOT.is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_ROOT), name="assets")

    @app.get("/", include_in_schema=False)
    def frontend_index() -> FileResponse:
        return FileResponse(FRONTEND_ROOT / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
