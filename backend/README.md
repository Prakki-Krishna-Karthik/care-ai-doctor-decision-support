# Local FastAPI backend

This is a local research MVP for reliability-aware multimodal chest X-ray
decision support. It reads MIMIC-CXR-JPG files and CSVs from the configured
local directories; it does not upload data anywhere.

## Endpoints

- `GET /api/health` — service, data-root, and model status.
- `GET /api/stats` — discovered image/CSV/label counts.
- `GET /api/cases?offset=0&limit=50&search=p10` — paged local image cases.
- `GET /api/fairness` — view-position cohort counts and metadata availability warnings.
- `POST /api/feedback` — append optional review feedback to a local JSONL file.
- `POST /api/analyze` — multipart `image` plus optional `clinical_data` JSON form field.

Example request:

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/analyze `
  -F "image=@C:\path\to\xray.jpg" `
  -F 'clinical_data={"age":65,"sex":"F","view_position":"AP"}'
```

The service computes image quality from contrast, edge sharpness, and image
size, clinical completeness from structured fields, then uses
`wc=Rc/(Rc+Ri+eps)` and `wi=Ri/(Rc+Ri+eps)` for label-wise fusion. Without a
checkpoint the response is explicitly `demo_mode` and every label abstains;
the response is not a diagnosis. The explainability field is a placeholder
metadata region until a localization-capable checkpoint is supplied.

Each `/api/analyze` response also contains `clinical_guidance`. This is a
plain-language clinician checklist with urgency flags, context-dependent
investigation prompts, and a treatment-safety block. It never returns a
medicine, dose, or prescription and must not be used to place an order without
independent clinical assessment.

When `mimic-cxr-reports.zip` is available and `study_id` or
`radiology_report` is supplied in `clinical_data`, the response also includes
local TF-IDF evidence retrieval and a report-versus-model comparison. The
retrieval is offline and does not generate unsupported clinical claims.

## Run

```powershell
cd doctor_decision_support
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r .\backend\requirements.txt
$env:MIMIC_IMAGE_ROOT = 'D:\MIMIC\jpg'
$env:MIMIC_DATA_ROOT = 'C:\Users\Krishna Karthik\Downloads'
python .\backend\app.py
```

## Optional baseline training

Training only uses locally discovered image/label matches, skips unknown
CheXpert `-1` labels, and splits by patient ID before fitting lightweight
one-vs-rest logistic regressions. It writes `baseline.joblib`,
`thresholds.json`, and `metrics.json` under `backend/artifacts`.

```powershell
python .\scripts\inspect_data.py
python .\scripts\train_baseline.py --max-images 5000
```

Restart the API after training so it loads the new checkpoint. This baseline
is for engineering validation, not clinical use.
