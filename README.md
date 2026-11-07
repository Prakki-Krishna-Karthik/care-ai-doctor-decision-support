# Reliability-Aware Multimodal Chest X-ray Decision Support

This is a research/prototype decision-support application for MIMIC-CXR-JPG images and report-derived clinical labels. It is not a diagnostic device and must not be used as a substitute for clinician judgment.

## What is implemented

- Image and clinical-data branches with explicit data-quality scores.
- Reliability-aware fusion: the less reliable modality receives less influence.
- Per-finding uncertainty and an abstention/review state.
- Image quality indicators, view-position context, real Grad-CAM image explanations when DenseNet is active, and evidence cards.
- FastAPI backend and a no-build browser frontend.
- Optional lightweight training utility for the downloaded subset.
- Patient-level comparative study across DenseNet-121, EfficientNet-B0, ConvNeXt-Tiny, elastic-net logistic regression, random forest, and histogram gradient boosting.
- Research artifacts: AUROC/AUPRC/F1/Brier/ECE tables, per-label heatmaps, confusion heatmaps, calibration plots, feature-importance plots, compute trade-off plots, cohort/label distributions, and training curves.
- The API exposes `/api/comparison` and serves the figures under `/artifacts/comparison/`.
- A readiness gate audits validation–test gaps and sparse label support; current results are intentionally marked research-only until repeated patient-level validation is completed.
- Runtime data integrity: recursive image discovery collapses duplicate filenames to one record, prefers the shallow direct image, and preserves patient/study identity from a nested path when available.
- The API exposes the raw-versus-unique image count and the duplicate IDs excluded from training and case browsing.
- Confidence is support-aware: fused probabilities are shrunk toward 0.5 when image quality or clinical support is weak, and the case is routed to review when the support threshold is not met.
- Local evidence retrieval is now report-aware: when a permitted study ID or report is supplied, the app retrieves MIMIC-CXR report chunks with TF-IDF and compares mentioned/negated findings with model signals. No patient data leaves the local machine.
- Prediction stability is estimated with three anatomy-preserving image views (original, contrast-adjusted, and brightness-adjusted); the spread is surfaced as an uncertainty indicator and review priority.
- Clinician-facing guidance translates model output into a plain-language summary, urgency flag, independent review checklist, and possible tests to discuss based on the supplied context.
- Treatment safety is explicit: the application never generates a medication, dose, or prescription. It asks the clinician to confirm the diagnosis and follow local protocols.
- The API exposes a view-position cohort audit at `/api/fairness` and a local `/api/feedback` endpoint for optional clinician-review feedback. MIMIC-CXR metadata does not include age or sex, so those demographic audits are reported as unavailable rather than fabricated.

## Run locally

From this directory in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r .\backend\requirements.txt
$env:MIMIC_IMAGE_ROOT = 'D:\MIMIC\jpg'
$env:MIMIC_DATA_ROOT = 'C:\Users\Krishna Karthik\Downloads'
python .\backend\app.py
```

Open `http://127.0.0.1:8000` in a browser. The backend serves the frontend as well as `/api/*` endpoints.

The main readout is intentionally written for clinician review rather than
machine-learning terminology. It separates four things:

1. What the model noticed in plain language.
2. What the clinician should verify independently.
3. Possible investigations to discuss when the entered symptoms or risk flags make them relevant. These are prompts, not automatic orders.
4. A medication safety block that explicitly prevents prescribing from the AI result alone.

For TB-related context, the checklist points to a complete clinical evaluation
and appropriate microbiologic testing rather than presenting the X-ray as a
confirmation. Local and national protocols always take precedence.

## Research framing

The differentiating process is quality-gated fusion with abstention: image quality and clinical completeness are measured before combining modalities. The application also preserves the reasons for a low-confidence result, so a clinician sees what additional information would improve the estimate.

## Comparative training and model selection

Run the comparison after the local image and metadata files are available:

```powershell
python .\backend\comparative_study.py --batch-size 32
python .\backend\model_readiness.py
```

The study uses patient-level splits and treats uncertain or missing labels as unknown. It compares image-only, structured-data-only, simple averaging, and reliability-aware fusion. The current local cohort selects the fine-tuned DenseNet-121 image branch. The structured acquisition-metadata branch is trained and retained for audit, but its held-out discrimination is close to chance, so the API does not allow it to degrade the image probability until richer clinical variables are available. This is an evidence-based deployment choice, not a claim of clinical validity.

The main files are:

- `backend/artifacts/comparative_results.json` — machine-readable comparison and selected configuration.
- `backend/artifacts/comparative_analysis_report.md` — paper-ready summary table.
- `backend/artifacts/comparison/` — generated figures.
- `backend/artifacts/clinical_model.joblib` — selected structured-data candidate.
- `backend/rag.py` — local TF-IDF retrieval, report lookup, and report–prediction comparison.

## Regularized training and safety gate

The regularized training path is available for a separate candidate checkpoint:

```powershell
python -u .\backend\deep_train.py --regularized --epochs 4 --batch-size 64 --output .\backend\artifacts\densenet121_regularized.pt
python .\backend\model_readiness.py
```

It freezes the convolutional backbone, avoids laterality-destroying flips, applies conservative augmentation, clipped class weighting, label smoothing, gradient clipping, and validation-loss early stopping. The API does not automatically deploy a candidate merely because it trained; the readiness audit must first pass the validation–test gap and label-support checks.

The regularized candidate also uses a dropout classification head. The inference loader supports both the original linear-head checkpoint and the regularized dropout-head checkpoint, so an existing saved model remains compatible.

## DenseNet-121 training

The downloaded ImageNet-initialized DenseNet-121 branch can be trained with:

```powershell
python .\backend\deep_train.py --max-images 1242 --epochs 2 --batch-size 32
```

When `backend/artifacts/densenet121.pt` exists, the API automatically prefers it over the lightweight baseline. The DenseNet metrics are written to `backend/artifacts/densenet_metrics.json`.

## Data handling

The app reads local files only. It never uploads MIMIC data to a third-party service. MIMIC-CXR data remain subject to the PhysioNet data-use agreement.
