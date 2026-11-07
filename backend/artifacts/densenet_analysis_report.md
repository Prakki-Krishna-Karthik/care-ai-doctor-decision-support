# DenseNet-121 ML Analysis

This is a preliminary research baseline report, not a clinical validation report.

## Training configuration

- Architecture: ImageNet-initialized DenseNet-121
- Fine-tuned components: classifier and final dense block
- Device: CUDA
- Patient-level split: 789 train / 239 validation / 237 test
- Label-matched images used: 1,265
- Temperature scaling: fitted on the validation split
- Threshold selection: validation F1 grid, then fixed on the test split

## Held-out test macro averages

| Metric | Value |
|---|---:|
| Accuracy | 0.8413 |
| Precision | 0.7599 |
| Sensitivity | 0.8026 |
| Specificity | 0.5321 |
| F1-score | 0.7770 |
| Balanced accuracy | 0.7067 |
| AUROC | 0.8471 |
| AUPRC | 0.8765 |
| Brier score | 0.1487 |
| ECE, 10 bins | 0.1843 |

These results are preliminary because the downloaded subset is small and incomplete compared with the full MIMIC-CXR-JPG collection. The expanded download changed the patient-level split, so the metrics should be compared with the earlier run only as a sensitivity check, not as a direct benchmark. The model should be retrained after the download is complete and evaluated on an untouched external dataset before any clinical interpretation.
