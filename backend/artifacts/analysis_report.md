# MIMIC-CXR Baseline ML Analysis

This report describes the locally trained research baseline. It is not a clinical validation report.

## Dataset and split

- Images discovered: 1242
- Label-matched images used: 1216
- Unique patient groups: 685
- Patient-level split: 722 train / 208 validation / 286 test

## Held-out test macro averages

| Metric | Value |
|---|---:|
| accuracy | 0.7646 |
| precision | 0.7001 |
| recall_sensitivity | 0.8633 |
| specificity | 0.6144 |
| f1 | 0.7330 |
| balanced_accuracy | 0.7883 |
| auroc | 0.7087 |
| auprc | 0.7008 |
| brier_score | 0.1847 |
| ece_10bin | 0.2391 |

## Interpretation

Thresholds were selected on the validation split by maximizing F1 over a fixed candidate grid, then applied unchanged to the held-out test split. AUROC and AUPRC use continuous probabilities. Specificity is TN/(TN+FP), while balanced accuracy is the mean of sensitivity and specificity. Brier score and ECE summarize probability quality, not only class decisions.

The baseline uses image-quality and structured clinical features with one-vs-rest logistic regression. It is intended for pipeline verification and comparison against the future DenseNet-121 image branch; it must not be interpreted as a clinical performance claim.
