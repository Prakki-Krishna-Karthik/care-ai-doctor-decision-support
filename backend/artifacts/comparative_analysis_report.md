# Comparative model study

This study uses patient-level splits and treats missing/uncertain labels as unknown rather than definite negatives.

- Images discovered: 1295
- Usable label-matched images: 1271
- Patient groups: 739
- Split: 814 train / 216 validation / 241 test
- Structured branch: acquisition metadata only; labels and report text excluded to avoid leakage.

## Selected configuration

- Image model: `densenet121_finetuned_current`
- Clinical model: `random_forest`
- Fusion: `image_only`

## Candidate comparison

| Model | AUROC | AUPRC | F1 | Brier | ECE | Params M | Time s |
|---|---:|---:|---:|---:|---:|---:|---:|
| densenet121_frozen | 0.9675 | 0.9600 | 0.9208 | 0.0472 | 0.0649 | 7.994 | 0.82 |
| efficientnet_b0_frozen | 0.9266 | 0.9327 | 0.8902 | 0.0665 | 0.0848 | 5.308 | 0.82 |
| convnext_tiny_frozen | 0.9207 | 0.9238 | 0.8926 | 0.0753 | 0.0890 | 28.601 | 0.76 |
| densenet121_finetuned_current | 0.9443 | 0.9630 | 0.9258 | 0.0689 | 0.1132 | 6.968 | 27.78 |
| elastic_net_logistic | 0.4861 | 0.6699 | 0.7011 | 0.2480 | 0.2657 | 0.029 | 1.09 |
| random_forest | 0.5167 | 0.6759 | 0.7639 | 0.2016 | 0.2174 | 2.882 | 6.85 |
| hist_gradient_boosting | 0.5141 | 0.6654 | 0.7233 | 0.1439 | 0.0756 | 0.676 | 3.37 |
| image_only | 0.9443 | 0.9630 | 0.9143 | 0.0689 | 0.1132 | 0.000 | 0.00 |
| clinical_only | 0.5167 | 0.6759 | 0.7639 | 0.2016 | 0.2174 | 0.000 | 0.00 |
| simple_average | 0.8085 | 0.9107 | 0.8867 | 0.1044 | 0.1834 | 0.000 | 0.00 |
| reliability_aware_fusion | 0.8786 | 0.9256 | 0.8685 | 0.0816 | 0.1117 | 0.000 | 0.00 |

## Figures

The comparison directory contains performance bars, AUROC heatmaps, confusion heatmaps, calibration plots, feature importance, performance-versus-compute, cohort split, label distribution, and training curves.

These results are a research comparison on a small local subset and are not clinical validation.