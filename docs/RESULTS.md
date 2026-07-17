# Results and Benchmark Record

This document separates values published in the paper from metrics produced by new runs of this repository.

## Published reference results

| Model | Published F1 | Source |
|---|---:|---|
| BERT | 0.85 | Singla and Ramachandra (2020) |
| RoBERTa | 0.80 | Singla and Ramachandra (2020) |
| ALBERT | 0.78 | Singla and Ramachandra (2020) |

These are reference values from the paper. They are not automatically reproduced by installing or cloning the repository.

## Reproduced-results leaderboard

Populate this table only from completed runs with retained artifacts.

| Model | Seed | Accuracy | Macro-F1 | Weighted-F1 | Macro ROC-AUC | Hardware | Commit | Status |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| BERT | — | — | — | — | — | — | — | Awaiting verified run |
| RoBERTa | — | — | — | — | — | — | — | Awaiting verified run |
| ALBERT | — | — | — | — | — | — | — | Awaiting verified run |

## Multi-seed robustness table

| Model | Seeds | Macro-F1 mean | Macro-F1 std. dev. | Accuracy mean | Training time mean | Notes |
|---|---|---:|---:|---:|---:|---|
| BERT | — | — | — | — | — | Not yet evaluated |
| RoBERTa | — | — | — | — | — | Not yet evaluated |
| ALBERT | — | — | — | — | — | Not yet evaluated |

## Efficiency comparison

| Model | Parameters | Peak GPU memory | Training time | Inference samples/sec | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| BERT | — | — | — | — | — |
| RoBERTa | — | — | — | — | — |
| ALBERT | — | — | — | — | — |

## Required run record

For every inserted result, include:

```text
Model:
Run date:
Result category: reproduced or extended
Git commit:
Configuration:
Seed:
Train file SHA-256:
Test file SHA-256:
Python:
PyTorch:
Transformers:
CUDA/cuDNN:
GPU:
Output artifact location:
Known deviations from the paper:
```

## Interpretation guidance

- Macro-F1 should drive the main model comparison.
- Accuracy alone can hide poor performance on minority classes.
- Differences smaller than run-to-run variation should not be overinterpreted.
- Efficiency and parameter count matter alongside predictive quality.
- Results obtained after changing data, preprocessing, architecture, or training settings must be labelled extended.
