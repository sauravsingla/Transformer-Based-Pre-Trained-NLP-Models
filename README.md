# Comparative Analysis of Transformer-Based Pre-Trained NLP Models

[![CI](https://github.com/sauravsingla/Transformer-Based-Pre-Trained-NLP-Models/actions/workflows/ci.yml/badge.svg)](https://github.com/sauravsingla/Transformer-Based-Pre-Trained-NLP-Models/actions/workflows/ci.yml)
[![DOI](https://img.shields.io/badge/DOI-10.26438%2Fijcse%2Fv8i11.4044-blue)](https://doi.org/10.26438/ijcse/v8i11.4044)
[![License: MIT](https://img.shields.io/badge/Code-MIT-yellow.svg)](LICENSE)

Reproducible companion repository for the paper:

> S. Singla and N. Ramachandra, **“Comparative Analysis of Transformer Based Pre-Trained NLP Models,”** *International Journal of Computer Sciences and Engineering*, vol. 8, no. 11, pp. 40–44, Nov. 2020. DOI: [10.26438/ijcse/v8i11.4044](https://doi.org/10.26438/ijcse/v8i11.4044).

## Study objective

The study compares BERT, RoBERTa, and ALBERT for five-class sentiment classification on the Corona NLP tweets dataset. The repository preserves the original experiment notebooks and adds a clean, testable training pipeline that follows the paper’s experimental setup.

## Dataset

Use the Kaggle **Corona NLP Text Classification** dataset and place the files below in `data/raw/`:

```text
data/raw/Corona_NLP_train.csv
data/raw/Corona_NLP_test.csv
```

The paper reports:

- 41,157 training tweets
- 3,798 test tweets
- five labels: `Extremely Negative`, `Negative`, `Neutral`, `Positive`, and `Extremely Positive`

The dataset is not redistributed in this repository. Review and comply with its original license and terms before use.

## Paper-aligned experiment configuration

| Model | Hugging Face checkpoint | Learning rate | Dropout | Batch size | Max length | Epochs | Reported F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| BERT | `bert-base-uncased` | 2e-5 | 0.35 | 8 | 120 | 5 | 0.85 |
| RoBERTa | `roberta-base` | 2e-5 | 0.32 | 32 | 120 | 5 | 0.80 |
| ALBERT | `albert-base-v2` | 2e-5 | 0.35 | 8 | 120 | 5 | 0.78 |

The classifier mirrors the paper’s proposed architecture: transformer representation → dropout → hidden layer → fully connected output layer. Training uses AdamW, linear warm-up, early model selection on validation macro-F1, and deterministic seeds where supported.

## Repository structure

```text
.
├── bert_model.ipynb             # original BERT experiment
├── Roberta_model_new.ipynb      # original RoBERTa experiment
├── albert_model.ipynb           # original ALBERT experiment
├── configs/                     # paper-aligned YAML configurations
├── src/transformer_sentiment/   # reusable training and evaluation code
├── tests/                       # fast unit tests
└── .github/workflows/ci.yml     # lint and test automation
```

## Installation

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For development checks:

```bash
pip install -r requirements-dev.txt
```

## Run an experiment

```bash
python -m transformer_sentiment.train \
  --config configs/bert.yaml \
  --train-file data/raw/Corona_NLP_train.csv \
  --test-file data/raw/Corona_NLP_test.csv \
  --output-dir outputs/bert
```

Replace `configs/bert.yaml` with `configs/roberta.yaml` or `configs/albert.yaml` to reproduce the corresponding comparison.

The command writes:

- `metrics.json` with accuracy, macro/weighted precision, recall and F1, and one-vs-rest ROC-AUC when defined
- `classification_report.json`
- `predictions.csv`
- the best model/tokenizer checkpoint

## Reproducibility notes

The published scores are reference results, not guaranteed values. Exact results can vary with library versions, hardware, random seeds, dataset file version, and GPU nondeterminism. The clean pipeline records the resolved configuration, software versions, label mapping, and random seed in the output directory.

## Validate the repository

```bash
ruff check src tests
pytest -q
```

## Citation

```bibtex
@article{singla2020comparative,
  title={Comparative Analysis of Transformer Based Pre-Trained NLP Models},
  author={Singla, Saurav and Ramachandra, N.},
  journal={International Journal of Computer Sciences and Engineering},
  volume={8},
  number={11},
  pages={40--44},
  year={2020},
  doi={10.26438/ijcse/v8i11.4044}
}
```

## License

The source code in this repository is released under the MIT License. The paper, dataset, model weights, and third-party libraries remain subject to their respective licenses.