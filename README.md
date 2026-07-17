# Comparative Analysis of Transformer-Based Pre-Trained NLP Models

[![CI](https://github.com/sauravsingla/Transformer-Based-Pre-Trained-NLP-Models/actions/workflows/ci.yml/badge.svg)](https://github.com/sauravsingla/Transformer-Based-Pre-Trained-NLP-Models/actions/workflows/ci.yml)
[![DOI](https://img.shields.io/badge/DOI-10.26438%2Fijcse%2Fv8i11.4044-blue)](https://doi.org/10.26438/ijcse/v8i11.4044)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/Code-MIT-yellow.svg)](LICENSE)

A reproducible research and teaching repository for comparing **BERT**, **RoBERTa**, and **ALBERT** on five-class COVID-19 tweet sentiment classification.

This repository accompanies:

> S. Singla and N. Ramachandra, **“Comparative Analysis of Transformer Based Pre-Trained NLP Models,”** *International Journal of Computer Sciences and Engineering*, vol. 8, no. 11, pp. 40–44, Nov. 2020. DOI: [10.26438/ijcse/v8i11.4044](https://doi.org/10.26438/ijcse/v8i11.4044).

## Why this repository exists

The original notebooks are preserved for historical fidelity. A modern package has been added so students, researchers, and practitioners can run the same comparison through a clean command-line workflow with validated inputs, reusable model code, deterministic settings, automated tests, and structured experiment outputs.

## What is included

- Original paper-era notebooks for BERT, RoBERTa, and ALBERT.
- Paper-aligned YAML configurations.
- A reusable PyTorch and Hugging Face training package.
- Five-class evaluation with accuracy, macro and weighted metrics, and multiclass ROC-AUC.
- Best-checkpoint selection using validation macro-F1.
- Prediction, classification-report, metric, and environment metadata exports.
- Unit tests, Ruff linting, and multi-version GitHub Actions CI.
- Reproducibility, model-card, contribution, security, and citation documentation.

## Study objective

The study compares three transformer families for sentiment classification on the **Corona NLP Text Classification** dataset:

1. **BERT** — bidirectional masked-language pre-training.
2. **RoBERTa** — a more aggressively trained BERT variant.
3. **ALBERT** — parameter sharing and factorized embeddings for improved parameter efficiency.

The target labels are:

```text
Extremely Negative
Negative
Neutral
Positive
Extremely Positive
```

## Published reference configuration

| Model | Checkpoint | Learning rate | Dropout | Batch size | Max length | Epochs | Published F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| BERT | `bert-base-uncased` | `2e-5` | `0.35` | `8` | `120` | `5` | `0.85` |
| RoBERTa | `roberta-base` | `2e-5` | `0.32` | `32` | `120` | `5` | `0.80` |
| ALBERT | `albert-base-v2` | `2e-5` | `0.35` | `8` | `120` | `5` | `0.78` |

The classifier head follows the paper’s intended structure:

```text
Transformer representation
        ↓
      Dropout
        ↓
   Dense hidden layer
        ↓
       ReLU
        ↓
 Five-class output layer
```

The published scores are included as reference values. They are not represented as newly reproduced results until fresh experiment artifacts are generated.

## Dataset

Download the Kaggle **Corona NLP Text Classification** dataset and place the files here:

```text
data/raw/Corona_NLP_train.csv
data/raw/Corona_NLP_test.csv
```

The paper-era files contain:

- 41,157 training tweets.
- 3,798 test tweets.
- Five sentiment classes.

The dataset is not redistributed. Users must follow the dataset’s original terms and licence.

## Repository structure

```text
.
├── bert_model.ipynb
├── Roberta_model_new.ipynb
├── albert_model.ipynb
├── configs/
│   ├── bert.yaml
│   ├── roberta.yaml
│   └── albert.yaml
├── docs/
│   ├── MODEL_CARD.md
│   ├── REPRODUCIBILITY.md
│   └── RESULTS.md
├── src/transformer_sentiment/
│   ├── data.py
│   ├── metrics.py
│   ├── model.py
│   └── train.py
├── tests/
├── CITATION.cff
├── CONTRIBUTING.md
├── SECURITY.md
└── .github/workflows/ci.yml
```

## Quick start

### 1. Clone and create an environment

```bash
git clone https://github.com/sauravsingla/Transformer-Based-Pre-Trained-NLP-Models.git
cd Transformer-Based-Pre-Trained-NLP-Models
python -m venv .venv
```

Activate it:

```bash
# Linux or macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

### 2. Install

```bash
python -m pip install --upgrade pip
pip install -e .
```

For development:

```bash
pip install -r requirements-dev.txt
```

### 3. Add the dataset

```text
data/raw/Corona_NLP_train.csv
data/raw/Corona_NLP_test.csv
```

### 4. Run BERT

```bash
transformer-sentiment \
  --config configs/bert.yaml \
  --train-file data/raw/Corona_NLP_train.csv \
  --test-file data/raw/Corona_NLP_test.csv \
  --output-dir outputs/bert
```

The module form is equivalent:

```bash
python -m transformer_sentiment.train \
  --config configs/bert.yaml \
  --train-file data/raw/Corona_NLP_train.csv \
  --test-file data/raw/Corona_NLP_test.csv \
  --output-dir outputs/bert
```

Run the other models by replacing the configuration with `configs/roberta.yaml` or `configs/albert.yaml`.

## Experiment outputs

Each run creates a self-contained output directory:

```text
outputs/bert/
├── model/
├── checkpoints/
├── metrics.json
├── classification_report.json
├── predictions.csv
└── run_metadata.json
```

- `metrics.json` contains accuracy, macro/weighted precision, recall, F1, and ROC-AUC when defined.
- `classification_report.json` contains per-class performance.
- `predictions.csv` contains text, actual label, predicted label, and prediction confidence.
- `run_metadata.json` records configuration, package versions, hardware availability, dataset sizes, and label mapping.
- `model/` contains the reloadable best model and tokenizer.

## Recommended comparison protocol

For a fair leaderboard:

1. Use the same train and test files for all models.
2. Do not tune on the test set.
3. Keep the seed and validation split fixed.
4. Report macro-F1 as the primary metric because the classes are not perfectly balanced.
5. Run multiple seeds when estimating robustness.
6. Record hardware, package versions, and elapsed training time.
7. Keep published values separate from reproduced values.

A results template is available in [`docs/RESULTS.md`](docs/RESULTS.md).

## Reproducibility boundaries

Exact numerical reproduction can be affected by:

- Dataset file revisions.
- GPU model and CUDA/cuDNN versions.
- PyTorch and Transformers versions.
- Random initialization and nondeterministic GPU operations.
- Details not fully specified in the original publication.

The repository therefore distinguishes:

- **Published reference results** — values reported in the paper.
- **Reproduced results** — values generated by a documented run of this code.
- **Extended results** — additional seeds, datasets, models, or ablations beyond the paper.

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the full protocol.

## Quality checks

```bash
ruff check src tests
pytest -q
python -m pip check
```

CI runs these checks on Python 3.10, 3.11, and 3.12.

## Responsible use

This repository is intended for research and education. COVID-19 tweets may contain sensitive, offensive, political, or personally identifying content. Model predictions can reflect dataset and language biases and should not be used as the sole basis for decisions about individuals.

See [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) for limitations and intended use.

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

GitHub citation metadata is also available through [`CITATION.cff`](CITATION.cff).

## Contributing

Contributions that improve reproducibility, documentation, tests, dataset adapters, benchmarking, or model coverage are welcome. Read [`CONTRIBUTING.md`](CONTRIBUTING.md) before opening a pull request.

## Licence

Repository source code is released under the [MIT License](LICENSE). The paper, dataset, pretrained model weights, and third-party libraries remain governed by their respective licences.
