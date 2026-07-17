# Model Card

## Model overview

This repository trains five-class sentiment classifiers using pretrained BERT, RoBERTa, and ALBERT encoders with a paper-aligned dense classification head.

## Intended use

The models are intended for:

- Reproducing and studying the accompanying paper.
- Teaching transformer fine-tuning and evaluation.
- Comparing pretrained language-model families.
- Prototyping sentiment-classification pipelines.
- Research on robustness, efficiency, and error analysis.

## Out-of-scope use

The models should not be used as the sole basis for:

- Employment, credit, insurance, healthcare, education, immigration, policing, or legal decisions.
- Mental-health diagnosis or risk assessment.
- Moderation decisions without human review.
- Claims about an individual’s beliefs, health, intent, or emotional state.
- Production deployment without domain-specific validation and monitoring.

## Training data

The experiments use the Corona NLP Text Classification dataset, containing English-language tweets collected during the COVID-19 period. The repository does not redistribute the dataset.

## Labels

- Extremely Negative
- Negative
- Neutral
- Positive
- Extremely Positive

## Architecture

Each model uses:

1. A pretrained transformer encoder.
2. A dropout layer.
3. A dense hidden layer.
4. ReLU activation.
5. A five-class output layer.

## Evaluation

The repository reports:

- Accuracy.
- Macro and weighted precision.
- Macro and weighted recall.
- Macro and weighted F1.
- Per-class classification metrics.
- Multiclass ROC-AUC when defined.

Macro-F1 is the preferred primary metric because it gives equal importance to each class.

## Limitations

- The data reflects a specific pandemic-era topic and time period.
- Tweets contain informal language, sarcasm, abbreviations, misspellings, URLs, and mentions.
- Sentiment labels may be subjective or noisy.
- The dataset may not represent all demographic groups, dialects, countries, topics, or platforms.
- Performance can degrade under domain shift.
- Confidence values are not guaranteed to be calibrated probabilities.
- Fine-tuned transformers can reproduce social and linguistic biases from their pretraining and task data.
- Five sentiment classes simplify emotions and opinions that may be mixed or context-dependent.

## Privacy and content considerations

Tweets may contain usernames, personal information, political opinions, health-related statements, or offensive content. The preprocessing removes URLs and user mentions from model text, but users remain responsible for secure and lawful dataset handling.

## Recommended deployment safeguards

A production adaptation should include:

- Domain-specific test data.
- Bias and subgroup evaluation.
- Human review for consequential decisions.
- Confidence calibration and abstention rules.
- Input drift and output-distribution monitoring.
- Error analysis by class and language pattern.
- Regular retraining and rollback procedures.
- Clear user-facing disclosures about limitations.

## Environmental considerations

Transformer fine-tuning consumes compute and energy. Reuse checkpoints, avoid unnecessary repeated runs, prefer mixed precision where validated, and record hardware and training duration for meaningful efficiency comparisons.

## Citation

Please cite the accompanying paper and this repository when using the implementation or experimental protocol.
