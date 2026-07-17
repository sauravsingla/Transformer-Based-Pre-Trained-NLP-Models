"""Command-line training entry point for the paper-aligned experiments."""

from __future__ import annotations

import argparse
import json
import platform
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
import yaml
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, Trainer, TrainingArguments

from .data import ID_TO_LABEL, LABELS, LABEL_TO_ID, TweetDataset, read_corona_file
from .metrics import classification_report_dict, compute_classification_metrics
from .model import PaperTransformerClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--train-file", required=True, type=Path)
    parser.add_argument("--test-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = {
        "model_name",
        "learning_rate",
        "dropout",
        "batch_size",
        "max_length",
        "epochs",
        "validation_size",
        "seed",
        "hidden_size",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"Missing config keys: {sorted(missing)}")
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config["seed"])
    set_seed(seed)

    full_train = read_corona_file(args.train_file)
    test_frame = read_corona_file(args.test_file)
    train_frame, validation_frame = train_test_split(
        full_train,
        test_size=float(config["validation_size"]),
        random_state=seed,
        stratify=full_train["label"],
    )

    tokenizer = AutoTokenizer.from_pretrained(str(config["model_name"]), use_fast=True)
    model = PaperTransformerClassifier.from_pretrained_with_head(
        str(config["model_name"]),
        num_labels=len(LABELS),
        hidden_size=int(config["hidden_size"]),
        dropout=float(config["dropout"]),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )
    train_dataset = TweetDataset(train_frame, tokenizer, int(config["max_length"]))
    validation_dataset = TweetDataset(validation_frame, tokenizer, int(config["max_length"]))
    test_dataset = TweetDataset(test_frame, tokenizer, int(config["max_length"]))

    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        learning_rate=float(config["learning_rate"]),
        per_device_train_batch_size=int(config["batch_size"]),
        per_device_eval_batch_size=int(config["batch_size"]),
        num_train_epochs=float(config["epochs"]),
        weight_decay=float(config.get("weight_decay", 0.01)),
        warmup_ratio=float(config.get("warmup_ratio", 0.1)),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=50,
        report_to=[],
        seed=seed,
        data_seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=lambda prediction: compute_classification_metrics(
            prediction.label_ids, prediction.predictions, len(LABELS)
        ),
    )
    trainer.train()
    prediction = trainer.predict(test_dataset)
    predicted_labels = prediction.predictions.argmax(axis=1)
    metrics = compute_classification_metrics(
        prediction.label_ids, prediction.predictions, len(LABELS)
    )
    report = classification_report_dict(prediction.label_ids, predicted_labels, LABELS)

    trainer.save_model(str(args.output_dir / "model"))
    tokenizer.save_pretrained(str(args.output_dir / "model"))
    pd.DataFrame(
        {
            "text": test_frame["text"],
            "actual": [ID_TO_LABEL[int(value)] for value in prediction.label_ids],
            "predicted": [ID_TO_LABEL[int(value)] for value in predicted_labels],
        }
    ).to_csv(args.output_dir / "predictions.csv", index=False)
    _write_json(args.output_dir / "metrics.json", metrics)
    _write_json(args.output_dir / "classification_report.json", report)
    _write_json(
        args.output_dir / "run_metadata.json",
        {
            "config": config,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_available": torch.cuda.is_available(),
            "labels": LABEL_TO_ID,
        },
    )


def _write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=True)


if __name__ == "__main__":
    main()
