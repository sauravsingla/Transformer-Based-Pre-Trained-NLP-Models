"""Dataset loading, cleaning and tokenization."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

LABELS = [
    "Extremely Negative",
    "Negative",
    "Neutral",
    "Positive",
    "Extremely Positive",
]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}

_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_USER = re.compile(r"@\w+")
_SPACE = re.compile(r"\s+")


def clean_tweet(value: object) -> str:
    """Apply conservative preprocessing while retaining sentiment-bearing text."""
    text = "" if pd.isna(value) else str(value)
    text = _URL.sub(" ", text)
    text = _USER.sub(" ", text)
    return _SPACE.sub(" ", text).strip()


def read_corona_file(path: str | Path, require_labels: bool = True) -> pd.DataFrame:
    """Read a CSV/XLSX Corona NLP file and return normalized text and labels."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Dataset file not found: {source}")
    if source.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(source)
    elif source.suffix.lower() == ".csv":
        frame = pd.read_csv(source, encoding="latin-1")
    else:
        raise ValueError("Dataset must be a CSV or Excel file.")

    text_column = next(
        (name for name in ("OriginalTweet", "text", "Text") if name in frame.columns), None
    )
    if text_column is None:
        raise ValueError("Missing tweet column; expected OriginalTweet, text, or Text.")

    result = pd.DataFrame({"text": frame[text_column].map(clean_tweet)})
    if require_labels:
        if "Sentiment" not in frame.columns:
            raise ValueError("Missing required Sentiment column.")
        unknown = sorted(set(frame["Sentiment"].dropna()) - set(LABELS))
        if unknown:
            raise ValueError(f"Unknown sentiment labels: {unknown}")
        result["label"] = frame["Sentiment"].map(LABEL_TO_ID)
        if result["label"].isna().any():
            raise ValueError("Sentiment contains missing values.")
        result["label"] = result["label"].astype(int)

    if result["text"].eq("").any():
        result = result.loc[result["text"].ne("")].reset_index(drop=True)
    return result


class TweetDataset(Dataset):
    """Tokenized tweet dataset compatible with Hugging Face Trainer."""

    def __init__(self, frame: pd.DataFrame, tokenizer: object, max_length: int) -> None:
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.frame.iloc[index]
        encoded = self.tokenizer(
            row["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        if "label" in row:
            item["labels"] = torch.tensor(int(row["label"]), dtype=torch.long)
        return item
