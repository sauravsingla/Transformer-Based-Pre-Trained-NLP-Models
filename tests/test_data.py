from pathlib import Path

import pandas as pd
import pytest

from transformer_sentiment.data import LABEL_TO_ID, clean_tweet, read_corona_file


def test_clean_tweet_removes_urls_mentions_and_extra_spaces() -> None:
    value = "  @person Great service! https://example.com   Thanks  "
    assert clean_tweet(value) == "Great service! Thanks"


def test_read_corona_csv_maps_all_paper_labels(tmp_path: Path) -> None:
    source = tmp_path / "tweets.csv"
    pd.DataFrame(
        {
            "OriginalTweet": ["a", "b", "c", "d", "e"],
            "Sentiment": list(LABEL_TO_ID),
        }
    ).to_csv(source, index=False)
    result = read_corona_file(source)
    assert result.columns.tolist() == ["text", "label"]
    assert result["label"].tolist() == list(range(5))


def test_read_corona_file_rejects_unknown_label(tmp_path: Path) -> None:
    source = tmp_path / "tweets.csv"
    pd.DataFrame({"OriginalTweet": ["text"], "Sentiment": ["Mixed"]}).to_csv(
        source, index=False
    )
    with pytest.raises(ValueError, match="Unknown sentiment labels"):
        read_corona_file(source)
