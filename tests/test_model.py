from __future__ import annotations

import torch
from transformers import BertConfig

from transformer_sentiment.model import PaperTransformerClassifier


def test_custom_head_configuration_survives_save_and_reload(tmp_path) -> None:
    config = BertConfig(
        vocab_size=101,
        hidden_size=24,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
        num_labels=5,
        classifier_hidden_size=13,
        classifier_dropout=0.27,
    )
    model = PaperTransformerClassifier(config)
    model.save_pretrained(tmp_path)

    reloaded = PaperTransformerClassifier.from_pretrained(tmp_path)

    assert reloaded.hidden.in_features == 24
    assert reloaded.hidden.out_features == 13
    assert reloaded.classifier.out_features == 5
    assert reloaded.dropout.p == 0.27


def test_forward_returns_five_class_logits_and_loss() -> None:
    config = BertConfig(
        vocab_size=101,
        hidden_size=24,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
        num_labels=5,
        classifier_hidden_size=13,
        classifier_dropout=0.27,
    )
    model = PaperTransformerClassifier(config)
    input_ids = torch.randint(0, config.vocab_size, (3, 8))
    attention_mask = torch.ones_like(input_ids)
    labels = torch.tensor([0, 2, 4])

    output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    assert output.logits.shape == (3, 5)
    assert output.loss is not None
    assert torch.isfinite(output.loss)
