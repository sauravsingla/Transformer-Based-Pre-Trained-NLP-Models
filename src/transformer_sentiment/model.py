"""Transformer classifier matching the paper's proposed head."""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class PaperTransformerClassifier(PreTrainedModel):
    """Transformer encoder followed by dropout, hidden and output layers."""

    config_class = AutoConfig

    def __init__(self, config: AutoConfig, hidden_size: int = 256, dropout: float = 0.35):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.encoder = AutoModel.from_config(config)
        encoder_size = getattr(config, "hidden_size", None) or getattr(config, "dim")
        self.dropout = nn.Dropout(dropout)
        self.hidden = nn.Linear(encoder_size, hidden_size)
        self.activation = nn.ReLU()
        self.classifier = nn.Linear(hidden_size, config.num_labels)
        self.post_init()

    @classmethod
    def from_pretrained_with_head(
        cls,
        model_name: str,
        *,
        num_labels: int,
        hidden_size: int,
        dropout: float,
        id2label: dict[int, str],
        label2id: dict[str, int],
    ) -> "PaperTransformerClassifier":
        config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )
        model = cls(config, hidden_size=hidden_size, dropout=dropout)
        model.encoder = AutoModel.from_pretrained(model_name, config=config)
        return model

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: object,
    ) -> SequenceClassifierOutput:
        encoder_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "return_dict": True,
        }
        if token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids
        outputs = self.encoder(**encoder_kwargs)
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0]
        logits = self.classifier(self.activation(self.hidden(self.dropout(pooled))))
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
