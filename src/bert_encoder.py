"""BERT text encoder for music context strings.

Wraps a HuggingFace encoder and exposes both the pooled CLS vector (used by the
Task 1 classifier and the contrastive dual encoder) and the full token sequence
(needed as keys/values for Task 3 cross-attention).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer


class BertTextEncoder(nn.Module):
    """Contextual text encoder with configurable partial freezing.

    Parameters
    ----------
    model_name : HuggingFace checkpoint, e.g. ``bert-base-uncased``.
    freeze : freeze the entire encoder (only the downstream head trains).
    unfreeze_last_n : when not fully frozen, keep only the last N transformer
        layers trainable. Fine-tuning all 12 layers of BERT-base alongside a GNN
        overfits MTAT's fairly repetitive metadata strings and costs GPU memory,
        so the default config unfreezes 4.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        freeze: bool = False,
        unfreeze_last_n: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size: int = self.config.hidden_size
        self.dropout = nn.Dropout(dropout)

        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False
        elif unfreeze_last_n is not None and unfreeze_last_n >= 0:
            self._unfreeze_last_n(unfreeze_last_n)

    def _unfreeze_last_n(self, n: int) -> None:
        for param in self.bert.parameters():
            param.requires_grad = False
        layers = self._transformer_layers()
        for layer in layers[len(layers) - n :] if n else []:
            for param in layer.parameters():
                param.requires_grad = True
        # The pooler is randomly initialised for some checkpoints; always train it.
        if getattr(self.bert, "pooler", None) is not None:
            for param in self.bert.pooler.parameters():
                param.requires_grad = True

    def _transformer_layers(self) -> nn.ModuleList:
        """Locate the encoder layer stack across BERT/DistilBERT naming."""
        if hasattr(self.bert, "encoder") and hasattr(self.bert.encoder, "layer"):
            return self.bert.encoder.layer          # BERT, RoBERTa
        if hasattr(self.bert, "transformer") and hasattr(self.bert.transformer, "layer"):
            return self.bert.transformer.layer      # DistilBERT
        raise AttributeError(f"Unrecognised encoder layout for {self.model_name}")

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (cls_vector (B, H), token_states (B, L, H))."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_states = outputs.last_hidden_state
        # Mean-pool over real tokens rather than using the raw [CLS] slot: without
        # a next-sentence-prediction objective the CLS vector of an off-the-shelf
        # checkpoint is a weak sentence summary, and masked mean pooling is a
        # consistently stronger sentence representation.
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (token_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)
        return self.dropout(pooled), token_states


class BertTagClassifier(nn.Module):
    """Task 1: BERT + linear head for multi-label tag prediction.

    Outputs raw logits; the training loop applies BCEWithLogitsLoss for
    numerical stability rather than sigmoid + BCE.
    """

    def __init__(
        self,
        num_tags: int,
        model_name: str = "bert-base-uncased",
        freeze: bool = False,
        unfreeze_last_n: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = BertTextEncoder(model_name, freeze, unfreeze_last_n, dropout)
        self.head = nn.Linear(self.encoder.hidden_size, num_tags)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        pooled, _ = self.encoder(input_ids, attention_mask)
        return self.head(pooled)


def build_tokenizer(model_name: str = "bert-base-uncased"):
    return AutoTokenizer.from_pretrained(model_name)


def build_context_string(
    title: str | None,
    artist: str | None,
    album: str | None,
    auxiliary_tags: list[str] | None = None,
    use_metadata: bool = True,
    use_auxiliary_tags: bool = True,
) -> str:
    """Compose the natural-language music-context string fed to BERT.

    MagnaTagATune ships no lyrics or captions, and its tags are the prediction
    target, so using tags as input would make Task 1 circular. The text modality
    is therefore built from (a) catalogue metadata and (b) the long-tail
    annotations that fall *outside* the top-K label set. Both are disjoint from
    the supervision signal.
    """
    parts: list[str] = []
    if use_metadata:
        clean = lambda value: str(value).strip() if value and str(value).strip().lower() != "nan" else ""
        title_s, artist_s, album_s = clean(title), clean(artist), clean(album)
        if title_s:
            sentence = f"The track is titled '{title_s}'"
            if artist_s:
                sentence += f" by {artist_s}"
            if album_s:
                sentence += f", from the album '{album_s}'"
            parts.append(sentence + ".")
        elif artist_s:
            parts.append(f"A track by {artist_s}.")

    if use_auxiliary_tags and auxiliary_tags:
        # Deterministic order keeps the input reproducible across runs.
        listed = ", ".join(sorted(auxiliary_tags))
        parts.append(f"Listeners also described it as: {listed}.")

    return " ".join(parts) if parts else "An untitled instrumental music clip."
