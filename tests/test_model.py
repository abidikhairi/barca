"""Tests for selective embedding training (new-token fine-tuning)."""

from __future__ import annotations

import torch
import torch.nn as nn

from barca.config import Config
from barca.model import freeze_for_token_training, resolve_trainable_token_ids


class TinyLM(nn.Module):
    """Minimal module exposing the embedding API freeze_for_token_training uses."""

    def __init__(self, vocab: int = 20, hidden: int = 4, tied: bool = True):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        if tied:
            self.lm_head.weight = self.embed.weight
        self.proj = nn.Linear(hidden, hidden)

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, idx):
        return self.lm_head(self.proj(self.embed(idx)))


class FakeTokenizer:
    unk_token_id = 0

    def __init__(self):
        self._vocab = {"<pad>": 0, "A": 3, "B": 5, "C": 7}

    def convert_tokens_to_ids(self, token):
        return self._vocab.get(token, self.unk_token_id)


def test_resolve_from_token_strings_and_range():
    cfg = Config.from_dict(
        {
            "data": {"path": "/x"},
            "model": {"trainable_tokens": ["A", "C"], "trainable_token_ids": [5]},
        }
    )
    assert resolve_trainable_token_ids(cfg.model, FakeTokenizer()) == [3, 5, 7]

    cfg2 = Config.from_dict(
        {"data": {"path": "/x"}, "model": {"trainable_token_range": [3, 7]}}
    )
    assert resolve_trainable_token_ids(cfg2.model, FakeTokenizer()) == [3, 4, 5, 6, 7]


def test_unknown_token_raises():
    cfg = Config.from_dict({"data": {"path": "/x"}, "model": {"trainable_tokens": ["Z"]}})
    try:
        resolve_trainable_token_ids(cfg.model, FakeTokenizer())
    except ValueError as exc:
        assert "not in the tokenizer" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_freeze_keeps_only_selected_rows_trainable():
    model = TinyLM(vocab=20, hidden=4, tied=True)
    freeze_for_token_training(model, [3, 5, 7])

    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"embed.weight"}  # tied lm_head shares the same parameter

    logits = model(torch.tensor([[3, 4, 5, 6]])).sum()
    logits.backward()
    grad = model.embed.weight.grad
    assert grad is not None
    # Selected rows get gradients; everything else is masked to zero.
    for row in (3, 5, 7):
        assert grad[row].abs().sum() > 0
    for row in (0, 1, 2, 4, 6, 8, 19):
        assert grad[row].abs().sum() == 0


def test_untied_lm_head_rows_are_trainable():
    model = TinyLM(vocab=20, hidden=4, tied=False)
    freeze_for_token_training(model, [2, 4])
    assert model.embed.weight.requires_grad
    assert model.lm_head.weight.requires_grad
    assert not model.proj.weight.requires_grad

    model(torch.tensor([[2, 3, 4]])).sum().backward()
    assert model.embed.weight.grad[2].abs().sum() > 0
    assert model.embed.weight.grad[0].abs().sum() == 0
    assert model.lm_head.weight.grad[4].abs().sum() > 0
    assert model.lm_head.weight.grad[1].abs().sum() == 0
