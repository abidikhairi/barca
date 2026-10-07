"""Tests for CLI overrides in barca.train."""

from __future__ import annotations

from types import SimpleNamespace

from barca.config import Config
from barca.train import _coerce, apply_overrides


def _args(**overrides):
    base = dict(
        output_dir=None,
        run_name=None,
        seed=None,
        no_wandb=False,
        max_steps=None,
        num_train_epochs=None,
        per_device_train_batch_size=None,
        gradient_accumulation_steps=None,
        learning_rate=None,
        max_train_samples=None,
        max_eval_samples=None,
        set=[],
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_coerce_types():
    assert _coerce("true") is True
    assert _coerce("false") is False
    assert _coerce("16") == 16
    assert _coerce("1e-3") == 1e-3
    assert _coerce("adamw_torch") == "adamw_torch"


def test_apply_overrides_training_and_data():
    cfg = Config.from_dict({"data": {"path": "/x"}})
    apply_overrides(
        cfg,
        _args(
            max_steps=5,
            per_device_train_batch_size=64,
            learning_rate=2e-3,
            max_train_samples=2000,
            max_eval_samples=100,
            set=["gradient_checkpointing=false", "weight_decay=0.0"],
        ),
    )
    assert cfg.training["max_steps"] == 5
    assert cfg.training["per_device_train_batch_size"] == 64
    assert cfg.training["learning_rate"] == 2e-3
    assert cfg.training["gradient_checkpointing"] is False
    assert cfg.training["weight_decay"] == 0.0
    assert cfg.data.max_train_samples == 2000
    assert cfg.data.max_eval_samples == 100


def test_no_wandb_flag():
    cfg = Config.from_dict({"data": {"path": "/x"}})
    apply_overrides(cfg, _args(no_wandb=True))
    assert cfg.wandb.enabled is False
