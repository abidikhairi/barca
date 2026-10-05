"""Tests for recipe parsing (no heavy dependencies beyond PyYAML)."""

from __future__ import annotations

import textwrap

import pytest

from barca.config import Config, DataConfig


def test_load_recipe_and_nested_quant(tmp_path):
    path = tmp_path / "recipe.yaml"
    path.write_text(
        textwrap.dedent(
            """
            run:
              name: unit-test
              output_dir: outputs/unit-test
              seed: 7
            data:
              path: /data/proteins
              text_column: seq
              max_seq_length: 128
            tokenizer:
              name_or_path: khairi/barca-0.8b-base
            model:
              dtype: float32
            peft:
              enabled: true
              method: qlora
              r: 8
              quant:
                bnb_4bit_quant_type: fp4
            training:
              learning_rate: 1.0e-4
              num_train_epochs: 2
            checkpointing:
              top_k: 3
            wandb:
              enabled: false
              project: barca-test
            """
        ),
        encoding="utf-8",
    )

    cfg = Config.from_yaml(path)
    assert cfg.run.name == "unit-test"
    assert cfg.run.seed == 7
    assert cfg.data.text_column == "seq"
    assert cfg.data.max_seq_length == 128
    assert cfg.model.dtype == "float32"
    assert cfg.peft.enabled is True
    assert cfg.peft.method == "qlora"
    assert cfg.peft.quant.bnb_4bit_quant_type == "fp4"
    assert cfg.training["learning_rate"] == 1.0e-4
    assert cfg.checkpointing.top_k == 3
    assert cfg.wandb.enabled is False
    assert cfg.source_path == str(path)


def test_unknown_top_level_key_raises(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("data: {path: /x}\ntrainingg: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown top-level config keys"):
        Config.from_yaml(path)


def test_unknown_nested_key_raises(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("data: {path: /x, bogus: 1}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown keys for DataConfig"):
        Config.from_yaml(path)


def test_data_requires_a_source():
    with pytest.raises(ValueError, match="data.path or data.tokenized_path"):
        DataConfig()


def test_qlora_requires_peft_enabled(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "data: {path: /x}\npeft: {enabled: false, method: qlora}\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="requires peft.enabled"):
        Config.from_yaml(path)


def test_roundtrip_yaml(tmp_path):
    cfg = Config.from_dict(
        {"data": {"path": "/data/proteins"}, "run": {"name": "roundtrip"}}
    )
    out = tmp_path / "out.yaml"
    cfg.to_yaml(out)
    reloaded = Config.from_yaml(out)
    assert reloaded.run.name == "roundtrip"
    assert reloaded.data.path == "/data/proteins"
