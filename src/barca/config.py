"""Typed configuration objects for Barca training recipes.

Recipes are YAML files (see ``configs/``). This module parses them into
dataclasses so that typos in structured sections fail loudly, while the
``training`` section is intentionally free-form: it is forwarded to
``transformers.TrainingArguments``.

Example
-------
>>> cfg = Config.from_yaml("configs/lora.yaml")
>>> cfg.model.name_or_path
'khairi/barca-0.8b-base'
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# --------------------------------------------------------------------------- #
# Structured sections
# --------------------------------------------------------------------------- #
@dataclass
class RunConfig:
    """Bookkeeping for a single training run."""

    name: str = "barca-finetune"
    output_dir: str = "outputs/barca-finetune"
    seed: int = 42
    # Optional free-text note recorded in the run config / W&B.
    description: Optional[str] = None


@dataclass
class DataConfig:
    """Where and how to load the protein dataset."""

    # Path passed to ``datasets.load_from_disk``.
    path: Optional[str] = None
    # If set, load an already-tokenized dataset (output of scripts/preprocess.py)
    # instead of tokenizing ``path`` on the fly.
    tokenized_path: Optional[str] = None

    train_split: str = "train"
    eval_split: Optional[str] = "validation"
    # Optional test split to evaluate after training.
    test_split: Optional[str] = None
    # Column holding the protein sequence (only used for raw, untokenized data).
    text_column: str = "sequence"
    # Whether the source already provides input_ids/attention_mask
    # (pre-tokenized / pre-packed). None => auto-detect from the train split.
    tokenized: Optional[bool] = None

    max_seq_length: int = 512
    # Concatenate sequences and chunk into ``max_seq_length`` blocks.
    packing: bool = False
    # Insert EOS between packed documents (keeps sequence boundaries).
    packing_add_eos: bool = True

    # If ``eval_split`` is None/absent and the source is a single Dataset,
    # hold out this fraction for evaluation (0.0 => no evaluation).
    eval_fraction: float = 0.0
    # If ``test_split`` is None and a held-out test fraction is desired.
    test_fraction: float = 0.0

    num_proc: Optional[int] = None
    # Cap on examples for quick smoke runs (None => all).
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None

    def __post_init__(self) -> None:
        if self.path is None and self.tokenized_path is None:
            raise ValueError("data.path or data.tokenized_path must be set")
        if self.max_seq_length < 2:
            raise ValueError("data.max_seq_length must be >= 2")
        if not 0.0 <= self.eval_fraction < 1.0:
            raise ValueError("data.eval_fraction must be in [0, 1)")
        if not 0.0 <= self.test_fraction < 1.0:
            raise ValueError("data.test_fraction must be in [0, 1)")


@dataclass
class TokenizerConfig:
    """Tokenizer loading options."""

    name_or_path: str = "khairi/barca-0.8b-base"
    trust_remote_code: bool = True
    use_fast: bool = True
    # Overrides the tokenizer's own max length when set.
    model_max_length: Optional[int] = None
    # Explicit special tokens (useful when the checkpoint ships without them).
    pad_token: Optional[str] = None
    eos_token: Optional[str] = None
    bos_token: Optional[str] = None


@dataclass
class ModelConfig:
    """Base model loading options."""

    name_or_path: str = "khairi/barca-0.8b-base"
    trust_remote_code: bool = True
    # float32 | float16 | bfloat16 | auto
    dtype: str = "bfloat16"
    # attention implementation: flash_attention_2 | sdpa | eager | None
    attn_implementation: Optional[str] = None
    # Delete non-peft modules not needed for inference when using LoRA/QLoRA.
    gradient_checkpointing: bool = True
    # Passed through to ``gradient_checkpointing_enable``.
    gradient_checkpointing_use_reentrant: bool = False

    # -- selective embedding training --------------------------------------
    # When any of the three options below is set, the whole model is frozen and
    # only the embedding rows for the selected tokens (plus their tied lm_head
    # rows) are trained. This is how the new protein tokens are fine-tuned
    # without touching the pretrained vocabulary embeddings.
    #
    # Token strings to train, e.g. ["<protein>", "</protein>", "ζA", ...].
    trainable_tokens: Optional[List[str]] = None
    # Explicit token ids to train (combined with trainable_tokens if both set).
    trainable_token_ids: Optional[List[int]] = None
    # Inclusive [start, end] id range to train (e.g. added protein tokens).
    trainable_token_range: Optional[List[int]] = None

    def __post_init__(self) -> None:
        allowed = {"float32", "float16", "bfloat16", "auto", "fp32", "fp16", "bf16"}
        if self.dtype not in allowed:
            raise ValueError(f"model.dtype must be one of {sorted(allowed)}, got {self.dtype!r}")
        rng = self.trainable_token_range
        if rng is not None and (len(rng) != 2 or int(rng[0]) > int(rng[1])):
            raise ValueError(
                "model.trainable_token_range must be an inclusive [start, end] with start <= end"
            )

    @property
    def selective_token_training(self) -> bool:
        """True when only selected token embeddings should be trained."""
        return bool(self.trainable_tokens or self.trainable_token_ids or self.trainable_token_range)


@dataclass
class QuantConfig:
    """BitsAndBytes quantization options (QLoRA)."""

    load_in_4bit: bool = True
    load_in_8bit: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class PeftConfig:
    """Parameter-efficient fine-tuning options (LoRA / QLoRA)."""

    enabled: bool = False
    # lora | qlora
    method: str = "lora"
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    # None => let peft infer the target modules for the architecture.
    target_modules: Optional[List[str]] = None
    bias: str = "none"
    use_rslora: bool = False
    quant: QuantConfig = field(default_factory=QuantConfig)

    def __post_init__(self) -> None:
        if self.method not in {"lora", "qlora"}:
            raise ValueError("peft.method must be 'lora' or 'qlora'")
        if self.method == "qlora" and not self.enabled:
            raise ValueError("peft.method='qlora' requires peft.enabled=true")


@dataclass
class CheckpointConfig:
    """Top-k checkpoint retention policy."""

    # Keep this many best checkpoints by ``metric``.
    top_k: int = 3
    metric: str = "eval_loss"
    greater_is_better: bool = False
    # Also always keep the most recent checkpoint (in addition to top-k).
    keep_last: bool = False

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("checkpointing.top_k must be >= 1")


@dataclass
class WandbConfig:
    """Weights & Biases logging options."""

    enabled: bool = True
    project: str = "barca-protein"
    entity: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    # Log the resolved recipe YAML to the run.
    log_config: bool = True
    # W&B model watching: false | gradients | parameters | all
    watch: str = "false"


# --------------------------------------------------------------------------- #
# Top-level config
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    """Entire training recipe."""

    run: RunConfig = field(default_factory=RunConfig)
    data: DataConfig = field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    peft: PeftConfig = field(default_factory=PeftConfig)
    # Free-form mapping forwarded to transformers.TrainingArguments.
    training: Dict[str, Any] = field(default_factory=dict)
    checkpointing: CheckpointConfig = field(default_factory=CheckpointConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    # Not part of the YAML schema; records where the config was loaded from.
    source_path: Optional[str] = None

    # -- construction ------------------------------------------------------ #
    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "Config":
        data = dict(data or {})
        known = {
            "run",
            "data",
            "tokenizer",
            "model",
            "peft",
            "training",
            "checkpointing",
            "wandb",
        }
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown top-level config keys: {sorted(unknown)}")
        return cls(
            run=_construct(RunConfig, data.get("run")),
            data=_construct(DataConfig, data.get("data")),
            tokenizer=_construct(TokenizerConfig, data.get("tokenizer")),
            model=_construct(ModelConfig, data.get("model")),
            peft=_construct(PeftConfig, data.get("peft")),
            training=dict(data.get("training") or {}),
            checkpointing=_construct(CheckpointConfig, data.get("checkpointing")),
            wandb=_construct(WandbConfig, data.get("wandb")),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        path = Path(path)
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        cfg = cls.from_dict(data)
        cfg.source_path = str(path)
        return cfg

    # -- serialization ----------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        data = dataclasses.asdict(self)
        data.pop("source_path", None)
        return data

    def to_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False, indent=2)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _construct(cls: type, data: Optional[Dict[str, Any]]):
    """Build a dataclass from a mapping, validating keys and nesting dataclasses."""
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise TypeError(f"{cls.__name__} expects a mapping, got {type(data).__name__}")

    field_map = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(field_map)
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")

    kwargs: Dict[str, Any] = {}
    for name, value in data.items():
        target = _nested_dataclass(field_map[name])
        if target is not None and isinstance(value, dict):
            kwargs[name] = _construct(target, value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _nested_dataclass(f: dataclasses.Field):
    """Return the dataclass type of a field whose default_factory makes one."""
    if f.default_factory is not dataclasses.MISSING:  # type: ignore[attr-defined]
        try:
            default = f.default_factory()  # type: ignore[misc]
        except Exception:
            return None
        if dataclasses.is_dataclass(default):
            return type(default)
    return None
