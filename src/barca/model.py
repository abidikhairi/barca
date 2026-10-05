"""Tokenizer and model loading for the Barca protein LM family.

``barca-0.8b-base`` is treated as a standard decoder-only (Llama-style) causal
language model, so everything here goes through the ``transformers`` Auto
classes. Architecture-specific behaviour can still be reached through
``trust_remote_code`` or by overriding ``model.name_or_path``.
"""

from __future__ import annotations

from typing import Optional

from .config import Config
from .utils import LOGGER, resolve_dtype


def load_tokenizer(cfg: Config):
    """Load and minimally normalize the tokenizer."""
    from transformers import AutoTokenizer

    tok = cfg.tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tok.name_or_path,
        trust_remote_code=tok.trust_remote_code,
        use_fast=tok.use_fast,
    )

    # Protein LMs frequently ship without a padding token. Causal LM training
    # pads only dynamically, so falling back to EOS is safe.
    if tok.pad_token is not None:
        tokenizer.pad_token = tok.pad_token
    elif tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError(
                "Tokenizer has no pad_token and no eos_token; set tokenizer.pad_token explicitly."
            )
        tokenizer.pad_token = tokenizer.eos_token

    if tok.eos_token is not None:
        tokenizer.eos_token = tok.eos_token
    if tok.bos_token is not None:
        tokenizer.bos_token = tok.bos_token
    if tok.model_max_length is not None:
        tokenizer.model_max_length = tok.model_max_length

    LOGGER.info(
        "Tokenizer: %s | vocab=%d | pad=%r eos=%r bos=%r",
        cfg.tokenizer.name_or_path,
        tokenizer.vocab_size,
        tokenizer.pad_token,
        tokenizer.eos_token,
        tokenizer.bos_token,
    )
    return tokenizer


def load_model(cfg: Config):
    """Load the base model and apply PEFT wrapping when configured."""
    from transformers import AutoModelForCausalLM

    mc = cfg.model
    dtype = resolve_dtype(mc.dtype)

    kwargs = {"trust_remote_code": mc.trust_remote_code}
    if mc.attn_implementation:
        kwargs["attn_implementation"] = mc.attn_implementation

    if cfg.peft.enabled and cfg.peft.method == "qlora":
        from transformers import BitsAndBytesConfig

        q = cfg.peft.quant
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=q.load_in_4bit,
            load_in_8bit=q.load_in_8bit,
            bnb_4bit_quant_type=q.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=resolve_dtype(q.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=q.bnb_4bit_use_double_quant,
        )
        kwargs["device_map"] = "auto"

    LOGGER.info("Loading model %s (dtype=%s)", mc.name_or_path, mc.dtype)
    model = _from_pretrained_autodtype(AutoModelForCausalLM, mc.name_or_path, dtype, kwargs)

    if cfg.peft.enabled:
        model = apply_peft(model, cfg)
    else:
        _log_trainable(model, label="full fine-tuning")

    return model


def apply_peft(model, cfg: Config):
    """Wrap ``model`` with a LoRA/QLoRA adapter according to ``cfg.peft``."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    peft_cfg = cfg.peft
    if peft_cfg.method == "qlora":
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=cfg.model.gradient_checkpointing,
        )
    elif cfg.model.gradient_checkpointing:
        # Required so gradients flow to LoRA adapters through frozen embeddings.
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=peft_cfg.r,
        lora_alpha=peft_cfg.alpha,
        lora_dropout=peft_cfg.dropout,
        target_modules=peft_cfg.target_modules,
        bias=peft_cfg.bias,
        use_rslora=peft_cfg.use_rslora,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    _log_trainable(model, label=f"{peft_cfg.method}")
    return model


def _from_pretrained_autodtype(auto_cls, name_or_path: str, dtype, kwargs: dict):
    """Load with ``dtype`` or the legacy ``torch_dtype`` kwarg as available."""
    if dtype is None:
        return auto_cls.from_pretrained(name_or_path, **kwargs)

    preferred, fallback = ("dtype", "torch_dtype")
    if not _prefers_dtype_kwarg():
        preferred, fallback = fallback, preferred
    try:
        return auto_cls.from_pretrained(name_or_path, **{preferred: dtype}, **kwargs)
    except TypeError:
        return auto_cls.from_pretrained(name_or_path, **{fallback: dtype}, **kwargs)


def _prefers_dtype_kwarg() -> bool:
    """True when the installed transformers uses ``dtype`` over ``torch_dtype``."""
    try:
        import transformers
        from packaging.version import Version

        return Version(transformers.__version__) >= Version("4.56.0")
    except Exception:  # pragma: no cover - packaging is always present in practice
        return False


def _log_trainable(model, label: str) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * trainable / total if total else 0.0
    LOGGER.info(
        "Model ready (%s): %s trainable / %s total params (%.2f%%)",
        label,
        f"{trainable:,}",
        f"{total:,}",
        pct,
    )
