"""LoRA / QLoRA adapter wrapping."""

from __future__ import annotations

from ..config import Config
from ..utils import log_trainable_parameters


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
    log_trainable_parameters(model, label=f"{peft_cfg.method}")
    return model
