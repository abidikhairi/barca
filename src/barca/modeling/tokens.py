"""Selective token training: resolve token ids, freeze the rest, mask gradients."""

from __future__ import annotations

from ..utils import LOGGER


def resolve_trainable_token_ids(model_cfg, tokenizer) -> list:
    """Resolve ``trainable_tokens`` / ``trainable_token_ids`` / range to ids."""
    ids = set()

    for token in model_cfg.trainable_tokens or []:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id == tokenizer.unk_token_id:
            raise ValueError(
                f"model.trainable_tokens contains {token!r}, which is not in the tokenizer "
                f"({model_cfg.name_or_path})."
            )
        ids.add(int(token_id))

    for token_id in model_cfg.trainable_token_ids or []:
        ids.add(int(token_id))

    rng = model_cfg.trainable_token_range
    if rng is not None:
        ids.update(range(int(rng[0]), int(rng[1]) + 1))

    if not ids:
        raise ValueError("Selective token training requested but no token ids were resolved.")
    return sorted(ids)


def freeze_for_token_training(model, token_ids: list) -> None:
    """Freeze every parameter except the embedding rows for ``token_ids``.

    Both the input embedding and (when untied) the output ``lm_head`` rows are
    trained. Non-selected rows are kept fixed by masking their gradients, so
    the pretrained vocabulary embeddings are untouched even though the whole
    embedding matrix is a single parameter.
    """

    for param in model.parameters():
        param.requires_grad_(False)

    input_emb = model.get_input_embeddings()
    output_emb = model.get_output_embeddings()

    weights = []
    if input_emb is not None:
        weights.append(input_emb.weight)
    if output_emb is not None and (input_emb is None or output_emb.weight is not input_emb.weight):
        weights.append(output_emb.weight)

    if not weights:
        raise ValueError("Model exposes no embeddings to train.")

    token_ids = [int(i) for i in token_ids]
    for weight in weights:
        if weight.shape[0] <= max(token_ids):
            raise ValueError(
                f"Token id {max(token_ids)} is out of range for embedding of size {weight.shape[0]}."
            )
        weight.requires_grad_(True)
        _mask_embedding_gradients(weight, token_ids)

    # Snapshot the full embedding so a trainer callback can restore the frozen
    # rows after each optimizer step (decoupled weight decay would otherwise
    # shrink every row, not just the selected ones).
    snapshots = [(weight, weight.detach().clone(), tuple(token_ids)) for weight in weights]
    model._barca_selective_state = {"token_ids": token_ids, "snapshots": snapshots}

    _log_selective(model, token_ids)


def _mask_embedding_gradients(weight, token_ids: list) -> None:
    """Register a hook that zeroes the gradient of non-selected embedding rows."""
    import torch

    ids = torch.as_tensor(token_ids, dtype=torch.long)

    def _post_accumulate(param) -> None:
        if param.grad is None:
            return
        keep = torch.zeros(param.grad.shape[0], dtype=torch.bool, device=param.grad.device)
        keep[ids.to(param.grad.device)] = True
        param.grad[~keep] = 0.0

    if hasattr(weight, "register_post_accumulate_grad_hook"):
        weight.register_post_accumulate_grad_hook(_post_accumulate)
    else:  # pragma: no cover - older torch fallback

        def _hook(grad):
            keep = torch.zeros(grad.shape[0], dtype=torch.bool, device=grad.device)
            keep[ids.to(grad.device)] = True
            grad = grad.clone()
            grad[~keep] = 0.0
            return grad

        weight.register_hook(_hook)


def _log_selective(model, token_ids: list) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOGGER.info(
        "Selective token training: %d/%d embedding rows trainable, %s params (%.4f%% of %s).",
        len(token_ids),
        model.get_input_embeddings().weight.shape[0],
        f"{trainable:,}",
        100.0 * trainable / total if total else 0.0,
        f"{total:,}",
    )
