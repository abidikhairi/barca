"""Trainer callbacks: top-k checkpoint retention and W&B config logging."""

from __future__ import annotations

import os
import shutil
from typing import List, Optional, Tuple

from transformers import TrainerCallback

from ..utils import LOGGER


class TopKCheckpointCallback(TrainerCallback):
    """Keep only the ``top_k`` best checkpoints by a metric (default eval loss).

    ``transformers`` can cap the *number* of checkpoints (``save_total_limit``)
    but always drops the oldest ones. This callback instead ranks every saved
    checkpoint by ``metric`` and deletes all but the best ``top_k``, which is
    what we want when disk is scarce and the best checkpoints matter most.

    It is assumed that ``save_strategy == eval_strategy`` and (for step-based
    saving) ``save_steps == eval_steps``, so the metric produced by the latest
    evaluation corresponds to the checkpoint being saved.
    """

    def __init__(
        self,
        metric: str = "eval_loss",
        greater_is_better: bool = False,
        top_k: int = 3,
        keep_last: bool = False,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        self.metric = metric
        self.greater_is_better = greater_is_better
        self.top_k = top_k
        self.keep_last = keep_last

        # (rank_key, raw_score, step, path) for retained checkpoints.
        self._kept: List[Tuple[float, float, int, str]] = []
        self._last_metric: Optional[float] = None
        self._last_metric_step: Optional[int] = None

    # -- helpers ----------------------------------------------------------- #
    def _rank_key(self, score: float) -> float:
        """Higher is better for sorting, independent of the metric direction."""
        return score if self.greater_is_better else -score

    def kept_paths(self) -> List[str]:
        """Currently retained checkpoint directories, best first."""
        return [entry[3] for entry in self._kept]

    def best_path(self) -> Optional[str]:
        """Path of the best retained checkpoint, or None if none were kept."""
        return self._kept[0][3] if self._kept else None

    # -- Trainer hooks ----------------------------------------------------- #
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and metrics.get(self.metric) is not None:
            self._last_metric = float(metrics[self.metric])
            self._last_metric_step = state.global_step
        return control

    def on_save(self, args, state, control, **kwargs):
        checkpoint = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(checkpoint):
            return control
        if self._last_metric is None or self._last_metric_step != state.global_step:
            # Save triggered without a matching evaluation; leave it alone.
            return control

        self._kept.append(
            (self._rank_key(self._last_metric), self._last_metric, state.global_step, checkpoint)
        )
        # Forget entries whose directories were removed (e.g. by the Trainer).
        self._kept = [entry for entry in self._kept if os.path.isdir(entry[3])]
        self._kept.sort(key=lambda entry: entry[0], reverse=True)

        keep = self._kept[: self.top_k]
        drop = self._kept[self.top_k :]

        if self.keep_last:
            latest = max(self._kept, key=lambda entry: entry[2])
            if latest not in keep:
                keep.append(latest)
                drop = [entry for entry in drop if entry is not latest]

        for _, score, step, path in drop:
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
                LOGGER.info("Removed checkpoint-%d (%s=%.6f) outside top-%d", step, self.metric, score, self.top_k)

        self._kept = keep[: self.top_k] if not self.keep_last else keep
        LOGGER.info(
            "Top-%d checkpoints: %s",
            self.top_k,
            [f"step {e[2]} ({self.metric}={e[1]:.6f})" for e in self._kept],
        )
        return control


class WandbConfigCallback(TrainerCallback):
    """Log the resolved recipe and (optionally) watch the model in W&B."""

    def __init__(self, config_file: Optional[str] = None, watch: str = "false") -> None:
        self.config_file = config_file
        self.watch = (watch or "false").lower()

    def on_train_begin(self, args, state, control, **kwargs):
        try:
            import wandb
        except ImportError:  # pragma: no cover
            return control
        if wandb.run is None:
            LOGGER.warning("wandb.run is None; skipping W&B config logging.")
            return control

        if self.config_file and os.path.isfile(self.config_file):
            try:
                wandb.save(self.config_file, base_path=os.path.dirname(self.config_file) or ".")
            except Exception as exc:  # pragma: no cover - non-fatal
                LOGGER.warning("Could not upload config to W&B: %s", exc)

        if self.watch not in {"false", "none", ""}:
            model = kwargs.get("model")
            if model is not None:
                try:
                    wandb.watch(model, log=self.watch if self.watch != "true" else "all")
                except Exception as exc:  # pragma: no cover - non-fatal
                    LOGGER.warning("wandb.watch failed: %s", exc)
        return control


class RestoreFrozenEmbeddingRowsCallback(TrainerCallback):
    """Keep non-selected embedding rows exactly frozen.

    Gradient masking already zeroes their gradients, but decoupled weight decay
    (AdamW) would still shrink every row. After each optimizer step we restore
    the frozen rows from a snapshot, so only the selected token rows move.
    """

    def __init__(self, model) -> None:
        state = getattr(model, "_barca_selective_state", None)
        self._entries = list(state["snapshots"]) if state else []

    def _restore(self) -> None:
        if not self._entries:
            return
        import torch

        with torch.no_grad():
            for weight, snapshot, token_ids in self._entries:
                keep = torch.zeros(weight.shape[0], dtype=torch.bool, device=weight.device)
                keep[torch.as_tensor(token_ids, device=weight.device)] = True
                frozen = ~keep
                weight.data[frozen] = snapshot.to(weight.device, weight.dtype)[frozen]

    def on_optimizer_step(self, args, state, control, **kwargs):
        self._restore()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        self._restore()
        return control


class TokenStatsCallback(TrainerCallback):
    """Log the number of training tokens seen (and the total budget)."""

    def __init__(self, seq_len: int, per_device_batch: int, grad_accum: int, train_size: int) -> None:
        self.tokens_per_step = max(1, per_device_batch) * max(1, grad_accum) * seq_len
        self.total_tokens = train_size * seq_len
        self.train_size = train_size
        self.seq_len = seq_len

    def on_train_begin(self, args, state, control, **kwargs):
        world = getattr(args, "world_size", 1) or 1
        self.tokens_per_step *= max(1, world)
        LOGGER.info(
            "Token budget: %s sequences x %d tokens = %s tokens; %s tokens per optimizer step.",
            f"{self.train_size:,}",
            self.seq_len,
            f"{self.total_tokens:,}",
            f"{self.tokens_per_step:,}",
        )
        self._log_wandb({"train/total_tokens": self.total_tokens}, step=state.global_step)
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        seen = getattr(state, "num_input_tokens_seen", None)
        if seen is None:
            seen = state.global_step * self.tokens_per_step
        self._log_wandb({"train/tokens_seen": seen}, step=state.global_step)
        return control

    @staticmethod
    def _log_wandb(payload: dict, step: int) -> None:
        try:
            import wandb

            if wandb.run is not None:
                wandb.log(payload, step=step)
        except Exception:  # pragma: no cover - logging must never break training
            pass
