"""Trainer callbacks: top-k checkpoint retention and W&B config logging."""

from __future__ import annotations

import os
import shutil
from typing import List, Optional, Tuple

from transformers import TrainerCallback

from .utils import LOGGER


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
