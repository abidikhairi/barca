"""Tests for the top-k checkpoint retention callback."""

from __future__ import annotations

from types import SimpleNamespace

from barca.callbacks import TopKCheckpointCallback


def _args(tmp_path):
    return SimpleNamespace(output_dir=str(tmp_path))


def _state(step, max_steps=1000):
    return SimpleNamespace(global_step=step, max_steps=max_steps)


def _save(cb, tmp_path, step, loss, metric="eval_loss"):
    checkpoint = tmp_path / f"checkpoint-{step}"
    checkpoint.mkdir(exist_ok=True)
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    cb.on_evaluate(_args(tmp_path), _state(step), None, metrics={metric: loss})
    cb.on_save(_args(tmp_path), _state(step), None)


def _remaining(tmp_path):
    return sorted(p.name for p in tmp_path.iterdir() if p.is_dir())


def test_keeps_best_three_by_eval_loss(tmp_path):
    cb = TopKCheckpointCallback(metric="eval_loss", greater_is_better=False, top_k=3)
    for step, loss in [(10, 0.5), (20, 0.3), (30, 0.4), (40, 0.2), (50, 0.6)]:
        _save(cb, tmp_path, step, loss)

    assert _remaining(tmp_path) == ["checkpoint-20", "checkpoint-30", "checkpoint-40"]
    assert len(cb.kept_paths()) == 3


def test_greater_is_better(tmp_path):
    cb = TopKCheckpointCallback(metric="eval_accuracy", greater_is_better=True, top_k=2)
    for step, acc in [(10, 0.6), (20, 0.9), (30, 0.7), (40, 0.8)]:
        _save(cb, tmp_path, step, acc, metric="eval_accuracy")

    assert _remaining(tmp_path) == ["checkpoint-20", "checkpoint-40"]


def test_keep_last_retains_most_recent(tmp_path):
    cb = TopKCheckpointCallback(
        metric="eval_loss", greater_is_better=False, top_k=2, keep_last=True
    )
    for step, loss in [(10, 0.1), (20, 0.2), (30, 0.3), (40, 0.4)]:
        _save(cb, tmp_path, step, loss)

    remaining = _remaining(tmp_path)
    assert "checkpoint-40" in remaining  # most recent is protected
    assert "checkpoint-10" in remaining  # best is kept
    assert len(remaining) == 3


def test_save_without_evaluation_is_ignored(tmp_path):
    cb = TopKCheckpointCallback(top_k=1)
    checkpoint = tmp_path / "checkpoint-5"
    checkpoint.mkdir()
    cb.on_save(_args(tmp_path), _state(5), None)
    assert _remaining(tmp_path) == ["checkpoint-5"]
