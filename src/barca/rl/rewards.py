"""GRPO reward functions for Barca protein design.

The reward suite scores one completion at a time and is called by TRL's
``GRPOTrainer`` with keyword arguments. Extra dataset columns (``reference`` and
``ref_len``) are forwarded per completion. Invalid completions receive zero from
every component; a fold-service outage returns ``None`` for the foldability
component so TRL's NaN-aware aggregation ignores it.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import httpx

LOGGER = logging.getLogger("barca.rl.rewards")
AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


@dataclass(frozen=True)
class RewardConfig:
    format_weight: float = 0.10
    ref_f1_weight: float = 0.30
    foldability_weight: float = 0.35
    diversity_weight: float = 0.15
    length_weight: float = 0.10
    include_foldability: bool = True
    min_len: int = 30
    max_len: int = 512
    require_zeta: bool = True
    uniq5_target: float = 0.8
    fold_service_url: str = "http://127.0.0.1:8010"
    fold_num_loops: int = 4
    fold_num_sampling_steps: int = 20
    fold_timeout_s: float = 1200.0
    fold_every_n_steps: int = 1
    extra: dict = field(default_factory=dict)


def completion_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    if isinstance(completion, (list, tuple)):
        if not completion:
            return ""
        return completion_text(completion[-1])
    return str(completion)


def parse_completion(
    text: str,
    min_len: int = 30,
    max_len: int = 512,
    require_zeta: bool = True,
) -> Optional[str]:
    """Return the canonical sequence inside one ``<protein>`` block, else ``None``."""
    if text.count("<protein>") != 1 or text.count("</protein>") != 1:
        return None
    start = text.index("<protein>") + len("<protein>")
    end = text.index("</protein>")
    if end <= start:
        return None
    raw = re.sub(r"\s+", "", text[start:end])
    if require_zeta:
        if re.fullmatch(r"(?:ζ[ACDEFGHIKLMNPQRSTVWY])+", raw) is None:
            return None
        sequence = raw.replace("ζ", "")
    else:
        sequence = raw.replace("ζ", "")
        if not sequence or set(sequence) - AMINO_ACIDS:
            return None
    if not (min_len <= len(sequence) <= max_len):
        return None
    return sequence


def kmers(sequence: str, k: int = 5) -> set[str]:
    return {sequence[i : i + k] for i in range(len(sequence) - k + 1)}


def reference_f1(sequence: str, reference: str) -> float:
    if len(sequence) < 5 or len(reference) < 5:
        return 0.0
    generated = kmers(sequence)
    expected = kmers(reference)
    denominator = len(generated) + len(expected)
    if denominator == 0:
        return 0.0
    return 2.0 * len(generated & expected) / denominator


def uniq5(sequence: str) -> float:
    if len(sequence) < 5:
        return 0.0
    return len(kmers(sequence)) / (len(sequence) - 4)


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("reward weights must sum to a positive number")
    return {name: value / total for name, value in weights.items()}


def make_rewards(
    config: RewardConfig,
    client_factory: Optional[Callable[[], httpx.AsyncClient]] = None,
) -> tuple[list[Callable], list[float]]:
    """Build reward functions and weights in the order expected by the trainer."""

    def _parse(completion) -> Optional[str]:
        return parse_completion(
            completion_text(completion),
            min_len=config.min_len,
            max_len=config.max_len,
            require_zeta=config.require_zeta,
        )

    def format_reward(completions=None, **kwargs) -> list[float]:
        del kwargs
        return [1.0 if _parse(completion) is not None else 0.0 for completion in completions]

    def ref_f1_reward(completions=None, reference=None, **kwargs) -> list[float]:
        del kwargs
        references = reference if reference is not None else [""] * len(completions)
        rewards = []
        for completion, expected in zip(completions, references, strict=True):
            sequence = _parse(completion)
            rewards.append(0.0 if sequence is None else reference_f1(sequence, expected))
        return rewards

    def diversity_reward(completions=None, **kwargs) -> list[float]:
        del kwargs
        rewards = []
        for completion in completions:
            sequence = _parse(completion)
            if sequence is None:
                rewards.append(0.0)
            else:
                rewards.append(min(1.0, uniq5(sequence) / config.uniq5_target))
        return rewards

    def length_reward(completions=None, reference=None, ref_len=None, **kwargs) -> list[float]:
        del kwargs
        if ref_len is not None:
            targets = [float(value) for value in ref_len]
        elif reference is not None:
            targets = [float(len(value)) for value in reference]
        else:
            targets = [0.0] * len(completions)
        rewards = []
        for completion, target in zip(completions, targets, strict=True):
            sequence = _parse(completion)
            if sequence is None or target <= 0:
                rewards.append(0.0)
            else:
                rewards.append(math.exp(-abs(len(sequence) - target) / target))
        return rewards

    functions: list[Callable] = [format_reward, ref_f1_reward, diversity_reward, length_reward]
    weights = {
        "format_reward": config.format_weight,
        "ref_f1_reward": config.ref_f1_weight,
        "diversity_reward": config.diversity_weight,
        "length_reward": config.length_weight,
    }

    if config.include_foldability:
        functions.insert(2, make_foldability_reward(config, client_factory))
        weights["foldability_reward"] = config.foldability_weight

    normalized = normalize_weights(weights)
    return functions, [normalized[function.__name__] for function in functions]


def make_foldability_reward(
    config: RewardConfig,
    client_factory: Optional[Callable[[], httpx.AsyncClient]] = None,
) -> Callable:
    """Return an async reward that scores parsed completions with ESMFold2 pLDDT."""

    def _client() -> httpx.AsyncClient:
        if client_factory is not None:
            return client_factory()
        return httpx.AsyncClient(timeout=config.fold_timeout_s)

    async def foldability_reward(completions=None, trainer_state=None, **kwargs) -> list[Optional[float]]:
        del kwargs
        parsed = [
            parse_completion(
                completion_text(completion),
                min_len=config.min_len,
                max_len=config.max_len,
                require_zeta=config.require_zeta,
            )
            for completion in completions
        ]
        valid_positions = [index for index, sequence in enumerate(parsed) if sequence is not None]
        rewards: list[Optional[float]] = [0.0] * len(parsed)
        if not valid_positions:
            return rewards
        step = int(getattr(trainer_state, "global_step", 0)) if trainer_state is not None else 0
        if config.fold_every_n_steps > 1 and step % config.fold_every_n_steps != 0:
            for index in valid_positions:
                rewards[index] = None
            return rewards

        sequences = [parsed[index] for index in valid_positions]
        try:
            async with _client() as client:
                response = await client.post(
                    f"{config.fold_service_url.rstrip('/')}/fold_batch",
                    json={
                        "sequences": sequences,
                        "num_loops": config.fold_num_loops,
                        "num_sampling_steps": config.fold_num_sampling_steps,
                        "seed": 0,
                    },
                )
                response.raise_for_status()
                results = response.json().get("results", [])
        except Exception:  # noqa: BLE001 - an outage must not penalize valid sequences
            LOGGER.exception("foldability service request failed")
            for index in valid_positions:
                rewards[index] = None
            return rewards

        if len(results) != len(valid_positions):
            LOGGER.error(
                "foldability service returned %d results for %d sequences",
                len(results),
                len(valid_positions),
            )
            for index in valid_positions:
                rewards[index] = None
            return rewards

        for index, result in zip(valid_positions, results, strict=True):
            if "error" in result or "plddt_mean" not in result:
                LOGGER.warning("foldability error for candidate %d: %s", index, result.get("error"))
                rewards[index] = None
            else:
                rewards[index] = float(result["plddt_mean"])
        return rewards

    return foldability_reward


def mean(values: Iterable[Optional[float]]) -> float:
    materialized = [value for value in values if value is not None]
    return sum(materialized) / len(materialized) if materialized else float("nan")
