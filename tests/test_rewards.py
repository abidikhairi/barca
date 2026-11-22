import asyncio
import math
from types import SimpleNamespace

import httpx
import pytest

from barca.rl.rewards import (
    RewardConfig,
    make_rewards,
    parse_completion,
    reference_f1,
    uniq5,
)

AA = "ACDEFGHIKLMNPQRSTVWY"
REFERENCE = "MPQCKTSPLSNWHTFLFEYKVYFLEDMSVENQMYHVSRTKCVADPAYS"


def completion(sequence: str, lead: str = "The design is:\n") -> str:
    return lead + "<protein>" + "".join("ζ" + residue for residue in sequence) + "</protein>"


def test_parse_completion_valid() -> None:
    assert parse_completion(completion(REFERENCE)) == REFERENCE


def test_parse_completion_rejects_invalid() -> None:
    assert parse_completion(completion(REFERENCE).replace("</protein>", "")) is None
    assert parse_completion(completion(REFERENCE) + completion(REFERENCE)) is None
    assert parse_completion(completion(REFERENCE).replace("ζA", "A", 1)) is None
    assert parse_completion(completion("A" * 29)) is None
    assert parse_completion(completion("A" * 513)) is None


def test_parse_without_zeta() -> None:
    text = "<protein>" + REFERENCE + "</protein>"
    assert parse_completion(text, require_zeta=False) == REFERENCE
    assert parse_completion(text, require_zeta=True) is None


def test_f1_and_uniq5() -> None:
    assert reference_f1(REFERENCE, REFERENCE) == pytest.approx(1.0)
    assert reference_f1("W" * 50, REFERENCE) == pytest.approx(0.0)
    assert uniq5("M" * 50) < 0.1
    assert uniq5(REFERENCE) > 0.8


def test_reward_functions() -> None:
    valid = completion(REFERENCE)
    short = completion("A" * 10)
    config = RewardConfig(include_foldability=False)
    functions, weights = make_rewards(config)
    assert [function.__name__ for function in functions] == [
        "format_reward",
        "ref_f1_reward",
        "diversity_reward",
        "length_reward",
    ]
    assert sum(weights) == pytest.approx(1.0)

    completions = [valid, short]
    references = [REFERENCE, REFERENCE]
    assert functions[0](completions=completions) == [1.0, 0.0]
    assert functions[1](completions=completions, reference=references) == pytest.approx([1.0, 0.0])
    assert functions[2](completions=completions) == pytest.approx([1.0, 0.0])
    assert functions[3](
        completions=completions, reference=references, ref_len=[len(REFERENCE), len(REFERENCE)]
    ) == pytest.approx([1.0, 0.0])


def test_length_reward_shape() -> None:
    config = RewardConfig(include_foldability=False)
    functions, _ = make_rewards(config)
    length_reward = functions[-1]
    assert length_reward(completions=[completion(REFERENCE)], ref_len=[len(REFERENCE)]) == pytest.approx(
        [1.0]
    )
    doubled = REFERENCE + REFERENCE
    assert length_reward(completions=[completion(doubled)], ref_len=[len(REFERENCE)]) == pytest.approx(
        [math.exp(-1.0)]
    )


def _client_factory(handler):
    return lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=5.0)


def test_foldability_reward_async() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = __import__("json").loads(request.content)
        results = [
            {"plddt_mean": 0.5 + 0.01 * index, "ptm": 0.4}
            for index, _ in enumerate(payload["sequences"])
        ]
        return httpx.Response(200, json={"results": results})

    config = RewardConfig(fold_service_url="http://test")
    funcs, weights = make_rewards(config, client_factory=_client_factory(handler))
    foldability = funcs[2]
    assert foldability.__name__ == "foldability_reward"
    assert weights[2] == pytest.approx(0.35)

    completions = [completion(REFERENCE), completion("A" * 10), completion(REFERENCE[::-1])]
    rewards = asyncio.run(foldability(completions=completions, reference=[REFERENCE] * 3))
    assert rewards[0] == pytest.approx(0.5)
    assert rewards[1] == 0.0
    assert rewards[2] == pytest.approx(0.51)


def test_foldability_service_failure_returns_none() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    config = RewardConfig(fold_service_url="http://test")
    funcs, _ = make_rewards(config, client_factory=_client_factory(handler))
    rewards = asyncio.run(funcs[2](completions=[completion(REFERENCE)], reference=[REFERENCE]))
    assert rewards == [None]


def test_foldability_skip_steps() -> None:
    calls = {"count": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(200, json={"results": [{"plddt_mean": 0.7}]})

    config = RewardConfig(fold_service_url="http://test", fold_every_n_steps=5)
    funcs, _ = make_rewards(config, client_factory=_client_factory(handler))
    foldability = funcs[2]
    rewards = asyncio.run(
        foldability(
            completions=[completion(REFERENCE)],
            reference=[REFERENCE],
            trainer_state=SimpleNamespace(global_step=1),
        )
    )
    assert rewards == [None]
    assert calls["count"] == 0
    rewards = asyncio.run(
        foldability(
            completions=[completion(REFERENCE)],
            reference=[REFERENCE],
            trainer_state=SimpleNamespace(global_step=0),
        )
    )
    assert rewards == [pytest.approx(0.7)]
    assert calls["count"] == 1
