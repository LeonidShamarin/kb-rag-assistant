"""Фолбек між Gemini-ключами для генерації: порядок, перемикання, без-ключа.

Без мережі й без ключа: ланцюжок — на стабах, білдер — на env-словниках.
"""

from __future__ import annotations

import asyncio

import pytest

from src.backoff import DailyQuotaExceeded, TransportError
from src.llm import FallbackLLM, GeminiLLM, LLMResult, build_llm
from src.schema import Answer


class _Stub:
    def __init__(self, name: str, *, exc: Exception | None = None) -> None:
        self.name, self.exc, self.calls = name, exc, 0

    async def generate(self, system_prompt, user_prompt, schema, max_repairs: int = 1) -> LLMResult:
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        return LLMResult(parsed=schema.model_construct(), prompt_tokens=1, output_tokens=1, repairs=0)


def _run(members):
    return asyncio.run(FallbackLLM(members).generate("s", "u", Answer))


def test_first_working_key_short_circuits() -> None:
    a, b = _Stub("a"), _Stub("b")
    _run([("a", a), ("b", b)])
    assert (a.calls, b.calls) == (1, 0)


def test_transport_then_quota_falls_through() -> None:
    a = _Stub("gemini:GEMINI_API_KEY", exc=TransportError("boom"))
    b = _Stub("gemini:GEMINI_API_KEY_2", exc=DailyQuotaExceeded("per-day"))
    c = _Stub("gemini:GEMINI_API_KEY_3")
    _run([("gemini:GEMINI_API_KEY", a), ("gemini:GEMINI_API_KEY_2", b), ("gemini:GEMINI_API_KEY_3", c)])
    assert (a.calls, b.calls, c.calls) == (1, 1, 1)


def test_all_keys_failing_raises_transporterror_naming_each() -> None:
    a = _Stub("gemini:GEMINI_API_KEY", exc=TransportError("x"))
    b = _Stub("gemini:GEMINI_API_KEY_2", exc=DailyQuotaExceeded("y"))
    with pytest.raises(TransportError) as e:
        _run([("gemini:GEMINI_API_KEY", a), ("gemini:GEMINI_API_KEY_2", b)])
    assert "GEMINI_API_KEY" in str(e.value) and "GEMINI_API_KEY_2" in str(e.value)


def test_build_llm_orders_and_dedups_single_and_none() -> None:
    members = build_llm("m", env={"GEMINI_API_KEY": "k1", "GEMINI_API_KEY_2": "k2",
                                  "GEMINI_API_KEY_3": "k3"}).members
    assert [n for n, _ in members] == ["gemini:GEMINI_API_KEY", "gemini:GEMINI_API_KEY_2",
                                       "gemini:GEMINI_API_KEY_3"]
    assert isinstance(build_llm("m", env={"GEMINI_API_KEY": "only"}), GeminiLLM)
    assert build_llm("m", env={}) is None
    # duplicate key value collapses to one -> bare client
    assert isinstance(build_llm("m", env={"GEMINI_API_KEY": "s", "GEMINI_API_KEY_2": "s"}), GeminiLLM)
