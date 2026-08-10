from __future__ import annotations

import pytest

from src.backoff import (
    Breaker,
    DailyQuotaExceeded,
    RateLimiter,
    TransportError,
    call_with_backoff,
    is_daily_quota,
    is_retryable,
    server_retry_delay,
)
from src.llm import parse_json_object


class FakeApiError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


DAILY_429 = (
    "429 RESOURCE_EXHAUSTED. {'quotaId': 'GenerateRequestsPerDayPerProjectPerModel', "
    "'retryDelay': '25s'}"
)
MINUTE_429 = (
    "429 RESOURCE_EXHAUSTED. {'quotaId': 'GenerateRequestsPerMinutePerProject', "
    "'retryDelay': '12s'}"
)


def test_daily_quota_recognised_by_quota_id_not_by_retry_delay():
    """
    Пастка, заради якої існує ця перевірка: Gemini повертає retryDelay у БУДЬ-ЯКОМУ
    429, включно з денною квотою. Слухати його наосліп = годинами крутити
    безнадійні спроби.
    """
    assert is_daily_quota(FakeApiError(429, DAILY_429)) is True
    assert is_daily_quota(FakeApiError(429, MINUTE_429)) is False
    assert server_retry_delay(FakeApiError(429, DAILY_429)) == 25.0


def test_retryable_classification():
    assert is_retryable(FakeApiError(429, MINUTE_429)) is True
    assert is_retryable(FakeApiError(429, DAILY_429)) is False
    assert is_retryable(FakeApiError(503, "unavailable")) is True
    assert is_retryable(FakeApiError(400, "INVALID_ARGUMENT")) is False
    assert is_retryable(FakeApiError(403, "forbidden")) is False
    assert is_retryable(ConnectionError("reset")) is True


async def test_daily_quota_opens_breaker_and_stops_further_calls():
    breaker = Breaker()
    limiter = RateLimiter(0)
    calls = 0

    async def failing():
        nonlocal calls
        calls += 1
        raise FakeApiError(429, DAILY_429)

    with pytest.raises(DailyQuotaExceeded):
        await call_with_backoff(failing, limiter, breaker)

    assert breaker.daily_quota_hit is True
    assert calls == 1  # без повторів — денну квоту перечекати не можна

    with pytest.raises(DailyQuotaExceeded):
        await call_with_backoff(failing, limiter, breaker)
    assert calls == 1  # другий виклик навіть не пішов у мережу


async def test_non_retryable_error_propagates_immediately():
    calls = 0

    async def failing():
        nonlocal calls
        calls += 1
        raise FakeApiError(400, "INVALID_ARGUMENT")

    with pytest.raises(FakeApiError):
        await call_with_backoff(failing, RateLimiter(0), Breaker())
    assert calls == 1


async def test_transport_error_after_retries(instant_sleep):
    calls = 0

    async def flaky():
        nonlocal calls
        calls += 1
        raise FakeApiError(503, "unavailable")

    with pytest.raises(TransportError):
        await call_with_backoff(flaky, RateLimiter(0), Breaker())
    assert calls == 4


async def test_recovers_after_transient_failure(instant_sleep):
    calls = 0

    async def flaky():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise FakeApiError(503, "unavailable")
        return "ok"

    assert await call_with_backoff(flaky, RateLimiter(0), Breaker()) == "ok"


async def test_resource_exhaustion_is_not_retried(instant_sleep):
    """
    Регресія на інцидент 2026-08-10. RecursionError потрапляв у загальний
    `except Exception`, вважався транспортною помилкою й запускав retry по колу
    — прогін тестів виїдав десятки гігабайт і клав машину. Такі помилки мають
    виходити назовні з першої спроби.
    """
    for fatal in (RecursionError, MemoryError, SystemError):
        calls = 0

        async def exhausted(exc_type=fatal):
            nonlocal calls
            calls += 1
            raise exc_type("вичерпано ресурс процесу")

        assert is_retryable(fatal("x")) is False
        with pytest.raises(fatal):
            await call_with_backoff(exhausted, RateLimiter(0), Breaker())
        assert calls == 1, f"{fatal.__name__} пішов у повтор"


def test_json_parser_strips_markdown_fence():
    assert parse_json_object('```json\n{"answered": true}\n```') == {"answered": True}


def test_json_parser_rejects_non_object():
    with pytest.raises(ValueError, match="JSON-об'єкт"):
        parse_json_object("[1, 2, 3]")
