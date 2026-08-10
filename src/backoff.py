"""
Спільні запобіжники для всіх викликів Gemini (і embeddings, і генерації).

Рівні розділені за класом проблеми, а не звалені в один retry:

1. **Rate limiter** — рознесення викликів у часі під квоту безкоштовного тіру,
   щоб узагалі не влітати в 429.
2. **Backoff** — транспорт: хвилинний 429, 5xx, обрив мережі.
3. **Circuit breaker** — денна квота. Її перечекати не можна, тому решта
   викликів падає миттєво, без марних походів у мережу.

Пастка, заради якої існує `is_daily_quota`: Gemini повертає `retryDelay`
(напр. `25s`) у **будь-якому** 429 — включно з денною квотою. Якщо слухати цю
пораду наосліп, прогін годинами крутить безнадійні спроби.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
import time
from typing import Awaitable, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

MAX_ATTEMPTS = 4
BASE_DELAY = 2.0
MAX_SERVER_RETRY_DELAY = 65.0

_RETRY_DELAY_RE = re.compile(r"'retryDelay':\s*'(\d+(?:\.\d+)?)s'")


class DailyQuotaExceeded(Exception):
    """Вичерпано денну квоту — до скидання квоти прогін не відновиться."""


class TransportError(Exception):
    """Вичерпані всі транспортні спроби."""


def is_daily_quota(exc: Exception) -> bool:
    return "PerDay" in str(exc)


def server_retry_delay(exc: Exception) -> float | None:
    match = _RETRY_DELAY_RE.search(str(exc))
    if not match:
        return None
    return min(float(match.group(1)), MAX_SERVER_RETRY_DELAY)


# Помилки вичерпання ресурсів процесу. Ловити їх у загальний `except Exception`
# і повторювати виклик — найгірше, що можна зробити: збій, який мав завершити
# спробу, перетворюється на нескінченний цикл. Саме так тестовий прогін цього
# модуля виїв 27 ГБ і поклав машину (2026-08-10): RecursionError від зіпсованого
# monkeypatch потрапляв сюди й запускав retry по колу.
FATAL_ERRORS = (RecursionError, MemoryError, SystemError)


def is_retryable(exc: Exception) -> bool:
    """Транспортна помилка, яку має сенс повторити після паузи."""
    if isinstance(exc, FATAL_ERRORS):
        return False
    name = type(exc).__name__
    code = getattr(exc, "code", None)
    if name == "ServerError" or (isinstance(code, int) and 500 <= code < 600):
        return True
    if code == 429:
        # Єдина 4xx, яку варто повторювати — хвилинний ліміт.
        # 400/401/403 самі не розсмокчуться, денна квота — тим паче.
        return not is_daily_quota(exc)
    if isinstance(code, int):
        return False
    return isinstance(exc, (asyncio.TimeoutError, ConnectionError, OSError))


class RateLimiter:
    """
    Черга на N викликів за хвилину. Паралелізм не обмежує — лише рознесення
    стартів у часі; за паралелізм відповідає семафор на рівні пайплайна.
    """

    def __init__(self, rpm: int):
        self._interval = 60.0 / rpm if rpm > 0 else 0.0
        self._lock = asyncio.Lock()
        self._next_slot = 0.0

    async def acquire(self) -> None:
        if self._interval <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            wait = self._next_slot - now
            if wait > 0:
                await asyncio.sleep(wait)
                now = time.monotonic()
            self._next_slot = now + self._interval


class Breaker:
    """Стан денної квоти, спільний для всіх викликів одного клієнта."""

    def __init__(self) -> None:
        self.daily_quota_hit = False


async def call_with_backoff(
    fn: Callable[[], Awaitable[T]],
    limiter: RateLimiter,
    breaker: Breaker,
    what: str = "call",
) -> T:
    last_error: Exception | None = None

    for attempt in range(MAX_ATTEMPTS):
        if breaker.daily_quota_hit:
            raise DailyQuotaExceeded("Денна квота вичерпана — виклики зупинено")
        try:
            await limiter.acquire()
            return await fn()
        except FATAL_ERRORS:
            # Вичерпано стек або памʼять — повторювати нічого, віддаємо назовні.
            raise
        except Exception as exc:  # noqa: BLE001 — тип розбираємо нижче
            if is_daily_quota(exc):
                breaker.daily_quota_hit = True
                raise DailyQuotaExceeded(
                    "Денна квота безкоштовного тіру вичерпана. "
                    "Спробуйте іншу модель (--model) або зачекайте до скидання."
                ) from exc
            if not is_retryable(exc):
                raise
            last_error = exc
            if attempt == MAX_ATTEMPTS - 1:
                break
            # Якщо сервер сам сказав, скільки чекати — слухаємо його;
            # exponential backoff тут лише запасний варіант.
            delay = server_retry_delay(exc) or BASE_DELAY * (2**attempt)
            delay += random.uniform(0, 1)
            logger.warning(
                "%s: транспортна помилка (спроба %d/%d), пауза %.1fs: %s",
                what,
                attempt + 1,
                MAX_ATTEMPTS,
                delay,
                str(exc)[:200],
            )
            await asyncio.sleep(delay)

    raise TransportError(f"{what}: {MAX_ATTEMPTS} спроб вичерпано: {last_error}")
