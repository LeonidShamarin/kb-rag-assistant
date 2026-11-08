"""
Тонка обгортка над Gemini для structured output.

Робить рівно три речі: викликає модель зі схемою, лічить токени і при невалідній
відповіді дає моделі один шанс виправитись, показавши їй її ж помилку
(self-repair). Транспортні проблеми — не сюди, вони в backoff.py.

Важливо: навіть коли модель отримала `response_schema`, відповідь усе одно
проганяється через `model_validate()`. Structured output — це сильна підказка,
а не гарантія.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import TypeVar

from google.genai import errors as genai_errors
from pydantic import BaseModel, ValidationError

from src.backoff import Breaker, DailyQuotaExceeded, RateLimiter, TransportError, call_with_backoff

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


@dataclass
class LLMResult:
    parsed: BaseModel
    prompt_tokens: int
    output_tokens: int
    repairs: int


class InvalidLLMOutput(Exception):
    """Модель не змогла віддати валідний JSON навіть після self-repair."""


def parse_json_object(raw: str) -> dict:
    raw = raw.strip()
    # захист від випадків, коли модель усе ж обгортає у ```json ... ```
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(f"Очікували JSON-об'єкт, отримали {type(data).__name__}")
    return data


class GeminiLLM:
    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0.0,
        rpm: int = 5,
        breaker: Breaker | None = None,
    ):
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self._limiter = RateLimiter(rpm)
        self.breaker = breaker or Breaker()

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: type[T],
        max_repairs: int = 1,
    ) -> LLMResult:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=self.temperature,
            response_mime_type="application/json",
            response_schema=schema,
            # Переказати наданий контекст — не задача для глибокого reasoning.
            # Саме thinking_level, а не thinking_budget: Gemini 3.x відповідає
            # на budget=0 помилкою 400 INVALID_ARGUMENT.
            thinking_config=types.ThinkingConfig(thinking_level="low"),
        )

        prompt = user_prompt
        prompt_tokens = output_tokens = 0
        last_error: Exception | None = None

        for attempt in range(max_repairs + 1):

            async def call(prompt: str = prompt):
                return await self._client.aio.models.generate_content(
                    model=self.model, contents=prompt, config=config
                )

            response = await call_with_backoff(
                call, self._limiter, self.breaker, what=f"generate[{self.model}]"
            )

            usage = getattr(response, "usage_metadata", None)
            if usage:
                prompt_tokens += getattr(usage, "prompt_token_count", 0) or 0
                output_tokens += (getattr(usage, "candidates_token_count", 0) or 0) + (
                    getattr(usage, "thoughts_token_count", 0) or 0
                )

            text = response.text
            try:
                if not text:
                    # Порожня відповідь — напр. спрацював safety-фільтр. Це вже
                    # не транспорт, а зміст: лікується self-repair-ом.
                    raise ValueError("Модель повернула порожню відповідь")
                parsed = schema.model_validate(parse_json_object(text))
                return LLMResult(parsed, prompt_tokens, output_tokens, repairs=attempt)
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                last_error = exc
                logger.warning("Невалідна відповідь (спроба %d): %s", attempt + 1, exc)
                prompt = (
                    f"{user_prompt}\n\n"
                    f"Попередня відповідь була невалідною: {exc}\n"
                    f"Виправ і поверни ТІЛЬКИ коректний JSON за схемою."
                )

        raise InvalidLLMOutput(f"Валідна відповідь не отримана: {last_error}")


# --- фолбек між кількома Gemini-ключами --------------------------------------
#
# kb-rag і генерацію, і embeddings робить через Gemini, тож крос-провайдерний
# фолбек тут не підходить (ембединги іншого провайдера дали б несумісні вектори
# й зіпсували індекс). Натомість ротуємо Gemini-ключі: основний GEMINI_API_KEY,
# далі ключі з ключі2.txt (GEMINI_API_KEY_2/_3) в останню чергу. Перемикаємося на
# транспортному збої та денній квоті; невалідний JSON (InvalidLLMOutput) — це
# зміст, його лікує self-repair, а не зміна ключа. Кожен ключ має свій Breaker.

_GEMINI_KEY_VARS = ("GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3")


class FallbackLLM:
    """Пробує Gemini-ключі по черзі; на `TransportError`/`DailyQuotaExceeded`
    бере наступний. Кидає помилку лише коли відмовили всі ключі."""

    def __init__(self, members: list[tuple[str, GeminiLLM]]):
        if not members:
            raise ValueError("порожній ланцюжок Gemini-ключів")
        self.members = members

    async def generate(self, system_prompt: str, user_prompt: str, schema: type[T],
                        max_repairs: int = 1) -> LLMResult:
        errors: list[str] = []
        for name, client in self.members:
            try:
                return await client.generate(system_prompt, user_prompt, schema, max_repairs)
            except (TransportError, DailyQuotaExceeded, genai_errors.APIError) as exc:
                errors.append(f"{name}: {str(exc)[:150]}")
        raise TransportError("усі Gemini-ключі відмовили -> " + " | ".join(errors))


def _gemini_keys(env: dict) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for var in _GEMINI_KEY_VARS:
        key = (env.get(var) or "").strip()
        if key and key not in seen:
            seen.add(key)
            out.append((var, key))
    return out


def build_llm(model: str, temperature: float = 0.0, rpm: int = 5, *, env: dict | None = None):
    """Збирає генеративний клієнт із ланцюжком Gemini-ключів, або `None`, якщо
    жодного ключа немає (як і раніше). Один ключ — звичайний `GeminiLLM`, кілька —
    обгортка з фолбеком."""
    env = os.environ if env is None else env
    keys = _gemini_keys(env)
    if not keys:
        return None
    members = [(f"gemini:{var}", GeminiLLM(key, model, temperature, rpm)) for var, key in keys]
    if len(members) == 1:
        return members[0][1]
    return FallbackLLM(members)
