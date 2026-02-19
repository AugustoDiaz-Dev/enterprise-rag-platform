from __future__ import annotations

import logging
from dataclasses import dataclass, field

from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.llm.base import BaseLLM, LLMResponse

logger = logging.getLogger(__name__)


@dataclass
class OpenAILLM(BaseLLM):
    """GPT-4o-mini provider via the official AsyncOpenAI SDK.

    Retries up to 3 times with exponential backoff on transient errors.
    """

    model: str = field(default_factory=lambda: settings.llm_model)
    temperature: float = field(default_factory=lambda: settings.llm_temperature)
    max_tokens: int = field(default_factory=lambda: settings.llm_max_tokens)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def complete(self, *, system: str, user: str) -> LLMResponse:
        from openai import AsyncOpenAI  # lazy import

        client = AsyncOpenAI(api_key=settings.openai_api_key)

        logger.debug("llm_request", extra={"model": self.model})

        response = await client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )

        choice = response.choices[0]
        usage = response.usage

        logger.info(
            "llm_response",
            extra={
                "model": response.model,
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
            },
        )

        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
        )
