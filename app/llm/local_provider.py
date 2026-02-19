from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.llm.base import BaseLLM, LLMResponse

logger = logging.getLogger(__name__)


@dataclass
class LocalModelProvider(BaseLLM):
    """LLM provider for locally-hosted models via Ollama (#11).

    Ollama exposes an OpenAI-compatible HTTP API at ``/api/chat``.
    Ensure Ollama is running and the model is pulled before use:

        ollama pull llama3
        OLLAMA_URL=http://localhost:11434 LLM_PROVIDER=local docker compose up

    Config (via environment variables / .env):
        LLM_PROVIDER=local
        LOCAL_MODEL_NAME=llama3          # any model available in Ollama
        LOCAL_MODEL_URL=http://localhost:11434
    """

    model: str = field(default="llama3")
    base_url: str = field(default="http://localhost:11434")

    async def complete(self, *, system: str, user: str) -> LLMResponse:
        import httpx  # already in dependencies

        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

        logger.debug("local_llm_request", extra={"model": self.model, "url": url})

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        message = data.get("message", {})
        content = message.get("content", "")

        # Ollama token counts (present in non-stream responses)
        eval_count: int = data.get("eval_count", 0)          # completion tokens
        prompt_eval_count: int = data.get("prompt_eval_count", 0)  # prompt tokens
        total = prompt_eval_count + eval_count

        logger.info(
            "local_llm_response",
            extra={
                "model": self.model,
                "prompt_tokens": prompt_eval_count,
                "completion_tokens": eval_count,
            },
        )

        return LLMResponse(
            content=content,
            model=self.model,
            prompt_tokens=prompt_eval_count,
            completion_tokens=eval_count,
            total_tokens=total,
        )
