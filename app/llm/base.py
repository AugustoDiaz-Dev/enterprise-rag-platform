from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class BaseLLM(ABC):
    """Base interface for all LLM providers.

    Each provider must implement `complete()` which accepts a system prompt
    and a user message and returns a structured LLMResponse.
    """

    @abstractmethod
    async def complete(self, *, system: str, user: str) -> LLMResponse:
        """Send a chat completion request and return the response."""
        ...
