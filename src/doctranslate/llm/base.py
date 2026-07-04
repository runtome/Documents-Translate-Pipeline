from abc import ABC, abstractmethod
from typing import Any, Optional

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    provider: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.2
    timeout_s: float = 60.0
    max_retries: int = 3
    extra: dict[str, Any] = Field(default_factory=dict)


class LLMClient(ABC):
    def __init__(self, config: LLMConfig):
        self.config = config

    @abstractmethod
    def translate_chunk(self, system_prompt: str, user_prompt: str) -> str:
        """Send one chunk to the provider and return its raw text response."""
