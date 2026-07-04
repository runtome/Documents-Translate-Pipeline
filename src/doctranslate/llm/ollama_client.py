from ollama import Client

from ..exceptions import LLMRequestError
from .base import LLMClient, LLMConfig

DEFAULT_HOST = "http://localhost:11434"


class OllamaClient(LLMClient):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client = Client(host=config.base_url or DEFAULT_HOST, timeout=config.timeout_s)

    def translate_chunk(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self._client.chat(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                options={"temperature": self.config.temperature},
            )
        except Exception as exc:
            raise LLMRequestError(str(exc), retriable=True) from exc
        return response["message"]["content"]
