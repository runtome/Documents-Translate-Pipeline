from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError

from ..exceptions import LLMRequestError
from .base import LLMClient, LLMConfig


class OpenAIClient(LLMClient):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client = OpenAI(
            api_key=config.api_key, base_url=config.base_url, timeout=config.timeout_s
        )

    def translate_chunk(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self.config.model,
                temperature=self.config.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except (RateLimitError, APITimeoutError, APIConnectionError) as exc:
            raise LLMRequestError(str(exc), retriable=True) from exc
        except APIStatusError as exc:
            raise LLMRequestError(str(exc), retriable=exc.status_code >= 500) from exc
        return response.choices[0].message.content
