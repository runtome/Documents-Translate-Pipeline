from anthropic import Anthropic, APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from ..exceptions import LLMRequestError
from .base import LLMClient, LLMConfig

DEFAULT_MAX_OUTPUT_TOKENS = 8192


class AnthropicClient(LLMClient):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client = Anthropic(
            api_key=config.api_key, base_url=config.base_url, timeout=config.timeout_s
        )
        self._max_tokens = config.extra.get("max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS)

    def translate_chunk(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self._client.messages.create(
                model=self.config.model,
                max_tokens=self._max_tokens,
                temperature=self.config.temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
        except (RateLimitError, APITimeoutError, APIConnectionError) as exc:
            raise LLMRequestError(str(exc), retriable=True) from exc
        except APIStatusError as exc:
            raise LLMRequestError(str(exc), retriable=exc.status_code >= 500) from exc
        return response.content[0].text
