import httpx

from ..exceptions import LLMRequestError
from .base import LLMClient, LLMConfig

DEFAULT_CHAT_PATH = "/v1/chat/completions"
DEFAULT_AUTH_HEADER = "Authorization"
DEFAULT_AUTH_SCHEME = "Bearer "
DEFAULT_RESPONSE_PATH = ["choices", 0, "message", "content"]


class ThaiLLMClient(LLMClient):
    """Adapter for thaillm.or.th.

    # TODO(thaillm): confirm the real request/response schema once thaillm.or.th
    # docs/API access are available. This currently assumes an OpenAI-chat-like
    # contract: POST {base_url}{chat_path} {"model", "messages", "temperature"}
    # -> {"choices": [{"message": {"content": ...}}]}.
    #
    # If the real contract differs, fix it via LLMConfig.extra (config/default.yaml
    # or --config), not by editing pipeline/cli code:
    #   chat_path: request path (default "/v1/chat/completions")
    #   auth_header: header carrying the API key (default "Authorization")
    #   auth_scheme: prefix before the key (default "Bearer ")
    #   response_path: keys/indices walked to reach the translated text
    #     (default ["choices", 0, "message", "content"])
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._chat_path = config.extra.get("chat_path", DEFAULT_CHAT_PATH)
        self._auth_header = config.extra.get("auth_header", DEFAULT_AUTH_HEADER)
        self._auth_scheme = config.extra.get("auth_scheme", DEFAULT_AUTH_SCHEME)
        self._response_path = config.extra.get("response_path", DEFAULT_RESPONSE_PATH)
        self._client = httpx.Client(base_url=config.base_url or "", timeout=config.timeout_s)

    def translate_chunk(self, system_prompt: str, user_prompt: str) -> str:
        headers = {}
        if self.config.api_key:
            headers[self._auth_header] = f"{self._auth_scheme}{self.config.api_key}"

        payload = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        try:
            response = self._client.post(self._chat_path, json=payload, headers=headers)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            raise LLMRequestError(str(exc), retriable=(status == 429 or status >= 500)) from exc
        except httpx.HTTPError as exc:
            raise LLMRequestError(str(exc), retriable=True) from exc

        value = response.json()
        for key in self._response_path:
            value = value[key]
        return value
