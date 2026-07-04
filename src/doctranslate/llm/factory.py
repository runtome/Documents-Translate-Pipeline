from .anthropic_client import AnthropicClient
from .base import LLMClient, LLMConfig
from .ollama_client import OllamaClient
from .openai_client import OpenAIClient
from .thaillm_client import ThaiLLMClient

_REGISTRY: dict[str, type[LLMClient]] = {
    "ollama": OllamaClient,
    "openai": OpenAIClient,
    "anthropic": AnthropicClient,
    "thaillm": ThaiLLMClient,
}


def get_client(config: LLMConfig) -> LLMClient:
    try:
        client_cls = _REGISTRY[config.provider]
    except KeyError:
        supported = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"unknown provider '{config.provider}' (supported: {supported})") from None
    return client_cls(config)
