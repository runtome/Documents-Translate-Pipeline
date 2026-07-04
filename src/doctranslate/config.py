import os
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv

from .llm.base import LLMConfig

DEFAULT_CONFIG_PATH = Path("config/default.yaml")

DEFAULTS: dict[str, Any] = {
    "chunk_token_budget": 3000,
    "max_retries": 3,
    "on_error": "abort",
    "temperature": 0.2,
    "timeout_s": 60.0,
    "output_pattern": "{stem}.{target_lang}{ext}",
    "providers": {
        "ollama": {"base_url": "http://localhost:11434", "model": "llama3.1"},
        "openai": {"model": "gpt-4o-mini"},
        "anthropic": {"model": "claude-sonnet-5"},
        "thaillm": {
            "base_url": "https://thaillm.or.th",
            "chat_path": "/v1/chat/completions",
            "model": "thaillm-default",
        },
    },
    "fonts": {
        "th": "assets/fonts/NotoSansThai-Regular.ttf",
        "ja": "assets/fonts/NotoSansJP-Regular.ttf",
        "en": "assets/fonts/NotoSansJP-Regular.ttf",
    },
}

API_KEY_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "thaillm": "THAILLM_API_KEY",
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_settings(config_path: Optional[str] = None) -> dict[str, Any]:
    load_dotenv()
    settings = dict(DEFAULTS)
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if path.exists():
        with open(path, encoding="utf-8") as f:
            file_settings = yaml.safe_load(f) or {}
        settings = _deep_merge(settings, file_settings)
    return settings


def build_llm_config(
    settings: dict[str, Any],
    provider: str,
    model_override: Optional[str] = None,
    base_url_override: Optional[str] = None,
    temperature_override: Optional[float] = None,
) -> LLMConfig:
    provider_settings = dict(settings.get("providers", {}).get(provider, {}))
    model = model_override or provider_settings.pop("model", None)
    if not model:
        raise ValueError(f"no model configured for provider '{provider}'; pass --model")

    base_url = base_url_override or provider_settings.pop("base_url", None)
    if provider == "ollama" and not base_url:
        base_url = os.environ.get("OLLAMA_HOST")

    api_key = os.environ.get(API_KEY_ENV_VARS.get(provider, ""))

    return LLMConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=(
            temperature_override if temperature_override is not None else settings.get("temperature", 0.2)
        ),
        timeout_s=settings.get("timeout_s", 60.0),
        max_retries=settings.get("max_retries", 3),
        extra=provider_settings,
    )
