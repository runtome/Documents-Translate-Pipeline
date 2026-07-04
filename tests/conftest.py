import re

import pytest

from doctranslate.llm.base import LLMClient, LLMConfig

_SEG_PATTERN = re.compile(r'<SEG id="([^"]+)">(.*?)</SEG>', re.DOTALL)


class FakeLLMClient(LLMClient):
    """Deterministic offline stand-in: echoes each segment prefixed with [TR]."""

    def __init__(self):
        super().__init__(LLMConfig(provider="fake", model="fake"))

    def translate_chunk(self, system_prompt: str, user_prompt: str) -> str:
        parts = [f'<SEG id="{seg_id}">[TR]{text}</SEG>' for seg_id, text in _SEG_PATTERN.findall(user_prompt)]
        return "\n".join(parts)


@pytest.fixture
def fake_llm_client():
    return FakeLLMClient()
