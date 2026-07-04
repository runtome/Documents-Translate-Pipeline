import httpx
import pytest

from doctranslate.exceptions import LLMRequestError
from doctranslate.llm.anthropic_client import AnthropicClient
from doctranslate.llm.base import LLMConfig
from doctranslate.llm.factory import get_client
from doctranslate.llm.ollama_client import OllamaClient
from doctranslate.llm.openai_client import OpenAIClient
from doctranslate.llm.thaillm_client import ThaiLLMClient


def _fake_response(status_code: int) -> httpx.Response:
    request = httpx.Request("POST", "https://example.invalid/v1/chat")
    return httpx.Response(status_code=status_code, request=request)


def test_factory_dispatches_to_registered_providers():
    assert isinstance(get_client(LLMConfig(provider="ollama", model="llama3.2")), OllamaClient)
    assert isinstance(
        get_client(LLMConfig(provider="openai", model="gpt-4o-mini", api_key="test")), OpenAIClient
    )
    assert isinstance(
        get_client(LLMConfig(provider="anthropic", model="claude-sonnet-5", api_key="test")),
        AnthropicClient,
    )
    assert isinstance(
        get_client(LLMConfig(provider="thaillm", model="thaillm-default", api_key="test")),
        ThaiLLMClient,
    )


def test_factory_raises_on_unknown_provider():
    with pytest.raises(ValueError, match="unknown provider"):
        get_client(LLMConfig(provider="not-a-provider", model="x"))


def test_openai_client_returns_message_content(mocker):
    client = OpenAIClient(LLMConfig(provider="openai", model="gpt-4o-mini", api_key="test"))
    fake_response = mocker.Mock()
    fake_response.choices = [mocker.Mock(message=mocker.Mock(content="translated text"))]
    mocker.patch.object(client._client.chat.completions, "create", return_value=fake_response)

    assert client.translate_chunk("system", "user") == "translated text"


def test_openai_client_wraps_rate_limit_as_retriable(mocker):
    import openai

    client = OpenAIClient(LLMConfig(provider="openai", model="gpt-4o-mini", api_key="test"))
    error = openai.RateLimitError("rate limited", response=_fake_response(429), body=None)
    mocker.patch.object(client._client.chat.completions, "create", side_effect=error)

    with pytest.raises(LLMRequestError) as exc_info:
        client.translate_chunk("system", "user")
    assert exc_info.value.retriable is True


def test_openai_client_wraps_bad_request_as_non_retriable(mocker):
    import openai

    client = OpenAIClient(LLMConfig(provider="openai", model="gpt-4o-mini", api_key="test"))
    error = openai.BadRequestError("bad request", response=_fake_response(400), body=None)
    mocker.patch.object(client._client.chat.completions, "create", side_effect=error)

    with pytest.raises(LLMRequestError) as exc_info:
        client.translate_chunk("system", "user")
    assert exc_info.value.retriable is False


def test_anthropic_client_returns_first_content_block_text(mocker):
    client = AnthropicClient(LLMConfig(provider="anthropic", model="claude-sonnet-5", api_key="test"))
    fake_block = mocker.Mock(text="translated text")
    fake_response = mocker.Mock(content=[fake_block])
    mocker.patch.object(client._client.messages, "create", return_value=fake_response)

    assert client.translate_chunk("system", "user") == "translated text"


def test_anthropic_client_wraps_overloaded_error_as_retriable(mocker):
    import anthropic

    client = AnthropicClient(LLMConfig(provider="anthropic", model="claude-sonnet-5", api_key="test"))
    error = anthropic.InternalServerError("overloaded", response=_fake_response(529), body=None)
    mocker.patch.object(client._client.messages, "create", side_effect=error)

    with pytest.raises(LLMRequestError) as exc_info:
        client.translate_chunk("system", "user")
    assert exc_info.value.retriable is True


def _thaillm_config(**extra) -> LLMConfig:
    return LLMConfig(
        provider="thaillm",
        model="thaillm-default",
        api_key="secret-key",
        base_url="https://thaillm.or.th",
        extra=extra,
    )


def test_thaillm_client_parses_default_response_path(mocker):
    client = ThaiLLMClient(_thaillm_config())
    fake_response = mocker.Mock()
    fake_response.json.return_value = {"choices": [{"message": {"content": "translated text"}}]}
    mocker.patch.object(client._client, "post", return_value=fake_response)

    assert client.translate_chunk("system", "user") == "translated text"


def test_thaillm_client_sends_bearer_auth_header(mocker):
    client = ThaiLLMClient(_thaillm_config())
    fake_response = mocker.Mock()
    fake_response.json.return_value = {"choices": [{"message": {"content": "x"}}]}
    post_mock = mocker.patch.object(client._client, "post", return_value=fake_response)

    client.translate_chunk("system", "user")

    _, kwargs = post_mock.call_args
    assert kwargs["headers"]["Authorization"] == "Bearer secret-key"


def test_thaillm_client_supports_custom_response_path(mocker):
    client = ThaiLLMClient(_thaillm_config(response_path=["result", "text"]))
    fake_response = mocker.Mock()
    fake_response.json.return_value = {"result": {"text": "translated via custom schema"}}
    mocker.patch.object(client._client, "post", return_value=fake_response)

    assert client.translate_chunk("system", "user") == "translated via custom schema"


def test_thaillm_client_wraps_5xx_as_retriable(mocker):
    client = ThaiLLMClient(_thaillm_config())
    request = httpx.Request("POST", "https://thaillm.or.th/v1/chat/completions")
    response = httpx.Response(status_code=503, request=request)
    error = httpx.HTTPStatusError("server error", request=request, response=response)
    fake_response = mocker.Mock()
    fake_response.raise_for_status.side_effect = error
    mocker.patch.object(client._client, "post", return_value=fake_response)

    with pytest.raises(LLMRequestError) as exc_info:
        client.translate_chunk("system", "user")
    assert exc_info.value.retriable is True


def test_thaillm_client_wraps_4xx_as_non_retriable(mocker):
    client = ThaiLLMClient(_thaillm_config())
    request = httpx.Request("POST", "https://thaillm.or.th/v1/chat/completions")
    response = httpx.Response(status_code=400, request=request)
    error = httpx.HTTPStatusError("bad request", request=request, response=response)
    fake_response = mocker.Mock()
    fake_response.raise_for_status.side_effect = error
    mocker.patch.object(client._client, "post", return_value=fake_response)

    with pytest.raises(LLMRequestError) as exc_info:
        client.translate_chunk("system", "user")
    assert exc_info.value.retriable is False
