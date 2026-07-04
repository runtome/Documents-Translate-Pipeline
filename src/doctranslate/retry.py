from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from .exceptions import ChunkValidationError, LLMRequestError
from .llm.base import LLMClient
from .validator import validate_response


def _is_retriable(exc: BaseException) -> bool:
    if isinstance(exc, LLMRequestError):
        return exc.retriable
    if isinstance(exc, ChunkValidationError):
        return True
    return False


def translate_chunk_with_retry(
    client: LLMClient,
    system_prompt: str,
    user_prompt: str,
    expected_ids: list[str],
    max_retries: int = 3,
) -> dict[str, str]:
    @retry(
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception(_is_retriable),
        reraise=True,
    )
    def _attempt() -> dict[str, str]:
        response = client.translate_chunk(system_prompt, user_prompt)
        result = validate_response(response, expected_ids)
        if not result.ok:
            raise ChunkValidationError(result)
        return result.translations

    return _attempt()
