import logging

from .models import Segment

TAG_OVERHEAD = 15

logger = logging.getLogger(__name__)

_encoding = None
_encoding_load_attempted = False


def _get_encoding():
    global _encoding, _encoding_load_attempted
    if not _encoding_load_attempted:
        _encoding_load_attempted = True
        try:
            import tiktoken

            _encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _encoding = None
    return _encoding


def estimate_tokens(text: str) -> int:
    encoding = _get_encoding()
    if encoding is not None:
        return len(encoding.encode(text))
    return max(1, len(text) // 3)


def build_chunks(segments: list[Segment], token_budget: int = 3000) -> list[list[Segment]]:
    chunks: list[list[Segment]] = []
    current: list[Segment] = []
    current_tokens = 0

    for seg in segments:
        cost = estimate_tokens(seg.source_text) + TAG_OVERHEAD

        if cost > token_budget:
            if current:
                chunks.append(current)
                current = []
                current_tokens = 0
            logger.warning(
                "segment %s (%d est. tokens) exceeds chunk_token_budget=%d; sending alone",
                seg.id,
                cost,
                token_budget,
            )
            chunks.append([seg])
            continue

        if current and current_tokens + cost > token_budget:
            chunks.append(current)
            current = []
            current_tokens = 0

        current.append(seg)
        current_tokens += cost

    if current:
        chunks.append(current)

    return chunks
