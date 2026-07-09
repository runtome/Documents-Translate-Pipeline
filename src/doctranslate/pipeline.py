import logging
from pathlib import Path
from typing import Optional

from . import exporter
from .chunker import MAX_SEGMENTS_PER_CHUNK, build_chunks, estimate_tokens
from .exceptions import ChunkProcessingError, ChunkValidationError
from .glossary import build_glossary_block
from .llm.base import LLMClient
from .models import Segment
from .prompts import build_system_prompt, build_user_prompt
from .replacer import docx_replacer, pdf_replacer, pptx_replacer, xlsx_replacer
from .retry import translate_chunk_with_retry
from .segmenter import docx_segmenter, pdf_segmenter, pptx_segmenter, xlsx_segmenter

logger = logging.getLogger(__name__)

SEGMENTERS = {
    "docx": docx_segmenter,
    "pptx": pptx_segmenter,
    "xlsx": xlsx_segmenter,
    "pdf": pdf_segmenter,
}
REPLACERS = {
    "docx": docx_replacer,
    "pptx": pptx_replacer,
    "xlsx": xlsx_replacer,
    "pdf": pdf_replacer,
}


def detect_doc_type(path: str) -> str:
    ext = Path(path).suffix.lower().lstrip(".")
    if ext not in SEGMENTERS:
        supported = ", ".join(sorted(SEGMENTERS))
        raise ValueError(f"unsupported file type '.{ext}' (supported: {supported})")
    return ext


def _translate_segments(
    client: LLMClient,
    source_lang: str,
    target_lang: str,
    glossary: Optional[dict[str, str]],
    segments: list[Segment],
    max_retries: int,
) -> dict[str, str]:
    # Short per-chunk numeric aliases, not the full UUIDs, are what's sent to the
    # LLM: small local models reliably corrupt long hex ids when copying dozens of
    # them verbatim, which fails validation. Aliases only need to be unique within
    # a single request/response round trip.
    aliases = [str(idx) for idx in range(1, len(segments) + 1)]
    alias_to_segment_id = dict(zip(aliases, (seg.id for seg in segments)))

    chunk_text = "\n".join(seg.source_text for seg in segments)
    glossary_block = build_glossary_block(chunk_text, glossary)
    system_prompt = build_system_prompt(source_lang, target_lang, glossary_block)
    user_prompt = build_user_prompt(segments, aliases)

    translated = translate_chunk_with_retry(client, system_prompt, user_prompt, aliases, max_retries=max_retries)
    return {alias_to_segment_id[alias]: text for alias, text in translated.items()}


def _looks_corrupted(translations: dict[str, str]) -> bool:
    """A response with an unclosed `<SEG id="...">` tag is flagged malformed, but
    the regex that parses it can go one of two ways: the unclosed id simply
    produces no match (safe — it just shows up as "missing"), or its content
    swallows a later tag's `<SEG id="...">` opener too before matching the
    later tag's `</SEG>` (unsafe — the earlier id's value is now corrupted and
    the later id vanishes from the response entirely without going "missing").
    A stray `<seg` inside an otherwise-parsed value is the signature of the
    second, unsafe case.
    """
    return any("<seg" in text.lower() for text in translations.values())


def _translate_with_gap_fill(
    client: LLMClient,
    source_lang: str,
    target_lang: str,
    glossary: Optional[dict[str, str]],
    segments: list[Segment],
    max_retries: int,
) -> tuple[dict[str, str], Optional[Exception]]:
    """Translate a group of segments; on a partial miss, retry only the missing ones.

    A chunk that comes back with a handful of ids missing (but none extra, and
    none of the successfully-parsed translations corrupted) shouldn't sacrifice
    the rest of a large, otherwise-successful chunk just to satisfy an
    all-or-nothing retry. This also covers the common small-model failure mode
    of leaving one `<SEG id="...">` tag unclosed (flagged as malformed) while
    every other tag in the same chunk parses cleanly. Retrying a tiny follow-up
    request for just the missing segments is both cheap and, empirically, far
    more reliable for small local models than re-sending the whole chunk again.
    """
    try:
        return _translate_segments(client, source_lang, target_lang, glossary, segments, max_retries), None
    except ChunkValidationError as exc:
        result = exc.validation_result
        eligible_for_gap_fill = (
            result.missing_ids
            and not result.extra_ids
            and len(result.missing_ids) < len(segments)
            and not _looks_corrupted(result.translations)
        )
        if not eligible_for_gap_fill:
            return {}, exc

        segment_by_alias = {str(idx): seg for idx, seg in enumerate(segments, start=1)}
        partial = {segment_by_alias[alias].id: text for alias, text in result.translations.items()}
        missing_segments = [segment_by_alias[alias] for alias in result.missing_ids]

        logger.warning(
            "%d/%d segments missing after retries; retrying just the missing ones",
            len(missing_segments),
            len(segments),
        )
        gap_translations, gap_error = _translate_with_gap_fill(
            client, source_lang, target_lang, glossary, missing_segments, max_retries
        )
        partial.update(gap_translations)
        return partial, gap_error
    except Exception as exc:
        return {}, exc


def run_pipeline(
    input_path: str,
    source_lang: str,
    target_lang: str,
    client: LLMClient,
    *,
    output_path: Optional[str] = None,
    chunk_token_budget: int = 3000,
    max_segments_per_chunk: int = MAX_SEGMENTS_PER_CHUNK,
    max_retries: int = 3,
    on_error: str = "abort",
    output_pattern: str = "{stem}.{target_lang}{ext}",
    font_paths: Optional[dict[str, str]] = None,
    glossary: Optional[dict[str, str]] = None,
) -> Path:
    doc_type = detect_doc_type(input_path)
    extraction = SEGMENTERS[doc_type].extract(input_path)

    chunks = build_chunks(extraction.segments, chunk_token_budget, max_segments_per_chunk)
    logger.info("extracted %d segments into %d chunks", len(extraction.segments), len(chunks))

    translations: dict[str, str] = {}
    for i, chunk in enumerate(chunks):
        partial, error = _translate_with_gap_fill(client, source_lang, target_lang, glossary, chunk, max_retries)
        translations.update(partial)
        if error is not None:
            if on_error == "abort":
                raise ChunkProcessingError(f"chunk {i} failed: {error}") from error
            logger.warning(
                "chunk %d: %d/%d segments could not be translated and are left as source text: %s",
                i,
                len(chunk) - len(partial),
                len(chunk),
                error,
            )

    REPLACERS[doc_type].apply_translations(
        extraction, translations, target_lang=target_lang, font_paths=font_paths
    )

    out_path = exporter.compute_output_path(input_path, target_lang, output_pattern, output_path)
    exporter.save(doc_type, extraction.doc_handle, out_path)
    logger.info("saved translated document to %s", out_path)
    return out_path


def dry_run_stats(
    input_path: str,
    chunk_token_budget: int = 3000,
    max_segments_per_chunk: int = MAX_SEGMENTS_PER_CHUNK,
) -> dict:
    doc_type = detect_doc_type(input_path)
    extraction = SEGMENTERS[doc_type].extract(input_path)
    chunks = build_chunks(extraction.segments, chunk_token_budget, max_segments_per_chunk)
    return {
        "doc_type": doc_type,
        "segment_count": len(extraction.segments),
        "chunk_count": len(chunks),
        "estimated_input_tokens": sum(estimate_tokens(seg.source_text) for seg in extraction.segments),
    }
