import logging
from pathlib import Path
from typing import Optional

from . import exporter
from .chunker import build_chunks, estimate_tokens
from .exceptions import ChunkProcessingError
from .glossary import build_glossary_block
from .llm.base import LLMClient
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


def run_pipeline(
    input_path: str,
    source_lang: str,
    target_lang: str,
    client: LLMClient,
    *,
    output_path: Optional[str] = None,
    chunk_token_budget: int = 3000,
    max_retries: int = 3,
    on_error: str = "abort",
    output_pattern: str = "{stem}.{target_lang}{ext}",
    font_paths: Optional[dict[str, str]] = None,
    glossary: Optional[dict[str, str]] = None,
) -> Path:
    doc_type = detect_doc_type(input_path)
    extraction = SEGMENTERS[doc_type].extract(input_path)

    chunks = build_chunks(extraction.segments, chunk_token_budget)
    logger.info("extracted %d segments into %d chunks", len(extraction.segments), len(chunks))

    translations: dict[str, str] = {}
    for i, chunk in enumerate(chunks):
        expected_ids = [seg.id for seg in chunk]
        chunk_text = "\n".join(seg.source_text for seg in chunk)
        glossary_block = build_glossary_block(chunk_text, glossary)
        system_prompt = build_system_prompt(source_lang, target_lang, glossary_block)
        user_prompt = build_user_prompt(chunk)
        try:
            chunk_translations = translate_chunk_with_retry(
                client, system_prompt, user_prompt, expected_ids, max_retries=max_retries
            )
            translations.update(chunk_translations)
        except Exception as exc:
            if on_error == "abort":
                raise ChunkProcessingError(f"chunk {i} failed: {exc}") from exc
            logger.warning("chunk %d failed, leaving %d segments untranslated: %s", i, len(chunk), exc)
            continue

    REPLACERS[doc_type].apply_translations(
        extraction, translations, target_lang=target_lang, font_paths=font_paths
    )

    out_path = exporter.compute_output_path(input_path, target_lang, output_pattern, output_path)
    exporter.save(doc_type, extraction.doc_handle, out_path)
    logger.info("saved translated document to %s", out_path)
    return out_path


def dry_run_stats(input_path: str, chunk_token_budget: int = 3000) -> dict:
    doc_type = detect_doc_type(input_path)
    extraction = SEGMENTERS[doc_type].extract(input_path)
    chunks = build_chunks(extraction.segments, chunk_token_budget)
    return {
        "doc_type": doc_type,
        "segment_count": len(extraction.segments),
        "chunk_count": len(chunks),
        "estimated_input_tokens": sum(estimate_tokens(seg.source_text) for seg in extraction.segments),
    }
