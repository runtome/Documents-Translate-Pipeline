import logging
from typing import Any, Optional

import fitz

from ..models import ExtractionResult

logger = logging.getLogger(__name__)

# Default PDF (base-14) fonts have no Thai glyphs and unreliable CJK coverage,
# so th/ja/en targets all get an embedded Unicode font. NotoSansJP also covers
# Latin, so "en" reuses it rather than needing a third font file.
DEFAULT_FONT_PATHS = {
    "th": "assets/fonts/NotoSansThai-Regular.ttf",
    "ja": "assets/fonts/NotoSansJP-Regular.ttf",
    "en": "assets/fonts/NotoSansJP-Regular.ttf",
}
FONT_NAMES = {"th": "notosans-th", "ja": "notosans-jp", "en": "notosans-jp"}

MIN_FONT_SIZE = 5.0
FONT_STEP = 0.5
REDACTION_FILL = (1, 1, 1)  # assumes a white page background; a known v1 limitation


def _fit_font_size(scratch_page, rect: "fitz.Rect", text: str, fontname: str, fontfile: str, start_size: float):
    size = start_size
    while size >= MIN_FONT_SIZE:
        fits = scratch_page.insert_textbox(rect, text, fontname=fontname, fontfile=fontfile, fontsize=size) >= 0
        if fits:
            return size
        size -= FONT_STEP
    return None


def apply_translations(
    extraction_result: ExtractionResult,
    translations: dict[str, str],
    *,
    target_lang: str,
    font_paths: Optional[dict[str, str]] = None,
    **_context: Any,
) -> None:
    doc = extraction_result.doc_handle
    font_name = FONT_NAMES.get(target_lang, FONT_NAMES["en"])
    font_path = str((font_paths or {}).get(target_lang) or DEFAULT_FONT_PATHS.get(target_lang, DEFAULT_FONT_PATHS["en"]))

    by_page: dict[int, list[tuple[str, dict]]] = {}
    for seg_id, ref in extraction_result.refs.items():
        if seg_id not in translations:
            continue
        by_page.setdefault(ref["page_index"], []).append((seg_id, ref))

    scratch_doc = fitz.open()
    scratch_page = scratch_doc.new_page()
    overflow_ids: list[str] = []

    try:
        for page_index, items in by_page.items():
            page = doc[page_index]
            page.insert_font(fontname=font_name, fontfile=font_path)

            for _, ref in items:
                page.add_redact_annot(fitz.Rect(*ref["bbox"]), fill=REDACTION_FILL)
            page.apply_redactions()

            for seg_id, ref in items:
                rect = fitz.Rect(*ref["bbox"])
                text = translations[seg_id]
                color = fitz.sRGB_to_pdf(ref["color"])
                fitted_size = _fit_font_size(scratch_page, rect, text, font_name, font_path, ref["size"])
                if fitted_size is None:
                    overflow_ids.append(seg_id)
                    fitted_size = MIN_FONT_SIZE
                page.insert_textbox(
                    rect, text, fontname=font_name, fontfile=font_path, fontsize=fitted_size, color=color
                )
    finally:
        scratch_doc.close()

    if overflow_ids:
        logger.warning(
            "%d segment(s) did not fit their original bounding box even at the %.1fpt floor: %s",
            len(overflow_ids),
            MIN_FONT_SIZE,
            overflow_ids,
        )
