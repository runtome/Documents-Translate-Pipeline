from typing import Any

from ..docx_utils import build_field_protection_map, replace_paragraph_text
from ..models import ExtractionResult


def apply_translations(
    extraction_result: ExtractionResult, translations: dict[str, str], **_context: Any
) -> None:
    protection = build_field_protection_map(extraction_result.doc_handle)
    for seg_id, paragraph in extraction_result.refs.items():
        if seg_id in translations:
            replace_paragraph_text(paragraph, translations[seg_id], protection)
