from typing import Any

from ..models import ExtractionResult


def apply_translations(
    extraction_result: ExtractionResult, translations: dict[str, str], **_context: Any
) -> None:
    for seg_id, cell in extraction_result.refs.items():
        if seg_id not in translations:
            continue
        cell.value = translations[seg_id]
