from typing import Any

from ..models import ExtractionResult
from .base import replace_via_dominant_run


def apply_translations(
    extraction_result: ExtractionResult, translations: dict[str, str], **_context: Any
) -> None:
    replace_via_dominant_run(extraction_result.refs, translations)
