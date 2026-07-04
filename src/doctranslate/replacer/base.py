from typing import Any, Protocol

from ..models import ExtractionResult


class Replacer(Protocol):
    def apply_translations(
        self, extraction_result: ExtractionResult, translations: dict[str, str], **context: Any
    ) -> None: ...


def replace_via_dominant_run(refs: dict[str, Any], translations: dict[str, str]) -> None:
    """Write translated text into the dominant (longest) run of each paragraph-like ref.

    Shared by docx/pptx replacers since python-docx and python-pptx paragraphs
    both expose a `.runs` list of run objects with a settable `.text`.
    """
    for seg_id, paragraph in refs.items():
        if seg_id not in translations:
            continue
        text = translations[seg_id]
        runs = paragraph.runs
        if not runs:
            paragraph.add_run().text = text
            continue
        dominant = max(runs, key=lambda r: len(r.text))
        dominant.text = text
        for run in runs:
            if run is not dominant:
                run.text = ""
