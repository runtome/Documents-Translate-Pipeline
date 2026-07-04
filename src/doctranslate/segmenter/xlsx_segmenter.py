import openpyxl

from ..models import ExtractionResult, Segment


def extract(path: str) -> ExtractionResult:
    wb = openpyxl.load_workbook(path)
    segments: list[Segment] = []
    refs: dict[str, object] = {}
    order = 0

    for sheet in wb.worksheets:
        for row in sheet.iter_rows():
            for cell in row:
                value = cell.value
                if not isinstance(value, str) or not value.strip() or value.startswith("="):
                    continue
                seg = Segment(
                    doc_type="xlsx",
                    source_text=value,
                    group_key=f"sheet:{sheet.title}",
                    order_hint=order,
                )
                order += 1
                segments.append(seg)
                refs[seg.id] = cell

    return ExtractionResult(doc_handle=wb, segments=segments, refs=refs)
