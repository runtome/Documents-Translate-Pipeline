import fitz

from ..models import ExtractionResult, Segment


def extract(path: str) -> ExtractionResult:
    doc = fitz.open(path)
    segments: list[Segment] = []
    refs: dict[str, dict] = {}
    order = 0

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_dict = page.get_text("dict")

        for block in page_dict["blocks"]:
            if block.get("type") != 0:
                continue

            line_texts = []
            spans = []
            for line in block["lines"]:
                line_texts.append("".join(span["text"] for span in line["spans"]))
                spans.extend(line["spans"])
            text = "\n".join(line_texts).strip()
            if not text:
                continue

            dominant_span = max(spans, key=lambda s: len(s["text"]))
            seg = Segment(
                doc_type="pdf",
                source_text=text,
                group_key=f"page{page_index}",
                order_hint=order,
            )
            order += 1
            segments.append(seg)
            refs[seg.id] = {
                "page_index": page_index,
                "bbox": tuple(block["bbox"]),
                "font": dominant_span["font"],
                "size": dominant_span["size"],
                "color": dominant_span["color"],
            }

    return ExtractionResult(doc_handle=doc, segments=segments, refs=refs)
