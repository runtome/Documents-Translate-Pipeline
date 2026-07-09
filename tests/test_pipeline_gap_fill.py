import re

import pytest
from docx import Document

from doctranslate.exceptions import ChunkProcessingError
from doctranslate.llm.base import LLMClient, LLMConfig
from doctranslate.pipeline import run_pipeline

_SEG_PATTERN = re.compile(r'<SEG id="([^"]+)">(.*?)</SEG>', re.DOTALL)


class DropsOneInBatchClient(LLMClient):
    """Always drops alias "3" when asked to translate more than one segment at once,
    but translates correctly when given a single segment - simulating a small local
    model that reliably fumbles exactly one tag out of a big batch, no matter how many
    times the whole batch is retried, yet handles a lone segment fine.
    """

    def __init__(self):
        super().__init__(LLMConfig(provider="fake", model="fake"))
        self.requests: list[list[str]] = []

    def translate_chunk(self, system_prompt: str, user_prompt: str) -> str:
        matches = _SEG_PATTERN.findall(user_prompt)
        self.requests.append([seg_id for seg_id, _ in matches])
        parts = []
        for seg_id, text in matches:
            if len(matches) > 1 and seg_id == "3":
                continue
            parts.append(f'<SEG id="{seg_id}">[TR]{text}</SEG>')
        return "\n".join(parts)


def test_gap_fill_recovers_single_dropped_segment_without_losing_the_rest(tmp_path):
    src = tmp_path / "sample.docx"
    doc = Document()
    for i in range(5):
        doc.add_paragraph(f"Paragraph {i}")
    doc.save(src)

    client = DropsOneInBatchClient()
    out_path = run_pipeline(str(src), "en", "th", client, max_retries=3)

    out_doc = Document(str(out_path))
    out_texts = [p.text for p in out_doc.paragraphs if p.text.strip()]
    assert out_texts == [f"[TR]Paragraph {i}" for i in range(5)]

    # first request covers all 5, then retries the full batch, then a final
    # single-segment gap-fill request recovers the one that kept getting dropped
    assert len(client.requests) >= 2
    assert client.requests[-1] == ["1"]


class AlwaysDropsOneOfTwoClient(LLMClient):
    """Even the gap-fill sub-chunk still drops one id - the missing segment should be
    left untranslated (not silently invented), and on-error=skip should keep the rest.
    """

    def __init__(self):
        super().__init__(LLMConfig(provider="fake", model="fake"))

    def translate_chunk(self, system_prompt: str, user_prompt: str) -> str:
        matches = _SEG_PATTERN.findall(user_prompt)
        parts = [f'<SEG id="{seg_id}">[TR]{text}</SEG>' for seg_id, text in matches[:-1]]
        return "\n".join(parts)


def test_unrecoverable_segment_is_left_untranslated_when_on_error_is_skip(tmp_path):
    src = tmp_path / "sample.docx"
    doc = Document()
    doc.add_paragraph("First paragraph")
    doc.add_paragraph("Second paragraph")
    doc.save(src)

    client = AlwaysDropsOneOfTwoClient()
    out_path = run_pipeline(str(src), "en", "th", client, on_error="skip", max_retries=1)

    out_doc = Document(str(out_path))
    out_texts = [p.text for p in out_doc.paragraphs if p.text.strip()]
    assert out_texts == ["[TR]First paragraph", "Second paragraph"]


class UnclosedTrailingTagClient(LLMClient):
    """Simulates a model that forgets to close the very last <SEG> tag in a
    batch response (e.g. it hit a stop/token limit mid-generation). This is
    flagged malformed (raw "<SEG" count != parsed pairs), but since nothing
    follows the unclosed tag, exactly one id is genuinely absent from the
    parsed translations and nothing else is corrupted - it should still be
    gap-filled. Closes tags correctly when given a single segment.
    """

    def __init__(self):
        super().__init__(LLMConfig(provider="fake", model="fake"))

    def translate_chunk(self, system_prompt: str, user_prompt: str) -> str:
        matches = _SEG_PATTERN.findall(user_prompt)
        parts = []
        for i, (seg_id, text) in enumerate(matches):
            if len(matches) > 1 and i == len(matches) - 1:
                parts.append(f'<SEG id="{seg_id}">[TR]{text}')  # deliberately unclosed
            else:
                parts.append(f'<SEG id="{seg_id}">[TR]{text}</SEG>')
        return "\n".join(parts)


def test_gap_fill_recovers_from_a_malformed_response_with_an_unclosed_trailing_tag(tmp_path):
    src = tmp_path / "sample.docx"
    doc = Document()
    for i in range(5):
        doc.add_paragraph(f"Paragraph {i}")
    doc.save(src)

    client = UnclosedTrailingTagClient()
    out_path = run_pipeline(str(src), "en", "th", client, max_retries=1)

    out_doc = Document(str(out_path))
    out_texts = [p.text for p in out_doc.paragraphs if p.text.strip()]
    assert out_texts == [f"[TR]Paragraph {i}" for i in range(5)]


class CorruptedMergeClient(LLMClient):
    """Leaves a *non-trailing* <SEG> tag unclosed, so the parser's non-greedy
    match swallows the next tag's opener into this segment's content before
    finding a real closing tag - producing a corrupted (not merely missing)
    value. This must NOT be silently gap-filled, since we can't trust which id
    actually owns that content.
    """

    def __init__(self):
        super().__init__(LLMConfig(provider="fake", model="fake"))

    def translate_chunk(self, system_prompt: str, user_prompt: str) -> str:
        matches = _SEG_PATTERN.findall(user_prompt)
        if len(matches) == 1:
            seg_id, text = matches[0]
            return f'<SEG id="{seg_id}">[TR]{text}</SEG>'
        parts = []
        for i, (seg_id, text) in enumerate(matches):
            if i == 0:
                parts.append(f'<SEG id="{seg_id}">[TR]{text}')  # unclosed, not last
            else:
                parts.append(f'<SEG id="{seg_id}">[TR]{text}</SEG>')
        return "\n".join(parts)


def test_corrupted_merge_from_a_non_trailing_unclosed_tag_is_not_gap_filled(tmp_path):
    src = tmp_path / "sample.docx"
    doc = Document()
    for i in range(3):
        doc.add_paragraph(f"Paragraph {i}")
    doc.save(src)

    client = CorruptedMergeClient()

    with pytest.raises(ChunkProcessingError):
        run_pipeline(str(src), "en", "th", client, max_retries=1)
