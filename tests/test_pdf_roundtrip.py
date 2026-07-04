import fitz

from doctranslate.pipeline import run_pipeline


def _normalize_whitespace(text: str) -> str:
    return text.replace("\xa0", " ")


def _make_pdf(path, lines):
    doc = fitz.open()
    page = doc.new_page()
    y = 72
    for line in lines:
        page.insert_text((72, y), line, fontsize=14)
        y += 24
    doc.save(str(path))
    doc.close()


def test_pdf_roundtrip(tmp_path, fake_llm_client):
    src = tmp_path / "sample.pdf"
    _make_pdf(src, ["Hello world", "Second line of text"])

    out_path = run_pipeline(str(src), "en", "th", fake_llm_client)

    assert out_path.exists()
    out_doc = fitz.open(str(out_path))
    assert out_doc.page_count == 1

    page_text = _normalize_whitespace(out_doc[0].get_text())
    assert "[TR]Hello world" in page_text
    assert "[TR]Second line of text" in page_text
    out_doc.close()


def test_pdf_roundtrip_preserves_page_count(tmp_path, fake_llm_client):
    src = tmp_path / "two_pages.pdf"
    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text((72, 72), "Page one text", fontsize=14)
    page2 = doc.new_page()
    page2.insert_text((72, 72), "Page two text", fontsize=14)
    doc.save(str(src))
    doc.close()

    out_path = run_pipeline(str(src), "en", "ja", fake_llm_client)

    out_doc = fitz.open(str(out_path))
    assert out_doc.page_count == 2
    assert "[TR]Page one text" in _normalize_whitespace(out_doc[0].get_text())
    assert "[TR]Page two text" in _normalize_whitespace(out_doc[1].get_text())
    out_doc.close()
