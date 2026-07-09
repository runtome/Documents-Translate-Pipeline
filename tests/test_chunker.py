from doctranslate.chunker import TAG_OVERHEAD, build_chunks, estimate_tokens
from doctranslate.models import Segment


def make_segment(text: str, group_key: str = "body", order_hint: int = 0) -> Segment:
    return Segment(doc_type="docx", source_text=text, group_key=group_key, order_hint=order_hint)


def test_estimate_tokens_nonzero_for_nonempty_text():
    assert estimate_tokens("hello world") > 0


def test_build_chunks_packs_small_segments_together():
    segments = [make_segment("short", order_hint=i) for i in range(5)]
    chunks = build_chunks(segments, token_budget=1000)
    assert len(chunks) == 1
    assert len(chunks[0]) == 5


def test_build_chunks_never_splits_a_segment_across_chunks():
    segments = [make_segment("word " * 50, order_hint=i) for i in range(10)]
    small_budget = estimate_tokens("word " * 50) + TAG_OVERHEAD + 5
    chunks = build_chunks(segments, token_budget=small_budget)

    all_ids = [seg.id for chunk in chunks for seg in chunk]
    assert all_ids == [seg.id for seg in segments]
    for chunk in chunks:
        assert len(chunk) <= 2


def test_build_chunks_oversized_segment_gets_its_own_chunk():
    huge = make_segment("word " * 5000, order_hint=0)
    normal = make_segment("short", order_hint=1)
    chunks = build_chunks([huge, normal], token_budget=100)

    assert any(chunk == [huge] for chunk in chunks)


def test_build_chunks_caps_segment_count_even_within_token_budget():
    segments = [make_segment("short", order_hint=i) for i in range(100)]
    chunks = build_chunks(segments, token_budget=100_000, max_segments=40)

    assert len(chunks) == 3
    assert [len(chunk) for chunk in chunks] == [40, 40, 20]
