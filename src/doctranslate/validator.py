import re
from dataclasses import dataclass, field

SEG_PATTERN = re.compile(r'<SEG id="([^"]+)">(.*?)</SEG>', re.DOTALL)
RAW_TAG_PATTERN = re.compile(r"<SEG\b")


@dataclass
class ValidationResult:
    ok: bool
    translations: dict[str, str]
    missing_ids: list[str] = field(default_factory=list)
    extra_ids: list[str] = field(default_factory=list)
    corrupted_ids: list[str] = field(default_factory=list)
    malformed: bool = False


def validate_response(response_text: str, expected_ids: list[str]) -> ValidationResult:
    raw_count = len(RAW_TAG_PATTERN.findall(response_text))
    matches = SEG_PATTERN.findall(response_text)
    # `malformed` is a coarse heuristic (raw "<SEG" occurrences vs. successfully
    # parsed pairs) kept for diagnostics/logging only. It over-fires on harmless
    # cases like a stray trailing "<SEG" fragment after generation stopped, even
    # when every expected id parsed cleanly — so it no longer gates `ok` below.
    # What actually determines trustworthiness is missing ids plus
    # `corrupted_ids`, computed straight from the parsed content.
    malformed = raw_count != len(matches)

    translations: dict[str, str] = {}
    for seg_id, text in matches:
        translations[seg_id] = text.strip()

    expected = set(expected_ids)
    seen = set(translations.keys())
    missing_ids = sorted(expected - seen)
    # `extra_ids` (ids not asked for — e.g. a small model inventing one extra
    # numbered entry) are recorded for diagnostics only, not treated as a
    # failure: they have no corresponding segment to write back to, so callers
    # simply discard them rather than distrusting an otherwise-complete response.
    extra_ids = sorted(seen - expected)

    # An unclosed <SEG id="..."> tag can swallow a neighboring tag's opener into
    # its own content before finding a real closing tag: the swallowed id then
    # vanishes entirely (shows up as missing) while this id's value is left
    # holding a literal stray "<seg" substring — not safe to trust even though
    # the id itself was "found". Only checked for *expected* ids: a bogus/extra
    # id (e.g. a model echoing a literal "..." placeholder from the prompt's
    # example format) is discarded regardless of its content, so whether it
    # looks corrupted doesn't matter and shouldn't affect `ok`.
    corrupted_ids = sorted(
        seg_id for seg_id, text in translations.items() if seg_id in expected and "<seg" in text.lower()
    )

    ok = not missing_ids and not corrupted_ids

    return ValidationResult(
        ok=ok,
        translations=translations,
        missing_ids=missing_ids,
        extra_ids=extra_ids,
        corrupted_ids=corrupted_ids,
        malformed=malformed,
    )
