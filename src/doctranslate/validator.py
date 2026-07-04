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
    malformed: bool = False


def validate_response(response_text: str, expected_ids: list[str]) -> ValidationResult:
    raw_count = len(RAW_TAG_PATTERN.findall(response_text))
    matches = SEG_PATTERN.findall(response_text)
    malformed = raw_count != len(matches)

    translations: dict[str, str] = {}
    for seg_id, text in matches:
        translations[seg_id] = text.strip()

    expected = set(expected_ids)
    seen = set(translations.keys())
    missing_ids = sorted(expected - seen)
    extra_ids = sorted(seen - expected)

    ok = not malformed and not missing_ids and not extra_ids

    return ValidationResult(
        ok=ok,
        translations=translations,
        missing_ids=missing_ids,
        extra_ids=extra_ids,
        malformed=malformed,
    )
