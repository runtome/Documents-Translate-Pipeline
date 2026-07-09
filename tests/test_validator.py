from doctranslate.validator import validate_response


def test_validate_response_ok_when_ids_match():
    response = '<SEG id="a">translated a</SEG>\n<SEG id="b">translated b</SEG>'
    result = validate_response(response, ["a", "b"])
    assert result.ok
    assert result.translations == {"a": "translated a", "b": "translated b"}
    assert not result.missing_ids
    assert not result.extra_ids
    assert not result.malformed


def test_validate_response_detects_missing_id():
    response = '<SEG id="a">translated a</SEG>'
    result = validate_response(response, ["a", "b"])
    assert not result.ok
    assert result.missing_ids == ["b"]


def test_validate_response_tolerates_a_harmless_extra_id():
    # An id that wasn't asked for (e.g. a small model inventing one extra
    # numbered entry) has no segment to write back to, so it's simply
    # discarded by callers rather than distrusting an otherwise-complete
    # response - it's recorded for diagnostics but doesn't fail validation.
    response = '<SEG id="a">translated a</SEG>\n<SEG id="c">translated c</SEG>'
    result = validate_response(response, ["a"])
    assert result.ok
    assert result.extra_ids == ["c"]


def test_validate_response_detects_malformed_tag():
    response = '<SEG id="a">translated a</SEG>\n<SEG id="b">unclosed'
    result = validate_response(response, ["a", "b"])
    assert not result.ok
    assert result.malformed
    assert result.missing_ids == ["b"]


def test_validate_response_ok_despite_harmless_trailing_incomplete_fragment():
    # A stray, dangling "<SEG" fragment after all real content (e.g. the model
    # started to emit another tag right as generation stopped) trips the raw
    # tag-count heuristic (malformed=True), but every expected id still parsed
    # cleanly - this must not be treated as a failure.
    response = '<SEG id="a">translated a</SEG>\n<SEG id="b">translated b</SEG>\n<SEG'
    result = validate_response(response, ["a", "b"])
    assert result.ok
    assert result.malformed
    assert not result.missing_ids
    assert not result.extra_ids
    assert not result.corrupted_ids


def test_validate_response_flags_corrupted_id_when_unclosed_tag_swallows_the_next():
    # id "a" is left unclosed, so the non-greedy regex swallows id "b"'s opener
    # into "a"'s content before finding a real closing tag - "a" is present but
    # corrupted, "b" vanishes entirely (shows up as missing).
    response = '<SEG id="a">translated a<SEG id="b">translated b</SEG>'
    result = validate_response(response, ["a", "b"])
    assert not result.ok
    assert result.missing_ids == ["b"]
    assert result.corrupted_ids == ["a"]


def test_validate_response_ignores_corruption_in_a_non_numeric_extra_id():
    # A bogus id that was never asked for (e.g. a model echoing a literal
    # "..." placeholder from the prompt's example format) has nowhere to be
    # written back to regardless of its content, so it shouldn't be flagged as
    # corrupted (that field is reserved for *expected* ids we'd otherwise
    # trust and use) or fail validation for an otherwise-complete response.
    response = '<SEG id="a">translated a</SEG>\n<SEG id="...">junk <SEG more junk</SEG>'
    result = validate_response(response, ["a"])
    assert result.ok
    assert result.corrupted_ids == []
    assert result.extra_ids == ["..."]
