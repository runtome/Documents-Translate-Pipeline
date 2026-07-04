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


def test_validate_response_detects_extra_id():
    response = '<SEG id="a">translated a</SEG>\n<SEG id="c">translated c</SEG>'
    result = validate_response(response, ["a"])
    assert not result.ok
    assert result.extra_ids == ["c"]


def test_validate_response_detects_malformed_tag():
    response = '<SEG id="a">translated a</SEG>\n<SEG id="b">unclosed'
    result = validate_response(response, ["a", "b"])
    assert not result.ok
    assert result.malformed
