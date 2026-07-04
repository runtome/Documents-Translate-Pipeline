"""Terminology-consistency glossary support: a JSON {source: target} term map."""

import json


def load_glossary(path: str) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"glossary file {path} must contain a JSON object of {{source: target}}")
    return {str(k): str(v) for k, v in data.items()}


def build_glossary_block(chunk_text: str, glossary: dict[str, str] | None) -> str:
    if not glossary:
        return ""
    matches = {src: tgt for src, tgt in glossary.items() if src in chunk_text}
    if not matches:
        return ""
    lines = "\n".join(f"- {src} -> {tgt}" for src, tgt in matches.items())
    return f"\nUse this glossary for consistent terminology:\n{lines}"
