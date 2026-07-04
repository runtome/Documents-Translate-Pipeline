from pathlib import Path
from typing import Any, Optional


def compute_output_path(
    input_path: str,
    target_lang: str,
    pattern: str,
    output_override: Optional[str] = None,
) -> Path:
    if output_override:
        return Path(output_override)
    p = Path(input_path)
    name = pattern.format(stem=p.stem, target_lang=target_lang, ext=p.suffix)
    return p.with_name(name)


def save(doc_type: str, doc_handle: Any, output_path: Path) -> None:
    if doc_type == "pdf":
        doc_handle.save(str(output_path), garbage=4, deflate=True)
    else:
        doc_handle.save(str(output_path))
