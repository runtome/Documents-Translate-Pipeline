from .models import Segment

LANGUAGE_NAMES = {"th": "Thai", "ja": "Japanese", "en": "English"}

SYSTEM_PROMPT_TEMPLATE = """You are a professional document translator.
Translate the text inside each <SEG id="..."> tag from {source_lang} to {target_lang}.

Rules:
- Translate only the text content inside each <SEG> tag.
- Keep every <SEG id="..."> tag and its id attribute EXACTLY as given.
- Return the same number of <SEG> tags, in the same order, with the same ids. Never merge, split, add, omit, or reorder tags.
- Preserve numbers, URLs, and placeholder tokens as-is.
- Do not add commentary, explanations, or any text outside the <SEG> tags.{glossary_block}"""


def language_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code, code)


def build_system_prompt(source_lang: str, target_lang: str, glossary_block: str = "") -> str:
    block = f"\n{glossary_block}" if glossary_block else ""
    return SYSTEM_PROMPT_TEMPLATE.format(
        source_lang=language_name(source_lang),
        target_lang=language_name(target_lang),
        glossary_block=block,
    )


def build_user_prompt(chunk_segments: list[Segment], aliases: list[str]) -> str:
    return "\n".join(
        f'<SEG id="{alias}">{seg.source_text}</SEG>' for alias, seg in zip(aliases, chunk_segments)
    )
