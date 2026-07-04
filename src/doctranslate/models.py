import uuid
from dataclasses import dataclass
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

DocType = Literal["docx", "pptx", "xlsx", "pdf"]


class Segment(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    doc_type: DocType
    source_text: str
    translated_text: Optional[str] = None
    group_key: str
    order_hint: int
    context: Optional[str] = None


@dataclass
class ExtractionResult:
    doc_handle: Any
    segments: list[Segment]
    refs: dict[str, Any]
