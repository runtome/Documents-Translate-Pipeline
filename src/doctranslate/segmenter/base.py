from typing import Protocol

from ..models import ExtractionResult


class Segmenter(Protocol):
    def extract(self, path: str) -> ExtractionResult: ...
