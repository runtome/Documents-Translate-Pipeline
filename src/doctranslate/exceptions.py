class LLMRequestError(Exception):
    def __init__(self, message: str, retriable: bool = False):
        super().__init__(message)
        self.retriable = retriable


class ChunkValidationError(Exception):
    def __init__(self, validation_result):
        super().__init__(
            f"missing={validation_result.missing_ids} "
            f"extra={validation_result.extra_ids} "
            f"malformed={validation_result.malformed}"
        )
        self.validation_result = validation_result


class ChunkProcessingError(Exception):
    pass
