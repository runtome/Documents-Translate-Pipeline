import logging
from datetime import datetime
from pathlib import Path

# httpcore's DEBUG-level logging is raw protocol-level connection tracing
# (send_request_headers.started, receive_response_body.complete, ...), not
# useful application signal — it drowns out our own -v/--verbose segment
# before/after output on the console. httpx's own one-line-per-request INFO
# summary ("HTTP Request: POST ... 200 OK") is unaffected and still shown.
_CONSOLE_NOISY_LOGGERS = ("httpcore", "urllib3")

_LOG_FORMAT = "%(levelname)s %(name)s: %(message)s"
_LOGS_DIR = Path("logs")


class _ExcludeLoggerPrefixes(logging.Filter):
    def __init__(self, prefixes: tuple[str, ...]):
        super().__init__()
        self._prefixes = prefixes

    def filter(self, record: logging.LogRecord) -> bool:
        return not any(record.name == p or record.name.startswith(f"{p}.") for p in self._prefixes)


class _OnlyLoggerPrefix(logging.Filter):
    def __init__(self, prefix: str):
        super().__init__()
        self._prefix = prefix

    def filter(self, record: logging.LogRecord) -> bool:
        return record.name == self._prefix or record.name.startswith(f"{self._prefix}.")


def configure_logging(verbose: bool = False) -> None:
    """Set up console + on-disk logging for a CLI run.

    Every run writes two timestamped files under logs/, regardless of
    --verbose, so a translation can always be debugged after the fact:
      - {timestamp}_all.log: everything, unfiltered — including the raw
        httpcore connection tracing that's deliberately hidden from the
        console, since that noise is still occasionally useful for diagnosing
        a real transport problem.
      - {timestamp}_translate.log: just this package's own logging (segment
        before/after text, chunk/retry/gap-fill progress), with third-party
        library noise filtered out.
    The console keeps its existing, human-friendly behavior: INFO by default,
    DEBUG with -v, always excluding httpcore/urllib3's protocol tracing.
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    formatter = logging.Formatter(_LOG_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(_ExcludeLoggerPrefixes(_CONSOLE_NOISY_LOGGERS))
    root.addHandler(console_handler)

    _LOGS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")

    all_handler = logging.FileHandler(_LOGS_DIR / f"{timestamp}_all.log", encoding="utf-8")
    all_handler.setLevel(logging.DEBUG)
    all_handler.setFormatter(formatter)
    root.addHandler(all_handler)

    translate_handler = logging.FileHandler(_LOGS_DIR / f"{timestamp}_translate.log", encoding="utf-8")
    translate_handler.setLevel(logging.DEBUG)
    translate_handler.setFormatter(formatter)
    translate_handler.addFilter(_OnlyLoggerPrefix("doctranslate"))
    root.addHandler(translate_handler)

    logging.getLogger("doctranslate.logging_utils").debug(
        "logging to %s and %s", all_handler.baseFilename, translate_handler.baseFilename
    )
