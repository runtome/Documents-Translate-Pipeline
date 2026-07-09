import logging

# httpcore's DEBUG-level logging is raw protocol-level connection tracing
# (send_request_headers.started, receive_response_body.complete, ...), not
# useful application signal — it drowns out our own -v/--verbose segment
# before/after output, so keep it capped regardless of verbosity. httpx's own
# one-line-per-request INFO summary ("HTTP Request: POST ... 200 OK") stays.
_NOISY_LOGGERS = ("httpcore", "urllib3")


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
