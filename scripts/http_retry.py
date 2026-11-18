#!/usr/bin/env python3
"""Bounded full-jitter retry helpers for the Dialpad HTTP clients.

Reads are idempotent and get bounded retries; the send path is not idempotent
and retries only failures that provably happened before the request was
accepted (the SAFE-RETRY rule in :func:`classify_send_stderr` and
:func:`send_url_error_safe`). Sleep, monotonic, and randomness are injected so
tests stay instant.
"""

from __future__ import annotations

import email.utils
import os
import random
import re
import socket
import time
from datetime import datetime, timezone
from typing import Any, Callable, NamedTuple

RETRIES_ENV = "DIALPAD_HTTP_RETRIES"
MAX_SLEEP_ENV = "DIALPAD_HTTP_RETRY_MAX_SLEEP"

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BASE_SECONDS = 0.5
DEFAULT_MAX_SLEEP_SECONDS = 8.0
DEFAULT_BUDGET_SECONDS = 20.0


def configured_max_attempts(env: dict[str, str] | None = None) -> int:
    raw = (os.environ if env is None else env).get(RETRIES_ENV, "").strip()
    try:
        value = int(raw) if raw else DEFAULT_MAX_ATTEMPTS
    except ValueError:
        return DEFAULT_MAX_ATTEMPTS
    return max(1, value)


def configured_max_sleep(env: dict[str, str] | None = None) -> float:
    raw = (os.environ if env is None else env).get(MAX_SLEEP_ENV, "").strip()
    try:
        value = float(raw) if raw else DEFAULT_MAX_SLEEP_SECONDS
    except ValueError:
        return DEFAULT_MAX_SLEEP_SECONDS
    return value if value > 0 else DEFAULT_MAX_SLEEP_SECONDS


def parse_retry_after(value: str | None, *, now: float | None = None) -> float | None:
    """Seconds from a Retry-After value (delta-seconds or HTTP-date), else None."""
    if not value:
        return None
    text = str(value).strip()
    if text.isdigit():
        return float(text)
    try:
        parsed = email.utils.parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    reference = datetime.fromtimestamp(now if now is not None else time.time(), tz=timezone.utc)
    return max(0.0, (parsed - reference).total_seconds())


def retry_after_from_headers(headers: Any) -> float | None:
    if headers is None:
        return None
    try:
        value = headers.get("Retry-After")
    except AttributeError:
        return None
    return parse_retry_after(value)


_RETRY_AFTER_TEXT_RE = re.compile(r"retry-after[\"']?\s*[:=]\s*(\d+(?:\.\d+)?)", re.IGNORECASE)


def retry_after_in_text(text: str) -> float | None:
    """A Retry-After seconds hint embedded in upstream error text, if visible."""
    match = _RETRY_AFTER_TEXT_RE.search(text or "")
    return float(match.group(1)) if match else None


class RetryDecision(NamedTuple):
    """How long to wait before the next attempt; retry_after=None means backoff."""

    retry_after: float | None = None


def _default_sleep(seconds: float) -> None:
    time.sleep(seconds)


def backoff_seconds(
    attempt_index: int,
    *,
    base: float = DEFAULT_BASE_SECONDS,
    cap: float = DEFAULT_MAX_SLEEP_SECONDS,
    random_fn: Callable[[], float] | None = None,
) -> float:
    """Full-jitter exponential backoff: uniform(0, min(cap, base * 2**attempt))."""
    bound = min(cap, base * (2**attempt_index))
    draw = random_fn() if random_fn is not None else random.random()
    return bound * max(0.0, min(1.0, draw))


def run(
    operation: Callable[[], Any],
    *,
    retry_decision: Callable[[BaseException], RetryDecision | None],
    attempts: int | None = None,
    budget_seconds: float | None = DEFAULT_BUDGET_SECONDS,
    base_seconds: float = DEFAULT_BASE_SECONDS,
    max_sleep_seconds: float | None = None,
    sleep: Callable[[float], None] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    random_fn: Callable[[], float] | None = None,
) -> Any:
    """Run ``operation`` until it succeeds or ``retry_decision`` returns None.

    ``retry_decision(exc)`` returns a RetryDecision to retry (carrying an
    optional Retry-After wait, capped at the max sleep) or None for a permanent
    failure that propagates immediately. The last failure propagates once
    ``attempts`` or ``budget_seconds`` is exhausted.
    """
    sleep_fn = sleep if sleep is not None else _default_sleep
    max_attempts = attempts if attempts is not None else configured_max_attempts()
    cap = max_sleep_seconds if max_sleep_seconds is not None else configured_max_sleep()
    deadline = None if budget_seconds is None else monotonic() + budget_seconds
    attempt_index = 0
    while True:
        try:
            return operation()
        except Exception as exc:
            decision = retry_decision(exc)
            attempt_index += 1
            if decision is None or attempt_index >= max_attempts:
                raise
            if decision.retry_after is not None:
                delay = min(cap, max(0.0, decision.retry_after))
            else:
                delay = backoff_seconds(
                    attempt_index - 1, base=base_seconds, cap=cap, random_fn=random_fn
                )
            if deadline is not None and monotonic() + delay > deadline:
                raise
            sleep_fn(delay)


# --------------------------------------------------------------------------
# SAFE-RETRY classification for sends.
#
# SMS send is not idempotent: a retry is only allowed when the request
# provably never reached Dialpad. The safe markers are connect-stage failures
# (DNS, refused connection, connect timeout, NewConnectionError) or an HTTP
# 429, which rejects the request before accepting it. Anything after request
# bytes may have gone out — read timeout, reset/disconnect mid-flight, any
# 5xx — is outcome_unknown: Dialpad may have queued the message, so a blind
# retry could double-send. A 4xx other than 429 is permanent: the server
# rejected the request, so it was definitively not queued.
# --------------------------------------------------------------------------

SEND_SAFE_RETRY = "safe_retry"
SEND_PERMANENT = "permanent"
SEND_OUTCOME_UNKNOWN = "outcome_unknown"

_SEND_SAFE_MARKERS = (
    "429 client error",
    "429 too many requests",
    "newconnectionerror",
    "connecttimeouterror",
    "connect timeout",
    "failed to establish a new connection",
    "connection refused",
    "errno 111",
    "name or service not known",
    "nameresolutionerror",
    "failed to resolve",
    "temporary failure in name resolution",
)

_SEND_UNKNOWN_MARKERS = (
    "read timed out",
    "readtimeout",
    "remotedisconnected",
    "remote end closed connection",
    "connectionreseterror",
    "connection reset",
    "connection aborted",
    "brokenpipeerror",
    "broken pipe",
    "server error",
)

_STATUS_RE = re.compile(r"(\d{3})\s+(?:client|server) error", re.IGNORECASE)


def classify_send_stderr(text: str) -> str:
    """Classify a generated-CLI send failure under the SAFE-RETRY rule.

    The generated CLI prints ``Error: <requests exception>`` on stderr, so the
    classifier works on requests/urllib3 exception text:

    - ``safe_retry``: provably not accepted — HTTP 429 or a connect-stage
      failure before any request bytes were sent.
    - ``permanent``: the server rejected the request with a non-429 4xx — it
      was seen and refused, so it is definitively not queued.
    - ``outcome_unknown``: anything else. Unrecognized failures stay here on
      purpose: a send failure that cannot be classified never claims the
      message was not sent.
    """
    lowered = (text or "").lower()
    for marker in _SEND_SAFE_MARKERS:
        if marker in lowered:
            return SEND_SAFE_RETRY
    for marker in _SEND_UNKNOWN_MARKERS:
        if marker in lowered:
            return SEND_OUTCOME_UNKNOWN
    status_match = _STATUS_RE.search(lowered)
    if status_match:
        code = int(status_match.group(1))
        if code == 429:
            return SEND_SAFE_RETRY
        if 400 <= code < 500:
            return SEND_PERMANENT
        return SEND_OUTCOME_UNKNOWN
    return SEND_OUTCOME_UNKNOWN


def stderr_is_transient(text: str) -> bool:
    """Whether generated-CLI stderr looks like a transient upstream failure.

    Used for non-send generated commands, which may retry on any retryable
    failure: every connect-stage and post-connect transient marker, any 429,
    and any 5xx. 4xx and unrecognized text do not retry.
    """
    lowered = (text or "").lower()
    if any(marker in lowered for marker in _SEND_SAFE_MARKERS + _SEND_UNKNOWN_MARKERS):
        return True
    status_match = _STATUS_RE.search(lowered)
    return bool(status_match and int(status_match.group(1)) >= 500)


def send_url_error_safe(error: BaseException) -> bool:
    """Whether a urllib ``URLError`` proves the request never left the client.

    Only DNS failures (``socket.gaierror``) and refused connections qualify —
    they happen before any request bytes are sent. A bare timeout or reset is
    deliberately excluded: urllib cannot pin it to the connect stage, so for a
    non-idempotent send it stays unsafe.
    """
    reason = getattr(error, "reason", None)
    return isinstance(reason, (socket.gaierror, ConnectionRefusedError))
