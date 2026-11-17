#!/usr/bin/env python3
"""Read-only Dialpad call API helpers for the team MCP connector.

A small stdlib-only client around the four GET endpoints the connector
uses — ``GET /api/v2/call`` (cursor-paginated list), ``GET /api/v2/call/{id}``,
``GET /api/v2/transcripts/{call_id}``, and
``GET /api/v2/call/{id}/ai_recap`` — plus pure shaping helpers that turn raw
Dialpad payloads into tool responses. Only GET requests are ever made and
the transport is injectable so tests never touch the network. The list
endpoint has no line filter, so callers must scope results themselves.

Call scoping rule (enforced by the tools on every request, never by the
cache): a call belongs to the lines ``{internal_number,
entry_point_target.phone}`` normalized through :func:`call_lines`. A caller
may see a call only when at least one of those lines is in its send-gate
allowed lines and none of them is the support line. Empirically, calls to
the office/sales shared lines that are forwarded to a teammate carry
``internal_number`` = the teammate's direct line and
``entry_point_target.phone`` = the shared line, so a caller holding the
shared-line grant sees them via the entry point, while a call that stays on
a teammate's direct line and never came through a granted shared line is
invisible to the others.

Recording/voicemail URLs are permanent dialpad.com/blob links (no expiry
parameter) that open only for a signed-in Dialpad user with access; they are
not pre-signed. They are only ever returned inside the get_call ``recording``
section — never in list output and never logged.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Mapping

import send_gate
import sms_approval

DIALPAD_API_BASE = "https://dialpad.com/api/v2"
CALLS_TIMEOUT_SECONDS = 12.0
CACHE_TTL_SECONDS = 300.0
CACHE_MAX_ENTRIES = 128
TRANSCRIPT_TEXT_MAX_CHARS = 20000

GET_CALL_SECTIONS = ("transcript", "recording", "recap")
SUMMARY_FORMATS = ("short", "medium", "long", "bullet")

RECORDING_URL_NOTE = (
    "Recording/voicemail URLs are permanent dialpad.com/blob links (no expiry "
    "parameter) that open only for a signed-in Dialpad user with access; "
    "they are not pre-signed."
)

_REASON_FOR_SECTION = {
    "not_found": "not_available",
    "not_permitted": "not_permitted",
    "rate_limited": "rate_limited",
}


class CallApiError(Exception):
    """A Dialpad call endpoint failure, pre-mapped to a public reason.

    ``reason`` is one of ``not_configured``, ``not_found``, ``not_permitted``,
    ``rate_limited``, or ``upstream_error``; ``retryable`` says whether a
    retry may succeed.
    """

    def __init__(
        self,
        reason: str,
        message: str,
        *,
        status: int | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.status = status
        self.retryable = retryable


def _urllib_transport(url: str, headers: Mapping[str, str], timeout: float) -> tuple[int, bytes]:
    """GET-only transport: returns (status, body) for any completed request."""
    request = urllib.request.Request(url, headers=dict(headers), method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(getattr(response, "status", 200) or 200), response.read()
    except urllib.error.HTTPError as exc:
        body = b""
        try:
            body = exc.read()
        except Exception:  # noqa: BLE001 - an unreadable error body loses no detail.
            pass
        return exc.code, body


class _ResponseCache:
    """Thread-safe bounded TTL cache for raw upstream payloads.

    Only raw Dialpad responses are cached — never an allow/refuse decision,
    so a cached call is re-scoped for every caller on every request.
    """

    def __init__(self, ttl: float, max_entries: int, *, now: Any = time.time) -> None:
        self._ttl = ttl
        self._max = max_entries
        self._now = now
        self._lock = threading.Lock()
        self._items: OrderedDict[tuple, tuple[float, Any]] = OrderedDict()

    def get(self, key: tuple) -> Any:
        with self._lock:
            hit = self._items.get(key)
            if hit is None:
                return None
            expires, value = hit
            if expires <= self._now():
                del self._items[key]
                return None
            self._items.move_to_end(key)
            return value

    def set(self, key: tuple, value: Any) -> None:
        with self._lock:
            self._items[key] = (self._now() + self._ttl, value)
            self._items.move_to_end(key)
            while len(self._items) > self._max:
                self._items.popitem(last=False)


class DialpadCallsClient:
    """GET-only client for the call endpoints, with a raw-response TTL cache.

    The API key resolves lazily from ``DIALPAD_API_KEY`` (falling back to
    ``DIALPAD_TOKEN``) on each request, so the client builds fine with no key
    configured and only the call tools then fail with ``not_configured``.
    """

    def __init__(
        self,
        *,
        environ: Mapping[str, str] | None = None,
        transport: Any = None,
        base_url: str = DIALPAD_API_BASE,
        timeout: float = CALLS_TIMEOUT_SECONDS,
        cache_ttl: float = CACHE_TTL_SECONDS,
        cache_max_entries: int = CACHE_MAX_ENTRIES,
        now: Any = time.time,
    ) -> None:
        self._environ = os.environ if environ is None else environ
        self._transport = transport or _urllib_transport
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._cache = _ResponseCache(cache_ttl, cache_max_entries, now=now)

    def _api_key(self) -> str:
        key = (self._environ.get("DIALPAD_API_KEY") or self._environ.get("DIALPAD_TOKEN") or "").strip()
        if not key:
            raise CallApiError(
                "not_configured",
                "DIALPAD_API_KEY is not set; call tools are not configured on this server",
            )
        return key

    def _get(self, path: str, query: Mapping[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        if query:
            url = f"{url}?{urllib.parse.urlencode({k: v for k, v in query.items() if v is not None})}"
        headers = {"Authorization": f"Bearer {self._api_key()}", "Accept": "application/json"}
        try:
            status, raw = self._transport(url, headers, self._timeout)
        except CallApiError:
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise CallApiError(
                "upstream_error", f"dialpad is unreachable: {exc}", retryable=True
            ) from exc
        message = _error_message(raw) or f"dialpad HTTP {status}"
        if status == 429 or (status == 400 and _is_rate_limit_body(raw)):
            raise CallApiError(
                "rate_limited",
                "dialpad rate limit reached; retry shortly",
                status=status,
                retryable=True,
            )
        if status == 404:
            raise CallApiError("not_found", "dialpad resource not found", status=status)
        if status in (401, 403):
            raise CallApiError(
                "not_permitted",
                f"dialpad refused the request ({message}); the API key may lack the required scope",
                status=status,
            )
        if status < 200 or status >= 300:
            raise CallApiError(
                "upstream_error",
                f"dialpad error (HTTP {status}): {message}",
                status=status,
                retryable=status >= 500,
            )
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except (UnicodeDecodeError, ValueError) as exc:
            raise CallApiError("upstream_error", "dialpad returned a malformed body") from exc
        if not isinstance(payload, dict):
            raise CallApiError("upstream_error", "dialpad returned a malformed body")
        return payload

    def _cached(self, key: tuple, fetch: Any) -> dict[str, Any]:
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        payload = fetch()
        self._cache.set(key, payload)
        return payload

    def list_call_page(
        self,
        *,
        started_after: int,
        started_before: int,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """One page of company-wide calls; ``limit`` is not a supported param."""
        query: dict[str, Any] = {
            "started_after": int(started_after),
            "started_before": int(started_before),
        }
        if cursor:
            query["cursor"] = cursor
        payload = self._get("/call", query)
        items = payload.get("items")
        next_cursor = payload.get("cursor")
        return {
            "items": [item for item in items if isinstance(item, dict)] if isinstance(items, list) else [],
            "cursor": str(next_cursor) if next_cursor else None,
        }

    def get_call(self, call_id: str) -> dict[str, Any]:
        return self._cached(("call", str(call_id)), lambda: self._get(f"/call/{call_id}"))

    def get_transcript(self, call_id: str) -> dict[str, Any]:
        return self._cached(("transcript", str(call_id)), lambda: self._get(f"/transcripts/{call_id}"))

    def get_ai_recap(self, call_id: str, *, summary_format: str | None = None) -> dict[str, Any]:
        query = {"summary_format": summary_format} if summary_format else None
        return self._cached(
            ("recap", str(call_id), summary_format or ""),
            lambda: self._get(f"/call/{call_id}/ai_recap", query),
        )


def _error_message(raw: bytes) -> str | None:
    try:
        parsed = json.loads(raw.decode("utf-8")) if raw else {}
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    error = parsed.get("error")
    if isinstance(error, dict):
        for key in ("message", "code"):
            value = error.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    for key in ("message", "error"):
        value = parsed.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _is_rate_limit_body(raw: bytes) -> bool:
    """Dialpad reports rate limiting as HTTP 400 with ``rate_limit_exceeded``."""
    return b"rate_limit_exceeded" in (raw or b"")


# --------------------------------------------------------------------------
# Pure shaping helpers
# --------------------------------------------------------------------------


def _first_str(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _ms_int(value: Any) -> int | None:
    """Epoch-ms fields arrive as JSON strings or floats; parse defensively."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number <= 0:
        return None
    return int(number)


def _iso_ms(value: Any) -> str | None:
    ms = _ms_int(value)
    if ms is None:
        return None
    try:
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    except (OverflowError, OSError, ValueError):
        return None


def _duration_seconds(value: Any) -> int | None:
    ms = _ms_int(value)
    return None if ms is None else int(ms / 1000)


def _normalized(number: Any) -> str | None:
    return sms_approval.normalize_phone_number(number)


def call_lines(call: Mapping[str, Any]) -> set[str]:
    """Normalized line keys this call belongs to.

    ``{internal_number, entry_point_target.phone}`` — the set the scoping
    rule intersects with the caller's allowed lines.
    """
    keys: set[str] = set()
    internal = _normalized(call.get("internal_number"))
    if internal:
        keys.add(internal)
    entry_point = call.get("entry_point_target")
    if isinstance(entry_point, dict):
        entry = _normalized(entry_point.get("phone"))
        if entry:
            keys.add(entry)
    return keys


def has_voicemail(call: Mapping[str, Any]) -> bool:
    if any(call.get(key) for key in ("voicemail_link", "voicemail_share_link", "voicemail_recording_id")):
        return True
    details = call.get("recording_details")
    if isinstance(details, list):
        for detail in details:
            if isinstance(detail, dict) and str(detail.get("recording_type") or "").lower() == "voicemail":
                return True
    return False


def has_recording(call: Mapping[str, Any]) -> bool:
    if call.get("was_recorded"):
        return True
    if isinstance(call.get("recording_details"), list) and call["recording_details"]:
        return True
    if isinstance(call.get("admin_recording_urls"), list) and call["admin_recording_urls"]:
        return True
    return False


def call_status(call: Mapping[str, Any]) -> str:
    """Derive a public status; live data only ever shows raw state ``hangup``."""
    raw = str(call.get("state") or "").strip().lower()
    if raw == "voicemail":
        return "voicemail"
    if raw in {"missed", "abandoned"}:
        return "missed"
    if has_voicemail(call):
        return "voicemail"
    if _ms_int(call.get("date_connected")) is not None:
        return "answered"
    direction = str(call.get("direction") or "").strip().lower()
    return "missed" if direction == "inbound" else "no_answer"


def call_summary(call: Mapping[str, Any], *, line_names: Mapping[str, str] | None = None) -> dict[str, Any]:
    """The per-call shape shared by ``list_calls`` and ``get_call``.

    Never carries recording or voicemail URLs — those only appear in the
    get_call ``recording`` section.
    """
    names = line_names or {}
    internal = send_gate._canonical_sender(call.get("internal_number"))
    entry_point = call.get("entry_point_target")
    entry_line = None
    entry_name = None
    if isinstance(entry_point, dict):
        entry_line = send_gate._canonical_sender(entry_point.get("phone")) if entry_point.get("phone") else None
        entry_name = names.get(entry_line or "") or _first_str(entry_point.get("name"))
    line = internal or entry_line
    contact = call.get("contact")
    if not isinstance(contact, dict):
        contact = {}
    target = call.get("target")
    if not isinstance(target, dict):
        target = {}
    direction = str(call.get("direction") or "").strip().lower() or None
    call_id = call.get("call_id") or call.get("id")
    if isinstance(call_id, (int, float)) and not isinstance(call_id, bool):
        call_id = str(int(call_id))
    elif call_id is not None:
        call_id = str(call_id).strip() or None
    summary: dict[str, Any] = {
        "call_id": call_id,
        "direction": direction,
        "line": line,
        "line_name": names.get(line or ""),
        "external_number": _first_str(call.get("external_number")),
        "contact_name": _first_str(contact.get("name"), call.get("external_display_name")),
        "answered_by": _first_str(target.get("name"), target.get("phone")),
        "started_at": _iso_ms(call.get("date_started")),
        "duration_seconds": _duration_seconds(call.get("duration")),
        "total_duration_seconds": _duration_seconds(call.get("total_duration")),
        "status": call_status(call),
        "raw_state": _first_str(call.get("state")),
        "has_recording": has_recording(call),
        "has_voicemail": has_voicemail(call),
        "was_transferred": bool(call.get("is_transferred") or call.get("was_transferred")),
    }
    if entry_line:
        summary["entry_point_line"] = entry_line
        summary["entry_point_name"] = entry_name
    return summary


def unavailable_section(exc: CallApiError) -> dict[str, Any]:
    """A degraded section for a failed optional fetch."""
    return {
        "available": False,
        "reason": _REASON_FOR_SECTION.get(exc.reason, "upstream_error"),
        "message": str(exc),
        "retryable": exc.retryable,
    }


def transcript_section(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Split ``transcript`` lines from moment/other typed lines."""
    raw_lines = payload.get("lines")
    lines: list[dict[str, Any]] = []
    moments: list[dict[str, Any]] = []
    for item in raw_lines if isinstance(raw_lines, list) else []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "transcript":
            lines.append(
                {
                    "speaker": _first_str(item.get("name")),
                    "time": _first_str(item.get("time")),
                    "text": _first_str(item.get("content")),
                }
            )
        else:
            moments.append(
                {
                    "name": _first_str(item.get("name")),
                    "time": _first_str(item.get("time")),
                    "text": _first_str(item.get("content")),
                    "type": _first_str(item.get("type")),
                }
            )
    text = "\n".join(
        f"{line['speaker']}: {line['text']}" if line.get("speaker") else str(line.get("text"))
        for line in lines
        if line.get("text")
    )
    truncated = len(text) > TRANSCRIPT_TEXT_MAX_CHARS
    if truncated:
        text = text[:TRANSCRIPT_TEXT_MAX_CHARS]
    return {
        "available": True,
        "lines": lines,
        "moments": moments,
        "transcript_text": text,
        "truncated": truncated,
    }


def recording_section(call: Mapping[str, Any]) -> dict[str, Any]:
    """Recording/voicemail URLs from the call payload; get_call output only."""
    urls: list[dict[str, Any]] = []
    details = call.get("recording_details")
    for detail in details if isinstance(details, list) else []:
        if not isinstance(detail, dict):
            continue
        url = _first_str(detail.get("url"))
        if not url:
            continue
        urls.append(
            {
                "type": _first_str(detail.get("recording_type")) or "callrecording",
                "url": url,
                "duration_seconds": _duration_seconds(detail.get("duration")),
                "start_time": _iso_ms(detail.get("start_time")),
            }
        )
    admin_urls = call.get("admin_recording_urls")
    for url in admin_urls if isinstance(admin_urls, list) else []:
        url = _first_str(url)
        if url:
            urls.append(
                {"type": "admincallrecording", "url": url, "duration_seconds": None, "start_time": None}
            )
    for key in ("voicemail_link", "voicemail_share_link"):
        url = _first_str(call.get(key))
        if url:
            urls.append({"type": "voicemail", "url": url, "duration_seconds": None, "start_time": None})
    section: dict[str, Any] = {"available": bool(urls), "urls": urls, "note": RECORDING_URL_NOTE}
    if not urls:
        section.update(
            {
                "reason": "not_available",
                "message": "no recording or voicemail is available for this call",
                "retryable": False,
            }
        )
    return section


def _recap_item(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {}
    return {key: item.get(key) for key in ("content", "format", "edited") if item.get(key) is not None}


def recap_section(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary")
    summary = summary if isinstance(summary, dict) else {}
    action_items = payload.get("action_items")
    purposes = payload.get("purposes")
    disposition = payload.get("disposition")
    return {
        "available": True,
        "summary": {"format": _first_str(summary.get("format")), "content": _first_str(summary.get("content"))},
        "action_items": [_recap_item(item) for item in action_items if isinstance(item, dict)]
        if isinstance(action_items, list)
        else [],
        "purposes": [
            {"category": _first_str(item.get("category")), "content": _first_str(item.get("content"))}
            for item in purposes
            if isinstance(item, dict)
        ]
        if isinstance(purposes, list)
        else [],
        "disposition": _recap_item(disposition) if isinstance(disposition, dict) else None,
    }
