"""Tests for the read-only call tools in scripts/team_mcp_server.py.

All Dialpad access goes through a fake client (or a fake transport for the
real client); nothing here touches the network.
"""

from __future__ import annotations

import hashlib
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(ROOT / "scripts"))

import send_api_server
import sms_sqlite
import team_calls
from team_calls import CallApiError
from team_mcp_server import create_server

SCOTT_LINE = "+14152230323"
LILLA_LINE = "+14158701945"
OFFICE_LINE = "+14159065785"
SALES_LINE = "+14155201316"
SUPPORT_LINE = "+14159917155"
CONTACT = "+14155550111"

GATE_CONFIG = {
    "shared_lines": {"office": OFFICE_LINE, "sales": SALES_LINE},
    "teammates": {"scott": SCOTT_LINE, "lilla": LILLA_LINE},
    "callers": {
        "team-scott": {"shared_lines": ["office", "sales"], "acts_for": "scott"},
        "team-lilla": {"shared_lines": ["office"], "acts_for": "lilla"},
        "operator": {"shared_lines": ["office", "sales"]},
        "service-bot": {"shared_lines": ["office"]},
    },
}

SCOTT_TOKEN = "fake-scott-mcp-token-0123456789abcdef"
LILLA_TOKEN = "fake-lilla-mcp-token-0123456789abcdef"


def _mcp_config(tmp_path: Path, **overrides) -> dict:
    config = {
        "access": {"team_domain": "team.cloudflareaccess.com", "aud": "aud-tag-123"},
        "email_callers": {},
        "allow_bearer_fallback": True,
        "allowed_origins": [],
    }
    config.update(overrides)
    return config


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _map_entries() -> dict:
    return {
        _digest(SCOTT_TOKEN): {"caller": "team-scott", "label": "scott-mcp"},
        _digest(LILLA_TOKEN): {"caller": "team-lilla", "label": "lilla-mcp"},
    }


def _request(server, path: str, *, method="POST", body=None, token=SCOTT_TOKEN):
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        f"http://{server.server_address[0]}:{server.server_address[1]}{path}",
        data=data,
        headers=headers,
        method=method,
    )
    return urllib.request.urlopen(request, timeout=5)


def _tool_call(server, name, arguments=None, **kwargs):
    params = {"name": name}
    if arguments is not None:
        params["arguments"] = arguments
    payload = {"jsonrpc": "2.0", "method": "tools/call", "id": 1, "params": params}
    response = json.load(_request(server, "/mcp", body=payload, **kwargs))
    return response["result"]


def _tool_payload(result) -> dict:
    return json.loads(result["content"][0]["text"])


# ---------------------------------------------------------------------------
# Fake Dialpad client
# ---------------------------------------------------------------------------


def _call(call_id, *, internal=OFFICE_LINE, entry=None, direction="inbound", **fields):
    call = {
        "call_id": str(call_id),
        "direction": direction,
        "internal_number": internal,
        "external_number": CONTACT,
        "state": "hangup",
        "date_started": "1763300000000",
        "duration": 45000.0,
        "total_duration": 90000.0,
    }
    if entry is not None:
        call["entry_point_target"] = {"phone": entry, "name": "entry"}
    call.update(fields)
    return call


class FakeDialpadCalls:
    """Records every request; raises CallApiError for mapped errors.

    ``pages`` maps the request cursor ("" for the first page) to a page so
    repeated calls replay identically for every caller.
    """

    def __init__(self, *, pages=None, calls=None, transcripts=None, recaps=None):
        self.pages = dict(pages or {})
        self.page_requests: list[tuple] = []
        self.calls = {str(key): value for key, value in (calls or {}).items()}
        self.transcripts = dict(transcripts or {})
        self.recaps = dict(recaps or {})
        self.requests: list[tuple] = []

    def list_call_page(self, *, started_after, started_before, cursor=None):
        self.page_requests.append({"after": started_after, "before": started_before, "cursor": cursor})
        return self.pages.get(cursor or "", {"items": [], "cursor": None})

    def _resolve(self, mapping, key):
        value = mapping.get(str(key))
        if isinstance(value, Exception):
            raise value
        if value is None:
            raise CallApiError("not_found", "dialpad resource not found", status=404)
        return value

    def get_call(self, call_id):
        self.requests.append(("call", str(call_id)))
        return self._resolve(self.calls, call_id)

    def get_transcript(self, call_id):
        self.requests.append(("transcript", str(call_id)))
        return self._resolve(self.transcripts, call_id)

    def get_ai_recap(self, call_id, *, summary_format=None):
        self.requests.append(("recap", str(call_id), summary_format))
        return self._resolve(self.recaps, call_id)


@pytest.fixture
def token_map_path(tmp_path: Path) -> Path:
    path = tmp_path / "send-api-tokens.json"
    send_api_server.write_token_map(path, _map_entries())
    return path


@pytest.fixture
def sms_db(tmp_path: Path) -> Path:
    path = tmp_path / "sms.db"
    conn = sms_sqlite.init_db(path)
    conn.close()
    return path


@pytest.fixture
def make_server(tmp_path, token_map_path, sms_db):
    servers: list = []

    def _make(client=None, config=None, env=None, **overrides):
        kwargs = {
            "bind": "127.0.0.1",
            "port": 0,
            "config": config if config is not None else _mcp_config(tmp_path),
            "gate_config": GATE_CONFIG,
            "tokens_path": token_map_path,
            "send_runner": lambda argv, **kw: None,
            "env": env if env is not None else {"DIALPAD_SMS_DB": str(sms_db)},
            "dialpad_client": client if client is not None else FakeDialpadCalls(),
        }
        kwargs.update(overrides)
        server = create_server(**kwargs)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append((server, thread))
        return server

    yield _make
    for server, thread in servers:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


# ---------------------------------------------------------------------------
# tools/list
# ---------------------------------------------------------------------------


def test_tools_list_has_call_tools_and_no_write_tools(make_server) -> None:
    server = make_server()
    payload = {"jsonrpc": "2.0", "method": "tools/list", "id": 1}
    response = json.load(_request(server, "/mcp", body=payload))
    tools = {tool["name"]: tool for tool in response["result"]["tools"]}
    assert set(tools) == {
        "send_sms",
        "list_conversations",
        "read_thread",
        "set_sms_alert_url",
        "list_calls",
        "get_call",
    }
    assert not {"dial_call", "make_call", "transfer_call", "hangup", "assign_call"} & set(tools)
    for tool in tools.values():
        assert tool["inputSchema"]["additionalProperties"] is False
    assert set(tools["list_calls"]["inputSchema"]["properties"]) == {"line", "since", "until", "limit"}
    assert set(tools["get_call"]["inputSchema"]["properties"]) == {"call_id", "include", "summary_format"}
    assert tools["get_call"]["inputSchema"]["required"] == ["call_id"]


# ---------------------------------------------------------------------------
# list_calls
# ---------------------------------------------------------------------------


FORWARDED_TO_SCOTT = _call("100", internal=SCOTT_LINE, entry=SALES_LINE, direction="inbound")
SCOTT_DIRECT = _call("101", internal=SCOTT_LINE, direction="inbound")
OFFICE_CALL = _call("102", internal=OFFICE_LINE, direction="outbound", date_connected="1763300005000")
LILLA_DIRECT = _call("103", internal=LILLA_LINE, direction="inbound")
SUPPORT_INTERNAL = _call("104", internal=SUPPORT_LINE, entry=SALES_LINE, direction="inbound")
SUPPORT_ENTRY = _call("105", internal=SCOTT_LINE, entry=SUPPORT_LINE, direction="inbound")
LILLA_VIA_SALES = _call("106", internal=LILLA_LINE, entry=SALES_LINE, direction="inbound")


def _list_server(make_server, calls) -> tuple:
    client = FakeDialpadCalls(pages={"": {"items": list(calls), "cursor": None}})
    server = make_server(client=client)
    return server, client


def test_list_calls_scoped_to_allowed_lines(make_server) -> None:
    server, _client = _list_server(
        make_server,
        [FORWARDED_TO_SCOTT, SCOTT_DIRECT, OFFICE_CALL, LILLA_DIRECT, SUPPORT_INTERNAL, SUPPORT_ENTRY],
    )
    result = _tool_call(server, "list_calls", {})
    assert result["isError"] is False
    payload = _tool_payload(result)
    ids = {call["call_id"] for call in payload["calls"]}
    # scott sees: his own calls, office calls, and the sales->scott forward.
    assert ids == {"100", "101", "102"}
    assert payload["lines"] == sorted([SCOTT_LINE, OFFICE_LINE, SALES_LINE])


def test_list_calls_support_line_never_visible_via_entry_point(make_server) -> None:
    """A support-line call stays hidden even when its other end is a granted line."""
    server, _client = _list_server(make_server, [SUPPORT_INTERNAL, SUPPORT_ENTRY])
    payload = _tool_payload(_tool_call(server, "list_calls", {}))
    assert payload["calls"] == []
    dumped = json.dumps(payload)
    assert SUPPORT_LINE not in dumped


def test_list_calls_forwarded_call_visibility_by_grant(make_server) -> None:
    """sales->scott forwarded call: scott (sales grant) sees it, lilla does not."""
    server, _client = _list_server(make_server, [FORWARDED_TO_SCOTT, LILLA_VIA_SALES, LILLA_DIRECT])
    scott_ids = {c["call_id"] for c in _tool_payload(_tool_call(server, "list_calls", {}))["calls"]}
    assert scott_ids == {"100", "106"}
    lilla_result = _tool_call(server, "list_calls", {}, token=LILLA_TOKEN)
    lilla_ids = {c["call_id"] for c in _tool_payload(lilla_result)["calls"]}
    # lilla sees the sales->lilla forward via her own line plus her direct call,
    # never the sales->scott forward (no scott line, no sales grant).
    assert lilla_ids == {"106", "103"}


def test_list_calls_line_filter(make_server) -> None:
    server, _client = _list_server(
        make_server, [FORWARDED_TO_SCOTT, SCOTT_DIRECT, OFFICE_CALL, LILLA_DIRECT]
    )
    for line_arg in (SALES_LINE, "sales"):
        payload = _tool_payload(_tool_call(server, "list_calls", {"line": line_arg}))
        assert {c["call_id"] for c in payload["calls"]} == {"100"}
        assert payload["lines"] == [SALES_LINE]
    payload = _tool_payload(_tool_call(server, "list_calls", {"line": "own"}))
    assert {c["call_id"] for c in payload["calls"]} == {"100", "101"}
    result = _tool_call(server, "list_calls", {"line": LILLA_LINE})
    assert _tool_payload(result)["error"]["code"] == "line_not_allowed"
    result = _tool_call(server, "list_calls", {"line": SUPPORT_LINE})
    assert _tool_payload(result)["error"]["code"] == "line_not_allowed"


def test_list_calls_pagination_and_truncation(make_server) -> None:
    pages = {
        "": {"items": [_call("200", internal=LILLA_LINE), SCOTT_DIRECT], "cursor": "c2"},
        "c2": {"items": [FORWARDED_TO_SCOTT], "cursor": "c3"},
        "c3": {"items": [OFFICE_CALL], "cursor": None},
    }
    client = FakeDialpadCalls(pages=pages)
    server = make_server(client=client)
    payload = _tool_payload(_tool_call(server, "list_calls", {"limit": 10}))
    assert {c["call_id"] for c in payload["calls"]} == {"101", "100", "102"}
    assert payload["scanned"] == 4
    assert payload["truncated"] is False
    assert [req["cursor"] for req in client.page_requests] == [None, "c2", "c3"]


def test_list_calls_truncated_when_page_cap_hits(make_server) -> None:
    pages = {
        ("" if i == 0 else f"c{i}"): {
            "items": [_call(str(900 + i), internal=LILLA_LINE)],
            "cursor": f"c{i + 1}",
        }
        for i in range(12)
    }
    client = FakeDialpadCalls(pages=pages)
    server = make_server(client=client)
    payload = _tool_payload(_tool_call(server, "list_calls", {"limit": 5}))
    assert payload["calls"] == []
    assert payload["scanned"] == 12
    assert payload["truncated"] is True
    assert len(client.page_requests) == 12


def test_list_calls_window_and_validation(make_server) -> None:
    server, client = _list_server(make_server, [])
    _tool_call(server, "list_calls", {"since": "2025-11-01T00:00:00Z", "until": 1763300000000})
    assert client.page_requests[0]["after"] == 1761955200000
    assert client.page_requests[0]["before"] == 1763300000000

    for arguments, code in [
        ({"since": "not-a-time"}, "invalid_argument"),
        ({"until": "not-a-time"}, "invalid_argument"),
        ({"since": "2026-11-10T00:00:00Z", "until": "2026-11-01T00:00:00Z"}, "invalid_argument"),
        ({"limit": 0}, "invalid_argument"),
        ({"limit": 101}, "invalid_argument"),
        ({"limit": "many"}, "invalid_argument"),
        ({"caller": "team-lilla"}, "invalid_argument"),
        ({"cursor": "x"}, "invalid_argument"),
        ({"line": 123}, "invalid_argument"),
    ]:
        result = _tool_call(server, "list_calls", arguments)
        assert result["isError"] is True, arguments
        assert _tool_payload(result)["error"]["code"] == code, arguments


def test_list_calls_status_and_flags(make_server) -> None:
    calls = [
        _call("300", internal=SCOTT_LINE, date_connected="1763300005000"),
        _call("301", internal=SCOTT_LINE, direction="inbound"),
        _call("302", internal=SCOTT_LINE, direction="outbound"),
        _call("303", internal=SCOTT_LINE, voicemail_link="https://dialpad.com/blob/abc"),
        _call(
            "304",
            internal=SCOTT_LINE,
            recording_details=[
                {"id": "r1", "recording_type": "voicemail", "url": "https://dialpad.com/blob/vm",
                 "duration": 5000, "start_time": "1763300006000"}
            ],
        ),
        _call("305", internal=SCOTT_LINE, state="abandoned"),
        _call("306", internal=SCOTT_LINE, direction="outbound", state="missed"),
        _call(
            "307",
            internal=SCOTT_LINE,
            date_connected="1763300005000",
            was_recorded=True,
            is_transferred=True,
            contact={"name": "Pat Example", "phone": CONTACT},
            target={"name": "Scott", "phone": SCOTT_LINE},
        ),
    ]
    server, _client = _list_server(make_server, calls)
    payload = _tool_payload(_tool_call(server, "list_calls", {"limit": 20}))
    by_id = {call["call_id"]: call for call in payload["calls"]}
    assert by_id["300"]["status"] == "answered"
    assert by_id["301"]["status"] == "missed"
    assert by_id["302"]["status"] == "no_answer"
    assert by_id["303"]["status"] == "voicemail"
    assert by_id["304"]["status"] == "voicemail"
    assert by_id["305"]["status"] == "missed"
    assert by_id["306"]["status"] == "missed"
    answered = by_id["307"]
    assert answered["has_recording"] is True
    assert answered["was_transferred"] is True
    assert answered["contact_name"] == "Pat Example"
    assert answered["answered_by"] == "Scott"
    assert answered["duration_seconds"] == 45
    assert answered["total_duration_seconds"] == 90
    assert answered["raw_state"] == "hangup"
    forwarded = by_id["300"]
    assert forwarded["line"] == SCOTT_LINE
    assert forwarded["line_name"] == "scott"


def test_list_calls_no_urls_and_line_names(make_server) -> None:
    calls = [
        _call(
            "400",
            internal=SCOTT_LINE,
            entry=SALES_LINE,
            voicemail_link="https://dialpad.com/blob/vm1",
            admin_recording_urls=["https://dialpad.com/blob/admin1"],
            recording_details=[
                {"id": "r2", "recording_type": "callrecording", "url": "https://dialpad.com/blob/rec1"}
            ],
        )
    ]
    server, _client = _list_server(make_server, calls)
    payload = _tool_payload(_tool_call(server, "list_calls", {}))
    entry = payload["calls"][0]
    assert entry["has_recording"] is True
    assert entry["has_voicemail"] is True
    assert entry["entry_point_line"] == SALES_LINE
    assert entry["entry_point_name"] == "sales"
    assert "dialpad.com" not in json.dumps(payload)


# ---------------------------------------------------------------------------
# get_call
# ---------------------------------------------------------------------------

ALLOWED_CALL = _call(
    "500",
    internal=SCOTT_LINE,
    entry=SALES_LINE,
    date_connected="1763300005000",
    was_recorded=True,
    contact={"name": "Pat Example", "phone": CONTACT},
    target={"name": "Scott", "phone": SCOTT_LINE},
    recording_details=[
        {
            "id": "r1",
            "recording_type": "callrecording",
            "url": "https://dialpad.com/blob/rec1",
            "duration": 45000,
            "start_time": "1763300001000",
        },
    ],
    admin_recording_urls=["https://dialpad.com/blob/admin1"],
)

VOICEMAIL_CALL = _call(
    "502",
    internal=SCOTT_LINE,
    voicemail_share_link="https://dialpad.com/blob/vm1",
)

TRANSCRIPT = {
    "call_id": "500",
    "lines": [
        {"content": "hello", "name": "Scott", "time": "2026-11-16T12:00:01Z", "type": "transcript", "user_id": "1"},
        {"content": "hi there", "name": "Pat", "time": "2026-11-16T12:00:04Z", "type": "transcript", "user_id": "2"},
        {"content": "price discussed", "name": "Pricing", "time": "2026-11-16T12:01:00Z", "type": "moment", "user_id": None},
        {"content": "asked about refund", "name": "Refund", "time": "2026-11-16T12:02:00Z", "type": "custom_moment", "user_id": None},
    ],
}

RECAP = {
    "call_id": "500",
    "summary": {"content": "short summary", "format": "short", "edited": False},
    "action_items": [{"content": "send the quote", "format": "bullet", "edited": False}],
    "purposes": [{"category": "sales", "content": "pricing question"}],
    "disposition": {"content": "resolved", "format": "short"},
}


def _get_server(make_server, *, calls=None, transcripts=None, recaps=None) -> tuple:
    client = FakeDialpadCalls(calls=calls, transcripts=transcripts, recaps=recaps)
    server = make_server(client=client)
    return server, client


def test_get_call_happy_path(make_server) -> None:
    server, client = _get_server(
        make_server, calls={"500": ALLOWED_CALL}, transcripts={"500": TRANSCRIPT}, recaps={"500": RECAP}
    )
    result = _tool_call(server, "get_call", {"call_id": "500"})
    assert result["isError"] is False
    payload = _tool_payload(result)

    summary = payload["call"]
    assert summary["call_id"] == "500"
    assert summary["status"] == "answered"
    assert summary["line"] == SCOTT_LINE
    assert summary["entry_point_line"] == SALES_LINE
    assert "dialpad.com" not in json.dumps(summary)

    transcript = payload["transcript"]
    assert transcript["available"] is True
    assert [line["text"] for line in transcript["lines"]] == ["hello", "hi there"]
    assert transcript["lines"][0]["speaker"] == "Scott"
    assert {moment["type"] for moment in transcript["moments"]} == {"moment", "custom_moment"}
    assert "Scott: hello" in transcript["transcript_text"]
    assert transcript["truncated"] is False

    recording = payload["recording"]
    assert recording["available"] is True
    types = {url["type"] for url in recording["urls"]}
    assert types == {"callrecording", "admincallrecording"}
    rec = next(url for url in recording["urls"] if url["type"] == "callrecording")
    assert rec["url"] == "https://dialpad.com/blob/rec1"
    assert rec["duration_seconds"] == 45
    assert rec["start_time"] == "2025-11-16T13:33:21Z"
    assert "not pre-signed" in recording["note"]

    recap = payload["recap"]
    assert recap["available"] is True
    assert recap["summary"] == {"format": "short", "content": "short summary"}
    assert recap["action_items"] == [{"content": "send the quote", "format": "bullet", "edited": False}]
    assert recap["purposes"] == [{"category": "sales", "content": "pricing question"}]
    assert recap["disposition"]["content"] == "resolved"
    assert ("call", "500") in client.requests
    assert ("transcript", "500") in client.requests
    assert ("recap", "500", None) in client.requests


def test_get_call_voicemail_url_type(make_server) -> None:
    server, _client = _get_server(make_server, calls={"502": VOICEMAIL_CALL})
    payload = _tool_payload(_tool_call(server, "get_call", {"call_id": "502", "include": ["recording"]}))
    assert payload["call"]["status"] == "voicemail"
    urls = payload["recording"]["urls"]
    assert urls == [
        {
            "type": "voicemail",
            "url": "https://dialpad.com/blob/vm1",
            "duration_seconds": None,
            "start_time": None,
        }
    ]


def test_get_call_include_subset_and_format(make_server) -> None:
    server, client = _get_server(
        make_server, calls={"500": ALLOWED_CALL}, transcripts={"500": TRANSCRIPT}, recaps={"500": RECAP}
    )
    payload = _tool_payload(
        _tool_call(server, "get_call", {"call_id": 500, "include": ["recap"], "summary_format": "bullet"})
    )
    assert set(payload) == {"call", "recap"}
    assert ("recap", "500", "bullet") in client.requests
    assert ("transcript", "500") not in client.requests


def test_get_call_disallowed_and_missing_are_identical(make_server) -> None:
    """Same generic refusal for a non-allowed line and a nonexistent id, and no
    section fetches happen for a refused call."""
    lilla_call = _call("600", internal=LILLA_LINE)
    support_call = _call("601", internal=SCOTT_LINE, entry=SUPPORT_LINE)
    server, client = _get_server(make_server, calls={"600": lilla_call, "601": support_call})

    refused = _tool_call(server, "get_call", {"call_id": "600"})
    missing = _tool_call(server, "get_call", {"call_id": "999"})
    support = _tool_call(server, "get_call", {"call_id": "601"})
    for result in (refused, missing, support):
        assert result["isError"] is True
    assert _tool_payload(refused) == _tool_payload(missing) == _tool_payload(support)
    assert _tool_payload(refused)["error"]["code"] == "call_not_found"
    # Only the call lookup ran — never a transcript/recap fetch for refused calls.
    assert ("transcript", "600") not in client.requests
    assert ("recap", "600", None) not in client.requests
    assert ("transcript", "601") not in client.requests


def test_get_call_section_degradation(make_server) -> None:
    server, _client = _get_server(
        make_server,
        calls={"500": ALLOWED_CALL},
        transcripts={"500": CallApiError("not_found", "dialpad resource not found", status=404)},
        recaps={"500": CallApiError("not_permitted", "scope missing", status=403)},
    )
    payload = _tool_payload(_tool_call(server, "get_call", {"call_id": "500"}))
    assert payload["call"]["call_id"] == "500"
    assert payload["transcript"]["available"] is False
    assert payload["transcript"]["reason"] == "not_available"
    assert payload["recap"]["available"] is False
    assert payload["recap"]["reason"] == "not_permitted"
    assert payload["recording"]["available"] is True


def test_get_call_recap_rate_limited(make_server) -> None:
    server, _client = _get_server(
        make_server,
        calls={"500": ALLOWED_CALL},
        recaps={"500": CallApiError("rate_limited", "dialpad rate limit reached; retry shortly", retryable=True)},
    )
    payload = _tool_payload(_tool_call(server, "get_call", {"call_id": "500"}))
    assert payload["recap"]["reason"] == "rate_limited"
    assert payload["recap"]["retryable"] is True


def test_get_call_lookup_hard_errors(make_server) -> None:
    server, _client = _get_server(
        make_server,
        calls={"500": CallApiError("rate_limited", "dialpad rate limit reached; retry shortly", retryable=True)},
    )
    result = _tool_call(server, "get_call", {"call_id": "500"})
    payload = _tool_payload(result)
    assert payload["error"]["code"] == "rate_limited"
    assert payload["error"]["retryable"] is True


def test_get_call_arg_validation(make_server) -> None:
    server, _client = _get_server(make_server, calls={"500": ALLOWED_CALL})
    for arguments in [
        {},
        {"call_id": "abc"},
        {"call_id": "12-34"},
        {"call_id": -1},
        {"call_id": "500", "include": "transcript"},
        {"call_id": "500", "include": ["transcript", "dial"]},
        {"call_id": "500", "summary_format": "huge"},
        {"call_id": "500", "caller": "team-lilla"},
        {"call_id": "500", "verbose": True},
    ]:
        result = _tool_call(server, "get_call", arguments)
        assert result["isError"] is True, arguments
        assert _tool_payload(result)["error"]["code"] == "invalid_argument", arguments


def test_get_call_empty_transcript_body_still_available(make_server) -> None:
    server, _client = _get_server(
        make_server, calls={"500": ALLOWED_CALL}, transcripts={"500": {"call_id": "500", "lines": []}}
    )
    payload = _tool_payload(_tool_call(server, "get_call", {"call_id": "500", "include": ["transcript"]}))
    assert payload["transcript"]["available"] is True
    assert payload["transcript"]["transcript_text"] == ""


# ---------------------------------------------------------------------------
# Real client: transport mapping, caching, missing key
# ---------------------------------------------------------------------------


class FakeTransport:
    """Routes url path -> (status, body); records requests."""

    def __init__(self, routes=None):
        self.routes = dict(routes or {})
        self.requests: list[str] = []

    def __call__(self, url, headers, timeout):
        from urllib.parse import urlsplit

        self.requests.append(url)
        assert headers["Authorization"] == "Bearer test-key"
        path = urlsplit(url).path
        route = self.routes.get(path.removeprefix("/api/v2"))
        if route is None:
            return 404, b'{"error": {"message": "not found"}}'
        if isinstance(route, Exception):
            raise route
        status, payload = route
        return status, json.dumps(payload).encode("utf-8")


def _real_client(routes, **kwargs) -> team_calls.DialpadCallsClient:
    return team_calls.DialpadCallsClient(
        environ={"DIALPAD_API_KEY": "test-key"}, transport=FakeTransport(routes), **kwargs
    )


def test_missing_api_key_is_a_clean_error(make_server) -> None:
    client = team_calls.DialpadCallsClient(environ={})
    server = make_server(client=client)
    result = _tool_call(server, "get_call", {"call_id": "500"})
    assert _tool_payload(result)["error"]["code"] == "not_configured"
    result = _tool_call(server, "list_calls", {})
    assert _tool_payload(result)["error"]["code"] == "not_configured"
    # The rest of the server still works.
    health = json.load(_request(server, "/health", method="GET"))
    assert health == {"caller": "team-scott"}


def test_client_transport_error_mapping() -> None:
    client = _real_client(
        {
            "/call/1": (429, {}),
            "/call/2": (400, {"error": {"message": "rate_limit_exceeded"}}),
            "/call/3": (403, {"error": {"message": "forbidden"}}),
            "/call/4": (500, {}),
            "/call/5": urllib.error.URLError("down"),
        }
    )
    for call_id, reason, retryable in [
        ("1", "rate_limited", True),
        ("2", "rate_limited", True),
        ("3", "not_permitted", False),
        ("4", "upstream_error", True),
        ("5", "upstream_error", True),
    ]:
        with pytest.raises(CallApiError) as raised:
            client.get_call(call_id)
        assert raised.value.reason == reason
        assert raised.value.retryable == retryable


def test_client_list_and_recap_requests() -> None:
    transport = FakeTransport(
        {
            "/call": (200, {"items": [_call("700")], "cursor": "next"}),
            "/call/700/ai_recap": (200, RECAP),
        }
    )
    client = team_calls.DialpadCallsClient(
        environ={"DIALPAD_API_KEY": "test-key"}, transport=transport
    )
    page = client.list_call_page(started_after=1, started_before=2)
    assert page["cursor"] == "next"
    assert len(page["items"]) == 1
    url = transport.requests[0]
    assert "started_after=1" in url and "started_before=2" in url and "limit=" not in url
    client.get_ai_recap("700", summary_format="long")
    assert "summary_format=long" in transport.requests[1]


def test_cache_does_not_leak_across_callers(make_server) -> None:
    """Caller A's allowed fetch caches the raw call; caller B's request for the
    same id is still refused by per-request scoping without a new fetch."""
    routes = {"/call/500": (200, ALLOWED_CALL)}
    client = _real_client(routes)
    server = make_server(client=client)

    result = _tool_call(server, "get_call", {"call_id": "500", "include": ["recording"]})
    assert result["isError"] is False
    assert len(client._transport.requests) == 1

    refused = _tool_call(server, "get_call", {"call_id": "500", "include": ["transcript"]}, token=LILLA_TOKEN)
    missing = _tool_call(server, "get_call", {"call_id": "999"}, token=LILLA_TOKEN)
    assert _tool_payload(refused) == _tool_payload(missing)
    assert _tool_payload(refused)["error"]["code"] == "call_not_found"
    # The cached raw call served B's lookup (no second fetch, no transcript
    # fetch) and B was still refused by per-request scoping; B's missing id
    # is the only new upstream request.
    assert [url for url in client._transport.requests] == [
        "https://dialpad.com/api/v2/call/500",
        "https://dialpad.com/api/v2/call/999",
    ]


def test_cached_sections_served_without_new_fetches(make_server) -> None:
    routes = {
        "/call/500": (200, ALLOWED_CALL),
        "/transcripts/500": (200, TRANSCRIPT),
        "/call/500/ai_recap": (200, RECAP),
    }
    client = _real_client(routes)
    server = make_server(client=client)
    for _ in range(2):
        payload = _tool_payload(_tool_call(server, "get_call", {"call_id": "500"}))
        assert payload["transcript"]["available"] is True
    assert len(client._transport.requests) == 3
