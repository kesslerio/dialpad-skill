"""Tests for transient-retry hardening: http_retry, log client retries,
send safe-retry classification, envelope audit, ledger paths, send_sms_at."""

from __future__ import annotations

import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import unittest
import urllib.error
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT / "scripts"), str(ROOT / "bin")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import http_retry
import log_api_client
import send_sms_at
import sms_receipts
import list_sms_thread
import _dialpad_compat
from _dialpad_compat import WrapperError

SCRIPTS_SEND_SMS = _dialpad_compat  # placeholder, replaced below


def _http_error(code: int, retry_after: str | None = None, body: bytes = b"") -> urllib.error.HTTPError:
    headers = {"Retry-After": retry_after} if retry_after is not None else {}
    return urllib.error.HTTPError(
        "https://log.example/v1", code, f"HTTP {code}", headers, io.BytesIO(body)
    )


class _FakeHTTPResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self, *args):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class HttpRetryHelperTests(unittest.TestCase):
    def test_parse_retry_after_seconds_and_date(self):
        self.assertEqual(http_retry.parse_retry_after("7"), 7.0)
        future = datetime.now(timezone.utc) + timedelta(seconds=30)
        parsed = http_retry.parse_retry_after(
            future.strftime("%a, %d %b %Y %H:%M:%S GMT"), now=None
        )
        self.assertIsNotNone(parsed)
        self.assertGreater(parsed, 0)
        self.assertLessEqual(parsed, 31)
        self.assertIsNone(http_retry.parse_retry_after("not-a-date"))
        self.assertIsNone(http_retry.parse_retry_after(None))

    def test_backoff_is_bounded_and_jittered(self):
        for attempt in range(4):
            delay = http_retry.backoff_seconds(attempt, base=0.5, cap=8.0, random_fn=lambda: 1.0)
            self.assertEqual(delay, min(8.0, 0.5 * (2**attempt)))
            delay = http_retry.backoff_seconds(attempt, base=0.5, cap=8.0, random_fn=lambda: 0.0)
            self.assertEqual(delay, 0.0)

    def test_env_overrides(self):
        self.assertEqual(http_retry.configured_max_attempts({"DIALPAD_HTTP_RETRIES": "5"}), 5)
        self.assertEqual(http_retry.configured_max_attempts({"DIALPAD_HTTP_RETRIES": "bogus"}), 3)
        self.assertEqual(http_retry.configured_max_attempts({"DIALPAD_HTTP_RETRIES": "0"}), 1)
        self.assertEqual(http_retry.configured_max_sleep({"DIALPAD_HTTP_RETRY_MAX_SLEEP": "2"}), 2.0)

    def test_run_retries_until_success_and_honors_retry_after(self):
        sleeps: list[float] = []
        attempts = {"n": 0}

        def op():
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise ValueError("boom")
            return "ok"

        def decision(exc):
            return http_retry.RetryDecision(retry_after=4.0) if isinstance(exc, ValueError) else None

        result = http_retry.run(op, retry_decision=decision, sleep=sleeps.append)
        self.assertEqual(result, "ok")
        self.assertEqual(attempts["n"], 3)
        self.assertEqual(sleeps, [4.0, 4.0])

    def test_run_stops_at_attempt_limit_and_propagates(self):
        attempts = {"n": 0}

        def op():
            attempts["n"] += 1
            raise ValueError("boom")

        with self.assertRaises(ValueError):
            http_retry.run(
                op,
                retry_decision=lambda exc: http_retry.RetryDecision(),
                attempts=3,
                sleep=lambda s: None,
            )
        self.assertEqual(attempts["n"], 3)

    def test_run_permanent_failure_does_not_retry(self):
        attempts = {"n": 0}

        def op():
            attempts["n"] += 1
            raise ValueError("nope")

        with self.assertRaises(ValueError):
            http_retry.run(op, retry_decision=lambda exc: None, sleep=lambda s: None)
        self.assertEqual(attempts["n"], 1)


class SendClassifierTests(unittest.TestCase):
    CASES = [
        ("429 Client Error: Too Many Requests for url: https://dialpad.com/api/v2/sms", http_retry.SAFE_RETRY),
        ("NewConnectionError: Failed to establish a new connection: [Errno 111] Connection refused", http_retry.SAFE_RETRY),
        ("ConnectTimeoutError: connect timeout", http_retry.SAFE_RETRY),
        ("Name or service not known", http_retry.SAFE_RETRY),
        ("Connection refused", http_retry.SAFE_RETRY),
        ("Read timed out. (read timeout=30)", http_retry.OUTCOME_UNKNOWN),
        ("RemoteDisconnected('Remote end closed connection without response')", http_retry.OUTCOME_UNKNOWN),
        ("ConnectionResetError: connection reset by peer", http_retry.OUTCOME_UNKNOWN),
        ("503 Server Error: Service Unavailable", http_retry.OUTCOME_UNKNOWN),
        ("500 Server Error: Internal Server Error", http_retry.OUTCOME_UNKNOWN),
        ("400 Client Error: Bad Request", http_retry.PERMANENT),
        ("401 Client Error: Unauthorized", http_retry.PERMANENT),
        ("403 Client Error: Forbidden", http_retry.PERMANENT),
        ("something entirely unexpected", http_retry.OUTCOME_UNKNOWN),
    ]

    def test_classifier_table(self):
        for text, expected in self.CASES:
            self.assertEqual(
                http_retry.classify_request_stderr(text), expected, f"stderr: {text!r}"
            )

    def test_stderr_is_transient_for_reads(self):
        self.assertTrue(http_retry.stderr_is_transient("503 Server Error"))
        self.assertTrue(http_retry.stderr_is_transient("connect timeout"))
        self.assertFalse(http_retry.stderr_is_transient("400 Client Error"))
        self.assertFalse(http_retry.stderr_is_transient("what is this"))

    def test_send_url_error_safe_only_for_connect_stage(self):
        safe = urllib.error.URLError(socket.gaierror("name resolution failed"))
        self.assertTrue(http_retry.send_url_error_safe(safe))
        refused = urllib.error.URLError(ConnectionRefusedError("refused"))
        self.assertTrue(http_retry.send_url_error_safe(refused))
        reset = urllib.error.URLError(ConnectionResetError("reset"))
        self.assertFalse(http_retry.send_url_error_safe(reset))
        timeout = urllib.error.URLError(socket.timeout("timed out"))
        self.assertFalse(http_retry.send_url_error_safe(timeout))


class _LogClientBase(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(
            os.environ,
            {"DIALPAD_LOG_URL": "https://log.example", "DIALPAD_LOG_TOKEN": "tok"},
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        sleeps: list[float] = []
        patcher_sleep = patch.object(log_api_client, "_sleep", sleeps.append)
        patcher_rand = patch.object(log_api_client, "_random_fn", lambda: 0.5)
        patcher_sleep.start()
        patcher_rand.start()
        self.addCleanup(patcher_sleep.stop)
        self.addCleanup(patcher_rand.stop)
        self.sleeps = sleeps
        self.calls = {"n": 0}

    def _urlopen_with(self, outcomes: list):
        queue = list(outcomes)

        def fake(request, timeout=None, **kwargs):
            self.calls["n"] += 1
            outcome = queue.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return _FakeHTTPResponse(outcome)

        return patch.object(log_api_client.urllib.request, "urlopen", side_effect=fake)


class LogApiClientRetryTests(_LogClientBase):
    def test_503_twice_then_200_succeeds_after_retries(self):
        outcomes = [_http_error(503), _http_error(503), {"ok": True, "data": {"items": []}, "meta": {}}]
        with self._urlopen_with(outcomes):
            data, _meta = log_api_client.get_data("/v1/sms/inbox")
        self.assertEqual(data, {"items": []})
        self.assertEqual(self.calls["n"], 3)
        self.assertEqual(len(self.sleeps), 2)

    def test_503_forever_raises_retryable(self):
        with self._urlopen_with([_http_error(503)] * 10):
            with self.assertRaises(log_api_client.LogApiError) as ctx:
                log_api_client.get_data("/v1/sms/inbox")
        self.assertTrue(ctx.exception.retryable)
        self.assertEqual(self.calls["n"], 3)

    def test_400_and_401_are_single_attempt_non_retryable(self):
        for code in (400, 401):
            self.calls["n"] = 0
            with self._urlopen_with([_http_error(code)]):
                with self.assertRaises(log_api_client.LogApiError) as ctx:
                    log_api_client.get_data("/v1/sms/inbox")
            self.assertFalse(ctx.exception.retryable)
            self.assertEqual(self.calls["n"], 1)

    def test_429_retry_after_is_honored_and_capped(self):
        outcomes = [_http_error(429, retry_after="120"), {"ok": True, "data": {"items": []}, "meta": {}}]
        with self._urlopen_with(outcomes):
            log_api_client.get_data("/v1/sms/inbox")
        self.assertEqual(self.sleeps, [8.0])  # capped at max sleep

    def test_200_missing_data_key_is_retryable_upstream_error(self):
        with self._urlopen_with([{"ok": True, "meta": {}}] * 3):
            with self.assertRaises(log_api_client.LogApiError) as ctx:
                log_api_client.get_data("/v1/sms/thread", required_lists=("messages",))
        self.assertEqual(ctx.exception.code, "upstream_error")
        self.assertTrue(ctx.exception.retryable)

    def test_200_missing_required_list_is_retryable_upstream_error(self):
        payload = {"ok": True, "data": {"phone": "+14155550111"}, "meta": {}}
        with self._urlopen_with([payload] * 3):
            with self.assertRaises(log_api_client.LogApiError) as ctx:
                log_api_client.get_data("/v1/sms/thread", required_lists=("messages",))
        self.assertEqual(ctx.exception.code, "upstream_error")
        self.assertTrue(ctx.exception.retryable)

    def test_url_error_is_retryable_and_retried(self):
        err = urllib.error.URLError(socket.timeout("timed out"))
        with self._urlopen_with([err, {"ok": True, "data": {}, "meta": {}}]):
            data, _meta = log_api_client.get_data("/v1/sms/thread")
        self.assertEqual(data, {})
        self.assertEqual(self.calls["n"], 2)

    def test_post_record_message_is_single_attempt(self):
        with self._urlopen_with([_http_error(503)] * 5):
            with self.assertRaises(log_api_client.LogApiError):
                log_api_client.record_message({"kind": "sms"})
        self.assertEqual(self.calls["n"], 1)


class ThreadWrapperEnvelopeTests(_LogClientBase):
    def _run_thread(self) -> tuple[int, dict]:
        argv = ["bin/list_sms_thread.py", "--phone", "+14155550111", "--json"]
        stdout = io.StringIO()
        with patch.object(sys, "argv", argv), redirect_stdout(stdout):
            code = list_sms_thread._run()
        return code, json.loads(stdout.getvalue())

    def test_envelope_success_after_retries(self):
        payload = {"ok": True, "data": {"phone": "+14155550111", "messages": []}, "meta": {}}
        outcomes = [_http_error(503), payload]
        with self._urlopen_with(outcomes):
            code, parsed = self._run_thread()
        self.assertEqual(code, 0)
        self.assertTrue(parsed["ok"])
        self.assertEqual(parsed["data"]["messages"], [])

    def test_envelope_retryable_error_after_exhaustion(self):
        with self._urlopen_with([_http_error(503)] * 10):
            code, parsed = self._run_thread()
        self.assertEqual(code, 2)
        self.assertFalse(parsed["ok"])
        self.assertTrue(parsed["error"]["retryable"])
        self.assertNotIn("data", parsed)

    def test_envelope_upstream_error_for_malformed_200(self):
        payload = {"ok": True, "data": {"phone": "+14155550111"}, "meta": {}}
        with self._urlopen_with([payload] * 3):
            code, parsed = self._run_thread()
        self.assertEqual(code, 2)
        self.assertFalse(parsed["ok"])
        self.assertEqual(parsed["error"]["code"], "upstream_error")
        self.assertTrue(parsed["error"]["retryable"])
        self.assertNotIn("data", parsed)

    def test_envelope_for_uncaught_exception(self):
        with patch.object(
            list_sms_thread, "read_thread_summary", side_effect=RuntimeError("totally broke")
        ):
            code, parsed = self._run_thread()
        self.assertEqual(code, 2)
        self.assertFalse(parsed["ok"])
        self.assertNotIn("data", parsed)


class RunGeneratedJsonSendTests(unittest.TestCase):
    def _proc(self, returncode: int, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(args=["generated"], returncode=returncode, stdout=stdout, stderr=stderr)

    def test_send_429_retries_then_succeeds(self):
        results = [
            self._proc(1, stderr="429 Client Error: Too Many Requests"),
            self._proc(1, stderr="429 Client Error: Too Many Requests"),
            self._proc(0, stdout='{"id": "msg-1"}'),
        ]
        calls = {"n": 0}

        def fake(cmd, capture_output=False):
            calls["n"] += 1
            return results.pop(0)

        with patch.object(_dialpad_compat, "run_generated", side_effect=fake):
            result = _dialpad_compat.run_generated_json(
                ["sms", "send", "--data", "{}"], sleep=lambda s: None
            )
        self.assertEqual(result, {"id": "msg-1"})
        self.assertEqual(calls["n"], 3)

    def test_send_read_timeout_is_outcome_unknown_no_retry(self):
        proc = self._proc(1, stderr="Error: Read timed out. (read timeout=30)")
        calls = {"n": 0}

        def fake(cmd, capture_output=False):
            calls["n"] += 1
            return proc

        with patch.object(_dialpad_compat, "run_generated", side_effect=fake):
            with self.assertRaises(WrapperError) as ctx:
                _dialpad_compat.run_generated_json(["sms", "send", "--data", "{}"], sleep=lambda s: None)
        self.assertEqual(ctx.exception.code, "send_outcome_unknown")
        self.assertFalse(ctx.exception.retryable)
        self.assertEqual(calls["n"], 1)

    def test_send_5xx_is_outcome_unknown_no_retry(self):
        proc = self._proc(1, stderr="503 Server Error: Service Unavailable")
        calls = {"n": 0}

        def fake(cmd, capture_output=False):
            calls["n"] += 1
            return proc

        with patch.object(_dialpad_compat, "run_generated", side_effect=fake):
            with self.assertRaises(WrapperError) as ctx:
                _dialpad_compat.run_generated_json(["sms", "send", "--data", "{}"], sleep=lambda s: None)
        self.assertEqual(ctx.exception.code, "send_outcome_unknown")
        self.assertEqual(calls["n"], 1)

    def test_send_4xx_is_permanent_no_retry(self):
        proc = self._proc(1, stderr="400 Client Error: Bad Request")
        calls = {"n": 0}

        def fake(cmd, capture_output=False):
            calls["n"] += 1
            return proc

        with patch.object(_dialpad_compat, "run_generated", side_effect=fake):
            with self.assertRaises(WrapperError) as ctx:
                _dialpad_compat.run_generated_json(["sms", "send", "--data", "{}"], sleep=lambda s: None)
        self.assertEqual(ctx.exception.code, "upstream_error")
        self.assertFalse(ctx.exception.retryable)
        self.assertEqual(calls["n"], 1)

    def test_non_send_retries_transient_then_succeeds(self):
        results = [
            self._proc(1, stderr="503 Server Error"),
            self._proc(0, stdout='{"items": []}'),
        ]
        calls = {"n": 0}

        def fake(cmd, capture_output=False):
            calls["n"] += 1
            return results.pop(0)

        with patch.object(_dialpad_compat, "run_generated", side_effect=fake):
            result = _dialpad_compat.run_generated_json(["call", "list"], sleep=lambda s: None)
        self.assertEqual(result, {"items": []})
        self.assertEqual(calls["n"], 2)

    def test_call_call_read_timeout_is_outcome_unknown_no_retry(self):
        # #170: call.call places a real phone call — a post-acceptance failure
        # must not blind-retry into a duplicate call.
        proc = self._proc(1, stderr="Error: Read timed out. (read timeout=30)")
        calls = {"n": 0}

        def fake(cmd, capture_output=False):
            calls["n"] += 1
            return proc

        with patch.object(_dialpad_compat, "run_generated", side_effect=fake):
            with self.assertRaises(WrapperError) as ctx:
                _dialpad_compat.run_generated_json(["call", "call.call", "--data", "{}"], sleep=lambda s: None)
        self.assertEqual(ctx.exception.code, "send_outcome_unknown")
        self.assertFalse(ctx.exception.retryable)
        self.assertIn("verify before repeating", str(ctx.exception))
        self.assertEqual(calls["n"], 1)

    def test_call_call_connect_error_retries_then_succeeds(self):
        results = [
            self._proc(1, stderr="NewConnectionError: Failed to establish a new connection: [Errno 111] Connection refused"),
            self._proc(0, stdout='{"call_id": "c-1"}'),
        ]
        calls = {"n": 0}

        def fake(cmd, capture_output=False):
            calls["n"] += 1
            return results.pop(0)

        with patch.object(_dialpad_compat, "run_generated", side_effect=fake):
            result = _dialpad_compat.run_generated_json(["call", "call.call", "--data", "{}"], sleep=lambda s: None)
        self.assertEqual(result, {"call_id": "c-1"})
        self.assertEqual(calls["n"], 2)

    def test_flag_only_writes_are_outcome_unknown_no_retry(self):
        # #170: webhooks.create / stats.create / *.delete carry flags, not
        # --data — still non-idempotent writes under the SAFE-RETRY rule.
        proc = self._proc(1, stderr="RemoteDisconnected('Remote end closed connection without response')")
        for args in (
            ["webhook", "create", "--hook-url", "https://example.invalid/hook"],
            ["stats", "stats.create", "--export-type", "sms", "--stat-type", "calls"],
            ["webhooks", "webhooks.delete", "--id", "7"],
        ):
            calls = {"n": 0}

            def fake(cmd, capture_output=False):
                calls["n"] += 1
                return proc

            with patch.object(_dialpad_compat, "run_generated", side_effect=fake):
                with self.assertRaises(WrapperError) as ctx:
                    _dialpad_compat.run_generated_json(args, sleep=lambda s: None)
            self.assertEqual(ctx.exception.code, "send_outcome_unknown", args)
            self.assertFalse(ctx.exception.retryable, args)
            self.assertIn("verify before repeating", str(ctx.exception), args)
            self.assertEqual(calls["n"], 1, args)

    def test_write_4xx_is_permanent_no_retry(self):
        proc = self._proc(1, stderr="400 Client Error: Bad Request")
        calls = {"n": 0}

        def fake(cmd, capture_output=False):
            calls["n"] += 1
            return proc

        with patch.object(_dialpad_compat, "run_generated", side_effect=fake):
            with self.assertRaises(WrapperError) as ctx:
                _dialpad_compat.run_generated_json(["call", "call.call", "--data", "{}"], sleep=lambda s: None)
        self.assertEqual(ctx.exception.code, "upstream_error")
        self.assertFalse(ctx.exception.retryable)
        self.assertEqual(calls["n"], 1)

    def test_read_command_read_timeout_still_retries(self):
        results = [
            self._proc(1, stderr="Error: Read timed out. (read timeout=30)"),
            self._proc(0, stdout='{"items": []}'),
        ]
        calls = {"n": 0}

        def fake(cmd, capture_output=False):
            calls["n"] += 1
            return results.pop(0)

        with patch.object(_dialpad_compat, "run_generated", side_effect=fake):
            result = _dialpad_compat.run_generated_json(["stats", "stats.get", "--id", "1"], sleep=lambda s: None)
        self.assertEqual(result, {"items": []})
        self.assertEqual(calls["n"], 2)

    def test_exhausted_send_retries_surface_retryable_with_retry_after(self):
        proc = self._proc(1, stderr='429 Client Error: Too Many Requests, Retry-After: 120')
        with patch.object(_dialpad_compat, "run_generated", return_value=proc):
            with self.assertRaises(WrapperError) as ctx:
                _dialpad_compat.run_generated_json(["sms", "send", "--data", "{}"], sleep=lambda s: None)
        self.assertTrue(ctx.exception.retryable)
        self.assertEqual(ctx.exception.retry_after_seconds, 120.0)


class LedgerPathTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.env = patch.dict(
            os.environ,
            {
                "XDG_STATE_HOME": str(Path(self.tmp.name) / "xdg"),
            },
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        os.environ.pop("DIALPAD_SMS_RECEIPT_LEDGER", None)
        self.payload = {"to_numbers": ["+14155550111"], "from_number": "+14155550140"}
        self.result = {"id": "msg-1", "message_status": "pending"}

    def _append(self) -> str:
        return sms_receipts.append_receipt(request_payload=self.payload, send_result=self.result, source="test")

    def test_env_override_writes_there(self):
        env_path = Path(self.tmp.name) / "env" / "receipts.jsonl"
        os.environ["DIALPAD_SMS_RECEIPT_LEDGER"] = str(env_path)
        self.addCleanup(lambda: os.environ.pop("DIALPAD_SMS_RECEIPT_LEDGER", None))
        self.assertEqual(self._append(), "appended")
        self.assertTrue(env_path.is_file())

    def test_legacy_unusable_falls_back_to_xdg(self):
        with patch.object(sms_receipts, "_dir_usable", return_value=False):
            self.assertEqual(self._append(), "appended")
        xdg = Path(self.tmp.name) / "xdg" / "dialpad" / "sms-receipts.jsonl"
        self.assertTrue(xdg.is_file())

    def test_legacy_usable_writes_legacy(self):
        legacy = Path(self.tmp.name) / "data" / "sms-receipts.jsonl"
        legacy.parent.mkdir(parents=True)
        with patch.object(sms_receipts, "DEFAULT_LEDGER_PATH", legacy):
            self.assertEqual(self._append(), "appended")
        self.assertTrue(legacy.is_file())

    def test_oserror_on_chosen_falls_back_to_xdg_once(self):
        blocker = Path(self.tmp.name) / "blocker"
        blocker.write_text("not a dir")
        bad = blocker / "receipts.jsonl"  # parent is a file -> OSError
        with patch.object(sms_receipts, "DEFAULT_LEDGER_PATH", bad), \
                patch.object(sms_receipts, "_dir_usable", return_value=True):
            self.assertEqual(self._append(), "appended")
        xdg = Path(self.tmp.name) / "xdg" / "dialpad" / "sms-receipts.jsonl"
        self.assertTrue(xdg.is_file())

    def test_all_candidates_fail_reports_append_failed_once(self):
        xdg_bad = Path(self.tmp.name) / "file_blocker"
        xdg_bad.write_text("x")
        with patch.object(sms_receipts, "_dir_usable", return_value=False), \
                patch.object(sms_receipts, "_xdg_ledger_path", return_value=xdg_bad / "r.jsonl"):
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                status = self._append()
        self.assertEqual(status, "append_failed")
        self.assertEqual(stderr.getvalue().count("Warning: failed to append"), 1)


class EvaluateRechecksTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime.now(timezone.utc)

    def _msg(self, direction: str, age: timedelta, text: str = "hi") -> dict:
        ts = int((self.now - age).timestamp() * 1000)
        return {"direction": direction, "timestamp": ts, "text": text}

    def test_empty_thread_sends(self):
        self.assertEqual(send_sms_at.evaluate_rechecks([], self.now), [])

    def test_inbound_opt_out_terms(self):
        for text in ("STOP", "please unsubscribe", "OPT OUT", "cancel", "QUIT"):
            msgs = [self._msg("inbound", timedelta(days=3), text)]
            reasons = send_sms_at.evaluate_rechecks(msgs, self.now)
            self.assertIn("inbound_opt_out", reasons, f"text={text!r}")

    def test_outbound_in_last_24h(self):
        msgs = [self._msg("outbound", timedelta(hours=2))]
        self.assertIn("outbound_last_24h", send_sms_at.evaluate_rechecks(msgs, self.now))

    def test_two_outbound_in_week_hits_cap(self):
        msgs = [self._msg("outbound", timedelta(days=2)), self._msg("outbound", timedelta(days=4))]
        self.assertIn("weekly_outbound_cap", send_sms_at.evaluate_rechecks(msgs, self.now))

    def test_one_outbound_in_week_under_cap(self):
        msgs = [self._msg("outbound", timedelta(days=4))]
        self.assertNotIn("weekly_outbound_cap", send_sms_at.evaluate_rechecks(msgs, self.now))

    def test_inbound_in_last_24h(self):
        msgs = [self._msg("inbound", timedelta(hours=3))]
        self.assertIn("inbound_last_24h", send_sms_at.evaluate_rechecks(msgs, self.now))

    def test_old_quiet_thread_sends(self):
        msgs = [
            self._msg("outbound", timedelta(days=10)),
            self._msg("inbound", timedelta(days=9)),
        ]
        self.assertEqual(send_sms_at.evaluate_rechecks(msgs, self.now), [])


class SendSmsAtCliTests(unittest.TestCase):
    def _run(self, argv: list[str]) -> tuple[int, str, str]:
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(sys, "argv", ["bin/send_sms_at.py", *argv]), \
                redirect_stdout(stdout), redirect_stderr(stderr):
            code = send_sms_at._run()
        return code, stdout.getvalue(), stderr.getvalue()

    BASE = ["--to", "+12125550100", "--from", "+14155550140", "--message", "hi"]

    def test_blocked_minute_rejected_never_overridable(self):
        for minute in ("00", "15", "30", "45"):
            code, out, _ = self._run(
                self.BASE + ["--at", f"2099-01-05 10:{minute}", "--tz", "America/New_York", "--json"]
            )
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(out)["error"]["code"], "invalid_argument")

    def test_unknown_timezone_rejected(self):
        code, out, _ = self._run(
            self.BASE + ["--at", "2099-01-05 10:07", "--tz", "Not/AZone", "--json"]
        )
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(out)["error"]["code"], "invalid_argument")

    def test_area_code_derives_timezone(self):
        target = send_sms_at._resolve_target(
            send_sms_at.build_parser().parse_args(
                self.BASE + ["--at", "2099-01-05 10:07"]
            )
        )
        self.assertEqual(
            target.astimezone(send_sms_at.ZoneInfo("America/New_York")).hour, 10
        )

    def test_unknown_area_code_without_tz_rejected(self):
        argv = ["--to", "+99999999999", "--from", "+14155550140", "--message", "hi",
                "--at", "2099-01-05 10:07", "--json"]
        code, out, _ = self._run(argv)
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(out)["error"]["code"], "invalid_argument")

    def test_outside_window_rejected_unless_override(self):
        code, out, _ = self._run(
            self.BASE + ["--at", "2099-01-05 23:07", "--tz", "America/New_York", "--json"]
        )
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(out)["error"]["code"], "send_gate_refused")

    def test_check_only_reports_would_send(self):
        summary = {"messages": []}
        with patch.object(list_sms_thread, "read_thread_summary", return_value=(summary, None)):
            code, out, _ = self._run(
                self.BASE + ["--at", "2099-01-05 10:07", "--tz", "America/New_York", "--check-only", "--json"]
            )
        self.assertEqual(code, 0)
        parsed = json.loads(out)
        self.assertEqual(parsed["data"]["status"], "would_send")

    def test_check_only_reports_skip_reasons(self):
        ts = int(datetime.now(timezone.utc).timestamp() * 1000)
        summary = {"messages": [{"direction": "inbound", "timestamp": ts, "text": "STOP"}]}
        with patch.object(list_sms_thread, "read_thread_summary", return_value=(summary, None)):
            code, out, _ = self._run(
                self.BASE + ["--at", "2099-01-05 10:07", "--tz", "America/New_York", "--check-only", "--json"]
            )
        self.assertEqual(code, 0)
        parsed = json.loads(out)
        self.assertEqual(parsed["data"]["status"], "skipped")
        self.assertIn("inbound_opt_out", parsed["data"]["reasons"])

    def test_unreadable_thread_skips_and_never_sends(self):
        with patch.object(
            list_sms_thread, "read_thread_summary", side_effect=WrapperError("down")
        ), patch.object(send_sms_at.subprocess, "run") as sub, \
                patch.object(send_sms_at, "_sleep"):
            code, out, _ = self._run(
                self.BASE + ["--at", "2000-01-05 10:07", "--tz", "America/New_York", "--json"]
            )
        self.assertEqual(code, 0)
        parsed = json.loads(out)
        self.assertEqual(parsed["data"]["status"], "skipped")
        self.assertIn("thread_unreadable", parsed["data"]["reasons"])
        sub.assert_not_called()

    def test_sends_via_subprocess_and_relays_envelope(self):
        summary = {"messages": []}
        env = {"ok": True, "command": "send_sms.send", "data": {"id": "msg-9"}, "meta": {}}
        proc = subprocess.CompletedProcess(args=[], returncode=0, stdout=json.dumps(env), stderr="")
        with patch.object(list_sms_thread, "read_thread_summary", return_value=(summary, None)), \
                patch.object(send_sms_at.subprocess, "run", return_value=proc) as sub:
            code, out, _ = self._run(
                self.BASE + ["--at", "2000-01-05 10:07", "--tz", "America/New_York", "--json"]
            )
        self.assertEqual(code, 0)
        parsed = json.loads(out)
        self.assertEqual(parsed["data"]["status"], "sent")
        self.assertEqual(parsed["data"]["send"]["data"]["id"], "msg-9")
        send_argv = sub.call_args[0][0]
        self.assertIn("--to", send_argv)
        self.assertIn("+12125550100", send_argv)
        self.assertIn("--json", send_argv)

    def test_dry_run_status_is_dry_run_not_sent(self):
        summary = {"messages": []}
        env = {"ok": True, "command": "send_sms.send", "data": {"mode": "dry_run"}, "meta": {}}
        proc = subprocess.CompletedProcess(args=[], returncode=0, stdout=json.dumps(env), stderr="")
        with patch.object(list_sms_thread, "read_thread_summary", return_value=(summary, None)), \
                patch.object(send_sms_at.subprocess, "run", return_value=proc) as sub:
            code, out, _ = self._run(
                self.BASE + ["--at", "2000-01-05 10:07", "--tz", "America/New_York", "--dry-run", "--json"]
            )
        self.assertEqual(code, 0)
        parsed = json.loads(out)
        self.assertEqual(parsed["data"]["status"], "dry_run")
        self.assertIn("--dry-run", sub.call_args[0][0])

    def test_subprocess_failure_relays_error_envelope(self):
        summary = {"messages": []}
        env = {"ok": False, "error": {"code": "send_gate_refused", "message": "no", "retryable": False}}
        proc = subprocess.CompletedProcess(args=[], returncode=2, stdout=json.dumps(env), stderr="")
        with patch.object(list_sms_thread, "read_thread_summary", return_value=(summary, None)), \
                patch.object(send_sms_at.subprocess, "run", return_value=proc):
            code, out, _ = self._run(
                self.BASE + ["--at", "2000-01-05 10:07", "--tz", "America/New_York", "--json"]
            )
        self.assertEqual(code, 2)
        self.assertFalse(json.loads(out)["ok"])

    def test_crashed_subprocess_is_outcome_unknown(self):
        summary = {"messages": []}
        proc = subprocess.CompletedProcess(args=[], returncode=1, stdout="not json", stderr="")
        with patch.object(list_sms_thread, "read_thread_summary", return_value=(summary, None)), \
                patch.object(send_sms_at.subprocess, "run", return_value=proc):
            code, out, _ = self._run(
                self.BASE + ["--at", "2000-01-05 10:07", "--tz", "America/New_York", "--json"]
            )
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(out)["error"]["code"], "send_outcome_unknown")


class DirectSendFallbackTests(unittest.TestCase):
    def setUp(self):
        spec = __import__("importlib.util", fromlist=["util"])
        import importlib.util as ilu

        self._spec = ilu.spec_from_file_location("scripts_send_sms_testmod", ROOT / "scripts" / "send_sms.py")
        self.mod = ilu.module_from_spec(self._spec)
        self.env = patch.dict(os.environ, {"DIALPAD_API_KEY": "test-key"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self._spec.loader.exec_module(self.mod)
        self.calls = {"n": 0}

    def _urlopen(self, outcomes: list):
        queue = list(outcomes)

        def fake(request, **kwargs):
            self.calls["n"] += 1
            outcome = queue.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return _FakeHTTPResponse(outcome)

        return patch.object(self.mod.urllib.request, "urlopen", side_effect=fake)

    def test_429_then_success_retries(self):
        outcomes = [
            urllib.error.HTTPError("u", 429, "x", {}, io.BytesIO(b"{}")),
            {"id": "msg-1"},
        ]
        with self._urlopen(outcomes):
            result = self.mod.send_sms(["+14155550111"], "hi", "+14155550140", _sleep=lambda s: None)
        self.assertEqual(result, {"id": "msg-1"})
        self.assertEqual(self.calls["n"], 2)

    def test_read_timeout_is_outcome_unknown(self):
        with self._urlopen([socket.timeout("read timed out")]):
            with self.assertRaises(self.mod.SendOutcomeUnknownError):
                self.mod.send_sms(["+14155550111"], "hi", "+14155550140", _sleep=lambda s: None)
        self.assertEqual(self.calls["n"], 1)

    def test_5xx_is_outcome_unknown(self):
        with self._urlopen([urllib.error.HTTPError("u", 503, "x", {}, io.BytesIO(b"{}"))]):
            with self.assertRaises(self.mod.SendOutcomeUnknownError):
                self.mod.send_sms(["+14155550111"], "hi", "+14155550140", _sleep=lambda s: None)
        self.assertEqual(self.calls["n"], 1)

    def test_4xx_non_429_is_permanent_runtime_error(self):
        with self._urlopen([urllib.error.HTTPError("u", 400, "x", {}, io.BytesIO(b"{}"))]):
            with self.assertRaises(RuntimeError) as ctx:
                self.mod.send_sms(["+14155550111"], "hi", "+14155550140", _sleep=lambda s: None)
        self.assertNotIsInstance(ctx.exception, self.mod.SendOutcomeUnknownError)
        self.assertEqual(self.calls["n"], 1)

    def test_connect_stage_url_error_retries(self):
        outcomes = [
            urllib.error.URLError(socket.gaierror("name or service not known")),
            {"id": "msg-2"},
        ]
        with self._urlopen(outcomes):
            result = self.mod.send_sms(["+14155550111"], "hi", "+14155550140", _sleep=lambda s: None)
        self.assertEqual(result, {"id": "msg-2"})
        self.assertEqual(self.calls["n"], 2)

    def test_reset_url_error_is_outcome_unknown(self):
        with self._urlopen([urllib.error.URLError(ConnectionResetError("reset"))]):
            with self.assertRaises(self.mod.SendOutcomeUnknownError):
                self.mod.send_sms(["+14155550111"], "hi", "+14155550140", _sleep=lambda s: None)
        self.assertEqual(self.calls["n"], 1)


if __name__ == "__main__":
    unittest.main()
