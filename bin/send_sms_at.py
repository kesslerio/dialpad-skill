#!/usr/bin/env python3
"""Scheduled SMS send with a just-in-time thread safety recheck.

Waits until a target recipient-local time, re-reads the thread through
list_sms_thread's retrying reader, then sends via bin/send_sms.py so the send
gate, receipts, and approval audit all still apply. Never sends blind: an
unreadable thread or any recheck reason skips the send with exit 0.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_PATH = str(ROOT / "scripts")
for path in (str(ROOT / "bin"), SCRIPTS_PATH):
    if path not in sys.path:
        sys.path.append(path)

from _dialpad_compat import (  # noqa: E402
    WrapperArgumentParser,
    WrapperError,
    emit_error,
    emit_success,
    handle_wrapper_exception,
    print_wrapper_error,
)

import send_gate  # noqa: E402
import list_sms_thread  # noqa: E402
from nanp_area_code_timezones import timezones_for_e164  # noqa: E402

COMMAND = "send_sms_at.send"
WRAPPER = "send_sms_at.py"

RECHECK_THREAD_LIMIT = 60
WEEKLY_OUTBOUND_CAP = 2
RECENT_WINDOW = timedelta(hours=24)
WEEKLY_WINDOW = timedelta(days=7)
WAIT_SLICE_SECONDS = 5.0
SEND_SMS_SCRIPT = ROOT / "bin" / "send_sms.py"

# Inbound keywords treated as an opt-out signal; a false positive only delays
# a scheduled send, a miss could re-message someone who asked us to stop.
_OPT_OUT_RE = re.compile(r"\b(STOP|UNSUBSCRIBE|OPT[\s-]?OUT|CANCEL|END|QUIT)\b")

_utcnow = lambda: datetime.now(timezone.utc)  # noqa: E731
_sleep = time.sleep


def build_parser() -> argparse.ArgumentParser:
    parser = WrapperArgumentParser(description="Send an SMS at a scheduled recipient-local time")
    parser.add_argument("--to", required=True, help="Recipient E.164 number")
    parser.add_argument("--from", dest="from_number", required=True, help="Sender E.164 number")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--message", help="SMS text content")
    group.add_argument("--message-file", help="Read SMS text from a UTF-8 file")
    parser.add_argument("--at", required=True, help='Target local time "YYYY-MM-DD HH:MM[:SS]"')
    parser.add_argument("--tz", help="IANA timezone for --at (default: recipient's NANP area code)")
    parser.add_argument("--caller", help="Send-gate caller (default: DIALPAD_SEND_CALLER env)")
    parser.add_argument("--allow-outside-window", action="store_true",
                        help="Allow --at outside the recommended 08:00-21:00 recipient-local window")
    parser.add_argument("--check-only", action="store_true",
                        help="Run the rechecks now and report the decision; no wait, no send")
    parser.add_argument("--dry-run", action="store_true", help="Pass --dry-run through to send_sms.py")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    return parser


def _bad_arg(message: str, code: str = "invalid_argument") -> WrapperError:
    return WrapperError(message, code=code, retryable=False)


def _resolve_target(args: argparse.Namespace) -> datetime:
    if args.tz:
        try:
            zone = ZoneInfo(args.tz.strip())
        except (KeyError, ValueError) as exc:
            raise _bad_arg(f"Unknown IANA timezone: {args.tz}") from exc
    else:
        zones = timezones_for_e164(args.to)
        if not zones:
            raise _bad_arg(f"Cannot derive a timezone from {args.to}: pass --tz explicitly")
        zone = zones[0]
    naive = None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            naive = datetime.strptime(args.at.strip(), fmt)
        except ValueError:
            continue
    if naive is None:
        raise _bad_arg('--at must be "YYYY-MM-DD HH:MM[:SS]"')
    local = naive.replace(tzinfo=zone)
    if local.minute in send_gate.BLOCKED_MINUTES:
        raise _bad_arg("--at falls on a blocked minute (:00/:15/:30/:45); pick another minute")
    if not args.allow_outside_window and not (
        send_gate.WINDOW_START <= local.time() < send_gate.WINDOW_END
    ):
        raise _bad_arg(
            "--at is outside the recommended 08:00-21:00 recipient-local window; "
            "pass --allow-outside-window to schedule anyway",
            code="send_gate_refused",
        )
    return local.astimezone(timezone.utc)


def _message_timestamp(message: dict) -> float | None:
    try:
        return float(str(message.get("timestamp"))) / 1000
    except (TypeError, ValueError):
        return None


def evaluate_rechecks(messages: list, now: datetime) -> list[str]:
    """Skip reasons for a scheduled send; an empty list means safe to send."""
    opt_out = inbound_recent = False
    outbound_recent = 0
    outbound_week = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        ts = _message_timestamp(message)
        age = now - datetime.fromtimestamp(ts, tz=timezone.utc) if ts is not None else None
        direction = message.get("direction")
        if direction == "inbound":
            if _OPT_OUT_RE.search(str(message.get("text") or "").upper()):
                opt_out = True
            if age is not None and age <= RECENT_WINDOW:
                inbound_recent = True
        elif direction == "outbound" and age is not None:
            if age <= RECENT_WINDOW:
                outbound_recent += 1
            if age <= WEEKLY_WINDOW:
                outbound_week += 1
    reasons: list[str] = []
    if opt_out:
        reasons.append("inbound_opt_out")
    if outbound_recent:
        reasons.append("outbound_last_24h")
    if outbound_week >= WEEKLY_OUTBOUND_CAP:
        reasons.append("weekly_outbound_cap")
    if inbound_recent:
        reasons.append("inbound_last_24h")
    return reasons


def _send_argv(args: argparse.Namespace) -> list[str]:
    argv = [
        sys.executable, str(SEND_SMS_SCRIPT),
        "--to", args.to,
        "--from", args.from_number,
        "--json",
    ]
    if args.message is not None:
        argv += ["--message", args.message]
    else:
        argv += ["--message-file", args.message_file]
    if args.caller:
        argv += ["--caller", args.caller]
    if args.dry_run:
        argv += ["--dry-run"]
    if args.allow_outside_window:
        argv += ["--allow-outside-window"]
    return argv


def _recheck_reasons(phone: str) -> list[str]:
    try:
        summary, _meta = list_sms_thread.read_thread_summary(phone, RECHECK_THREAD_LIMIT)
    except Exception:
        return ["thread_unreadable"]
    return evaluate_rechecks(summary.get("messages") or [], _utcnow())


def _emit_decision(json_mode: bool, status: str, reasons: list[str], target_utc: datetime) -> int:
    data = {"status": status, "reasons": reasons, "target_utc": target_utc.isoformat().replace("+00:00", "Z")}
    if json_mode:
        emit_success(COMMAND, WRAPPER, data)
    else:
        detail = f" ({', '.join(reasons)})" if reasons else ""
        print(f"Scheduled send: {status}{detail}")
    return 0


def _run() -> int:
    json_mode = "--json" in sys.argv
    try:
        args = build_parser().parse_args()
        json_mode = args.json
        target_utc = _resolve_target(args)

        if args.check_only:
            reasons = _recheck_reasons(args.to)
            return _emit_decision(json_mode, "skipped" if reasons else "would_send", reasons, target_utc)

        while True:
            remaining = (target_utc - _utcnow()).total_seconds()
            if remaining <= 0:
                break
            _sleep(min(WAIT_SLICE_SECONDS, remaining))

        reasons = _recheck_reasons(args.to)
        if reasons:
            return _emit_decision(json_mode, "skipped", reasons, target_utc)

        proc = subprocess.run(_send_argv(args), capture_output=True, text=True)
        try:
            envelope = json.loads(proc.stdout)
        except (json.JSONDecodeError, TypeError):
            envelope = None
        if proc.returncode != 0:
            if isinstance(envelope, dict) and envelope.get("ok") is False:
                sys.stdout.write(proc.stdout)
                return proc.returncode
            emit_error(
                COMMAND, WRAPPER, "send_outcome_unknown",
                f"send_sms exited {proc.returncode} without a result envelope",
                retryable=False,
            )
            return 2
        data = {"status": "sent", "send": envelope}
        if json_mode:
            emit_success(COMMAND, WRAPPER, data)
        else:
            print("Scheduled SMS sent")
            sys.stdout.write(proc.stdout)
        return 0
    except WrapperError as err:
        if json_mode:
            return handle_wrapper_exception(COMMAND, WRAPPER, err, True)
        print_wrapper_error(err)
        return 2
    except Exception as err:  # noqa: BLE001 - wrappers must return structured JSON in --json mode.
        if json_mode:
            return handle_wrapper_exception(COMMAND, WRAPPER, err, True)
        print_wrapper_error(err)
        return 2


if __name__ == "__main__":
    raise SystemExit(_run())
