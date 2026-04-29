from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

DEFAULT_CSV = Path("/Users/u1528314/repos/radioastro-ml/collect/small_subset/small_selection.csv")
DEFAULT_MAIL_ACCOUNT = "Google"

PRODUCT_VIEWER_RE = re.compile(r"/#/(?:productViewer|productviewer)/([^/?#\s]+)")
SDM_ID_RE = re.compile(r"\b\d{2}[AB]-[A-Za-z0-9_.-]+\b")
URL_TOKEN_RE = re.compile(r"https?://dl-dsoc\.nrao\.edu/[^/\s]+/(\d+)/([a-f0-9]{16,64})/?", re.IGNORECASE)
REQUEST_NAME_RE = re.compile(r"your\s+([A-Za-z0-9][A-Za-z0-9+\-_.]*)\s+is\s+complete\b", re.IGNORECASE)


@dataclass
class MailMessage:
    message_id: str
    subject: str
    received_at: str
    content: str


@dataclass
class MatchResult:
    row_idx: int
    reason: str
    score: int


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "")).strip()


def normalize_wget_command(command: str) -> str:
    cmd = (command or "").replace("\\\r\n", " ").replace("\\\n", " ").replace("\\\r", " ")
    cmd = re.sub(r"\s\\\s*", " ", cmd)
    cmd = normalize_text(cmd)
    cmd = cmd.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    return cmd


def extract_wget_commands(text: str) -> list[str]:
    commands: list[str] = []
    lines = (text or "").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not re.match(r"^\s*wget\b", line, re.IGNORECASE):
            i += 1
            continue

        parts = [line.strip()]
        while i + 1 < len(lines):
            current = lines[i].rstrip()
            nxt = lines[i + 1]
            if current.endswith("\\") or re.match(r"^\s+https?://", nxt):
                parts.append(nxt.strip())
                i += 1
            else:
                break

        cmd = normalize_wget_command(" ".join(parts))
        if "dl-dsoc.nrao.edu" in cmd:
            commands.append(cmd)
        i += 1

    deduped: list[str] = []
    seen: set[str] = set()
    for cmd in commands:
        if cmd not in seen:
            deduped.append(cmd)
            seen.add(cmd)
    return deduped


def extract_message_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    if not text:
        return tokens

    for match in PRODUCT_VIEWER_RE.finditer(text):
        tokens.add(match.group(1).lower())
    for match in SDM_ID_RE.finditer(text):
        tokens.add(match.group(0).lower())
    for match in URL_TOKEN_RE.finditer(text):
        tokens.add(match.group(1).lower())
        tokens.add(match.group(2).lower())
    for match in REQUEST_NAME_RE.finditer(text):
        tokens.add(match.group(1).lower())
    return tokens


def build_row_tokens(row: dict[str, str]) -> set[str]:
    tokens: set[str] = set()
    for key in ("name", "folder", "access_url"):
        value = str(row.get(key) or "").strip()
        if not value:
            continue
        lowered = value.lower()
        tokens.add(lowered)
        for match in PRODUCT_VIEWER_RE.finditer(value):
            tokens.add(match.group(1).lower())
        for match in SDM_ID_RE.finditer(value):
            tokens.add(match.group(0).lower())
    return tokens


def read_messages_from_file(path: Path) -> list[MailMessage]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError("message file must contain a JSON list")

    messages: list[MailMessage] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"message entry {idx} is not an object")
        messages.append(
            MailMessage(
                message_id=str(item.get("message_id") or idx),
                subject=str(item.get("subject") or ""),
                received_at=str(item.get("received_at") or ""),
                content=str(item.get("content") or ""),
            )
        )
    return messages


def fetch_outlook_messages(days_back: int, unread_only: bool, inbox_only: bool, debug: bool = False) -> list[MailMessage]:
    message_source = "messages of inbox" if inbox_only else "messages"
    source_label = "inbox" if inbox_only else "all_messages"
    applescript = f"""
set cutoffDate to (current date) - ({int(days_back)} * days)
set outText to ""
set scannedCount to 0
set includedCount to 0
set errorCount to 0

tell application "Microsoft Outlook"
    repeat with m in {message_source}
        try
            set scannedCount to scannedCount + 1
            set msgDate to time received of m
            set includeMsg to (msgDate is greater than or equal to cutoffDate)
            if includeMsg then
                if {str(bool(unread_only)).lower()} then
                    set includeMsg to (read status of m is false)
                end if
            end if

            if includeMsg then
                set includedCount to includedCount + 1
                set outText to outText & "<<<MSG>>>\\n"
                set outText to outText & (id of m as string) & "\\n"
                set outText to outText & "<<<SUBJECT>>>\\n"
                set outText to outText & (subject of m as string) & "\\n"
                set outText to outText & "<<<TIME>>>\\n"
                set outText to outText & (msgDate as string) & "\\n"
                set outText to outText & "<<<CONTENT>>>\\n"
                set outText to outText & (content of m as string) & "\\n"
                set outText to outText & "<<<END>>>\\n"
            end if
        on error errMsg
            set errorCount to errorCount + 1
        end try
    end repeat
end tell

return "<<<DEBUG>>>\\nsource={source_label}\\nscanned=" & scannedCount & "\\nincluded=" & includedCount & "\\nerrors=" & errorCount & "\\n<<<ENDDEBUG>>>\\n" & outText
"""

    proc = subprocess.run(
        ["osascript", "-e", applescript],
        check=True,
        text=True,
        capture_output=True,
    )
    if debug:
        print("[DEBUG] AppleScript raw output preview:")
        preview = proc.stdout[:1200].replace("\r", "\\r")
        print(preview if preview else "<empty>")
    return parse_applescript_messages(proc.stdout, debug=debug)


def fetch_mail_app_messages(
    days_back: int,
    *,
    account_name: str,
    mailbox_name: str,
    sender_filter: str | None,
    debug: bool = False,
) -> list[MailMessage]:
    sender_clause = ""
    sender_label = "any"
    if sender_filter:
        sender_value = sender_filter.replace('"', '\\"')
        sender_clause = f' whose sender contains "{sender_value}"'
        sender_label = sender_filter

    account_value = account_name.replace('"', '\\"')
    mailbox_value = mailbox_name.replace('"', '\\"')

    applescript = f"""
set cutoffDate to (current date) - ({int(days_back)} * days)
set outText to ""
set scannedCount to 0
set includedCount to 0
set errorCount to 0

tell application "Mail"
    set targetMailbox to mailbox "{mailbox_value}" of account "{account_value}"
    set candidateMessages to (messages of targetMailbox{sender_clause})
    repeat with m in candidateMessages
        try
            set scannedCount to scannedCount + 1
            set msgDate to date received of m
            set includeMsg to (msgDate is greater than or equal to cutoffDate)

            if includeMsg then
                set includedCount to includedCount + 1
                set outText to outText & "<<<MSG>>>\\n"
                set outText to outText & (id of m as string) & "\\n"
                set outText to outText & "<<<SUBJECT>>>\\n"
                set outText to outText & (subject of m as string) & "\\n"
                set outText to outText & "<<<TIME>>>\\n"
                set outText to outText & (msgDate as string) & "\\n"
                set outText to outText & "<<<CONTENT>>>\\n"
                set outText to outText & (content of m as string) & "\\n"
                set outText to outText & "<<<END>>>\\n"
            end if
        on error errMsg
            set errorCount to errorCount + 1
        end try
    end repeat
end tell

return "<<<DEBUG>>>\\nsource=mail_app\\naccount={account_value}\\nmailbox={mailbox_value}\\nsender_filter={sender_label}\\nscanned=" & scannedCount & "\\nincluded=" & includedCount & "\\nerrors=" & errorCount & "\\n<<<ENDDEBUG>>>\\n" & outText
"""

    proc = subprocess.run(
        ["osascript", "-e", applescript],
        check=True,
        text=True,
        capture_output=True,
    )
    if debug:
        print("[DEBUG] AppleScript raw output preview:")
        preview = proc.stdout[:1200].replace("\r", "\\r")
        print(preview if preview else "<empty>")
    return parse_applescript_messages(proc.stdout, debug=debug)


def parse_applescript_messages(raw: str, debug: bool = False) -> list[MailMessage]:
    messages: list[MailMessage] = []
    debug_block = ""
    if "<<<DEBUG>>>\n" in raw and "\n<<<ENDDEBUG>>>\n" in raw:
        debug_block = raw.split("<<<DEBUG>>>\n", 1)[1].split("\n<<<ENDDEBUG>>>\n", 1)[0]
        raw = raw.split("\n<<<ENDDEBUG>>>\n", 1)[1]
    if debug and debug_block:
        print("[DEBUG] AppleScript counters:")
        print(debug_block)

    for chunk in raw.split("<<<MSG>>>\n"):
        chunk = chunk.strip()
        if not chunk:
            continue

        try:
            message_id, rest = chunk.split("\n<<<SUBJECT>>>\n", 1)
            subject, rest = rest.split("\n<<<TIME>>>\n", 1)
            received_at, rest = rest.split("\n<<<CONTENT>>>\n", 1)
            content, _ = rest.split("\n<<<END>>>", 1)
        except ValueError:
            continue

        messages.append(
            MailMessage(
                message_id=message_id.strip(),
                subject=subject.strip(),
                received_at=received_at.strip(),
                content=content.strip(),
            )
        )
    return messages


def choose_row_for_message(message: MailMessage, rows: list[dict[str, str]], pending_rows: list[int]) -> MatchResult | None:
    body = f"{message.subject}\n{message.content}"
    body_lower = body.lower()
    msg_tokens = extract_message_tokens(body)

    matches: list[MatchResult] = []
    for idx in pending_rows:
        row = rows[idx]
        row_tokens = build_row_tokens(row)
        score = 0
        reasons: list[str] = []

        overlap = sorted(msg_tokens & row_tokens)
        if overlap:
            score += 100
            reasons.append(f"shared token {overlap[0]}")

        folder = str(row.get("folder") or row.get("name") or "").strip().lower()
        if folder:
            folder_re = re.compile(rf"(?<![A-Za-z0-9]){re.escape(folder)}(?![A-Za-z0-9])")
            if folder_re.search(body_lower):
                score += 40
                reasons.append(f"name/folder mention {folder}")

        if score:
            matches.append(MatchResult(row_idx=idx, reason=", ".join(reasons), score=score))

    if not matches:
        if len(pending_rows) == 1:
            return MatchResult(row_idx=pending_rows[0], reason="only pending row left", score=1)
        return None

    matches.sort(key=lambda item: item.score, reverse=True)
    if len(matches) > 1 and matches[0].score == matches[1].score:
        return None
    return matches[0]


def load_rows(csv_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]

    for col in ("wget_command", "status", "folder", "name", "access_url"):
        if col not in fieldnames:
            fieldnames.append(col)
        for row in rows:
            row.setdefault(col, "")

    return rows, fieldnames


def save_rows(csv_path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def iter_message_commands(messages: Iterable[MailMessage]) -> Iterable[tuple[MailMessage, str]]:
    seen: set[str] = set()
    for message in messages:
        for cmd in extract_wget_commands(message.content):
            if cmd in seen:
                continue
            seen.add(cmd)
            yield message, cmd


def update_csv_from_messages(csv_path: Path, messages: list[MailMessage], dry_run: bool) -> int:
    rows, fieldnames = load_rows(csv_path)
    pending_rows = [
        idx
        for idx, row in enumerate(rows)
        if not str(row.get("wget_command") or "").strip()
    ]

    updates: list[tuple[int, str, str]] = []
    unmatched: list[str] = []

    for message, command in iter_message_commands(messages):
        if not pending_rows:
            break

        match = choose_row_for_message(message, rows, pending_rows)
        if match is None:
            unmatched.append(message.subject or message.message_id)
            continue

        row_idx = match.row_idx
        rows[row_idx]["wget_command"] = command
        updates.append((row_idx, str(rows[row_idx].get("name") or ""), match.reason))
        pending_rows = [idx for idx in pending_rows if idx != row_idx]

    for row_idx, name, reason in updates:
        print(f"[MATCH] row={row_idx} name={name} reason={reason}")
        print(f"[WGET] {rows[row_idx]['wget_command']}")

    if unmatched:
        print(f"[WARN] {len(unmatched)} message(s) had a wget command but could not be matched automatically.")
        for item in unmatched[:10]:
            print(f"  - {item}")

    if dry_run:
        print(f"[DRY-RUN] Would write {len(updates)} wget command(s) to {csv_path}")
        return len(updates)

    if updates:
        save_rows(csv_path, rows, fieldnames)
        print(f"[OK] Wrote {len(updates)} wget command(s) to {csv_path}")
    else:
        print("[OK] No new wget commands were imported.")
    return len(updates)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import NRAO wget commands from Outlook mail into small_selection.csv"
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="CSV file to update")
    parser.add_argument(
        "--backend",
        choices=("mail", "outlook"),
        default="mail",
        help="Mail source backend. Defaults to macOS Mail because Outlook AppleScript is unreliable here.",
    )
    parser.add_argument(
        "--messages-json",
        type=Path,
        help="Optional JSON file with message objects for testing instead of reading Outlook live",
    )
    parser.add_argument("--days-back", type=int, default=14, help="How many days of mail to scan")
    parser.add_argument(
        "--inbox-only",
        action="store_true",
        help="Only scan the Inbox. By default all Outlook folders are scanned.",
    )
    parser.add_argument(
        "--unread-only",
        action="store_true",
        help="Only scan unread mail. By default read mail is included too.",
    )
    parser.add_argument(
        "--include-read",
        action="store_true",
        help="Deprecated no-op kept for compatibility; read mail is already included by default.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show matches without writing the CSV")
    parser.add_argument("--debug", action="store_true", help="Print Outlook query diagnostics")
    parser.add_argument(
        "--mail-account",
        default=DEFAULT_MAIL_ACCOUNT,
        help=f'Mail app account name (default: "{DEFAULT_MAIL_ACCOUNT}")',
    )
    parser.add_argument("--mailbox", default="INBOX", help='Mail app mailbox name (default: "INBOX")')
    parser.add_argument(
        "--sender",
        default="do-not-reply@nrao.edu",
        help="Optional sender substring filter for Mail app mode",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        if args.messages_json:
            messages = read_messages_from_file(args.messages_json)
        elif args.backend == "mail":
            messages = fetch_mail_app_messages(
                days_back=args.days_back,
                account_name=args.mail_account,
                mailbox_name=args.mailbox,
                sender_filter=args.sender,
                debug=args.debug,
            )
        else:
            messages = fetch_outlook_messages(
                days_back=args.days_back,
                unread_only=args.unread_only,
                inbox_only=args.inbox_only,
                debug=args.debug,
            )
    except subprocess.CalledProcessError as exc:
        print(exc.stderr.strip() or str(exc), file=sys.stderr)
        print("Outlook access failed. If Automation permissions are blocked, grant Codex/Terminal access to Outlook and try again.", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Failed to load messages: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    print(f"[INFO] Loaded {len(messages)} message(s)")
    return 0 if update_csv_from_messages(args.csv, messages, dry_run=args.dry_run) >= 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
