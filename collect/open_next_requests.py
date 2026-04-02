from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path


DEFAULT_CSV = Path("/Users/u1528314/repos/radioastro-ml/collect/small_subset/small_selection.csv")


def extract_request_id(access_url: str) -> str:
    match = re.search(r"/#/(?:productViewer|productviewer)/([^/?#\s]+)", access_url or "")
    if match:
        return match.group(1)
    return ""


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def is_pending_request(row: dict[str, str]) -> bool:
    access_url = (row.get("access_url") or "").strip()
    wget_command = (row.get("wget_command") or "").strip()
    return bool(access_url) and not wget_command


def choose_rows(rows: list[dict[str, str]], limit: int) -> list[tuple[int, dict[str, str]]]:
    picked: list[tuple[int, dict[str, str]]] = []
    for idx, row in enumerate(rows):
        if is_pending_request(row):
            picked.append((idx, row))
        if len(picked) >= limit:
            break
    return picked


def print_selection(selected: list[tuple[int, dict[str, str]]]) -> None:
    print("[INFO] Selected rows in CSV order:")
    for order, (idx, row) in enumerate(selected, start=1):
        name = (row.get("name") or "").strip()
        access_url = (row.get("access_url") or "").strip()
        print(f"{order:02d}. id={name} row={idx} url={access_url}")


def open_in_firefox(selected: list[tuple[int, dict[str, str]]]) -> None:
    script_lines = [
        'tell application "Firefox"',
        "activate",
    ]

    for idx, (_, row) in enumerate(selected):
        url = (row.get("access_url") or "").replace('"', '\\"')
        if idx == 0:
            script_lines.append(f'open location "{url}"')
        else:
            script_lines.append("tell application \"System Events\"")
            script_lines.append('keystroke "n" using {command down}')
            script_lines.append("end tell")
            script_lines.append("delay 0.2")
            script_lines.append(f'open location "{url}"')

    script_lines.append("end tell")
    script = "\n".join(script_lines)

    subprocess.run(["osascript", "-e", script], check=True, text=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Open the next pending NRAO request links from small_selection.csv in Firefox."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="CSV file to read")
    parser.add_argument("--limit", type=int, default=20, help="How many pending rows to select")
    parser.add_argument("--open", action="store_true", help="Open the selected URLs in Firefox")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = load_rows(args.csv)
    selected = choose_rows(rows, args.limit)

    if not selected:
        print("[OK] No pending rows with blank wget_command were found.")
        return 0

    print_selection(selected)

    if args.open:
        try:
            open_in_firefox(selected)
            print(f"[OK] Opened {len(selected)} URL(s) in Firefox.")
        except subprocess.CalledProcessError as exc:
            print(exc.stderr or str(exc), file=sys.stderr)
            return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
