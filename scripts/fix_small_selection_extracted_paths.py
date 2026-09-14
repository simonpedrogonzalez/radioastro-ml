from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


DEFAULT_CSV_PATH = Path(
    "/Users/u1528314/repos/radioastro-ml/collect/small_subset/small_selection.csv"
)
DEFAULT_EXTRACTED_ROOT = Path(
    "/Users/u1528314/repos/radioastro-ml/collect/extracted"
)
IGNORED_SUBDIRS = {"selfcal", "selfcal_test", "diagnostics", "experiments"}


def normalize_extracted_ms_path(path_str: str | Path) -> Path:
    path = Path(str(path_str).strip()).expanduser()
    parts = list(path.parts)
    for idx, part in enumerate(parts[:-1]):
        if part == "collect" and parts[idx + 1].startswith("extracted"):
            parts[idx + 1] = "extracted"
            return Path(*parts)
    return path


def is_ignored_candidate(ms_path: Path, sample_dir: Path) -> bool:
    try:
        rel_parts = ms_path.relative_to(sample_dir).parts
    except ValueError:
        return True
    return any(part in IGNORED_SUBDIRS for part in rel_parts) or ms_path.name.endswith("_imgprep.ms")


def scan_sample_dir(sample_dir: Path, folder: str) -> tuple[Path | None, str]:
    expected = sample_dir / folder / f"{folder}.ms"
    if expected.exists():
        return expected, "expected"

    hits = sorted(
        ms_path
        for ms_path in sample_dir.rglob("*.ms")
        if not is_ignored_candidate(ms_path, sample_dir)
    )
    if not hits:
        return None, "no_ms_found"
    if len(hits) == 1:
        return hits[0], "unique_scan"
    return None, f"ambiguous_scan:{len(hits)}"


def resolve_verified_ms_path(row: dict[str, str], extracted_root: Path) -> tuple[Path | None, str]:
    current_value = str(row.get("extracted_ms", "")).strip()
    if not current_value:
        return None, "blank"

    normalized = normalize_extracted_ms_path(current_value)
    if normalized.exists():
        if str(normalized) == current_value:
            return normalized, "current_exists"
        return normalized, "normalized_exists"

    folder = str(row.get("folder", "")).strip()
    if not folder:
        return None, "missing_folder"

    sample_dir = extracted_root / folder
    if not sample_dir.is_dir():
        return None, "sample_dir_missing"

    resolved, reason = scan_sample_dir(sample_dir, folder)
    if resolved is None:
        return None, reason
    return resolved, reason


def load_rows(csv_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if reader.fieldnames is None:
            raise SystemExit(f"No CSV header found in {csv_path}")
        return rows, list(reader.fieldnames)


def write_rows(csv_path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fix extracted_ms paths in small_selection.csv by verifying files under collect/extracted."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--extracted-root", type=Path, default=DEFAULT_EXTRACTED_ROOT)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write verified path updates back into the CSV. Without this flag, run as dry-run.",
    )
    args = parser.parse_args()

    csv_path = args.csv.expanduser()
    extracted_root = args.extracted_root.expanduser()
    rows, fieldnames = load_rows(csv_path)

    stats: Counter[str] = Counter()
    updated_rows = 0
    unresolved_examples: list[str] = []
    changed_examples: list[str] = []

    for row in rows:
        current_value = str(row.get("extracted_ms", "")).strip()
        resolved, reason = resolve_verified_ms_path(row, extracted_root)
        stats[reason] += 1

        if resolved is None:
            if current_value and len(unresolved_examples) < 20:
                folder = str(row.get("folder", "")).strip()
                unresolved_examples.append(f"{folder}: {current_value} [{reason}]")
            continue

        resolved_str = str(resolved)
        if resolved_str != current_value:
            if len(changed_examples) < 20:
                folder = str(row.get("folder", "")).strip()
                changed_examples.append(f"{folder}: {current_value} -> {resolved_str} [{reason}]")
            row["extracted_ms"] = resolved_str
            updated_rows += 1

    print(f"csv={csv_path}")
    print(f"extracted_root={extracted_root}")
    print(f"rows={len(rows)}")
    print(f"updated_rows={updated_rows}")
    for key in sorted(stats):
        print(f"{key}={stats[key]}")

    if changed_examples:
        print("\nChanged examples:")
        for item in changed_examples:
            print(item)

    if unresolved_examples:
        print("\nUnresolved examples:")
        for item in unresolved_examples:
            print(item)

    if args.apply:
        write_rows(csv_path, rows, fieldnames)
        print(f"\nWrote updates to {csv_path}")
    else:
        print("\nDry run only; no file was modified.")


if __name__ == "__main__":
    main()
