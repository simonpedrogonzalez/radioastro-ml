"""Validated, dry-run-first cleanup for compact simulation samples."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from .fits import validate_fits_triplet
from .schema import atomic_write_json, load_sample_manifest, referenced_files


@dataclass(frozen=True)
class CleanupEntry:
    path: Path
    bytes: int


@dataclass(frozen=True)
class CleanupReport:
    sample_dir: Path
    manifest: Path
    dry_run: bool
    removed: tuple[CleanupEntry, ...]
    retained: tuple[Path, ...]
    reclaimed_bytes: int
    unexpected: tuple[Path, ...]
    audit_path: Path | None


_REMOVABLE_DIRECTORY_SUFFIXES = (
    ".ms",
    ".cl",
    ".g",
    ".image",
    ".image.tt0",
    ".model",
    ".model.tt0",
    ".residual",
    ".residual.tt0",
    ".psf",
    ".psf.tt0",
    ".mask",
    ".mask.tt0",
    ".sumwt",
    ".sumwt.tt0",
    ".pb",
    ".pb.tt0",
    ".weight",
    ".weight.tt0",
)
_REMOVABLE_FILE_SUFFIXES = (".png", ".log", ".last", ".fits.tmp", ".tmp")
_REMOVABLE_DIRECTORY_NAMES = {"pipeline", "mask_probe", "flagversions"}
_REMOVABLE_FILE_NAMES = {"table.lock"}


def _tree_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _has_retained_descendant(path: Path, retained: set[Path]) -> bool:
    for item in retained:
        try:
            item.relative_to(path)
        except ValueError:
            continue
        return True
    return False


def _recognized(path: Path) -> bool:
    lower = path.name.casefold()
    if path.is_dir():
        return lower in _REMOVABLE_DIRECTORY_NAMES or lower.endswith(
            _REMOVABLE_DIRECTORY_SUFFIXES
        )
    return lower in _REMOVABLE_FILE_NAMES or lower.endswith(_REMOVABLE_FILE_SUFFIXES)


def _plan(sample_dir: Path, retained: set[Path]) -> tuple[list[CleanupEntry], list[Path]]:
    removed: list[CleanupEntry] = []
    unexpected: list[Path] = []
    covered: set[Path] = set()
    for path in sorted(sample_dir.rglob("*"), key=lambda item: (len(item.parts), str(item))):
        resolved = path.resolve(strict=False)
        if resolved in retained or any(parent in covered for parent in path.parents):
            continue
        if path.is_symlink():
            unexpected.append(path)
            continue
        if _recognized(path) and not _has_retained_descendant(resolved, retained):
            removed.append(CleanupEntry(path, _tree_size(path)))
            if path.is_dir():
                covered.add(path)
    selected = {entry.path for entry in removed}
    for path in sorted(sample_dir.rglob("*")):
        if path.is_dir():
            continue
        if (
            path.resolve(strict=False) in retained
            or path in selected
            or any(parent in covered for parent in path.parents)
            or path.name.startswith(".")
        ):
            continue
        if not _recognized(path):
            unexpected.append(path)
    return removed, sorted(set(unexpected))


def cleanup_simulation_sample(
    sample_dir: str | Path,
    *,
    manifest: str = "sample.json",
    dry_run: bool = True,
    strict: bool = True,
) -> CleanupReport:
    raw_root = Path(sample_dir).expanduser()
    if raw_root.is_symlink():
        raise ValueError(f"Sample directory cannot be a symlink: {raw_root}")
    root = raw_root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Sample directory does not exist: {root}")
    sample = load_sample_manifest(root / manifest, require_files=True, verify_integrity=True)
    validate_fits_triplet(sample.products)
    retained = {path.resolve() for path in referenced_files(sample)} | {sample.path.resolve()}
    existing_audit = root / "cleanup.audit.json"
    if existing_audit.is_file():
        retained.add(existing_audit.resolve())
    removed, unexpected = _plan(root, retained)
    if strict and unexpected:
        raise ValueError(
            "Unexpected files prevent strict cleanup:\n"
            + "\n".join(f"  - {path}" for path in unexpected)
        )
    audit_path: Path | None = None
    if not dry_run:
        for entry in sorted(removed, key=lambda item: len(item.path.parts), reverse=True):
            if entry.path.is_dir():
                shutil.rmtree(entry.path)
            else:
                entry.path.unlink(missing_ok=True)
        audit_path = root / "cleanup.audit.json"
        atomic_write_json(
            audit_path,
            {
                "schema_version": 1,
                "sample_id": sample.sample_id,
                "manifest": sample.path.name,
                "reclaimed_bytes": sum(entry.bytes for entry in removed),
                "removed": [
                    {"path": entry.path.relative_to(root).as_posix(), "bytes": entry.bytes}
                    for entry in removed
                ],
                "retained": sorted(path.relative_to(root).as_posix() for path in retained),
                "unexpected": [path.relative_to(root).as_posix() for path in unexpected],
            },
        )
        # Revalidate after mutation so cleanup cannot report success with a damaged sample.
        load_sample_manifest(sample.path, require_files=True, verify_integrity=True)
        validate_fits_triplet(sample.products)
    return CleanupReport(
        sample_dir=root,
        manifest=sample.path,
        dry_run=dry_run,
        removed=tuple(removed),
        retained=tuple(sorted(retained)),
        reclaimed_bytes=sum(entry.bytes for entry in removed),
        unexpected=tuple(unexpected),
        audit_path=audit_path,
    )


__all__ = ["CleanupEntry", "CleanupReport", "cleanup_simulation_sample"]
