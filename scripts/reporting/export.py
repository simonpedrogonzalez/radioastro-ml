"""Export a Quarto experiment report as one self-contained HTML file."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit


_RESOURCE_ATTRIBUTES = {
    "img": ("src", "srcset"),
    "image": ("href",),
    "script": ("src",),
    "source": ("src", "srcset"),
    "video": ("poster",),
}
_CSS_URL = re.compile(r"url\(\s*(['\"]?)(.*?)\1\s*\)", re.IGNORECASE)


@dataclass(frozen=True)
class ReportExport:
    html_path: Path
    mode: str
    exported_bytes: int
    resource_count: int


def _resource_attributes(tag: str, attrs: list[tuple[str, str | None]]):
    if tag != "link":
        return _RESOURCE_ATTRIBUTES.get(tag, ())
    relation = next((value or "" for name, value in attrs if name == "rel"), "")
    return ("href",) if {"stylesheet", "icon"} & set(relation.lower().split()) else ()


class _Resources(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.urls: list[str] = []
        self.has_html = False
        self.in_style = False

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        self.has_html |= tag == "html"
        self.in_style |= tag == "style"
        names = _resource_attributes(tag, attrs)
        for name, value in attrs:
            if value is None:
                continue
            if name in names:
                if name == "srcset" and not value.lstrip().startswith("data:"):
                    self.urls.extend(part.strip().split()[0] for part in value.split(","))
                else:
                    self.urls.append(value)
            if name == "style":
                self.urls.extend(match[1] for match in _CSS_URL.findall(value))

    def handle_endtag(self, tag):
        if tag.lower() == "style":
            self.in_style = False

    def handle_data(self, data):
        if self.in_style:
            self.urls.extend(match[1] for match in _CSS_URL.findall(data))


def _validate_standalone(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    resources = _Resources()
    resources.feed(text)
    if not text.strip() or not resources.has_html:
        raise ValueError(f"Quarto did not produce a valid HTML document: {path}")

    unresolved = []
    embedded = 0
    for value in resources.urls:
        value = value.strip()
        scheme = urlsplit(value).scheme
        if not value or value.startswith("#") or scheme in {"mailto", "tel"}:
            continue
        if scheme == "data":
            embedded += 1
        else:
            unresolved.append(value)
    if "file://" in text.lower() or unresolved:
        detail = ", ".join(repr(value) for value in unresolved[:5])
        raise ValueError(f"Detached report has unresolved resources: {detail}")
    return embedded


def export_detached_report(
    report: str | Path,
    destination: str | Path,
    *,
    overwrite: bool = False,
) -> ReportExport:
    """Render, validate, and atomically install a standalone report."""
    source = Path(report).expanduser().resolve()
    target = Path(destination).expanduser().resolve()
    if source.suffix.lower() != ".qmd" or not source.is_file():
        raise FileNotFoundError(f"Quarto report not found: {source}")
    if target.suffix.lower() != ".html":
        raise ValueError("destination must end in .html")
    if target == source.with_suffix(".html"):
        raise ValueError("destination must not replace the normal report.html")
    if target.exists() and not overwrite:
        raise FileExistsError(f"Export destination already exists: {target}")

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".report-export-", dir=source.parent) as temporary:
        staging = Path(temporary)
        try:
            result = subprocess.run(
                [
                    "quarto",
                    "render",
                    source.name,
                    "--to",
                    "html",
                    "--output-dir",
                    staging.name,
                    "--embed-resources",
                ],
                cwd=source.parent,
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            raise RuntimeError(f"Could not run Quarto: {exc}") from exc
        if result.returncode:
            detail = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"Quarto export failed: {detail or result.returncode}")

        standalone = staging / f"{source.stem}.html"
        if not standalone.is_file():
            raise RuntimeError(f"Quarto did not create the expected export: {standalone}")
        resource_count = _validate_standalone(standalone)

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", dir=target.parent
        )
        os.close(descriptor)
        temporary_target = Path(temporary_name)
        try:
            shutil.copyfile(standalone, temporary_target)
            os.replace(temporary_target, target)
        finally:
            temporary_target.unlink(missing_ok=True)

    return ReportExport(
        target,
        "single-html",
        target.stat().st_size,
        resource_count,
    )


def main(argv: list[str] | None = None) -> ReportExport:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report")
    parser.add_argument("destination")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    exported = export_detached_report(
        args.report, args.destination, overwrite=args.overwrite
    )
    print(f"Detached report: {exported.html_path}")
    return exported


if __name__ == "__main__":
    main()


__all__ = ["ReportExport", "export_detached_report"]
