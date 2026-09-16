from __future__ import annotations

import base64
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.reporting import ReportExport, export_detached_report


STANDALONE = (
    '<!DOCTYPE html><html><body><img src="data:image/png;base64,eA==">'
    '<a href="https://example.com">external link</a></body></html>'
)


class DetachedReportTests(unittest.TestCase):
    def _fixture(self, root: Path) -> Path:
        report = root / "report.qmd"
        report.write_text("# Source report\n", encoding="utf-8")
        return report

    @staticmethod
    def _rendering(html: str):
        def run(command, *, cwd, **kwargs):
            output_dir = Path(cwd) / command[command.index("--output-dir") + 1]
            (output_dir / "report.html").write_text(html, encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, "", "")

        return run

    def test_exports_standalone_html_without_changing_normal_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = self._fixture(root)
            normal = root / "report.html"
            normal.write_text("normal", encoding="utf-8")
            destination = root / "detached.html"

            with patch("scripts.reporting.export.subprocess.run", self._rendering(STANDALONE)):
                result = export_detached_report(report, destination)

            self.assertIsInstance(result, ReportExport)
            self.assertEqual(result.html_path, destination.resolve())
            self.assertEqual(result.resource_count, 1)
            self.assertEqual(result.exported_bytes, destination.stat().st_size)
            self.assertEqual(normal.read_text(encoding="utf-8"), "normal")

    def test_refuses_unresolved_resource_and_existing_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = self._fixture(root)
            destination = root / "detached.html"
            destination.write_text("existing", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                export_detached_report(report, destination)

            destination.unlink()
            linked = '<html><body><img src="images/dirty.png"></body></html>'
            with patch("scripts.reporting.export.subprocess.run", self._rendering(linked)):
                with self.assertRaisesRegex(ValueError, "unresolved resources"):
                    export_detached_report(report, destination)
            self.assertFalse(destination.exists())

    def test_overwrite_is_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = self._fixture(root)
            destination = root / "detached.html"
            destination.write_text("old", encoding="utf-8")
            with patch("scripts.reporting.export.subprocess.run", self._rendering(STANDALONE)):
                export_detached_report(report, destination, overwrite=True)
            self.assertEqual(destination.read_text(encoding="utf-8"), STANDALONE)

    @unittest.skipUnless(
        os.environ.get("RUN_QUARTO_INTEGRATION") == "1" and shutil.which("quarto"),
        "set RUN_QUARTO_INTEGRATION=1 to run the real Quarto export",
    )
    def test_real_quarto_export_embeds_image(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = root / "report.qmd"
            report.write_text(
                "---\nformat:\n  html:\n    embed-resources: false\n---\n\n![pixel](pixel.png)\n",
                encoding="utf-8",
            )
            (root / "pixel.png").write_bytes(
                base64.b64decode(
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
                )
            )
            destination = root / "detached.html"
            result = export_detached_report(report, destination)
            self.assertGreaterEqual(result.resource_count, 1)
            self.assertIn("data:image/png", destination.read_text(encoding="utf-8"))
            self.assertFalse((root / "report.html").exists())


if __name__ == "__main__":
    unittest.main()
