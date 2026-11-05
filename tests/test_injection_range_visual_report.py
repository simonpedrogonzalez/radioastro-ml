"""Focused checks for the explicit visual-report case and page contracts."""
import json
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.injection_range_visual_report import requested_cases, requested_stops, write_html


class VisualReportTest(unittest.TestCase):
    def setUp(self):
        self.source = dict(id="0449+113", nominal_source_snr=81.8530386613495,
                           selected_antennas=[8, 11])

    def test_physical_stops_and_equivalent_phase_endpoints(self):
        stops = requested_stops(self.source)
        for kind, control in (("phase", 0), ("amp", 1), ("noise", 1)):
            self.assertEqual([s["value"] for s in stops[kind]],
                             sorted(s["value"] for s in stops[kind]))
            self.assertIn(control, [s["value"] for s in stops[kind]])
        negative = next(s for s in stops["phase"] if s["value"] == -180)
        positive = next(s for s in stops["phase"] if s["value"] == 180)
        self.assertEqual(len(negative["badges"]), 4)
        self.assertEqual(len(positive["badges"]), 4)
        self.assertEqual(len([s for s in stops["noise"] if not s["admissible"]]), 1)
        self.assertTrue(all(abs(s["a"] / self.source["nominal_source_snr"] - s["r"]) < 1e-12
                            for s in stops["noise"]))
        sliders, cases = requested_cases(dict(sources=[self.source], seeds=[0, 1, 2]))
        phase = sliders[self.source["id"]]["phase"]
        self.assertEqual(phase[0]["cases"], phase[-1]["cases"])
        self.assertEqual(len(cases), len(set(cases)))

    def test_standalone_page_script_and_controls(self):
        sliders, cases = requested_cases(dict(sources=[self.source], seeds=[0, 1, 2]))
        sid = self.source["id"]
        manifest = dict(sources=[self.source], antenna_roles={sid: {"8": "low contribution",
                                                                    "11": "high contribution"}},
                        sliders=sliders, seeds=[0, 1, 2])
        data = {cid: dict(y=.1, frames={name: f"images/{name}.png" for name in
                                           ("truth", "recovered", "error", "dirty", "clean", "residual")})
                for cid in cases}
        with tempfile.TemporaryDirectory() as tmp:
            write_html(Path(tmp), manifest, data, {})
            page = (Path(tmp) / "report.html").read_text()
            self.assertEqual(page.count("<script>"), 1)
            self.assertIn("modalview", page)
            self.assertIn("touchmove", page)
            self.assertNotIn("__PAYLOAD__", page)
            payload = json.loads(page.split("const DATA=", 1)[1].split(";", 1)[0])
            self.assertEqual(len(payload["sources"][sid]["sliders"]), 3)
            script = re.search(r"<script>(.*?)</script>", page, re.S).group(1)
            script_path = Path(tmp) / "report.js"
            script_path.write_text(script)
            subprocess.run(["node", "--check", str(script_path)], check=True,
                           capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
