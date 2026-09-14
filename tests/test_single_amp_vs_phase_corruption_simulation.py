from __future__ import annotations

import json
import importlib.util
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from scripts import single_amp_vs_phase_corruption_simulation as experiment


class SingleAmplitudeVersusPhaseExperimentTests(unittest.TestCase):
    def _thermal_run(self, root: Path, name: str) -> Path:
        run = root / name
        simulation = run / "0012-399" / "simulation"
        ms = simulation / "0012-399_matched_snr.ms"
        result = simulation / "default_imaging"
        ms.mkdir(parents=True)
        result.mkdir()
        metadata = simulation / "0012-399_matched_snr.simulation.json"
        metadata.write_text("{}\n", encoding="utf-8")
        for filename in ("dirty.png", "clean.png", "residual.png"):
            (result / filename).write_bytes(b"png")
        (result / "qa.json").write_text("{}\n", encoding="utf-8")
        manifest = {
            "configuration": {"imsize": [256, 256]},
            "samples": [
                {
                    "id": "0012-399",
                    "simulation_ms": str(ms.relative_to(run)),
                    "simulation_result_dir": str(result.relative_to(run)),
                    "simulation_metadata": str(metadata.relative_to(run)),
                    "metric_region": {
                        "min_radius_beams": 3.0,
                        "max_radius_beams": 6.7,
                    },
                }
            ],
        }
        (run / "report.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return run

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is supplied by CASA")
    def test_constant_curve_obeys_current_builder_protocol(self):
        import numpy as np

        curve = experiment._ConstantCurve(1.1)
        sampled = curve.sample(np.random.default_rng(1), times=np.array([1.0, 2.0]))
        self.assertIs(sampled, curve)
        np.testing.assert_array_equal(
            sampled.eval(np.array([1.0, 5.0, 9.0])),
            np.array([1.1, 1.1, 1.1]),
        )

    def test_explicit_source_run_resolves_complete_products(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = self._thermal_run(Path(temporary), "thermal")
            source = experiment.find_source_simulation(run)

        self.assertEqual(source.experiment_dir, run.resolve())
        self.assertEqual(source.imsize, (256, 256))
        self.assertEqual(source.metric_min_radius_beams, 3.0)
        self.assertEqual(source.metric_max_radius_beams, 6.7)

    def test_discovery_uses_newest_qualifying_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            older = self._thermal_run(
                root, "extracted_thermal_simulation_comparison_20260101T000000"
            )
            newer = self._thermal_run(
                root, "extracted_thermal_simulation_comparison_20260102T000000"
            )
            with patch.object(experiment, "EXPERIMENTS_ROOT", root):
                source = experiment.find_source_simulation()

        self.assertNotEqual(older, newer)
        self.assertEqual(source.experiment_dir, newer.resolve())

    def test_missing_image_product_rejects_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = self._thermal_run(Path(temporary), "thermal")
            (run / "0012-399/simulation/default_imaging/residual.png").unlink()
            with self.assertRaisesRegex(RuntimeError, "residual.png"):
                experiment.find_source_simulation(run)

    def test_existing_output_is_refused_before_casa_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = self._thermal_run(root, "thermal")
            output = root / "already-there"
            output.mkdir()
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                experiment.run_experiment(source_run=run, output_dir=output)

    def test_errors_are_deliberately_significant(self):
        self.assertEqual(experiment.AMPLITUDE_GAIN, 1.50)
        self.assertEqual(experiment.PHASE_ERROR_DEGREES, 45.0)

    def test_manifest_requires_the_three_row_image_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = experiment.find_source_simulation(self._thermal_run(root, "thermal"))
            manifest = experiment._new_manifest(
                source,
                root / "experiment",
                root / "experiment/baseline/default_imaging",
                0,
                "ea01",
            )

        self.assertEqual(
            manifest["configuration"]["report_layout"],
            {
                "rows": [
                    "clean simulation with thermal noise",
                    "one-antenna constant amplitude-only error",
                    "one-antenna constant phase-only error",
                ],
                "columns": ["dirty", "clean", "residual"],
                "final_clean_comparison": [
                    "thermal simulation",
                    "one-antenna amplitude-only error",
                    "one-antenna phase-only error",
                ],
            },
        )
        self.assertIn("50%", manifest["description"])
        self.assertIn("+45 degree", manifest["description"])

    def test_report_template_fixes_case_order_and_image_columns(self):
        template = experiment.REPORT_TEMPLATE.read_text(encoding="utf-8")
        baseline = template.index("clean simulation with thermal noise")
        amplitude = template.index("one-antenna constant amplitude-only error")
        phase = template.index("one-antenna constant phase-only error")

        self.assertLess(baseline, amplitude)
        self.assertLess(amplitude, phase)
        self.assertIn('(\"dirty.png\", \"Dirty\")', template)
        self.assertIn('(\"clean.png\", \"Clean\")', template)
        self.assertIn('(\"residual.png\", \"Residual\")', template)

    def test_report_template_renders_exactly_three_image_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "baseline").mkdir()
            (root / "baseline/qa.json").write_text("{}\n", encoding="utf-8")
            manifest = {
                "source": {"baseline_result_dir": "baseline"},
                "configuration": {
                    "antenna_id": 0,
                    "antenna_name": "ea01",
                    "amplitude_gain": 1.5,
                    "phase_error_degrees": 45.0,
                },
                "variants": [
                    {
                        "name": "constant_amplitude",
                        "result_dir": "amplitude",
                        "corruption_plot": "amplitude-gain.png",
                    },
                    {
                        "name": "constant_phase",
                        "result_dir": "phase",
                        "corruption_plot": "phase-gain.png",
                    },
                ],
                "failures": [],
            }
            (root / "report.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            template = experiment.REPORT_TEMPLATE.read_text(encoding="utf-8")
            code = template.split("```{python}\n", 1)[1].rsplit("```", 1)[0]
            rendered = io.StringIO()
            with patch.object(Path, "cwd", return_value=root), redirect_stdout(rendered):
                exec(compile(code, str(experiment.REPORT_TEMPLATE), "exec"), {})

        report_html = rendered.getvalue()
        self.assertEqual(report_html.count('<div class="row-label">'), 3)
        self.assertEqual(report_html.count('<div class="image-cell">'), 9)
        self.assertEqual(report_html.count('<div class="clean-cell">'), 3)
        self.assertLess(
            report_html.index("Applied gain diagnostics"),
            report_html.index("Final CLEAN-only comparison"),
        )
        clean_panel = report_html.split("Final CLEAN-only comparison", 1)[1]
        self.assertLess(
            clean_panel.index("Thermal simulation"),
            clean_panel.index("Amplitude error"),
        )
        self.assertLess(
            clean_panel.index("Amplitude error"), clean_panel.index("Phase error")
        )


if __name__ == "__main__":
    unittest.main()
