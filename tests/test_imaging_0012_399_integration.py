"""Opt-in real-CASA regression tests for the documented reference MS.

Run direct imaging with:

    RUN_CASA_IMAGING_INTEGRATION=1 casa --nogui --nologger \
        -c tests/test_imaging_0012_399_integration.py

Run the VLA test with the pipeline-enabled CASA application and additionally set
``RUN_VLA_PIPELINE_INTEGRATION=1``.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.imaging import BeamRegion, DefaultImagingConfig, image_ms, image_ms_VLA_pipe
from scripts.imaging.metadata import DEFAULT_EXTRACTED_MS_ROOT, resolve_path


class Imaging0012399IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ms = resolve_path("0012-399", DEFAULT_EXTRACTED_MS_ROOT).path

    @unittest.skipUnless(
        os.environ.get("RUN_CASA_IMAGING_INTEGRATION") == "1",
        "set RUN_CASA_IMAGING_INTEGRATION=1 to run real imaging",
    )
    def test_direct_exterior_region_metric_contract(self):
        with tempfile.TemporaryDirectory(prefix="imaging-0012-399-") as temporary:
            # The checked-in canonical 0012-399.ms is a split MS with only DATA.
            # Keep the package default explicit and override it for this fixture.
            # Its 1.8% fractional bandwidth also leads the VLA Pipeline to use
            # one Taylor term; CASA 6.7 can abort native mtmfs cleaning here
            # with nterms=2, so match that known-good reference decision.
            config = replace(DefaultImagingConfig, datacolumn="data", nterms=1)
            region = BeamRegion(min_radius_beams=3.0)
            result = image_ms(
                "0012-399",
                config,
                temporary,
                metric_region_resolver=lambda _: region,
            )
            for path in (
                result.dirty_fits,
                result.clean_fits,
                result.residual_fits,
                result.qa_text,
                result.qa_json,
            ):
                self.assertTrue(path.exists(), path)
            self.assertIsNone(result.dirty_image)
            self.assertIsNone(result.clean_image)
            self.assertIsNone(result.residual_image)
            self.assertIsNone(result.dirty_png)
            payload = json.loads(result.qa_json.read_text(encoding="utf-8"))
            metrics = payload["metrics"]
            self.assertEqual(payload["schema_version"], 3)
            self.assertEqual(payload["visibility_id"], "0012-399")
            self.assertEqual(payload["engine"], "direct")
            self.assertEqual(
                payload["products"],
                {
                    "dirty_fits": "dirty.fits.gz",
                    "clean_fits": "clean.fits.gz",
                    "residual_fits": "residual.fits.gz",
                    "qa_text": "qa.txt",
                    "qa_json": "qa.json",
                },
            )
            self.assertFalse((result.output_dir / "dirty.png").exists())
            self.assertFalse((result.output_dir / "clean.png").exists())
            self.assertFalse((result.output_dir / "residual.png").exists())
            self.assertEqual(set(payload["plot_recipes"]), {"dirty", "clean", "residual"})
            self.assertEqual(
                metrics["region"],
                {"min_radius_beams": 3.0, "max_radius_beams": None},
            )
            self.assertGreater(metrics["residual"]["n_pixels"], 0)
            self.assertGreater(metrics["residual"]["area_synthesized_beams"], 0)
            self.assertEqual(payload["resolved_config"]["grid"]["image_grid"]["imsize"], [256, 256])
            self.assertAlmostEqual(
                metrics["residual"]["peak_over_scaled_mad"],
                metrics["residual"]["residual_abs_peak_jy_per_beam"]
                / metrics["residual"]["scaled_mad_jy_per_beam"],
            )
            self.assertIsNotNone(payload["tclean_summary"])

    @unittest.skipUnless(
        os.environ.get("RUN_CASA_IMAGING_INTEGRATION") == "1"
        and os.environ.get("RUN_VLA_PIPELINE_INTEGRATION") == "1",
        "set both integration flags and use pipeline-enabled CASA",
    )
    def test_vla_pipeline_same_contract_and_background_rms(self):
        with tempfile.TemporaryDirectory(prefix="imaging-vla-0012-399-") as temporary:
            result = image_ms_VLA_pipe(
                "0012-399",
                temporary,
                metric_region=BeamRegion(min_radius_beams=3.0),
            )
            payload = json.loads(result.qa_json.read_text(encoding="utf-8"))
            for path in (result.dirty_fits, result.clean_fits, result.residual_fits):
                self.assertTrue(path.exists(), path)
            self.assertIsNone(result.dirty_image)
            self.assertEqual(payload["engine"], "vla_pipeline")
            self.assertIsNone(payload["resolved_config"])
            self.assertEqual(payload["effective_imaging_parameters"]["requested_imsize"], [256, 256])
            self.assertEqual(
                payload["effective_imaging_parameters"]["measured_geometry"]["imsize"],
                [256, 256],
            )
            self.assertIsNotNone(payload["pipeline_background"])
            self.assertGreater(payload["pipeline_background"]["rms_jy_per_beam"], 0)
            self.assertEqual(set(payload["metrics"]), {
                "region",
                "clean_peak_jy_per_beam",
                "residual",
                "dynamic_range_rms",
                "dynamic_range_scaled_mad",
            })


if __name__ == "__main__":
    unittest.main()
