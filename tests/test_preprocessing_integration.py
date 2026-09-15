"""Opt-in end-to-end retention test using a real Measurement Set and CASA."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.imaging import CleanIterationsConfig, DefaultImagingConfig, image_ms
from scripts.imaging.metadata import DEFAULT_EXTRACTED_MS_ROOT, resolve_path
from scripts.preprocessing import finalize_simulation_sample, load_dataset_manifest
from scripts.simulation import phase_center_point_source, simulate_ms


@unittest.skipUnless(
    os.environ.get("RUN_PREPROCESSING_INTEGRATION") == "1",
    "set RUN_PREPROCESSING_INTEGRATION=1 and run inside CASA",
)
class PreprocessingIntegrationTests(unittest.TestCase):
    def test_simulate_image_finalize_and_read_without_casa(self):
        source_ms = resolve_path("0012-399", DEFAULT_EXTRACTED_MS_ROOT).path
        with tempfile.TemporaryDirectory(prefix="preprocessing-0012-399-") as temporary:
            dataset_root = Path(temporary)
            sample_root = dataset_root / "sample"
            simulation = simulate_ms(
                source_ms,
                [phase_center_point_source(source_ms, flux_jy=0.01)],
                sample_root / "simulation/sample.ms",
            )
            config = replace(
                DefaultImagingConfig,
                nterms=1,
                mask_nbeams=None,
                clean=CleanIterationsConfig(niter=0),
            )
            imaging = image_ms(
                simulation.ms_path,
                config,
                sample_root / "simulation/default_imaging",
                imsize=(64, 64),
            )
            finalized = finalize_simulation_sample(
                sample_root,
                sample_id="sample",
                label_id=0,
                label_name="baseline",
                imaging_result=imaging,
                simulation_result=simulation,
                dataset_index=dataset_root / "dataset.json",
            )
            self.assertFalse(simulation.ms_path.exists())
            self.assertTrue(finalized.cleanup.audit_path.is_file())
            self.assertEqual(
                load_dataset_manifest(dataset_root / "dataset.json").samples,
                (finalized.manifest.path,),
            )
            code = (
                "from scripts.preprocessing.schema import load_sample_manifest; "
                f"m=load_sample_manifest({str(finalized.manifest.path)!r}); "
                "assert m.sample_id == 'sample'; assert m.corruptions == ()"
            )
            completed = subprocess.run(
                ["python3", "-c", code],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
