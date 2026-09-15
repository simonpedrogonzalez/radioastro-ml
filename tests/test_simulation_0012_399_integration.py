"""Opt-in real-CASA integration tests for the simulation package.

Run with:

    RUN_CASA_SIMULATION_INTEGRATION=1 casa --nogui --nologger \
        -c tests/test_simulation_0012_399_integration.py
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.imaging import measure_pb_region, vla_pipeline_annulus_rms
from scripts.simulation import (
    phase_center_point_source,
    phase_center_point_source_from_snr,
    simulate_ms,
    theoretical_vla_simplenoise,
)


RUN_INTEGRATION = os.environ.get("RUN_CASA_SIMULATION_INTEGRATION") == "1"
FIXTURE = ROOT / "tests/fixtures/simulation/vla_pipeline_0012_399"
SOURCE_MS = ROOT / "collect/extracted/0012-399/0012-399/0012-399.ms"
EXPECTED_PIPELINE_RMS = 0.00024389197254258183
ECT_EQUIVALENT_SEFD_JY = 236.7
ECT_NOISE_PARAMETERS = {
    "band": "C",
    "sampler": "8bit",
    "sefd_jy": ECT_EQUIVALENT_SEFD_JY,
    "eta_c": 0.93,
}


def _persistent_tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        if item.name == "table.lock":
            continue
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        with item.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _casa_version() -> str:
    import casatools

    function = getattr(casatools, "version_string", None)
    return str(function()) if callable(function) else str(casatools.version())


def _visibility_noise_statistics(ms: Path) -> tuple[float, float, float, float, int]:
    import numpy as np
    from casatools import table

    count = 0
    real_sum = imag_sum = real_square_sum = imag_square_sum = 0.0
    tb = table()
    tb.open(str(ms), nomodify=True)
    try:
        columns = set(tb.colnames())
        for start in range(0, int(tb.nrows()), 2048):
            rows = min(2048, int(tb.nrows()) - start)
            data = np.asarray(tb.getcol("DATA", startrow=start, nrow=rows))
            flags = np.asarray(tb.getcol("FLAG", startrow=start, nrow=rows), dtype=bool)
            usable = ~flags
            if "FLAG_ROW" in columns:
                flag_row = np.asarray(
                    tb.getcol("FLAG_ROW", startrow=start, nrow=rows), dtype=bool
                ).reshape(1, 1, -1)
                usable &= ~flag_row
            values = data[usable]
            real = values.real.astype(float, copy=False)
            imag = values.imag.astype(float, copy=False)
            count += int(values.size)
            real_sum += float(real.sum())
            imag_sum += float(imag.sum())
            real_square_sum += float(np.square(real).sum())
            imag_square_sum += float(np.square(imag).sum())
    finally:
        tb.close()
    real_mean = real_sum / count
    imag_mean = imag_sum / count
    real_std = math.sqrt(real_square_sum / count - real_mean**2)
    imag_std = math.sqrt(imag_square_sum / count - imag_mean**2)
    return real_mean, imag_mean, real_std, imag_std, count


def _assert_weight_contract(test: unittest.TestCase, ms: Path, sigma: float | None) -> None:
    import numpy as np
    from casatools import table

    tb = table()
    tb.open(str(ms), nomodify=True)
    try:
        columns = set(tb.colnames())
        test.assertIn("SIGMA", columns)
        test.assertIn("WEIGHT", columns)
        test.assertIn("WEIGHT_SPECTRUM", columns)
        test.assertNotIn("SIGMA_SPECTRUM", columns)
        expected_sigma = 1.0 if sigma is None else sigma
        expected_weight = 1.0 if sigma is None else 1.0 / sigma**2
        for start in range(0, int(tb.nrows()), 4096):
            rows = min(4096, int(tb.nrows()) - start)
            actual_sigma = tb.getcol("SIGMA", startrow=start, nrow=rows)
            actual_weight = tb.getcol("WEIGHT", startrow=start, nrow=rows)
            actual_spectrum = tb.getcol("WEIGHT_SPECTRUM", startrow=start, nrow=rows)
            # MS SIGMA is commonly stored as float32, so compare at its precision.
            test.assertTrue(np.allclose(actual_sigma, expected_sigma, rtol=1e-6, atol=0))
            test.assertTrue(np.allclose(actual_weight, expected_weight, rtol=1e-6, atol=0))
            test.assertTrue(np.allclose(actual_spectrum, expected_weight, rtol=1e-6, atol=0))
    finally:
        tb.close()


@unittest.skipUnless(RUN_INTEGRATION, "set RUN_CASA_SIMULATION_INTEGRATION=1")
class PipelineRMSRegressionTests(unittest.TestCase):
    def test_pipeline_fixture_matches_the_producing_result(self):
        reference = json.loads((FIXTURE / "reference.json").read_text(encoding="utf-8"))
        qa = json.loads((FIXTURE / "qa.json").read_text(encoding="utf-8"))
        actual = vla_pipeline_annulus_rms(
            FIXTURE / "image", FIXTURE / "pb", FIXTURE / "clean_mask"
        )
        relative_tolerance = 1e-6 if _casa_version().startswith("6.6.6.18") else 5e-3
        self.assertTrue(
            math.isclose(actual, EXPECTED_PIPELINE_RMS, rel_tol=relative_tolerance, abs_tol=0),
            (actual, EXPECTED_PIPELINE_RMS, relative_tolerance),
        )
        self.assertEqual(reference["pipeline_rms_jy_per_beam_exact"], EXPECTED_PIPELINE_RMS)
        self.assertEqual(reference["casa_version"], "6.6.6.18")
        self.assertEqual(reference["pipeline_version"], "2025.1.0.36")
        self.assertEqual(qa["vla_background_rms_jy_per_beam"], 0.00024)

        ordinary = measure_pb_region(
            FIXTURE / "image",
            FIXTURE / "pb",
            pb_min=reference["pb_min_for_this_image"],
            pb_max=reference["pb_max_for_this_image"],
            exclude_mask=FIXTURE / "clean_mask",
        )
        self.assertGreater(abs(ordinary.rms_jy_per_beam - actual), 1e-6)


@unittest.skipUnless(RUN_INTEGRATION, "set RUN_CASA_SIMULATION_INTEGRATION=1")
class SimulationIntegrationTests(unittest.TestCase):
    def test_noise_only_data_weights_and_dirty_image_rms(self):
        import numpy as np
        from casatasks import tclean

        original_hash = _persistent_tree_hash(SOURCE_MS)
        sigma = theoretical_vla_simplenoise(
            SOURCE_MS, sefd_jy=ECT_EQUIVALENT_SEFD_JY, eta_c=0.93
        )
        with tempfile.TemporaryDirectory(prefix="simulation-noise-0012-399-") as temporary:
            root = Path(temporary)
            result = simulate_ms(
                SOURCE_MS,
                [],
                root / "noise.ms",
                noise_model="vla-thermal",
                noise_parameters=ECT_NOISE_PARAMETERS,
                seed=12345,
            )
            self.assertEqual(result.simplenoise_jy, sigma)
            self.assertEqual(result.seed, 12345)
            self.assertIsNone(result.component_list)
            self.assertTrue(result.metadata_json.is_file())
            self.assertTrue(result.metadata_text and result.metadata_text.is_file())
            self.assertEqual(
                json.loads(result.metadata_json.read_text(encoding="utf-8"))["schema_version"],
                2,
            )
            self.assertEqual(_persistent_tree_hash(SOURCE_MS), original_hash)

            real_mean, imag_mean, real_std, imag_std, count = _visibility_noise_statistics(
                result.ms_path
            )
            self.assertGreater(count, 1_000_000)
            self.assertLess(abs(real_mean), 0.01 * sigma)
            self.assertLess(abs(imag_mean), 0.01 * sigma)
            self.assertTrue(math.isclose(real_std, sigma, rel_tol=0.01))
            self.assertTrue(math.isclose(imag_std, sigma, rel_tol=0.01))
            _assert_weight_contract(self, result.ms_path, sigma)

            image_base = root / "noise_dirty"
            tclean(
                vis=str(result.ms_path),
                imagename=str(image_base),
                datacolumn="data",
                specmode="mfs",
                gridder="standard",
                stokes="I",
                deconvolver="hogbom",
                weighting="natural",
                imsize=[256, 256],
                cell="0.15arcsec",
                niter=0,
                interactive=False,
                parallel=False,
            )
            image = Path(f"{image_base}.image")
            pb = Path(f"{image_base}.pb")
            self.assertTrue(image.exists())
            self.assertTrue(pb.exists())
            central = measure_pb_region(image, pb, pb_min=0.5)
            expected = sigma / math.sqrt(5_738_686)
            self.assertTrue(
                math.isclose(central.rms_jy_per_beam, expected, rel_tol=0.1),
                (central.rms_jy_per_beam, expected),
            )
            self.assertTrue(
                math.isclose(
                    central.rms_jy_per_beam,
                    central.scaled_mad_jy_per_beam,
                    rel_tol=0.1,
                )
            )
            self.assertTrue(np.isfinite(central.rms_jy_per_beam))

    def test_component_prediction_and_noiseless_weights(self):
        from casatasks import ft
        from casatools import table
        import numpy as np

        original_hash = _persistent_tree_hash(SOURCE_MS)
        component = phase_center_point_source(SOURCE_MS, flux_jy=1.0)
        with tempfile.TemporaryDirectory(prefix="simulation-component-0012-399-") as temporary:
            root = Path(temporary)
            result = simulate_ms(SOURCE_MS, [component], root / "source.ms")
            self.assertTrue(result.component_list and result.component_list.is_dir())
            self.assertTrue(result.metadata_json.is_file())
            self.assertTrue(result.metadata_text and result.metadata_text.is_file())
            self.assertEqual(_persistent_tree_hash(SOURCE_MS), original_hash)
            _assert_weight_contract(self, result.ms_path, None)

            oracle = root / "oracle.ms"
            shutil.copytree(SOURCE_MS, oracle)
            ft(
                vis=str(oracle),
                complist=str(result.component_list),
                usescratch=True,
                incremental=False,
            )
            simulated_table = table()
            oracle_table = table()
            simulated_table.open(str(result.ms_path), nomodify=True)
            oracle_table.open(str(oracle), nomodify=True)
            try:
                for start in range(0, int(simulated_table.nrows()), 2048):
                    rows = min(2048, int(simulated_table.nrows()) - start)
                    simulated = simulated_table.getcol("DATA", startrow=start, nrow=rows)
                    expected = oracle_table.getcol("MODEL_DATA", startrow=start, nrow=rows)
                    flags = oracle_table.getcol("FLAG", startrow=start, nrow=rows)
                    self.assertTrue(np.allclose(simulated[~flags], expected[~flags], rtol=1e-6))
            finally:
                simulated_table.close()
                oracle_table.close()

    def test_full_source_noise_and_imaging_package_flow(self):
        from casatasks import tclean
        from scripts.imaging import CleanIterationsConfig, ImagingConfig, image_ms

        with tempfile.TemporaryDirectory(prefix="simulation-full-0012-399-") as temporary:
            root = Path(temporary)
            pilot = simulate_ms(
                SOURCE_MS,
                [],
                root / "pilot.ms",
                noise_model="vla-thermal",
                noise_parameters=ECT_NOISE_PARAMETERS,
                seed=24680,
            )
            pilot_base = root / "pilot_dirty"
            tclean(
                vis=str(pilot.ms_path),
                imagename=str(pilot_base),
                datacolumn="data",
                specmode="mfs",
                gridder="standard",
                stokes="I",
                deconvolver="hogbom",
                weighting="natural",
                imsize=[256, 256],
                cell="0.15arcsec",
                niter=0,
                interactive=False,
                parallel=False,
            )
            pilot_metrics = measure_pb_region(
                Path(f"{pilot_base}.image"), Path(f"{pilot_base}.pb"), pb_min=0.5
            )
            source = phase_center_point_source_from_snr(
                SOURCE_MS, 100.0, pilot_metrics.rms_jy_per_beam
            )
            simulated = simulate_ms(
                SOURCE_MS,
                [source],
                root / "source_noise.ms",
                noise_model="vla-thermal",
                noise_parameters=ECT_NOISE_PARAMETERS,
                seed=13579,
            )

            config = ImagingConfig(
                datacolumn="data",
                specmode="mfs",
                stokes="I",
                gridder="standard",
                deconvolver="hogbom",
                nterms=1,
                weighting="natural",
                robust=0.5,
                gain=0.1,
                mask_nbeams=None,
                clean=CleanIterationsConfig(niter=0),
            )
            original_image = image_ms(
                SOURCE_MS, config, root / "original_image", keep_intermediate_products=True
            )
            simulated_image = image_ms(
                simulated.ms_path,
                config,
                root / "simulated_image",
                keep_intermediate_products=True,
            )
            for imaging_result in (original_image, simulated_image):
                for path in (
                    imaging_result.dirty_image,
                    imaging_result.clean_image,
                    imaging_result.residual_image,
                    imaging_result.dirty_png,
                    imaging_result.clean_png,
                    imaging_result.residual_png,
                    imaging_result.qa_text,
                    imaging_result.qa_json,
                ):
                    self.assertTrue(path.exists(), path)

            # A 100-sigma source dominates a dirty residual's full-image MAD
            # through its sidelobes.  Use the independent noise-only pilot RMS,
            # which is the quantity supplied to the source constructor.
            measured_snr = (
                simulated_image.qa.metrics.residual.residual_abs_peak_jy_per_beam
                / pilot_metrics.rms_jy_per_beam
            )
            self.assertTrue(
                math.isclose(measured_snr, 100.0, rel_tol=0.1),
                {
                    "measured_snr": measured_snr,
                    "source_flux_jy": source["flux"][0],
                    "pilot_rms_jy_per_beam": pilot_metrics.rms_jy_per_beam,
                    "simulated_scaled_mad_jy_per_beam": (
                        simulated_image.qa.metrics.residual.scaled_mad_jy_per_beam
                    ),
                    "simulated_abs_peak_jy_per_beam": (
                        simulated_image.qa.metrics.residual.residual_abs_peak_jy_per_beam
                    ),
                },
            )
            self.assertNotEqual(
                original_image.qa.metrics.residual.scaled_mad_jy_per_beam,
                simulated_image.qa.metrics.residual.scaled_mad_jy_per_beam,
            )


if __name__ == "__main__":
    unittest.main()
