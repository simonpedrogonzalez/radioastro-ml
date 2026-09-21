from __future__ import annotations

import base64
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    import numpy as np
except ImportError:  # CASA supplies NumPy for the flag-counting tests.
    np = None


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import compare_extracted_thermal_simulations as experiment
from scripts.simulation import natural_image_rms_from_simplenoise
from scripts.simulation import noise as noise_module


class FlagTable:
    def __init__(self, tables):
        self.tables = tables
        self.current = None

    def open(self, path, nomodify=True):
        self.current = self.tables.get(Path(path).name, self.tables.get("MAIN"))
        if self.current is None:
            raise FileNotFoundError(path)

    def close(self):
        self.current = None

    def nrows(self):
        return self.current["nrows"]

    def colnames(self):
        return list(self.current["columns"])

    def getcell(self, column, row):
        return self.current["columns"][column][row]

    def getcol(self, column, *args, **kwargs):
        value = np.asarray(self.current["columns"][column])
        start = kwargs.get("startrow", args[0] if args else 0)
        count = kwargs.get("nrow", args[1] if len(args) > 1 else -1)
        stop = None if count is None or count < 0 else start + count
        return value[start:stop] if value.ndim == 1 else value[..., start:stop]


def flag_tables(correlation_rows, description_ids, flags, flag_rows=None):
    main = {
        "nrows": len(description_ids),
        "columns": {
            "DATA_DESC_ID": np.asarray(description_ids),
            "FLAG": np.asarray(flags, dtype=bool),
        },
    }
    if flag_rows is not None:
        main["columns"]["FLAG_ROW"] = np.asarray(flag_rows, dtype=bool)
    return {
        "MAIN": main,
        "POLARIZATION": {
            "nrows": len(correlation_rows),
            "columns": {"CORR_TYPE": list(correlation_rows)},
        },
        "DATA_DESCRIPTION": {
            "nrows": len(correlation_rows),
            "columns": {"POLARIZATION_ID": list(range(len(correlation_rows)))},
        },
    }


@unittest.skipIf(np is None, "NumPy is not installed in this Python runtime")
class NaturalImageRMSTests(unittest.TestCase):
    def test_rr_ll_count_honors_flag_and_flag_row(self):
        flags = np.zeros((4, 2, 3), dtype=bool)
        flags[0, 0, 0] = True
        flags[1, :, :] = True  # Cross-hand flags do not affect Stokes-I counting.
        tables = flag_tables([(5, 6, 7, 8)], [0, 0, 0], flags, [False, False, True])
        with patch.object(noise_module, "_new_table", lambda: FlagTable(tables)):
            actual = natural_image_rms_from_simplenoise("example.ms", 0.7, chunk_rows=2)
        self.assertAlmostEqual(actual, 0.7 / math.sqrt(7))

    def test_mixed_rr_ll_and_xx_yy_mappings(self):
        flags = np.zeros((4, 2, 2), dtype=bool)
        tables = flag_tables(
            [(5, 6, 7, 8), (9, 10, 11, 12)],
            [0, 1],
            flags,
        )
        with patch.object(noise_module, "_new_table", lambda: FlagTable(tables)):
            actual = natural_image_rms_from_simplenoise("example.ms", 0.8)
        self.assertAlmostEqual(actual, 0.8 / math.sqrt(8))


class ExperimentDriverTests(unittest.TestCase):
    def test_annulus_runs_from_three_beams_to_border_minus_one_beam(self):
        resolved = SimpleNamespace(
            grid=SimpleNamespace(
                beam=SimpleNamespace(major_arcsec=2.0),
                image_grid=SimpleNamespace(
                    imsize=(256, 192),
                    cell_arcsec=(0.5, 0.5),
                ),
            )
        )
        region = experiment.metric_region_for_resolved_grid(resolved)
        self.assertEqual(region.min_radius_beams, 3.0)
        self.assertAlmostEqual(region.max_radius_beams, 22.75)
        self.assertEqual(
            experiment.metric_region_policy()["outer_border_margin_beams"],
            1.0,
        )

    def test_annulus_rejects_an_image_without_room(self):
        resolved = SimpleNamespace(
            grid=SimpleNamespace(
                beam=SimpleNamespace(major_arcsec=2.0),
                image_grid=SimpleNamespace(
                    imsize=(32, 32),
                    cell_arcsec=(0.5, 0.5),
                ),
            )
        )
        with self.assertRaisesRegex(ValueError, "too narrow"):
            experiment.metric_region_for_resolved_grid(resolved)

    def test_find_samples_returns_only_canonical_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical = root / "0012-399" / "0012-399" / "0012-399.ms"
            canonical.mkdir(parents=True)
            (root / "0012-399" / "0012-399_imgprep.ms").mkdir()
            (root / "0012-399" / "simulation" / "0012-399_100sigma.ms").mkdir(
                parents=True
            )
            (root / "other" / "wrong.ms").mkdir(parents=True)
            self.assertEqual(experiment.find_samples(root), [canonical.resolve()])
            self.assertEqual(
                experiment.find_samples(root, ["0012-399"]), [canonical.resolve()]
            )
            with self.assertRaises(FileNotFoundError):
                experiment.find_samples(root, ["missing"])

    def test_seed_is_stable_positive_and_sample_specific(self):
        first = experiment.stable_sample_seed("0012-399")
        self.assertEqual(first, experiment.stable_sample_seed("0012-399"))
        self.assertNotEqual(first, experiment.stable_sample_seed("0205+322"))
        self.assertGreater(first, 0)
        self.assertLessEqual(first, experiment._SEED_MAX)

    def test_manifest_write_is_atomic_and_keeps_failures_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.json"
            manifest = experiment._new_manifest(2)
            manifest["samples"].append({"id": "complete"})
            manifest["failures"].append(
                {"id": "failed", "stage": "simulation", "error": "test"}
            )
            experiment.write_manifest(path, manifest)
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual([item["id"] for item in loaded["samples"]], ["complete"])
            self.assertEqual([item["id"] for item in loaded["failures"]], ["failed"])
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_manifest_records_the_experiment_grid(self):
        manifest = experiment._new_manifest(1)

        self.assertEqual(
            manifest["configuration"]["imsize"],
            list(experiment.IMAGING_IMSIZE),
        )
        self.assertEqual(experiment.IMAGING_IMSIZE, (256, 256))
        self.assertEqual(manifest["configuration"]["fits_invalid_policy"], "fill")

    def test_legacy_manifest_cannot_be_resumed_into_new_region_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.ms"
            source.mkdir()
            output = root / "experiment"
            output.mkdir()
            (output / "report.json").write_text(
                json.dumps(
                    {
                        "configuration": {},
                        "total_samples": 1,
                        "samples": [],
                        "failures": [],
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch.object(experiment, "find_samples", return_value=[source]),
                patch.object(experiment, "_check_disk_space"),
            ):
                with self.assertRaisesRegex(RuntimeError, "new experiment directory"):
                    experiment.main(output)

    def test_interrupted_uncommitted_sample_output_is_removed_for_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            experiment_dir = Path(temporary).resolve()
            sample_dir = experiment_dir / "0846-261"
            imaging_dir = sample_dir / "original" / "default_imaging"
            imaging_dir.mkdir(parents=True)
            (imaging_dir / "partial-product").write_text("interrupted", encoding="utf-8")

            experiment._remove_interrupted_sample_output(sample_dir, experiment_dir)

            self.assertFalse(sample_dir.exists())

    def test_interrupted_output_cleanup_rejects_unexpected_entries(self):
        with tempfile.TemporaryDirectory() as temporary:
            experiment_dir = Path(temporary).resolve()
            sample_dir = experiment_dir / "0846-261"
            sample_dir.mkdir()
            (sample_dir / "keep-me.txt").write_text("manual file", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "unexpected entries"):
                experiment._remove_interrupted_sample_output(sample_dir, experiment_dir)

            self.assertTrue(sample_dir.exists())

    def test_manifest_is_committed_before_reporter_notification(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.ms"
            source.mkdir()
            output = root / "experiment"
            observed_sample_ids = []

            class InspectingReporter:
                def __init__(self, report_path, every):
                    self.manifest_path = Path(report_path).with_name("report.json")

                def sample_completed(self):
                    manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                    observed_sample_ids.append(
                        [entry["id"] for entry in manifest["samples"]]
                    )

                def finish(self):
                    pass

            with (
                patch.object(experiment, "find_samples", return_value=[source]),
                patch.object(experiment, "_check_disk_space"),
                patch.object(
                    experiment,
                    "process_sample",
                    return_value={"id": "source"},
                ),
                patch.object(experiment, "QuartoReporter", InspectingReporter),
            ):
                experiment.main(output)

            self.assertEqual(observed_sample_ids, [["source"]])

    @unittest.skipUnless(shutil.which("quarto"), "Quarto is not installed")
    def test_report_renders_comparison_rows_histograms_and_six_images(self):
        tiny_png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copyfile(
                ROOT / "scripts/reporting/vla_original_simulation_comparison.qmd",
                root / "report.qmd",
            )
            for index, result_dir in enumerate(
                (root / "original", root / "simulation_image")
            ):
                result_dir.mkdir()
                for name in ("dirty.png", "clean.png", "residual.png"):
                    (result_dir / name).write_bytes(tiny_png)
                metrics = {
                    "scaled_mad_jy_per_beam": 1e-4,
                    "peak_over_scaled_mad": 5.0,
                    "p99_over_scaled_mad": 2.4,
                    "dynamic_range": 1200.0,
                }
                qa = {
                    "schema_version": 1,
                    "effective_imaging_parameters": {
                        "requested_imsize": [256, 256],
                        "mask_nbeams": 6.0,
                        "measured_geometry": {},
                    },
                    "metrics": metrics,
                    "warnings": [],
                }
                if index == 1:
                    qa["schema_version"] = 2
                    qa["metrics"] = {
                        "region": {
                            "min_radius_beams": 3.0,
                            "max_radius_beams": None,
                        },
                        "clean_peak_jy_per_beam": 0.13,
                        "dynamic_range_rms": 1000.0,
                        "dynamic_range_scaled_mad": 1300.0,
                        "residual": {
                            "n_pixels": 200,
                            "area_synthesized_beams": 25.0,
                            "rms_jy_per_beam": 1.3e-4,
                            "scaled_mad_jy_per_beam": 1e-4,
                            "residual_abs_peak_jy_per_beam": 6e-4,
                            "residual_min_jy_per_beam": -5e-4,
                            "residual_max_jy_per_beam": 6e-4,
                            "peak_over_scaled_mad": 6.0,
                            "p99_over_scaled_mad": 3.4,
                            "p99_5_over_scaled_mad": 3.8,
                            "rms_over_scaled_mad": 1.3,
                        },
                    }
                (result_dir / "qa.json").write_text(
                    json.dumps(qa), encoding="utf-8"
                )
            (root / "simulation.json").write_text(
                json.dumps(
                    {
                        "components": [{"flux": [0.01, 0, 0, 0]}],
                        "noise": {
                            "noise_model": "vla-thermal",
                            "simplenoise_jy": 0.1,
                            "seed": 1,
                            "resolved_parameters": {
                                "band": "C",
                                "sampler": "8bit",
                                "sefd_jy": 310,
                                "eta_c": 0.93,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            (root / "report.json").write_text(
                json.dumps(
                    {
                        "title": "Fixture",
                        "description": "Fixture description",
                        "total_samples": 1,
                        "samples": [
                            {
                                "id": "0012-399",
                                "original_ms": "/fixture/0012-399.ms",
                                "original_result_dir": "original",
                                "simulation_ms": "simulation.ms",
                                "simulation_component_list": "simulation.cl",
                                "simulation_metadata": "simulation.json",
                                "simulation_result_dir": "simulation_image",
                                "source_snr": 1200.0,
                                "source_snr_basis": "original_clean_peak_over_residual_scaled_mad",
                                "target_dynamic_range": 1200.0,
                                "predicted_image_rms_jy_per_beam": 1e-4,
                            }
                        ],
                        "failures": [],
                    }
                ),
                encoding="utf-8",
            )
            result = subprocess.run(
                ["quarto", "render", "report.qmd"],
                cwd=root,
                capture_output=True,
                text=True,
                env={**os.environ, "HOME": str(root / "home")},
            )
            self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
            rendered = (root / "report.html").read_text(encoding="utf-8")
            self.assertEqual(rendered.count('class="comparison-row'), 4)
            self.assertEqual(rendered.count('class="histogram-panel"'), 5)
            self.assertIn(
                "Population histograms are suppressed because the QA files use different metric schemas or beam-region bounds.",
                rendered,
            )
            self.assertIn("metric region=full image (schema 1)", rendered)
            self.assertIn("metric region=3 beams to image edge", rendered)
            self.assertIn("target S/N from original dynamic range=1200", rendered)
            self.assertIn("achieved simulation robust DR=1300", rendered)
            for relative in (
                "original/dirty.png",
                "original/clean.png",
                "original/residual.png",
                "simulation_image/dirty.png",
                "simulation_image/clean.png",
                "simulation_image/residual.png",
            ):
                self.assertIn(relative, rendered)


RUN_INTEGRATION = os.environ.get(
    "RUN_EXTRACTED_SIMULATION_COMPARISON_INTEGRATION"
) == "1"


@unittest.skipUnless(RUN_INTEGRATION, "enable the full 0012-399 direct-imaging integration")
class ComparisonDirectImagingIntegrationTests(unittest.TestCase):
    def test_0012_399_complete_pair(self):
        source = ROOT / "collect/extracted/0012-399/0012-399/0012-399.ms"

        def tree_hash(path):
            digest = __import__("hashlib").sha256()
            for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
                if item.name == "table.lock":
                    continue
                digest.update(str(item.relative_to(path)).encode())
                with item.open("rb") as handle:
                    while chunk := handle.read(1024 * 1024):
                        digest.update(chunk)
            return digest.hexdigest()

        before = tree_hash(source)
        with tempfile.TemporaryDirectory(prefix="comparison-0012-399-") as temporary, patch.object(
            experiment, "SAMPLE_IDS", ["0012-399"]
        ):
            output = experiment.main(temporary)
            manifest = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["samples"]), 1)
            self.assertEqual(manifest["failures"], [])
            entry = manifest["samples"][0]
            metadata = json.loads(
                (output / entry["simulation_metadata"]).read_text(encoding="utf-8")
            )
            original_qa = json.loads(
                (output / entry["original_result_dir"] / "qa.json").read_text()
            )
            simulation_qa = json.loads(
                (output / entry["simulation_result_dir"] / "qa.json").read_text()
            )
            target_dynamic_range = original_qa["metrics"]["dynamic_range_scaled_mad"]
            flux = metadata["components"][0]["flux"][0]
            predicted = entry["predicted_image_rms_jy_per_beam"]
            self.assertAlmostEqual(flux, target_dynamic_range * predicted)
            self.assertEqual(entry["target_dynamic_range"], target_dynamic_range)
            self.assertEqual(entry["source_snr"], target_dynamic_range)
            self.assertEqual(
                entry["source_snr_basis"],
                experiment.SOURCE_SNR_BASIS,
            )
            self.assertEqual(original_qa["engine"], "direct")
            self.assertEqual(simulation_qa["engine"], "direct")
            self.assertEqual(
                original_qa["metrics"]["region"],
                simulation_qa["metrics"]["region"],
            )
            self.assertEqual(
                original_qa["metrics"]["region"],
                entry["metric_region"],
            )
            self.assertEqual(entry["metric_region"]["min_radius_beams"], 3.0)
            self.assertGreater(
                entry["metric_region"]["max_radius_beams"],
                entry["metric_region"]["min_radius_beams"],
            )
            geometry = original_qa["resolved_config"]["grid"]
            nx, ny = geometry["image_grid"]["imsize"]
            cell_x, cell_y = geometry["image_grid"]["cell_arcsec"]
            beam_major = geometry["beam"]["major_arcsec"]
            nearest_border_beams = min(
                min(nx // 2, nx - 1 - nx // 2) * cell_x,
                min(ny // 2, ny - 1 - ny // 2) * cell_y,
            ) / beam_major
            self.assertAlmostEqual(
                entry["metric_region"]["max_radius_beams"],
                nearest_border_beams - 1.0,
            )
            self.assertEqual(
                original_qa["effective_imaging_parameters"]["imsize"],
                list(experiment.IMAGING_IMSIZE),
            )
            self.assertEqual(metadata["noise"]["noise_model"], "vla-thermal")
            resolved_noise = metadata["noise"]["resolved_parameters"]
            expected_sigma = experiment.theoretical_vla_simplenoise(
                source,
                sefd_jy=experiment.VLA_OSS_2026A_SEFD_JY[resolved_noise["band"]],
                eta_c=experiment.ETA_C,
            )
            self.assertTrue(
                math.isclose(
                    metadata["noise"]["simplenoise_jy"],
                    expected_sigma,
                    rel_tol=1e-12,
                )
            )

            from casatools import table

            simulated_ms = output / entry["simulation_ms"]
            tb = table()
            tb.open(str(simulated_ms), nomodify=True)
            try:
                columns = set(tb.colnames())
                expected_weight = 1.0 / expected_sigma**2
                for start in range(0, int(tb.nrows()), 4096):
                    count = min(4096, int(tb.nrows()) - start)
                    self.assertTrue(
                        np.allclose(
                            tb.getcol("SIGMA", startrow=start, nrow=count),
                            expected_sigma,
                            rtol=1e-6,
                            atol=0,
                        )
                    )
                    self.assertTrue(
                        np.allclose(
                            tb.getcol("WEIGHT", startrow=start, nrow=count),
                            expected_weight,
                            rtol=1e-6,
                            atol=0,
                        )
                    )
                    if "SIGMA_SPECTRUM" in columns:
                        self.assertTrue(
                            np.allclose(
                                tb.getcol(
                                    "SIGMA_SPECTRUM", startrow=start, nrow=count
                                ),
                                expected_sigma,
                                rtol=1e-6,
                                atol=0,
                            )
                        )
                    if "WEIGHT_SPECTRUM" in columns:
                        self.assertTrue(
                            np.allclose(
                                tb.getcol(
                                    "WEIGHT_SPECTRUM", startrow=start, nrow=count
                                ),
                                expected_weight,
                                rtol=1e-6,
                                atol=0,
                            )
                        )
            finally:
                tb.close()
            self.assertTrue((output / "report.html").is_file())
            rendered = (output / "report.html").read_text(encoding="utf-8")
            self.assertEqual(rendered.count('class="comparison-row'), 4)
            self.assertEqual(rendered.count('class="histogram-panel"'), 5)
            experiment._verify_completed_entry(entry, output)
            for key in experiment._MATCHED_IMAGING_KEYS:
                self.assertEqual(
                    original_qa["effective_imaging_parameters"].get(key),
                    simulation_qa["effective_imaging_parameters"].get(key),
                )
        self.assertEqual(tree_hash(source), before)


if __name__ == "__main__":
    unittest.main()
