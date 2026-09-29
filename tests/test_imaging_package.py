from __future__ import annotations

import json
import inspect
import math
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import call, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.imaging import (
    DEFAULT_IMSIZE,
    Beam,
    BeamRegion,
    CleanIterationsConfig,
    DefaultImagingConfig,
    GridConfig,
    ImageMetrics,
    IMAGING_METRIC_DEFINITIONS,
    PipelineBackground,
    QAReport,
    RegionMetrics,
    beam_region_mask,
    export_casa_fits,
    image_ms,
    image_ms_VLA_pipe,
    measure_image_metrics,
    summarize_residual_pixels,
)
from scripts.imaging.config import central_circle_mask, normalize_imsize
from scripts.imaging.metadata import (
    DEFAULT_CALIBRATOR_META_CSV,
    DEFAULT_EXTRACTED_MS_ROOT,
    repository_path,
    resolve_csv_meta,
    resolve_path,
)
from scripts.imaging.qa import json_safe, normalize_tclean_summary, render_qa_text
from scripts.imaging.plot_utils import (
    _fits_beam_geometry,
    write_fits_comparison_plots,
    write_individual_plots,
)
from scripts.imaging.metrics import _ImagePlane, _metric_validity
from scripts.imaging.vla_pipeline import (
    _pipeline_summary,
    _run_pipeline,
    _weblog_background_rms,
    parse_tclean_calls,
)


class DefaultsAndMetadataTests(unittest.TestCase):
    def test_canonical_defaults(self):
        config = DefaultImagingConfig
        self.assertEqual(config.datacolumn, "data")
        self.assertEqual(config.deconvolver, "mtmfs")
        self.assertEqual(config.nterms, 1)
        self.assertEqual(config.clean.niter, 1_000_000)
        self.assertEqual(config.clean.dirty_peak_fraction, 1e-7)
        self.assertEqual(config.clean.nsigma, 3.0)
        self.assertIsNone(config.clean.cycleniter)
        self.assertEqual(config.mask_nbeams, 6.0)
        self.assertEqual(config.grid, GridConfig())
        self.assertFalse(hasattr(config, "band_catalog_csv"))
        self.assertEqual(replace(config, robust=-0.5).robust, -0.5)

    def test_both_entrypoints_default_to_256_square(self):
        self.assertEqual(DEFAULT_IMSIZE, (256, 256))
        self.assertEqual(inspect.signature(image_ms).parameters["imsize"].default, (256, 256))
        self.assertEqual(
            inspect.signature(image_ms_VLA_pipe).parameters["imsize"].default,
            (256, 256),
        )
        self.assertEqual(inspect.signature(image_ms_VLA_pipe).parameters["mask_nbeams"].default, 6.0)
        self.assertEqual(
            inspect.signature(image_ms).parameters["metric_region"].default,
            BeamRegion(),
        )
        self.assertEqual(
            inspect.signature(image_ms_VLA_pipe).parameters["metric_region"].default,
            BeamRegion(),
        )
        self.assertEqual(
            inspect.signature(image_ms).parameters["fits_invalid_policy"].default,
            "error",
        )

    def test_fits_export_rejects_invalid_fill_configuration_before_casa(self):
        with self.assertRaisesRegex(ValueError, "invalid_policy"):
            export_casa_fits("unused", "unused.fits.gz", invalid_policy="ignore")
        with self.assertRaisesRegex(ValueError, "fill_value"):
            export_casa_fits("unused", "unused.fits.gz", fill_value=math.inf)

    def test_metric_region_resolver_is_exclusive_and_must_be_callable(self):
        # The comparison driver deliberately reloads local CASA packages when
        # test discovery imports it, so fetch the current class identities.
        from scripts.imaging import (
            BeamRegion as CurrentBeamRegion,
            DefaultImagingConfig as CurrentDefaultImagingConfig,
            image_ms as current_image_ms,
        )

        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(TypeError, "callable"):
                current_image_ms(
                    "unused",
                    CurrentDefaultImagingConfig,
                    temporary,
                    metric_region_resolver=object(),
                )
            with self.assertRaisesRegex(ValueError, "cannot both"):
                current_image_ms(
                    "unused",
                    CurrentDefaultImagingConfig,
                    temporary,
                    metric_region=CurrentBeamRegion(min_radius_beams=3.0),
                    metric_region_resolver=lambda _: CurrentBeamRegion(),
                )

    def test_imsize_accepts_a_scalar_or_two_axis_pair(self):
        self.assertEqual(normalize_imsize(384), (384, 384))
        self.assertEqual(normalize_imsize([384, 192]), (384, 192))
        with self.assertRaisesRegex(ValueError, "exactly two"):
            normalize_imsize([256])
        with self.assertRaisesRegex(ValueError, "positive"):
            normalize_imsize((256, 0))

    def test_central_mask_is_a_beam_scaled_circle(self):
        self.assertEqual(
            central_circle_mask((256, 192), 1.5, 6.0),
            "circle[[128pix,96pix],4.5arcsec]",
        )

    def test_repository_paths_do_not_depend_on_cwd(self):
        self.assertEqual(
            repository_path("collect/small_subset/small_selection.csv"),
            DEFAULT_CALIBRATOR_META_CSV,
        )

    def test_0012_399_id_and_catalog_resolution(self):
        resolved = resolve_path("0012-399", DEFAULT_EXTRACTED_MS_ROOT)
        self.assertEqual(resolved.path.name, "0012-399.ms")
        self.assertEqual(resolved.visibility_id, "0012-399")
        meta, preferred_spw = resolve_csv_meta(
            resolved.visibility_id, DEFAULT_CALIBRATOR_META_CSV
        )
        self.assertIsNotNone(meta)
        self.assertEqual(meta.array_configuration, "B")
        self.assertEqual(meta.catalog_band_codes, ("C",))
        self.assertAlmostEqual(meta.catalog_frequency_ghz, 7.291)
        self.assertEqual(preferred_spw, 13)

    def test_missing_id_error_includes_search_location(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, str(root)):
                resolve_path("9999+999", root)


class CleanControlTests(unittest.TestCase):
    def test_dirty_peak_fraction_is_staged(self):
        controls = CleanIterationsConfig(
            niter=50, dirty_peak_fraction=0.1, cycleniter=10
        ).resolve(Path("dirty.residual"), imstat_task=lambda **_: {"min": [-2.0], "max": [1.0]})
        self.assertEqual(controls.threshold, "0.2Jy")
        self.assertEqual(controls.niter, 50)

    def test_mutually_exclusive_threshold_controls(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            CleanIterationsConfig(threshold="1mJy", dirty_peak_fraction=0.1)

    def test_large_iteration_count_requires_stopper(self):
        with self.assertRaisesRegex(ValueError, "requires"):
            CleanIterationsConfig(niter=1_000_000)


class SummaryTests(unittest.TestCase):
    def test_tclean_summary_normalizes_case_and_arrays(self):
        record = {
            "iterDone": 37,
            "nMajorDone": 4,
            "stopCode": 2,
            "stopDescription": "threshold",
            "summaryMajor": [0, 10, 37],
            "summaryMinor": {
                "peakRes": [0.2, 0.01],
                "modelFlux": [0.0, 0.7],
                "cycleThresh": [0.1, 0.005],
                "newCasaField": [1, 2],
            },
            "futureKey": {"value": 3},
        }
        summary, warnings = normalize_tclean_summary(record)
        self.assertFalse(warnings)
        self.assertEqual(summary.iterdone, 37)
        self.assertEqual(summary.nmajordone, 4)
        self.assertEqual(summary.stopcode, 2)
        self.assertEqual(summary.major_cycle_iteration_counts, (0, 10, 37))
        self.assertEqual(summary.final_peak_residual_jy_per_beam, 0.01)
        self.assertEqual(summary.final_model_flux_jy, 0.7)
        self.assertEqual(summary.final_cycle_threshold_jy_per_beam, 0.005)
        self.assertIn("futureKey", summary.raw)
        json.dumps(json_safe(summary), allow_nan=False)

    def test_malformed_summary_keeps_a_warning(self):
        summary, warnings = normalize_tclean_summary(None)
        self.assertIsNone(summary.iterdone)
        self.assertTrue(warnings)

    def test_text_report_is_human_readable_and_keeps_json_out(self):
        report = QAReport(
            schema_version=2,
            engine="vla_pipeline",
            input_value="0012-399",
            ms_path=Path("/data/simulated_constant_0.7Jy_phasecenter.ms"),
            visibility_id="0012-399",
            resolved_config=None,
            effective_imaging_parameters={
                "pipeline_source_datacolumn": "DATA",
                "data_column_validation": {
                    "table_column": "DATA",
                    "rows_examined": 64,
                    "nonzero_finite_unflagged_samples": 4096,
                },
                "csv_meta": {"array_configuration": "B"},
                "ms_band_meta": {
                    "selected_band": "C",
                    "representative_frequency_ghz": 7.291,
                    "frequency_reference_spw": 13,
                },
                "field": "J0006-0623",
                "spw": ["0"],
                "uvrange": "",
                "measured_geometry": {
                    "imsize": [512, 512],
                    "cell_arcsec": [0.15, 0.15],
                    "field_of_view_arcsec": [76.8, 76.8],
                    "beam": {
                        "major_arcsec": 1.2,
                        "minor_arcsec": 0.8,
                        "position_angle_deg": 32.0,
                    },
                },
                "specmode": "mfs",
                "gridder": "standard",
                "stokes": "I",
                "deconvolver": "mtmfs",
                "nterms": 1,
                "gain": 0.1,
                "weighting": "briggs",
                "robust": 0.5,
                "usemask": "auto-multithresh",
                "sidelobethreshold": 2.0,
                "noisethreshold": 4.25,
                "lownoisethreshold": 1.5,
                "minbeamfrac": 0.3,
                "niter": 10_000,
                "threshold": "0.0Jy",
                "nsigma": 5.0,
                "nmajor": -1,
                "interactive": False,
            },
            products={},
            metrics=ImageMetrics(
                region=BeamRegion(min_radius_beams=3.0),
                clean_peak_jy_per_beam=0.7,
                residual=RegionMetrics(
                    n_pixels=100,
                    area_synthesized_beams=12.5,
                    rms_jy_per_beam=1.7e-4,
                    scaled_mad_jy_per_beam=1.5e-4,
                    residual_abs_peak_jy_per_beam=9.0e-4,
                    residual_min_jy_per_beam=-8.0e-4,
                    residual_max_jy_per_beam=9.0e-4,
                    peak_over_scaled_mad=6.0,
                    p99_over_scaled_mad=2.1,
                    p99_5_over_scaled_mad=2.5,
                    rms_over_scaled_mad=1.7 / 1.5,
                ),
                dynamic_range_rms=0.7 / 1.7e-4,
                dynamic_range_scaled_mad=4666.7,
            ),
            metric_units={},
            metric_validity={},
            tclean_summary=None,
            pipeline_background=PipelineBackground(
                rms_jy_per_beam=1.7e-4,
                region="PB annulus",
                algorithm="Chauvenet",
            ),
        )

        rendered = render_qa_text(report)
        self.assertIn("Engine: vla_pipeline", rendered)
        self.assertIn("Input MS: /data/simulated_constant_0.7Jy_phasecenter.ms", rendered)
        self.assertIn("deconvolver=mtmfs | nterms=1", rendered)
        self.assertIn(
            "sigma=1.4826*median(|R-median(R)|)=0.00015 Jy/beam",
            rendered,
        )
        self.assertIn("max=max(|R|)/sigma=6", rendered)
        self.assertNotIn("interactive", rendered)
        self.assertNotIn('"effective_imaging_parameters"', rendered)

    def test_metric_definitions_have_stable_text_and_latex_representations(self):
        definitions = {item.key: item for item in IMAGING_METRIC_DEFINITIONS}
        self.assertEqual(repr(definitions["max"]), "max=max(|R|)/sigma")
        self.assertEqual(definitions["DR"].latex, r"\mathrm{DR}=\max(I_{\mathrm{clean}})/\sigma")
        self.assertIn("description", definitions["p995"].to_report_dict())


class PipelineArtifactTests(unittest.TestCase):
    def test_pipeline_makeimlist_receives_requested_imsize(self):
        calls = {}

        class Context:
            clean_list_pending = [{"mask": None}]

            def set_state(self, *args):
                pass

        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)

            def save():
                (work / "pipeline-test").mkdir()

            def makeimages(**kwargs):
                calls["makeimages"] = kwargs
                return "result"

            tasks = {
                "h_init": Context,
                "h_save": save,
                "hifv_importdata": lambda **kwargs: None,
                "hifv_flagtargetsdata": lambda: None,
                "hif_checkproductsize": lambda **kwargs: None,
                "hif_makeimlist": lambda **kwargs: calls.update(kwargs),
                "hif_makeimages": makeimages,
            }
            with patch("scripts.imaging.vla_pipeline._write_central_mask") as write_mask:
                _, result, weblog = _run_pipeline(
                    work,
                    Path("input.ms"),
                    "target",
                    (384, 192),
                    tasks,
                    mask_radius_arcsec=3.0,
                )

        self.assertEqual(calls["hm_imsize"], [384, 192])
        self.assertEqual(calls["makeimages"]["hm_masking"], "manual")
        write_mask.assert_called_once_with(
            Context.clean_list_pending[0],
            work / "central_clean_0.mask",
            3.0,
        )
        self.assertEqual(
            Context.clean_list_pending[0]["mask"],
            str(work / "central_clean_0.mask"),
        )
        self.assertEqual(result, "result")
        self.assertEqual(weblog.name, "pipeline-test")

    def test_parses_multiline_tclean_calls(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "casa_commands.log"
            path.write_text(
                "# generated\n"
                "tclean(vis=['x.ms'], imagename='dirty',\n"
                "       niter=0, deconvolver='mtmfs')\n"
                "tclean(vis=['x.ms'], imagename='clean', niter=100,\n"
                "       threshold='1e-5Jy', fullsummary=True)\n",
                encoding="utf-8",
            )
            calls = parse_tclean_calls(path)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["parameters"]["niter"], 0)
        self.assertEqual(calls[1]["parameters"]["threshold"], "1e-5Jy")

    def test_weblog_background_rms_unit_conversion(self):
        with tempfile.TemporaryDirectory() as temporary:
            stage = Path(temporary) / "html/stage5"
            stage.mkdir(parents=True)
            (stage / "t2-4m_details.html").write_text(
                "<th>non-pbcor image RMS</th><td>1.7 uJy/beam</td>",
                encoding="utf-8",
            )
            self.assertAlmostEqual(_weblog_background_rms(Path(temporary)), 1.7e-6)

    def test_weblog_background_rms_has_no_fuzzy_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            stage = Path(temporary) / "html/stage5"
            stage.mkdir(parents=True)
            (stage / "t2-4m_details.html").write_text(
                "<th>residual robust RMS</th><td>1.7 uJy/beam</td>",
                encoding="utf-8",
            )
            self.assertIsNone(_weblog_background_rms(Path(temporary)))

    def test_pipeline_result_fields_are_preferred_and_normalized(self):
        class FakeTcleanResult:
            def __init__(self):
                self.iterations = {
                    2: {
                        "nmajordone": 3,
                        "nminordone_array": [0, 12, 32],
                        "summaryminor": {"peakRes": [0.1, 0.002]},
                    }
                }
                self._tclean_iterdone = 32
                self._tclean_stopcode = 8
                self._tclean_stopreason = "n-sigma criterion"

        result = FakeTcleanResult()
        summary, warnings = _pipeline_summary([result], None, Path("unused"))
        self.assertFalse(warnings)
        self.assertEqual(summary.source, "pipeline_context")
        self.assertEqual(summary.iterdone, 32)
        self.assertEqual(summary.nmajordone, 3)
        self.assertEqual(summary.stopcode, 8)
        self.assertEqual(summary.final_peak_residual_jy_per_beam, 0.002)


class PlotTests(unittest.TestCase):
    def test_clean_and_residual_plots_receive_the_clean_mask(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            mask = Path("clean.mask")
            region = BeamRegion(min_radius_beams=3.0, max_radius_beams=8.0)
            beam = Beam(2.0, 1.0, 30.0)
            with patch("scripts.imaging.plot_utils.casa_image_to_png") as render:
                write_individual_plots(
                    Path("dirty.image"),
                    Path("clean.image"),
                    Path("clean.residual"),
                    output,
                    visibility_id="0012-399",
                    mask_image=mask,
                    metric_region=region,
                    fallback_beam=beam,
                )

        self.assertEqual(
            render.call_args_list,
            [
                call(
                    Path("dirty.image"),
                    output / "dirty.png",
                    title="0012-399 dirty",
                    draw_beam=True,
                ),
                call(
                    Path("clean.image"),
                    output / "clean.png",
                    title="0012-399 clean",
                    mask_path=mask,
                    draw_beam=True,
                ),
                call(
                    Path("clean.residual"),
                    output / "residual.png",
                    title="0012-399 residual",
                    mask_path=mask,
                    draw_beam=True,
                    metric_region=region,
                    fallback_beam=beam,
                ),
            ],
        )

    def test_missing_fits_beam_uses_measured_beam_fallback(self):
        beam = Beam(major_arcsec=3.6, minor_arcsec=1.8, position_angle_deg=27.0)

        bmaj, bmin, bpa = _fits_beam_geometry({}, beam)

        self.assertAlmostEqual(bmaj, 0.001)
        self.assertAlmostEqual(bmin, 0.0005)
        self.assertEqual(bpa, 27.0)

    def test_fits_beam_takes_precedence_over_fallback(self):
        beam = Beam(major_arcsec=3.6, minor_arcsec=1.8, position_angle_deg=27.0)
        header = {"BMAJ": 0.002, "BMIN": 0.001, "BPA": 45.0}

        self.assertEqual(_fits_beam_geometry(header, beam), (0.002, 0.001, 45.0))

    def test_comparison_panels_reuse_canonical_plotter_and_channel_scales(self):
        region = BeamRegion(min_radius_beams=3.0, max_radius_beams=8.0)
        samples = [
            (
                "baseline",
                "rho_corr=0",
                {name: Path(f"baseline-{name}.fits.gz") for name in ("dirty", "clean", "residual", "psf")},
            ),
            (
                "variant",
                "rho_corr=10",
                {name: Path(f"variant-{name}.fits.gz") for name in ("dirty", "clean", "residual", "psf")},
            ),
        ]
        with tempfile.TemporaryDirectory() as temporary, patch(
            "scripts.imaging.plot_utils.shared_fits_display_limits",
            side_effect=[(-1.0, 1.0), (-2.0, 2.0), (-3.0, 3.0), (0.0, 1.0)],
        ), patch(
            "scripts.imaging.plot_utils.casa_image_to_png", return_value={"colormap": "inferno"}
        ) as render:
            rows, recipes = write_fits_comparison_plots(
                samples, Path(temporary), metric_region=region
            )

        self.assertEqual(len(rows), 2)
        self.assertEqual(render.call_count, 8)
        self.assertEqual(
            recipes["limits_mjy_per_beam"],
            {
                "dirty": [-1.0, 1.0],
                "clean": [-2.0, 2.0],
                "residual": [-3.0, 3.0],
                "psf": [0.0, 1.0],
            },
        )
        dirty_call = render.call_args_list[0]
        residual_call = render.call_args_list[2]
        self.assertEqual(dirty_call.kwargs["display_limits_mjy_per_beam"], (-1.0, 1.0))
        self.assertIsNone(dirty_call.kwargs["metric_region"])
        self.assertEqual(residual_call.kwargs["metric_region"], region)
        self.assertTrue(all(item.kwargs["draw_beam"] for item in render.call_args_list))


class BeamRegionMetricTests(unittest.TestCase):
    def test_region_validation(self):
        self.assertEqual(BeamRegion(), BeamRegion(None, None))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            BeamRegion(min_radius_beams=-1)
        with self.assertRaisesRegex(ValueError, "greater"):
            BeamRegion(min_radius_beams=3, max_radius_beams=3)
        with self.assertRaisesRegex(ValueError, "greater"):
            BeamRegion(min_radius_beams=4, max_radius_beams=3)
        for invalid in (float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "finite"):
                BeamRegion(max_radius_beams=invalid)

    def test_central_annular_and_exterior_masks_share_boundaries(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest(
                "NumPy is supplied by CASA, not the lightweight test interpreter"
            )

        full = beam_region_mask((9, 9), (1.0, 1.0), 1.0, BeamRegion())
        central = beam_region_mask((9, 9), (1.0, 1.0), 1.0, BeamRegion(max_radius_beams=3))
        annulus = beam_region_mask(
            (9, 9), (1.0, 1.0), 1.0, BeamRegion(min_radius_beams=3, max_radius_beams=4)
        )
        exterior = beam_region_mask((9, 9), (1.0, 1.0), 1.0, BeamRegion(min_radius_beams=4))
        self.assertTrue(central[4, 4])
        self.assertFalse(central[4, 7])
        self.assertTrue(annulus[4, 7])
        self.assertFalse(annulus[4, 8])
        self.assertTrue(exterior[4, 8])
        self.assertTrue(np.all(full))
        self.assertTrue(np.all(central | annulus | exterior))
        self.assertFalse(np.any(central & annulus))
        self.assertFalse(np.any(annulus & exterior))

        def center_row_transitions(mask):
            row = mask[mask.shape[0] // 2].astype(int)
            return int(np.count_nonzero(np.diff(row)))

        self.assertEqual(center_row_transitions(full), 0)
        self.assertEqual(center_row_transitions(central), 2)
        self.assertEqual(center_row_transitions(exterior), 2)
        self.assertEqual(center_row_transitions(annulus), 4)

    def test_non_square_cells_use_angular_not_pixel_radius(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is supplied by CASA, not the lightweight test interpreter")
        selected = beam_region_mask(
            (7, 7), (2.0, 1.0), 2.0, BeamRegion(max_radius_beams=1.1)
        )
        self.assertTrue(selected[4, 3])   # one arcsec vertically
        self.assertTrue(selected[3, 2])   # two arcsec horizontally
        self.assertFalse(selected[3, 1])  # four arcsec horizontally

    def test_shared_summarizer_uses_one_pixel_population(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is supplied by CASA, not the lightweight test interpreter")
        values = np.asarray([-1.0, 0.0, 1.0])
        metrics = summarize_residual_pixels(
            values,
            pixel_area_arcsec2=1.0,
            beam=Beam(2.0, 1.0, 0.0),
        )
        self.assertEqual(metrics.n_pixels, 3)
        self.assertAlmostEqual(metrics.rms_jy_per_beam, (2.0 / 3.0) ** 0.5)
        self.assertAlmostEqual(metrics.scaled_mad_jy_per_beam, 1.4826)
        self.assertAlmostEqual(
            metrics.area_synthesized_beams,
            3.0 / (math.pi * 2.0 / (4.0 * math.log(2.0))),
        )

    def test_shared_summarizer_rejects_invalid_geometry(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is supplied by CASA, not the lightweight test interpreter")
        values = np.asarray([-1.0, 0.0, 1.0])
        with self.assertRaisesRegex(ValueError, "positive"):
            summarize_residual_pixels(
                values, pixel_area_arcsec2=0.0, beam=Beam(2.0, 1.0, 0.0)
            )
        with self.assertRaisesRegex(ValueError, "positive"):
            summarize_residual_pixels(
                values, pixel_area_arcsec2=1.0, beam=Beam(2.0, 0.0, 0.0)
            )

    def test_metric_validity_preserves_nested_schema(self):
        metrics = ImageMetrics(
            region=BeamRegion(min_radius_beams=3.0),
            clean_peak_jy_per_beam=1.0,
            residual=RegionMetrics(
                n_pixels=10,
                area_synthesized_beams=2.0,
                rms_jy_per_beam=0.1,
                scaled_mad_jy_per_beam=0.09,
                residual_abs_peak_jy_per_beam=0.3,
                residual_min_jy_per_beam=-0.2,
                residual_max_jy_per_beam=0.3,
                peak_over_scaled_mad=3.0,
                p99_over_scaled_mad=2.5,
                p99_5_over_scaled_mad=2.8,
                rms_over_scaled_mad=float("nan"),
            ),
            dynamic_range_rms=10.0,
            dynamic_range_scaled_mad=float("nan"),
        )

        validity = _metric_validity(metrics)

        self.assertTrue(validity["clean_peak_jy_per_beam"])
        self.assertTrue(validity["residual"]["n_pixels"])
        self.assertFalse(validity["residual"]["rms_over_scaled_mad"])
        self.assertTrue(validity["dynamic_range_rms"])
        self.assertFalse(validity["dynamic_range_scaled_mad"])

    def test_outlier_changes_rms_and_peak_ratio_but_not_scaled_mad(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is supplied by CASA, not the lightweight test interpreter")
        beam = Beam(1.0, 1.0, 0.0)
        baseline = summarize_residual_pixels(
            np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0]),
            pixel_area_arcsec2=1.0,
            beam=beam,
        )
        contaminated = summarize_residual_pixels(
            np.asarray([-2.0, -1.0, 0.0, 1.0, 200.0]),
            pixel_area_arcsec2=1.0,
            beam=beam,
        )
        self.assertEqual(
            contaminated.scaled_mad_jy_per_beam,
            baseline.scaled_mad_jy_per_beam,
        )
        self.assertGreater(contaminated.rms_jy_per_beam, baseline.rms_jy_per_beam * 10)
        self.assertGreater(
            contaminated.peak_over_scaled_mad,
            baseline.peak_over_scaled_mad * 10,
        )

    def test_measurement_uses_internal_mask_region_and_global_positive_peak(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is supplied by CASA, not the lightweight test interpreter")

        clean_values = np.zeros((5, 5), dtype=float)
        clean_values[0, 0] = -100.0
        clean_values[4, 4] = 7.0
        clean_values[0, 4] = 50.0
        clean_valid = np.ones((5, 5), dtype=bool)
        clean_valid[0, 4] = False
        residual_values = np.arange(25, dtype=float).reshape(5, 5) - 12.0
        residual_values[2, 2] = float("nan")
        residual_valid = np.ones((5, 5), dtype=bool)
        residual_valid[1, 1] = False
        beam = Beam(1.0, 0.5, 0.0)
        planes = [
            _ImagePlane(clean_values, clean_valid, (1.0, 1.0), beam),
            _ImagePlane(residual_values, residual_valid, (1.0, 1.0), beam),
        ]
        region = BeamRegion(max_radius_beams=2.0)
        region_mask = beam_region_mask((5, 5), (1.0, 1.0), 1.0, region)
        expected = residual_values[
            region_mask & residual_valid & np.isfinite(residual_values)
        ]
        with patch("scripts.imaging.metrics._load_image_plane", side_effect=planes):
            metrics = measure_image_metrics("clean.image", "residual.image", region=region)

        self.assertIsInstance(metrics, ImageMetrics)
        self.assertEqual(metrics.region, region)
        self.assertEqual(metrics.clean_peak_jy_per_beam, 7.0)
        self.assertEqual(metrics.residual.n_pixels, expected.size)
        self.assertAlmostEqual(
            metrics.residual.rms_jy_per_beam,
            float(np.sqrt(np.mean(np.square(expected)))),
        )
        self.assertAlmostEqual(
            metrics.dynamic_range_rms,
            7.0 / metrics.residual.rms_jy_per_beam,
        )
        self.assertAlmostEqual(
            metrics.dynamic_range_scaled_mad,
            7.0 / metrics.residual.scaled_mad_jy_per_beam,
        )

    def test_empty_image_region_error_reports_bounds_and_available_radius(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is supplied by CASA, not the lightweight test interpreter")
        plane = _ImagePlane(
            np.ones((5, 5), dtype=float),
            np.ones((5, 5), dtype=bool),
            (1.0, 1.0),
            Beam(1.0, 1.0, 0.0),
        )
        with patch("scripts.imaging.metrics._load_image_plane", side_effect=[plane, plane]):
            with self.assertRaisesRegex(
                ValueError,
                r"min_radius_beams=10.*maximum corner radius=",
            ):
                measure_image_metrics(
                    "clean.image",
                    "residual.image",
                    region=BeamRegion(min_radius_beams=10.0),
                )

    def test_measurement_uses_finite_residuals_when_casa_mask_misses_region(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is supplied by CASA, not the lightweight test interpreter")
        values = np.arange(25, dtype=float).reshape(5, 5)
        clean = _ImagePlane(
            values,
            np.ones((5, 5), dtype=bool),
            (1.0, 1.0),
            Beam(1.0, 1.0, 0.0),
        )
        residual = _ImagePlane(
            values,
            np.zeros((5, 5), dtype=bool),
            (1.0, 1.0),
            Beam(1.0, 1.0, 0.0),
        )
        region = BeamRegion(min_radius_beams=2.0)
        expected = values[beam_region_mask((5, 5), (1.0, 1.0), 1.0, region)]
        with patch(
            "scripts.imaging.metrics._load_image_plane",
            side_effect=[clean, residual],
        ):
            metrics = measure_image_metrics("clean.image", "residual.image", region=region)
        self.assertEqual(metrics.residual.n_pixels, expected.size)


if __name__ == "__main__":
    unittest.main()
