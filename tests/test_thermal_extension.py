"""Optional thermal_dir source selection (EXT62 extension) and the thermal extension launcher."""
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import create_dataset_v1 as g
from scripts import create_dataset_v2 as v2
from scripts import create_dataset_v2_matched as matched
from scripts.dataset_v2 import DEFAULT as SPARSE_DEFAULT
from scripts.dataset_v2_matched import DEFAULT as MATCHED_DEFAULT
from scripts.injection_bounds_report import digest
from scripts.run_thermal_extension import check_output, parse, source_ids

TRAIN_ID, VAL_ID = "0005+383", "0022+002"   # existing partition members, used as synthetic sources

# Split-only config digests before thermal_dir support (docs/CHPC_EXT62_CODE_REVIEW.md).
LEGACY_DIGESTS = {
    "train": ("d575a4fe0ffff32cb5dc35a3b3a9193d3863408ea9d4f8517d54b4c81f94d13f",
              "e4728b89a8d00cc40ae5e2a3bebda47faa2dfdcc0fcc07332b0f3de7aaabca55"),
    "val": ("21a8905604c1f129786cf18ba8adb56113231527528fe48164a989edeccf0a81",
            "417e2b4b06677191b1cb5c92c54af3c5702ceb518584217921eb7226ad38b75c"),
    "test": ("7019fba8d3a550f60511180708b7b3b464bebdf4c7e6a59cb1a6b49a609edf82",
             "efeb282dbb67db0c87324b2423b369ba53408ce0cb1e01ab9c2fb890db0e18f4"),
}
LEGACY_AVAILABLE = (g.DATASET_DIR / "dataset.json").is_file() and (g.THERMAL_DIR / "report.json").is_file()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return path


def synthetic_thermal(root, ids=(TRAIN_ID, VAL_ID), flux=1.0, rms=1e-4, policy=None, decisions=None):
    """Minimal thermal comparison directory: report.json, simulation metadata and original QA.

    policy/decisions: extension thermal run (per-source flux_decision; sky flux = decided flux)."""
    samples = []
    for sid in ids:
        decision = (decisions or {}).get(sid)
        sky_flux = decision["flux_jy"] if decision else 0.5
        write_json(root / sid / "simulation" / "metadata.json",
                   dict(components=[dict(shape="point", flux=[sky_flux, 0, 0, 0])]))
        write_json(root / sid / "original" / "default_imaging" / "qa.json",
                   dict(tclean_summary=dict(final_model_flux_jy=flux), metrics=dict(clean_peak_jy_per_beam=flux)))
        (root / sid / "original_ms" / f"{sid}.ms").mkdir(parents=True, exist_ok=True)
        samples.append(dict(id=sid, original_ms=str(root / sid / "original_ms" / f"{sid}.ms"),
                            simulation_ms=f"{sid}/simulation/{sid}.ms",
                            simulation_metadata=f"{sid}/simulation/metadata.json",
                            simulation_result_dir=f"{sid}/simulation/default_imaging",
                            source_snr=flux / rms, predicted_image_rms_jy_per_beam=rms,
                            **(policy or {}), **({"flux_decision": decision} if decision else {})))
    write_json(root / "report.json", dict(configuration=dict(imsize=[256, 256], **(policy or {})), samples=samples,
                                          failures=[dict(id="9999+999", stage="simulation", error="x")]))
    return root


class ConfigurationLoaders(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def config_file(self, **overrides):
        return write_json(self.tmp / f"{len(list(self.tmp.iterdir()))}.json", overrides)

    def test_defaults_do_not_contain_thermal_dir(self):
        self.assertNotIn("thermal_dir", SPARSE_DEFAULT)
        self.assertNotIn("thermal_dir", MATCHED_DEFAULT)

    def test_split_only_digests_unchanged(self):
        for split, (sparse, matched_digest) in LEGACY_DIGESTS.items():
            path = self.config_file(split=split)
            self.assertEqual(digest(v2.configuration(path)), sparse)
            self.assertEqual(digest(matched.configuration(path)), matched_digest)
            self.assertNotIn("thermal_dir", matched.configuration(path))

    def test_both_loaders_accept_thermal_dir_and_reject_unknown(self):
        for loader in (v2.configuration, matched.configuration):
            config = loader(self.config_file(split="val", thermal_dir="collect/experiments/x"))
            self.assertEqual(config["thermal_dir"], "collect/experiments/x")
            with self.assertRaises(ValueError):
                loader(self.config_file(thermal_directory="x"))

    def test_thermal_root(self):
        self.assertEqual(v2.thermal_root({}), g.THERMAL_DIR)
        self.assertEqual(v2.thermal_root(dict(thermal_dir="collect/experiments/x")),
                         g.ROOT / "collect/experiments/x")
        self.assertEqual(v2.thermal_root(dict(thermal_dir="/abs/x")), Path("/abs/x"))


class ThermalDirSelection(unittest.TestCase):
    def setUp(self):
        self.thermal = synthetic_thermal(Path(tempfile.mkdtemp()))

    def config(self, split, **extra):
        return dict(matched.configuration(None), split=split, thermal_dir=str(self.thermal), **extra)

    def test_selects_successful_samples_of_the_split(self):
        ids, entries, audit = v2.indexed_sources(self.config("train"))
        self.assertEqual(ids, [TRAIN_ID])
        self.assertEqual(v2.indexed_sources(self.config("val"))[0], [VAL_ID])
        self.assertNotIn("9999+999", entries)   # failures are never selectable
        self.assertEqual(audit["source_selection"], "thermal_report_samples")
        self.assertEqual(audit["thermal_report"], str((self.thermal / "report.json").resolve()))
        self.assertEqual(audit["split_counts"], dict(train=1, val=1, test=0))
        self.assertNotIn("index", audit)

    def test_explicit_source_ids_must_belong_to_split(self):
        self.assertEqual(v2.indexed_sources(self.config("train", source_ids=[TRAIN_ID]))[0], [TRAIN_ID])
        with self.assertRaises(ValueError):
            v2.indexed_sources(self.config("train", source_ids=[VAL_ID]))
        with self.assertRaises(ValueError):
            v2.indexed_sources(self.config("test"))   # empty split

    def test_unpartitioned_source_is_rejected(self):
        synthetic_thermal(self.thermal, ids=(TRAIN_ID, "8888+888"))
        with self.assertRaisesRegex(ValueError, "partitions.py"):
            v2.indexed_sources(self.config("train"))

    def test_shared_parent_ms_is_rejected(self):
        report = json.loads((self.thermal / "report.json").read_text())
        report["samples"][1]["original_ms"] = report["samples"][0]["original_ms"]
        write_json(self.thermal / "report.json", report)
        with self.assertRaisesRegex(ValueError, "Parent MS shared"):
            v2.indexed_sources(self.config("train"))

    def test_source_and_original_flux_come_from_thermal_dir(self):
        config = self.config("train")
        source = v2.thermal_source(config, TRAIN_ID)
        self.assertEqual(source.simulation_metadata, (self.thermal / TRAIN_ID / "simulation/metadata.json").resolve())
        spec = v2.original_flux_spec(source, config)
        self.assertEqual(Path(spec["original_qa"]),
                         (self.thermal / TRAIN_ID / "original/default_imaging/qa.json").resolve())
        self.assertEqual(spec["sky"][0]["flux"][0], 1.0)
        self.assertAlmostEqual(spec["nominal_source_snr"], 1e4)

    def test_preview_helpers_use_thermal_dir(self):
        from scripts.report_dataset_v2 import preview as sparse_preview
        from scripts.report_dataset_v2_matched import preview as matched_preview

        out = Path(tempfile.mkdtemp())
        matched_preview(self.config("train"), out / "matched")
        data = json.loads((out / "matched" / "sampling_preview.json").read_text())
        self.assertEqual(list(data["nominal_source_snr"]), [TRAIN_ID])
        self.assertEqual(data["source_index"]["source_selection"], "thermal_report_samples")
        sparse = dict(v2.configuration(None), thermal_dir=str(self.thermal))
        sparse_preview(sparse, out / "sparse")
        data = json.loads((out / "sparse" / "sampling_preview.json").read_text())
        self.assertEqual([s["id"] for s in data["sources"]], [TRAIN_ID])


@unittest.skipUnless(LEGACY_AVAILABLE, "legacy Dataset 1 index / thermal report not present")
class LegacySelection(unittest.TestCase):
    def test_without_thermal_dir_uses_dataset1_index(self):
        counts = {}
        for split in ("train", "val", "test"):
            ids, _, audit = v2.indexed_sources(dict(matched.configuration(None), split=split))
            counts[split] = len(ids)
            self.assertIn("index", audit)
            self.assertEqual(audit["thermal_report"], str((g.THERMAL_DIR / "report.json").resolve()))
        self.assertEqual(counts, dict(train=100, val=21, test=22))


class Launcher(unittest.TestCase):
    def test_source_ids(self):
        tmp = Path(tempfile.mkdtemp())
        ids_file = tmp / "ids.txt"
        ids_file.write_text("# new\n0002+002\n\n0001+001  # comment\n")
        self.assertEqual(source_ids(["0003+003"], ids_file), ["0001+001", "0002+002", "0003+003"])
        with self.assertRaises(ValueError):
            source_ids([])
        with self.assertRaises(ValueError):
            source_ids(["0001+001", "0001+001"])

    def test_refuses_directory_with_other_sources(self):
        thermal = synthetic_thermal(Path(tempfile.mkdtemp()))
        check_output(thermal, [TRAIN_ID, VAL_ID, "9999+999"])   # same run: resume allowed
        with self.assertRaisesRegex(ValueError, "fresh directory"):
            check_output(thermal, [TRAIN_ID])
        if (g.THERMAL_DIR / "report.json").is_file():
            with self.assertRaises(ValueError):
                parse(["--output", str(g.THERMAL_DIR), "--ids", "0001+001"])

    def test_parse_relative_output_and_fresh_directory(self):
        output, ids, options = parse(["--output", "collect/experiments/does_not_exist_ext", "--ids", "0002+002,0001+001"])
        self.assertEqual(output, (g.ROOT / "collect/experiments/does_not_exist_ext").resolve())
        self.assertEqual(ids, ["0001+001", "0002+002"])
        self.assertEqual(options, dict(flux_policy=None, simulation_flags=None, reuse_qa_from=[], extracted_root=None))
        _, _, options = parse(["--output", "/tmp/x_ext", "--ids", "a", "--flux-policy", "nrao_then_clean",
                               "--simulation-flags", "unflag", "--reuse-qa-from", "/tmp/old"])
        self.assertEqual(options["flux_policy"], "nrao_then_clean")
        self.assertEqual(options["simulation_flags"], "unflag")
        self.assertEqual(options["reuse_qa_from"], [Path("/tmp/old").resolve()])
        for bad in (["--simulation-flags", "unflag"], ["--reuse-qa-from", "/tmp/old"],
                    ["--flux-policy", "nrao_then_clean", "--reuse-qa-from", "/tmp/x_ext"]):
            with self.assertRaises(SystemExit):
                parse(["--output", "/tmp/x_ext", "--ids", "a", *bad])


POLICY = dict(flux_policy="nrao_then_clean", simulation_flags="unflag")


def nrao_decision(flux=0.7):
    return dict(method="nrao_pipeline", flux_jy=flux, frequency_hz=5e9, provenance=dict(sidecar="x"))


def sidecar_ms(root, *, schema=2, status="available", field="J0001+0001", spw="3", coefficients=(-0.5, 0.3)):
    folder = root / "s" / "s"
    ms = folder / "s.ms"
    ms.mkdir(parents=True, exist_ok=True)
    write_json(folder / "extraction_metadata.json", dict(gain_calibrator_name="J0001+0001", selected_spw="3"))
    write_json(folder / "flux_estimate.json", dict(
        schema=schema, status=status, reason=None, gain_calibrator_name=field, selected_spw=spw,
        stokes_i_jy=2.0, reference_frequency_ghz=4.0, spectral_index_coefficients=list(coefficients),
        source_text="setjy(...)", source_file="casa_commands.log", recovered_from="logs", ambiguity_checked=True))
    return ms


class NraoFlux(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())

    def test_spectral_model_is_evaluated_at_simulation_frequency(self):
        from scripts.nrao_flux import nrao_flux_decision

        decision = nrao_flux_decision(sidecar_ms(self.root), 8e9)
        x = math.log10(2.0)
        self.assertAlmostEqual(decision["flux_jy"], 2.0 * 2.0 ** (-0.5 + 0.3 * x), places=14)
        self.assertEqual(decision["method"], "nrao_pipeline")
        self.assertEqual(decision["frequency_hz"], 8e9)
        self.assertEqual(decision["model"]["spectral_index_coefficients"], [-0.5, 0.3])
        self.assertEqual(len(decision["provenance"]["sidecar_sha256"]), 64)

    def test_unusable_sidecars_are_rejected(self):
        from scripts.nrao_flux import FluxUnavailable, nrao_flux_decision

        for kwargs in (dict(schema=None), dict(status="ambiguous"), dict(status="needs_logs"),
                       dict(field="J9999+9999"), dict(spw="4"), dict(coefficients=())):
            with self.subTest(**kwargs), self.assertRaises(FluxUnavailable):
                nrao_flux_decision(sidecar_ms(Path(tempfile.mkdtemp()), **kwargs), 5e9)
        with self.assertRaises(FluxUnavailable):
            nrao_flux_decision(self.root / "missing" / "x.ms", 5e9)


class FluxDecision(unittest.TestCase):
    """compare_extracted_thermal_simulations.decide_flux routes (CASA imaging mocked)."""

    def setUp(self):
        from scripts import compare_extracted_thermal_simulations as thermal
        from scripts.imaging import DefaultImagingConfig
        from dataclasses import asdict

        self.thermal = thermal
        self.root = Path(tempfile.mkdtemp())
        self.ms = sidecar_ms(self.root, schema=None)   # schema-1 sidecar: NRAO route unavailable
        self.out = self.root / "experiment"
        expected = json.loads(json.dumps(asdict(DefaultImagingConfig), default=str))
        self.params = {k: expected[k] for k in thermal._REUSED_QA_SETTINGS}
        self.params.update(imsize=[256, 256], requested_clean=expected["clean"])

    def old_qa(self, flux=0.4, **changes):
        old = self.root / "old_thermal"
        qa = dict(engine="direct", schema_version=2, visibility_id="s", ms_path=str(self.ms),
                  effective_imaging_parameters=dict(self.params), tclean_summary=dict(final_model_flux_jy=flux))
        qa.update(changes)
        write_json(old / "s" / "original" / "default_imaging" / "qa.json", qa)
        return old

    def test_nrao_route_skips_original_imaging(self):
        ms = sidecar_ms(self.root)
        with mock.patch.object(self.thermal, "image_ms", side_effect=AssertionError("imaged")):
            decision, original = self.thermal.decide_flux(ms, self.out, 5e9, [])
        self.assertEqual(decision["method"], "nrao_pipeline")
        self.assertIsNone(original)

    def test_malformed_or_overflowing_nrao_metadata_uses_clean_fallback(self):
        old = self.old_qa()
        for payload in ('{truncated', '[]', None):
            ms = sidecar_ms(self.root)
            path = ms.parent / 'flux_estimate.json'
            if payload is None:
                data = json.loads(path.read_text())
                data['spectral_index_coefficients'] = [1e308, 1e308]
                path.write_text(json.dumps(data))
            else:
                path.write_text(payload)
            with self.subTest(payload=payload), mock.patch.object(
                    self.thermal, 'image_ms', side_effect=AssertionError('imaged')):
                decision, original = self.thermal.decide_flux(ms, self.out, 8e9, [old])
            self.assertEqual(decision['method'], 'original_clean')
            self.assertIsNone(original)

    def test_clean_fallback_reuses_matching_qa_read_only(self):
        old = self.old_qa()
        before = (old / "s/original/default_imaging/qa.json").read_bytes()
        with mock.patch.object(self.thermal, "image_ms", side_effect=AssertionError("imaged")):
            decision, original = self.thermal.decide_flux(self.ms, self.out, 5e9, [old])
        self.assertEqual(decision["method"], "original_clean")
        self.assertEqual(decision["flux_jy"], 0.4)
        self.assertTrue(decision["provenance"]["reused"])
        self.assertIn("schema-1", decision["nrao_unavailable"])
        self.assertEqual(decision["provenance"]["qa_sha256"], hashlib.sha256(before).hexdigest())
        self.assertEqual((old / "s/original/default_imaging/qa.json").read_bytes(), before)
        self.assertIsNone(original)

    def test_qa_for_other_input_or_settings_is_not_reused(self):
        for changes in (dict(ms_path="/elsewhere/s.ms"), dict(visibility_id="t"),
                        dict(effective_imaging_parameters=dict(self.params, robust=0.0)),
                        dict(effective_imaging_parameters=dict(self.params, imsize=[512, 512]))):
            with self.subTest(changes=list(changes)):
                path, reasons = self.thermal.reusable_original_qa("s", self.ms, [self.old_qa(**changes)])
                self.assertIsNone(path)
                self.assertTrue(reasons)

    def test_clean_fallback_images_original_when_nothing_reusable(self):
        qa_path = write_json(self.out / "s/original/default_imaging/qa.json",
                             dict(tclean_summary=dict(final_model_flux_jy=0.25)))
        fake = mock.Mock(qa_json=qa_path)
        with mock.patch.object(self.thermal, "image_ms", return_value=fake) as image:
            decision, original = self.thermal.decide_flux(self.ms, self.out, 5e9, [])
        image.assert_called_once()
        self.assertIs(original, fake)
        self.assertEqual(decision["flux_jy"], 0.25)
        self.assertEqual(decision["provenance"]["qa_json"], "s/original/default_imaging/qa.json")
        self.assertFalse(decision["provenance"]["reused"])

    def test_both_routes_failing_raises(self):
        with mock.patch.object(self.thermal, "image_ms", side_effect=RuntimeError("fully flagged")):
            with self.assertRaisesRegex(Exception, "no NRAO flux .*fully flagged"):
                self.thermal.decide_flux(self.ms, self.out, 5e9, [])
        with mock.patch.object(self.thermal, "image_ms", side_effect=AssertionError("imaged")):
            with self.assertRaisesRegex(Exception, "CLEAN"):   # reused QA with an empty model
                self.thermal.decide_flux(self.ms, self.out, 5e9, [self.old_qa(flux=0.0)])

    def test_policy_validation(self):
        self.assertEqual(self.thermal.extension_policy(None, None), {})
        self.assertEqual(self.thermal.extension_policy("nrao_then_clean", "unflag"), POLICY)
        for bad in ((None, "unflag"), ("other", None), ("nrao_then_clean", "keep")):
            with self.assertRaises(ValueError):
                self.thermal.extension_policy(*bad)


class ExtensionPolicies(unittest.TestCase):
    def setUp(self):
        self.thermal = synthetic_thermal(Path(tempfile.mkdtemp()), policy=POLICY,
                                         decisions={TRAIN_ID: nrao_decision(), VAL_ID: nrao_decision(0.9)})

    def config(self, split="train", **extra):
        return dict(matched.configuration(None), split=split, thermal_dir=str(self.thermal), **POLICY, **extra)

    def test_loaders_validate_policies_without_changing_defaults(self):
        tmp = Path(tempfile.mkdtemp())
        for loader in (v2.configuration, matched.configuration):
            ok = loader(write_json(tmp / "ok.json", dict(thermal_dir="x", **POLICY)))
            self.assertEqual((ok["flux_policy"], ok["simulation_flags"]), ("nrao_then_clean", "unflag"))
            for bad in (POLICY, dict(thermal_dir="x", simulation_flags="unflag"),
                        dict(thermal_dir="x", flux_policy="other"),
                        dict(thermal_dir="x", flux_policy="nrao_then_clean", simulation_flags="keep")):
                with self.subTest(loader=loader.__module__, bad=bad), self.assertRaises(ValueError):
                    loader(write_json(tmp / "bad.json", bad))

    def test_selection_requires_matching_thermal_policy(self):
        self.assertEqual(v2.indexed_sources(self.config())[0], [TRAIN_ID])
        legacy = dict(matched.configuration(None), split="train", thermal_dir=str(self.thermal))
        with self.assertRaisesRegex(ValueError, "policy"):
            v2.indexed_sources(legacy)
        with self.assertRaisesRegex(ValueError, "policy"):
            v2.indexed_sources(dict(legacy, flux_policy="nrao_then_clean"))   # flags differ

    def test_nrao_decision_needs_no_original_qa(self):
        config = self.config()
        (self.thermal / TRAIN_ID / "original").rename(self.thermal / TRAIN_ID / "no_original")
        spec = v2.original_flux_spec(v2.thermal_source(config, TRAIN_ID), config)
        self.assertEqual(spec["flux_jy"], 0.7)
        self.assertEqual(spec["flux_method"], "nrao_pipeline")
        self.assertEqual(spec["simulation_flags"], "unflag")
        self.assertAlmostEqual(spec["nominal_source_snr"], 0.7 / 1e-4)
        self.assertNotIn("original_qa", spec)

    def test_clean_decision_pins_its_qa(self):
        qa = self.thermal / TRAIN_ID / "original/default_imaging/qa.json"
        decision = dict(method="original_clean", flux_jy=0.3, frequency_hz=5e9,
                        provenance=dict(qa_json=f"{TRAIN_ID}/original/default_imaging/qa.json",
                                        qa_sha256=hashlib.sha256(qa.read_bytes()).hexdigest()))
        synthetic_thermal(self.thermal, policy=POLICY, decisions={TRAIN_ID: decision, VAL_ID: nrao_decision()})
        qa.write_text(qa.read_text())   # unchanged content
        config = self.config()
        spec = v2.original_flux_spec(v2.thermal_source(config, TRAIN_ID), config)
        self.assertEqual((spec["flux_jy"], spec["original_qa"]), (0.3, str(qa.resolve())))
        qa.write_text("{}")
        with self.assertRaisesRegex(ValueError, "changed"):
            v2.original_flux_spec(v2.thermal_source(config, TRAIN_ID), config)

    def test_simulated_flux_must_equal_decision(self):
        write_json(self.thermal / TRAIN_ID / "simulation/metadata.json",
                   dict(components=[dict(shape="point", flux=[0.5, 0, 0, 0])]))
        config = self.config()
        with self.assertRaisesRegex(ValueError, "decided"):
            v2.original_flux_spec(v2.thermal_source(config, TRAIN_ID), config)

    def test_previews_use_recorded_decisions(self):
        from scripts.report_dataset_v2_matched import preview as matched_preview

        out = Path(tempfile.mkdtemp())
        matched_preview(self.config(), out)
        data = json.loads((out / "sampling_preview.json").read_text())
        self.assertAlmostEqual(data["nominal_source_snr"][TRAIN_ID], 0.7 / 1e-4)


if __name__ == "__main__":
    unittest.main()
