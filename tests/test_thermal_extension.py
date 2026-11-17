"""Optional thermal_dir source selection (EXT62 extension) and the thermal extension launcher."""
import json
import tempfile
import unittest
from pathlib import Path

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


def synthetic_thermal(root, ids=(TRAIN_ID, VAL_ID), flux=1.0, rms=1e-4):
    """Minimal thermal comparison directory: report.json, simulation metadata and original QA."""
    samples = []
    for sid in ids:
        write_json(root / sid / "simulation" / "metadata.json",
                   dict(components=[dict(shape="point", flux=[0.5, 0, 0, 0])]))
        write_json(root / sid / "original" / "default_imaging" / "qa.json",
                   dict(tclean_summary=dict(final_model_flux_jy=flux), metrics=dict(clean_peak_jy_per_beam=flux)))
        (root / sid / "original_ms" / f"{sid}.ms").mkdir(parents=True, exist_ok=True)
        samples.append(dict(id=sid, original_ms=str(root / sid / "original_ms" / f"{sid}.ms"),
                            simulation_ms=f"{sid}/simulation/{sid}.ms",
                            simulation_metadata=f"{sid}/simulation/metadata.json",
                            simulation_result_dir=f"{sid}/simulation/default_imaging",
                            source_snr=flux / rms, predicted_image_rms_jy_per_beam=rms))
    write_json(root / "report.json", dict(configuration=dict(imsize=[256, 256]), samples=samples,
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
        spec = v2.original_flux_spec(source, v2.thermal_root(config))
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
        output, ids = parse(["--output", "collect/experiments/does_not_exist_ext", "--ids", "0002+002,0001+001"])
        self.assertEqual(output, (g.ROOT / "collect/experiments/does_not_exist_ext").resolve())
        self.assertEqual(ids, ["0001+001", "0002+002"])


if __name__ == "__main__":
    unittest.main()
