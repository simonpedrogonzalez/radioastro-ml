from __future__ import annotations

import json
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.preprocessing import (
    ALL_IDS,
    TEST_IDS,
    TRAIN_IDS,
    VAL_IDS,
    add_sample_to_dataset,
    create_sample_manifest,
    finalize_simulation_sample,
    load_dataset_manifest,
    load_sample_manifest,
    normalize_partition,
    partition_for_sample,
    resolve_reference,
    source_dataset_id,
    cleanup_simulation_sample,
)


class PartitionTests(unittest.TestCase):
    def test_hard_coded_partitions_are_complete_ordered_and_disjoint(self):
        self.assertEqual((len(TRAIN_IDS), len(TEST_IDS), len(VAL_IDS)), (93, 20, 20))
        self.assertEqual(len(ALL_IDS), len(set(ALL_IDS)))
        self.assertEqual(ALL_IDS, tuple(sorted(ALL_IDS)))
        self.assertEqual((TRAIN_IDS[-1], TEST_IDS[0], VAL_IDS[0]), (
            "1513+236",
            "1513-102",
            "1927+612",
        ))

    def test_sample_variants_follow_their_source_dataset_partition(self):
        self.assertEqual(source_dataset_id("0012-399_phase_only"), "0012-399")
        self.assertEqual(partition_for_sample("0012-399_phase_only"), "train")
        self.assertEqual(partition_for_sample("1513-102"), "test")
        self.assertEqual(partition_for_sample("2357-114_amp_phase"), "val")
        self.assertEqual(normalize_partition(" TRAIN "), "train")
        with self.assertRaisesRegex(ValueError, "Unknown partition"):
            normalize_partition("validation")
        with self.assertRaisesRegex(ValueError, "does not match"):
            source_dataset_id("unknown_sample")

    def test_dataset_loads_only_the_requested_partition_fully(self):
        from scripts.preprocessing.dataset import FitsSimulationDataset

        paths = tuple(Path(f"/{name}.json") for name in ("train", "test", "val"))
        minimal = {
            paths[0]: SimpleNamespace(
                sample_id="0005+383_variant", label_name="baseline", label_id=0
            ),
            paths[1]: SimpleNamespace(
                sample_id="1513-102_variant", label_name="baseline", label_id=0
            ),
            paths[2]: SimpleNamespace(
                sample_id="1927+612_variant", label_name="baseline", label_id=0
            ),
        }
        calls = []

        def fake_load(path, *, require_files=True, verify_integrity=True):
            calls.append((path, require_files, verify_integrity))
            return minimal[path]

        manifest = SimpleNamespace(labels={"baseline": 0}, samples=paths)
        with (
            patch("scripts.preprocessing.dataset._require_torch"),
            patch(
                "scripts.preprocessing.dataset.load_dataset_manifest",
                return_value=manifest,
            ) as load_index,
            patch(
                "scripts.preprocessing.dataset.load_sample_manifest",
                side_effect=fake_load,
            ),
        ):
            dataset = FitsSimulationDataset(".", partition="test")

        self.assertEqual(
            [sample.sample_id for sample in dataset.samples], ["1513-102_variant"]
        )
        load_index.assert_called_once_with(
            dataset.root / "dataset.json", require_samples=False
        )
        self.assertEqual(
            calls,
            [
                (paths[0], False, False),
                (paths[1], False, False),
                (paths[1], True, True),
                (paths[2], False, False),
            ],
        )


class ManifestTests(unittest.TestCase):
    def make_sample(self, root: Path, *, corruptions: int = 0):
        image_dir = root / "simulation/default_imaging"
        image_dir.mkdir(parents=True)
        products = {}
        for name in ("dirty", "clean", "residual", "psf"):
            path = image_dir / f"{name}.fits.gz"
            path.write_bytes(f"{name}\n".encode())
            products[name] = path
        qa_json = image_dir / "qa.json"
        qa_json.write_text('{"schema_version": 3}\n', encoding="utf-8")
        qa_text = image_dir / "qa.txt"
        qa_text.write_text("QA\n", encoding="utf-8")
        simulation_json = root / "simulation/sample.simulation.json"
        simulation_json.write_text('{"schema_version": 2}\n', encoding="utf-8")
        simulation_text = root / "simulation/sample.simulation.txt"
        simulation_text.write_text("Simulation\n", encoding="utf-8")
        report_pairs = []
        for index in range(corruptions):
            directory = root / f"corruptions/{index:03d}"
            directory.mkdir(parents=True)
            report = directory / "corruption.json"
            report.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "context": {
                            "retained_sample_id": "sample",
                            "application_index": index,
                        },
                        "configuration": {"type": f"example-{index}"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            text = directory / "corruption.txt"
            text.write_text(f"Corruption {index}\n", encoding="utf-8")
            report_pairs.append((report, text))
        return create_sample_manifest(
            root / "sample.json",
            sample_id="sample",
            label_id=2,
            label_name="corrupted",
            products=products,
            imaging_qa=qa_json,
            imaging_text=qa_text,
            simulation=simulation_json,
            simulation_text=simulation_text,
            corruptions=report_pairs,
        )

    def test_zero_and_multiple_corruptions_are_explicit_and_ordered(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            zero = self.make_sample(root / "zero", corruptions=0)
            multiple = self.make_sample(root / "multiple", corruptions=2)
            self.assertEqual(zero.schema_version, 2)
            self.assertEqual(zero.channel_order, ("dirty", "clean", "residual", "psf"))
            self.assertEqual(zero.corruptions, ())
            decoded = [
                json.loads(reference.corruption.read_text())["configuration"]["type"]
                for reference in multiple.corruptions
            ]
            self.assertEqual(decoded, ["example-0", "example-1"])
            self.assertEqual(multiple.raw["metadata"]["corruptions"][0], {
                "corruption": "corruptions/000/corruption.json",
                "corruption_text": "corruptions/000/corruption.txt",
            })

    def test_integrity_detects_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            sample = self.make_sample(Path(temporary), corruptions=1)
            sample.products["dirty"].write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                load_sample_manifest(sample.path)

    def test_relative_references_cannot_escape_or_traverse_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "sample"
            root.mkdir()
            with self.assertRaisesRegex(ValueError, "escapes"):
                resolve_reference(root, "../outside", name="test")
            outside = Path(temporary) / "outside"
            outside.write_text("x", encoding="utf-8")
            (root / "link").symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "symlink"):
                resolve_reference(root, "link", name="test")

    def test_dataset_index_has_stable_order_and_label_map(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self.make_sample(root / "first")
            second = self.make_sample(root / "second")
            # Sample IDs must be unique within an index.
            second_payload = second.raw
            second_payload["sample_id"] = "sample-2"
            second.path.unlink()
            from scripts.preprocessing import write_sample_manifest

            second = write_sample_manifest(second.path, second_payload)
            index_path = root / "dataset.json"
            add_sample_to_dataset(index_path, first.path)
            result = add_sample_to_dataset(index_path, second.path)
            self.assertEqual(result.samples, (first.path, second.path))
            self.assertEqual(result.labels, {"corrupted": 2})
            self.assertEqual(load_dataset_manifest(index_path).samples, result.samples)

    def test_cleanup_is_dry_run_first_and_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_sample(root)
            generated_ms = root / "temporary.ms"
            generated_ms.mkdir()
            (generated_ms / "table.dat").write_bytes(b"large")
            png = root / "dirty.png"
            png.write_bytes(b"png")
            with patch("scripts.preprocessing.cleanup.validate_fits_products"):
                planned = cleanup_simulation_sample(root)
                self.assertTrue(generated_ms.exists())
                self.assertTrue(png.exists())
                self.assertGreater(planned.reclaimed_bytes, 0)
                completed = cleanup_simulation_sample(root, dry_run=False)
                self.assertFalse(generated_ms.exists())
                self.assertFalse(png.exists())
                self.assertTrue(completed.audit_path.is_file())
                repeated = cleanup_simulation_sample(root)
            self.assertEqual(repeated.removed, ())
            self.assertEqual(repeated.reclaimed_bytes, 0)

    def test_finalizer_compacts_before_atomically_indexing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample_root = root / "sample"
            image_dir = sample_root / "simulation/default_imaging"
            image_dir.mkdir(parents=True)
            products = {}
            for name in ("dirty", "clean", "residual", "psf"):
                products[name] = image_dir / f"{name}.fits.gz"
                products[name].write_bytes(name.encode())
            qa_json = image_dir / "qa.json"
            qa_json.write_text('{"schema_version": 3}\n', encoding="utf-8")
            qa_text = image_dir / "qa.txt"
            qa_text.write_text("QA\n", encoding="utf-8")
            simulation_json = sample_root / "simulation/sample.simulation.json"
            simulation_json.write_text('{"schema_version": 2}\n', encoding="utf-8")
            simulation_text = sample_root / "simulation/sample.simulation.txt"
            simulation_text.write_text("Simulation\n", encoding="utf-8")
            temporary_ms = sample_root / "simulation/sample.ms"
            temporary_ms.mkdir()
            (temporary_ms / "table.dat").write_bytes(b"temporary")
            imaging = SimpleNamespace(
                dirty_fits=products["dirty"],
                clean_fits=products["clean"],
                residual_fits=products["residual"],
                psf_fits=products["psf"],
                qa_json=qa_json,
                qa_text=qa_text,
            )
            simulation = SimpleNamespace(
                metadata_json=simulation_json,
                metadata_text=simulation_text,
            )
            with patch("scripts.preprocessing.cleanup.validate_fits_products"):
                finalized = finalize_simulation_sample(
                    sample_root,
                    sample_id="sample",
                    label_id=0,
                    label_name="baseline",
                    imaging_result=imaging,
                    simulation_result=simulation,
                    dataset_index=root / "dataset.json",
                )
            self.assertFalse(temporary_ms.exists())
            self.assertTrue(finalized.manifest.path.is_file())
            self.assertEqual(
                load_dataset_manifest(root / "dataset.json").samples,
                (finalized.manifest.path,),
            )


@unittest.skipUnless(
    importlib.util.find_spec("numpy")
    and importlib.util.find_spec("astropy")
    and importlib.util.find_spec("torch"),
    "NumPy, Astropy, and PyTorch are required for FITS dataset tests",
)
class FitsDatasetTests(unittest.TestCase):
    def test_synthetic_products_load_in_declared_order_and_batches(self):
        import numpy as np
        import torch
        from astropy.io import fits

        from scripts.preprocessing import FitsSimulationDataset, make_simulation_dataloader

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_dir = root / "sample/simulation/default_imaging"
            image_dir.mkdir(parents=True)
            products = {}
            for index, name in enumerate(("dirty", "clean", "residual", "psf"), start=1):
                header = fits.Header(
                    {
                        "CTYPE1": "RA---TAN",
                        "CTYPE2": "DEC--TAN",
                        "CUNIT1": "deg",
                        "CUNIT2": "deg",
                        "CRPIX1": 2.5,
                        "CRPIX2": 2.5,
                        "CRVAL1": 0.0,
                        "CRVAL2": 0.0,
                        "CDELT1": -0.001,
                        "CDELT2": 0.001,
                        "BUNIT": "1" if name == "psf" else "Jy/beam",
                        "BMAJ": 0.001,
                        "BMIN": 0.0005,
                        "BPA": 0.0,
                    }
                )
                path = image_dir / f"{name}.fits.gz"
                fits.PrimaryHDU(
                    np.full((1, 1, 4, 4), index, dtype=np.float32), header=header
                ).writeto(path)
                products[name] = path
            qa_json = image_dir / "qa.json"
            qa_json.write_text('{"schema_version": 3}\n', encoding="utf-8")
            qa_text = image_dir / "qa.txt"
            qa_text.write_text("QA\n", encoding="utf-8")
            simulation_json = root / "sample/simulation/sample.simulation.json"
            simulation_json.write_text('{"schema_version": 2}\n', encoding="utf-8")
            simulation_text = root / "sample/simulation/sample.simulation.txt"
            simulation_text.write_text("Simulation\n", encoding="utf-8")
            manifest = create_sample_manifest(
                root / "sample/sample.json",
                sample_id="0005+383_phase_only",
                label_id=0,
                label_name="baseline",
                products=products,
                imaging_qa=qa_json,
                imaging_text=qa_text,
                simulation=simulation_json,
                simulation_text=simulation_text,
            )
            add_sample_to_dataset(root / "dataset.json", manifest.path)
            dataset = FitsSimulationDataset(root, partition="train")
            item = dataset[0]
            self.assertEqual(tuple(item["image"].shape), (4, 4, 4))
            self.assertTrue(torch.all(item["image"][0] == 1))
            self.assertTrue(torch.all(item["image"][1] == 2))
            self.assertTrue(torch.all(item["image"][2] == 3))
            self.assertTrue(torch.all(item["image"][3] == 4))
            batch = next(iter(make_simulation_dataloader(dataset)))
            self.assertEqual(tuple(batch["image"].shape), (1, 4, 4, 4))
            self.assertEqual(batch["sample_id"], ["0005+383_phase_only"])
            self.assertEqual(batch["partition"], ["train"])
            self.assertEqual(batch["metadata"][0]["corruptions"], [])


if __name__ == "__main__":
    unittest.main()
