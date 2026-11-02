from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts import create_dataset_v1 as experiment
from scripts.preprocessing import TEST_IDS, TRAIN_IDS, VAL_IDS


class FullDatasetConfigurationTests(unittest.TestCase):
    def test_antenna_selection_is_seeded_and_limited_to_unflagged_choices(self):
        choices = ((2, "ea03"), (7, "ea08"), (11, "ea12"))

        first = experiment._select_antenna(choices, seed=12345)
        second = experiment._select_antenna(choices, seed=12345)

        self.assertEqual(first, second)
        self.assertIn(first, choices)

    def test_fixed_overrides_or_random_per_variant_seeds(self):
        first, second = experiment.VARIANTS[:2]

        self.assertIsNone(experiment.FIXED_ANTENNA_ID)
        self.assertIsNone(experiment.FIXED_ERROR_SIGN)

        with patch.object(experiment, "FIXED_ANTENNA_ID", 1):
            self.assertIsNone(experiment._antenna_seed("0012-399", first))
            self.assertEqual(
                experiment._select_fixed_antenna(((0, "ea01"), (1, "ea02")), 1),
                (1, "ea02"),
            )
        with patch.object(experiment, "FIXED_ANTENNA_ID", None):
            self.assertNotEqual(
                experiment._antenna_seed("0012-399", first),
                experiment._antenna_seed("0012-399", second),
            )
        with patch.object(experiment, "FIXED_ERROR_SIGN", 1):
            self.assertIsNone(experiment._error_direction_seed("0012-399", first))
        with patch.object(experiment, "FIXED_ERROR_SIGN", None):
            self.assertNotEqual(
                experiment._error_direction_seed("0012-399", first),
                experiment._error_direction_seed("0012-399", second),
            )

    def test_unflagged_antenna_choices_are_sorted(self):
        fake_corruption = SimpleNamespace(
            get_unflagged_antennas=Mock(
                return_value=(
                    {"ea08": 7, "ea03": 2},
                    {7: "ea08", 2: "ea03"},
                )
            )
        )
        with patch.dict(
            sys.modules,
            {"scripts.corruption": fake_corruption},
        ):
            choices = experiment._unflagged_antenna_choices(Path("/source.ms"))

        self.assertEqual(choices, ((2, "ea03"), (7, "ea08")))

    def test_unflagged_antenna_choices_reject_empty_set(self):
        fake_corruption = SimpleNamespace(
            get_unflagged_antennas=Mock(return_value=({}, {}))
        )
        with patch.dict(
            sys.modules,
            {"scripts.corruption": fake_corruption},
        ):
            with self.assertRaisesRegex(RuntimeError, "No antenna"):
                experiment._unflagged_antenna_choices(Path("/source.ms"))

    def test_manifest_covers_every_partition_and_variant(self):
        source_ids = (TRAIN_IDS[0], TEST_IDS[0], VAL_IDS[0])

        self.assertEqual(
            experiment._source_partition_counts(source_ids),
            {"train": 1, "test": 1, "val": 1},
        )
        self.assertEqual(len(experiment.LABELS), 17)
        self.assertEqual(
            set(experiment.LABELS),
            {"not_corrupted", *(v.label_name for v in experiment.VARIANTS)},
        )
        self.assertEqual(
            {name: experiment.LABELS[name] for name in experiment.LEGACY_LABELS},
            experiment.LEGACY_LABELS,
        )
        self.assertEqual(
            {
                name: experiment.LABELS[name]
                for name in (
                    "amp_snr_5",
                    "amp_snr_15",
                    "amp_snr_20",
                    "amp_snr_40",
                    "phase_snr_5",
                    "phase_snr_15",
                    "phase_snr_20",
                    "phase_snr_40",
                )
            },
            {
                "amp_snr_5": 9,
                "amp_snr_15": 10,
                "amp_snr_20": 11,
                "amp_snr_40": 12,
                "phase_snr_5": 13,
                "phase_snr_15": 14,
                "phase_snr_20": 15,
                "phase_snr_40": 16,
            },
        )

    def test_partially_complete_source_selects_only_missing_ids(self):
        source_id = TRAIN_IDS[0]
        requested = experiment._requested_samples(source_id)
        missing = {
            f"{source_id}_amp_snr_15",
            f"{source_id}_phase_snr_40",
        }
        completed = set(requested) - missing

        self.assertEqual(
            set(experiment._pending_samples(requested, completed, set())),
            missing,
        )
        repair = f"{source_id}_amp_snr_10"
        self.assertEqual(
            set(experiment._pending_samples(requested, completed, {repair})),
            missing | {repair},
        )

    def test_incremental_index_append_installs_full_stable_label_map(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample = SimpleNamespace(
                sample_id="source_amp_snr_5",
                label_name="amp_snr_5",
                label_id=9,
                path=root / "samples/source_amp_snr_5/sample.json",
            )
            experiment._append_sample_to_dataset(root / "dataset.json", sample, set())
            payload = json.loads((root / "dataset.json").read_text(encoding="utf-8"))

        self.assertEqual(payload["labels"], experiment.LABELS)
        self.assertEqual(
            payload["samples"], ["samples/source_amp_snr_5/sample.json"]
        )

    def test_missing_thermal_sources_are_explicit_exclusions(self):
        selected = tuple(
            source_id
            for source_id in (*TRAIN_IDS, *TEST_IDS, *VAL_IDS)
            if source_id not in {TRAIN_IDS[0], VAL_IDS[0]}
        )
        exclusions = experiment._thermal_exclusions(
            {
                "failures": [
                    {
                        "id": TRAIN_IDS[0],
                        "stage": "simulation",
                        "error": "upstream failure",
                    }
                ]
            },
            selected,
        )

        self.assertEqual([item["id"] for item in exclusions], [TRAIN_IDS[0], VAL_IDS[0]])
        self.assertEqual(exclusions[0]["partition"], "train")
        self.assertEqual(exclusions[0]["error"], "upstream failure")
        self.assertEqual(exclusions[1]["partition"], "val")

    def test_cli_no_longer_selects_one_source(self):
        arguments = experiment._parse_args([])
        self.assertFalse(hasattr(arguments, "source_id"))
        self.assertFalse(arguments.noise_controls)
        self.assertIsNone(arguments.noise_control_source_ids)

    def test_noise_control_scope_defaults_to_validation_and_rejects_training(self):
        source_ids = (TRAIN_IDS[0], TEST_IDS[0], VAL_IDS[0])
        self.assertEqual(
            experiment._select_noise_control_sources(
                source_ids, enabled=True, explicit_source_ids=None
            ),
            (VAL_IDS[0],),
        )
        self.assertEqual(
            experiment._select_noise_control_sources(
                source_ids,
                enabled=True,
                explicit_source_ids=(TEST_IDS[0],),
            ),
            (TEST_IDS[0],),
        )
        with self.assertRaisesRegex(ValueError, "Training"):
            experiment._select_noise_control_sources(
                source_ids,
                enabled=True,
                explicit_source_ids=(TRAIN_IDS[0],),
            )

    def test_noise_control_draw_is_normalized_and_records_total_sigma(self):
        source = SimpleNamespace(source_id=VAL_IDS[0])
        control = experiment.NoiseControl(20.0, "noise_snr_20")
        raw = SimpleNamespace(SNR_corr=1000.0, valid_sample_count=200)
        measured = SimpleNamespace(SNR_corr=20.00001, valid_sample_count=200)
        with patch(
            "scripts.simulation.add_thermal_noise_inplace",
            return_value={"simplenoise_jy": 2.0, "seed": 123},
        ), patch(
            "scripts.corruption.measure_corruption_metrics",
            side_effect=[raw, measured],
        ), patch.object(experiment, "_rescale_visibility_delta") as rescale:
            metadata = experiment._apply_noise_control(
                source,
                control,
                Path("/V.ms"),
                Path("/D.ms"),
                2.0,
                "simplenoise",
                {},
                experiment._stable_seed(VAL_IDS[0], "thermal_noise"),
            )

        rescale.assert_called_once_with(Path("/V.ms"), Path("/D.ms"), 0.02)
        self.assertEqual(metadata["kind"], "increased_noise")
        self.assertEqual(metadata["valid_sample_count"], 200)
        self.assertAlmostEqual(metadata["equivalent_extra_sigma_jy"], 2.0)
        self.assertAlmostEqual(metadata["final_total_sigma_jy"], 2.0 * 2**0.5)
        self.assertNotEqual(
            metadata["extra_noise_seed"], metadata["baseline_noise_seed"]
        )

    def test_source_ids_to_process_selects_and_validates_ids(self):
        self.assertIsNone(experiment.SOURCE_IDS_TO_PROCESS)
        selected = (TRAIN_IDS[0], TEST_IDS[0])
        with patch.object(experiment, "SOURCE_IDS_TO_PROCESS", selected):
            self.assertEqual(experiment._configured_source_ids(), selected)
        with patch.object(experiment, "SOURCE_IDS_TO_PROCESS", ("not-a-source",)):
            with self.assertRaisesRegex(ValueError, "unknown IDs"):
                experiment._configured_source_ids()

    def test_report_uses_realized_gain_and_signed_phase(self):
        template = experiment.REPORT_TEMPLATE.read_text(encoding="utf-8")

        self.assertIn('physical = solution.get("g_amp")', template)
        self.assertIn('physical = solution.get("phi_deg")', template)
        self.assertIn('"Amplitude gain"', template)
        self.assertIn('"Phase offset (deg)"', template)
        self.assertIn("overflow-x: auto", template)
        self.assertIn("repeat({columns}", template)

    def test_known_four_level_report_migrates_but_policy_changes_fail(self):
        source_ids = (TRAIN_IDS[0], TRAIN_IDS[1])
        expected = {
            "SNR_corr_targets": list(experiment.SNR_CORR_TARGETS),
            "labels": dict(experiment.LABELS),
            "base_seed": experiment.BASE_SEED,
            "antenna_selection_policy": experiment.ANTENNA_SELECTION_POLICY,
        }
        report = {
            "source_run": "/thermal",
            "source_ids": [source_ids[0]],
            "source_partition_counts": {"train": 1, "test": 0, "val": 0},
            "excluded_sources": [],
            "configuration": {
                **expected,
                "SNR_corr_targets": list(experiment.LEGACY_SNR_CORR_TARGETS),
                "labels": dict(experiment.LEGACY_LABELS),
            },
            "sources": [{"id": source_ids[0], "partition": "train"}],
            "dataset_iteration": {"sample_count": 9},
        }
        with patch.object(
            experiment,
            "_new_report_manifest",
            return_value={"configuration": expected},
        ):
            migrated = experiment._resume_report_manifest(
                report, Path("/thermal"), source_ids, []
            )
        self.assertTrue(migrated)
        self.assertEqual(report["configuration"], expected)
        self.assertEqual(report["source_ids"], list(source_ids))
        self.assertEqual(report["sources"], [])
        self.assertIsNone(report["dataset_iteration"])

        incompatible = {
            **report,
            "configuration": {**expected, "base_seed": experiment.BASE_SEED + 1},
        }
        with patch.object(
            experiment,
            "_new_report_manifest",
            return_value={"configuration": expected},
        ):
            with self.assertRaisesRegex(RuntimeError, "incompatible"):
                experiment._resume_report_manifest(
                    incompatible, Path("/thermal"), source_ids, []
                )

    def test_pb_support_detection_finds_border_mask_and_rejects_interior_hole(self):
        import numpy as np

        sample = SimpleNamespace(sample_id="source_amp_snr_10", products={})
        circular = np.ones((7, 7), dtype=np.float32)
        circular[0, :] = 0.0
        circular[-1, :] = 0.0
        circular[:, 0] = 0.0
        circular[:, -1] = 0.0
        planes = {
            name: SimpleNamespace(values=circular.copy())
            for name in ("dirty", "clean", "residual")
        }
        with patch("scripts.preprocessing.validate_fits_products", return_value=planes):
            self.assertTrue(experiment._has_border_connected_common_zeros(sample))

        ambiguous = np.ones((7, 7), dtype=np.float32)
        ambiguous[1, 1] = 0.0
        planes = {
            name: SimpleNamespace(values=ambiguous.copy())
            for name in ("dirty", "clean", "residual")
        }
        with patch("scripts.preprocessing.validate_fits_products", return_value=planes):
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                experiment._has_border_connected_common_zeros(sample)

    def test_prepared_repair_swap_keeps_recoverable_old_directory(self):
        sample_id = "source_amp_snr_10"
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            paths = experiment._repair_paths(output, sample_id)
            paths["live"].mkdir(parents=True)
            (paths["live"] / "marker").write_text("old", encoding="utf-8")
            paths["staged"].mkdir(parents=True)
            (paths["staged"] / "marker").write_text("new", encoding="utf-8")
            experiment._write_repair_state(paths, sample_id, "prepared")

            replacement = SimpleNamespace(path=paths["live"] / "sample.json")
            with patch.object(
                experiment, "_load_valid_replacement", return_value=replacement
            ):
                result = experiment._finish_repair_transaction(output, sample_id)

            self.assertEqual(result, replacement.path)
            self.assertEqual(
                (paths["live"] / "marker").read_text(encoding="utf-8"), "new"
            )
            self.assertEqual(
                (paths["backup"] / "marker").read_text(encoding="utf-8"), "old"
            )
            state = json.loads(paths["state"].read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "committed")

    def test_invalid_staged_repair_restores_the_live_sample(self):
        sample_id = "source_amp_snr_10"
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            paths = experiment._repair_paths(output, sample_id)
            paths["live"].mkdir(parents=True)
            (paths["live"] / "marker").write_text("old", encoding="utf-8")
            paths["staged"].mkdir(parents=True)
            (paths["staged"] / "marker").write_text("new", encoding="utf-8")
            experiment._write_repair_state(paths, sample_id, "prepared")

            with patch.object(
                experiment,
                "_load_valid_replacement",
                side_effect=[SimpleNamespace(), ValueError("invalid replacement")],
            ):
                with self.assertRaisesRegex(ValueError, "invalid replacement"):
                    experiment._finish_repair_transaction(output, sample_id)

            self.assertEqual(
                (paths["live"] / "marker").read_text(encoding="utf-8"), "old"
            )
            self.assertFalse(paths["backup"].exists())
            self.assertTrue(paths["failed"].is_dir())
            state = json.loads(paths["state"].read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "rolled_back")

    def test_repair_sample_preserves_rolled_back_failed_replacement(self):
        sample_id = "source_amp_snr_10"
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            paths = experiment._repair_paths(output, sample_id)
            paths["live"].mkdir(parents=True)
            old = SimpleNamespace(sample_id=sample_id)

            def fail_after_rollback(_output, _sample_id):
                paths["failed"].mkdir()
                experiment._write_repair_state(paths, sample_id, "rolled_back")
                raise ValueError("invalid replacement")

            with patch.object(
                experiment,
                "_finalize_sample",
                return_value=paths["staged"] / "sample.json",
            ), patch(
                "scripts.preprocessing.load_sample_manifest",
                return_value=SimpleNamespace(),
            ), patch.object(
                experiment, "_assert_square_pb_support"
            ), patch.object(
                experiment, "_assert_sample_pb_policy"
            ), patch.object(
                experiment, "_assert_repair_equivalent"
            ), patch.object(
                experiment,
                "_finish_repair_transaction",
                side_effect=fail_after_rollback,
            ):
                with self.assertRaisesRegex(ValueError, "invalid replacement"):
                    experiment._repair_sample(
                        SimpleNamespace(),
                        None,
                        None,
                        None,
                        None,
                        None,
                        Path("/V.ms"),
                        1.0,
                        None,
                        "simplenoise",
                        {},
                        1,
                        output,
                        old,
                    )

            self.assertTrue(paths["failed"].is_dir())
            state = json.loads(paths["state"].read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "rolled_back")

    def test_run_visits_every_source_and_variant(self):
        source_ids = (TRAIN_IDS[0], TEST_IDS[0], VAL_IDS[0])
        sources = {
            source_id: experiment.ThermalSource(
                source_id,
                Path(f"/{source_id}.ms"),
                Path(f"/{source_id}.json"),
                Path(f"/{source_id}-imaging"),
                (256, 256),
                3.0,
                10.0,
                100.0,
                1e-4,
            )
            for source_id in source_ids
        }
        finalized = Mock(return_value=Path("sample.json"))
        select_antenna = Mock(
            side_effect=lambda choices, seed: choices[seed % len(choices)]
        )

        class Reporter:
            def __init__(self, *args, **kwargs):
                self.completed = 0

            def sample_completed(self):
                self.completed += 1

            def finish(self):
                pass

        fake_corruption = SimpleNamespace(
            ConstantGainSpec=lambda *args, **kwargs: (args, kwargs),
            measure_constant_gain_norms=lambda spec: object(),
        )
        manifest = lambda source_run, ids, excluded: {
            "report_schema_version": experiment.REPORT_SCHEMA_VERSION,
            "source_run": str(source_run),
            "source_ids": list(ids),
            "source_partition_counts": experiment._source_partition_counts(ids),
            "configuration": {},
            "sources": [],
            "failures": [],
            "dataset_iteration": None,
            "noise_controls": {
                "dataset_index": experiment.NOISE_CONTROL_INDEX_NAME,
                "requested_source_ids": [],
                "failures": [],
                "dataset_iteration": None,
            },
        }
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            sys.modules, {"scripts.corruption": fake_corruption}
        ), patch("scripts.reporting.QuartoReporter", Reporter), patch.multiple(
            experiment,
            _ensure_thermal_entries=Mock(),
            find_thermal_run=Mock(return_value=(Path("/thermal"), {}, source_ids)),
            _thermal_source=Mock(side_effect=lambda run, report, source_id: sources[source_id]),
            _thermal_exclusions=Mock(return_value=[]),
            _new_report_manifest=Mock(side_effect=manifest),
            _check_disk_space=Mock(),
            _recover_repair_transactions=Mock(),
            _migrate_dataset_labels=Mock(),
            _audit_indexed_samples=Mock(
                return_value=experiment.IndexedSamples({}, frozenset())
            ),
            _audit_noise_controls=Mock(return_value={}),
            _recover_or_remove_sample=Mock(return_value=False),
            _unflagged_antenna_choices=Mock(
                return_value=((0, "ea01"), (1, "ea02"))
            ),
            _select_antenna=select_antenna,
            _prepare_V_ms=Mock(return_value=Path("/V.ms")),
            _noise_request=Mock(return_value=("simplenoise", {}, 1e-4)),
            _finalize_sample=finalized,
            _assert_square_pb_support=Mock(),
            _assert_sample_pb_policy=Mock(),
            _assert_noise_control_sample=Mock(),
            _append_sample_to_dataset=Mock(),
            _source_manifests=Mock(return_value={}),
            _write_comparison_plots=Mock(return_value={"panels": [], "recipes": {}}),
            _cleanup_V_work=Mock(),
            FIXED_ANTENNA_ID=None,
            FIXED_ERROR_SIGN=None,
        ), patch(
            "scripts.preprocessing.load_sample_manifest",
            return_value=SimpleNamespace(),
        ):
            output = experiment.run_experiment(
                output_dir=temporary, generate_noise_controls=True
            )
            report = json.loads(
                (output / "report.json").read_text(encoding="utf-8")
            )

        self.assertEqual(
            finalized.call_count,
            len(source_ids) * len(experiment.LABELS)
            + len(experiment.NOISE_CONTROLS),
        )
        self.assertEqual(
            select_antenna.call_count,
            len(source_ids) * len(experiment.VARIANTS),
        )
        expected_selection_seeds = {
            experiment._stable_seed(
                source_id,
                f"{variant.label_name}:antenna",
            )
            for source_id in source_ids
            for variant in experiment.VARIANTS
        }
        self.assertEqual(
            {call.kwargs["seed"] for call in select_antenna.call_args_list},
            expected_selection_seeds,
        )
        self.assertEqual(
            [(item["id"], item["partition"]) for item in report["sources"]],
            [(TRAIN_IDS[0], "train"), (TEST_IDS[0], "test"), (VAL_IDS[0], "val")],
        )
        self.assertEqual(
            report["noise_controls"]["requested_source_ids"], [VAL_IDS[0]]
        )
        control_calls = [
            call
            for call in finalized.call_args_list
            if call.kwargs.get("noise_control") is not None
        ]
        self.assertEqual(len(control_calls), len(experiment.NOISE_CONTROLS))


class TrainingNoiseTests(unittest.TestCase):
    def test_reindex_existing_noise_preserves_cleanup_audit(self):
        source_id = TRAIN_IDS[0]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary).resolve()
            (output / "dataset.json").write_text("{}\n")
            baseline_path = output / "baseline.json"
            baseline = SimpleNamespace(sample_id=source_id + "_not_corrupted", label_name="not_corrupted")
            samples = {}
            audits = {}
            for sid in experiment._requested_noise_controls(source_id):
                path = output / "samples" / sid / "sample.json"
                path.parent.mkdir(parents=True)
                path.write_text("{}")
                samples[path] = SimpleNamespace(sample_id=sid, path=path)
                audit = path.with_name("cleanup.audit.json")
                audit.write_text('{"original_cleanup_history": true}\n')
                audits[audit] = (audit.read_bytes(), audit.stat().st_mtime_ns)
            with patch("scripts.preprocessing.load_dataset_manifest", return_value=SimpleNamespace(samples=[baseline_path])), \
                 patch("scripts.preprocessing.load_sample_manifest", side_effect=lambda path, **kw: baseline if path == baseline_path else samples[path]), \
                 patch.multiple(experiment,
                    find_thermal_run=Mock(return_value=(Path("/thermal"), {}, (source_id,))),
                    _thermal_source=Mock(), _audit_noise_controls=Mock(return_value={}),
                    _assert_noise_control_sample=Mock(), _append_sample_to_dataset=Mock(),
                    _noise_control_payload=Mock(return_value={}),
                    _iterate_and_validate_noise_controls=Mock(return_value={})), \
                 patch.object(experiment, "_recover_or_remove_sample") as recovery, \
                 patch.object(experiment, "_prepare_V_ms") as prepare:
                experiment.run_noise_training(output_dir=output)
                recovery.assert_not_called()
                prepare.assert_not_called()
                self.assertEqual(experiment._append_sample_to_dataset.call_count, len(samples))
            for audit, before in audits.items():
                self.assertEqual((audit.read_bytes(), audit.stat().st_mtime_ns), before)

    def test_noise_extension_keeps_gain_levels_and_validation_split(self):
        old_levels = (5., 10., 15., 20., 30., 40., 50., 100.)
        self.assertEqual(experiment.SNR_CORR_TARGETS, old_levels)
        self.assertEqual(experiment.NOISE_SNR_CORR_TARGETS, (*old_levels, 1000., 2000., 3000., 4000.))
        available = [TRAIN_IDS[0], VAL_IDS[0], TEST_IDS[0]]
        self.assertEqual(experiment._select_noise_sources(available, available, partition="val"), (VAL_IDS[0],))
        for selected in ([TRAIN_IDS[0]], [TEST_IDS[0]]):
            with self.assertRaises(ValueError):
                experiment._select_noise_sources(available, available, selected, partition="val")
        with patch.object(experiment, "run_noise_validation") as validation, patch.object(experiment, "run_experiment") as main:
            experiment.main(["--noise-validation-only", "--noise-control-source", VAL_IDS[0]])
            validation.assert_called_once_with(source_run=None, output_dir=None, source_ids=[VAL_IDS[0]])
            main.assert_not_called()

    def test_selection_and_dispatch_are_train_only(self):
        available = [TRAIN_IDS[0], VAL_IDS[0], TEST_IDS[0]]
        self.assertEqual(experiment._select_noise_training_sources(available, available), (TRAIN_IDS[0],))
        for selected in ([VAL_IDS[0]], [TEST_IDS[0]], [TRAIN_IDS[0]] * 2, [TRAIN_IDS[1]]):
            with self.subTest(selected=selected), self.assertRaises(ValueError):
                experiment._select_noise_training_sources(available, available, selected)
        with patch.object(experiment, "run_noise_training") as training, patch.object(experiment, "run_experiment") as main:
            experiment.main(["--noise-training-source", TRAIN_IDS[0]])
            training.assert_called_once_with(source_run=None, output_dir=None, source_ids=[TRAIN_IDS[0]])
            main.assert_not_called()
        with self.assertRaises(SystemExit):
            experiment._parse_args(["--noise-training", "--noise-controls"])

    def test_training_generation_preserves_main_and_resumes_without_work(self):
        source_id = TRAIN_IDS[0]
        indexed = {}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary).resolve()
            main_index = output / "dataset.json"; main_index.write_text("{}\n")
            old_report = output / "report.json"; old_report.write_text('{"failures": ["old gain failure"]}\n')
            baseline_path = output / "baseline.json"
            baseline = SimpleNamespace(sample_id=source_id + "_not_corrupted", label_name="not_corrupted")
            source = SimpleNamespace(source_id=source_id)

            def finalize(*args, noise_control):
                sid = source_id + "_" + noise_control.label_name
                sample = SimpleNamespace(sample_id=sid, path=output / "samples" / sid / "sample.json")
                indexed[sid] = sample
                return sample.path

            def load_sample(path, **kwargs):
                return baseline if path == baseline_path else indexed[path.parent.name]

            with patch("scripts.preprocessing.load_dataset_manifest", return_value=SimpleNamespace(samples=[baseline_path])), \
                 patch("scripts.preprocessing.load_sample_manifest", side_effect=load_sample), \
                 patch.multiple(experiment,
                    find_thermal_run=Mock(return_value=(Path("/thermal"), {}, (source_id,))),
                    _thermal_source=Mock(return_value=source),
                    _audit_noise_controls=Mock(side_effect=lambda index: dict(indexed)),
                    _recover_or_remove_sample=Mock(return_value=False),
                    _prepare_V_ms=Mock(return_value=Path("/V.ms")),
                    _check_disk_space=Mock(), _cleanup_V_work=Mock(),
                    _noise_request=Mock(return_value=("simplenoise", {}, 1.)),
                    _assert_noise_control_sample=Mock(),
                    _noise_control_payload=Mock(return_value={"SNR_corr_measured": 20.}),
                    _iterate_and_validate_noise_controls=Mock(return_value={"sample_count": len(experiment.NOISE_CONTROLS)})), \
                 patch.object(experiment, "_finalize_sample", side_effect=finalize) as finalized, \
                 patch.object(experiment, "_append_sample_to_dataset") as append, \
                 patch.object(experiment, "_ensure_thermal_entries") as precursor, \
                 patch.object(experiment, "_migrate_dataset_labels") as migrate:
                experiment.run_noise_training(output_dir=output)
                self.assertEqual(finalized.call_count, len(experiment.NOISE_CONTROLS))
                self.assertTrue(all(call.args[0] == output / "dataset_noise_training.json" for call in append.call_args_list))
                report = output / "noise_training_report.json"
                before = (report.read_bytes(), report.stat().st_mtime_ns)
                finalized.reset_mock()
                experiment._prepare_V_ms.reset_mock()
                experiment.run_noise_training(output_dir=output)
                finalized.assert_not_called(); experiment._prepare_V_ms.assert_not_called()
                self.assertEqual(before, (report.read_bytes(), report.stat().st_mtime_ns))
                precursor.assert_not_called(); migrate.assert_not_called()
                # An existing eight-level run can be extended without regenerating it.
                saved = json.loads(report.read_text())
                saved["configuration"]["levels"] = list(experiment.SNR_CORR_TARGETS)
                report.write_text(json.dumps(saved))
                for sid in list(indexed):
                    if float(sid.rsplit("_", 1)[1]) > 100:
                        del indexed[sid]
                experiment.run_noise_training(output_dir=output)
                self.assertEqual(finalized.call_count, 4)
                self.assertEqual(json.loads(report.read_text())["configuration"]["levels"], list(experiment.NOISE_SNR_CORR_TARGETS))
            self.assertEqual(main_index.read_text(), "{}\n")
            self.assertEqual(old_report.read_text(), '{"failures": ["old gain failure"]}\n')


if __name__ == "__main__":
    unittest.main()
