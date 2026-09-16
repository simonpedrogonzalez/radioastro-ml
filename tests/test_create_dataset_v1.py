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
        self.assertEqual(len(experiment.LABELS), 9)
        self.assertEqual(
            set(experiment.LABELS),
            {"not_corrupted", *(v.label_name for v in experiment.VARIANTS)},
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

    def test_report_uses_realized_gain_and_signed_phase(self):
        template = experiment.REPORT_TEMPLATE.read_text(encoding="utf-8")

        self.assertIn('physical = solution.get("g_amp")', template)
        self.assertIn('physical = solution.get("phi_deg")', template)
        self.assertIn('"Amplitude gain"', template)
        self.assertIn('"Phase offset (deg)"', template)

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
            ConstantGainSpec=lambda *args: args,
            measure_constant_gain_norms=lambda spec: object(),
        )
        manifest = lambda source_run, ids, excluded: {
            "source_run": str(source_run),
            "source_ids": list(ids),
            "source_partition_counts": experiment._source_partition_counts(ids),
            "configuration": {},
            "sources": [],
            "failures": [],
            "dataset_iteration": None,
        }
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            sys.modules, {"scripts.corruption": fake_corruption}
        ), patch("scripts.reporting.QuartoReporter", Reporter), patch.multiple(
            experiment,
            find_thermal_run=Mock(return_value=(Path("/thermal"), {}, source_ids)),
            _thermal_source=Mock(side_effect=lambda run, report, source_id: sources[source_id]),
            _thermal_exclusions=Mock(return_value=[]),
            _new_report_manifest=Mock(side_effect=manifest),
            _check_disk_space=Mock(),
            _existing_sample_ids=Mock(return_value=set()),
            _recover_or_remove_sample=Mock(return_value=False),
            _unflagged_antenna_choices=Mock(
                return_value=((0, "ea01"), (1, "ea02"))
            ),
            _select_antenna=select_antenna,
            _prepare_V_ms=Mock(return_value=Path("/V.ms")),
            _noise_request=Mock(return_value=("simplenoise", {}, 1e-4)),
            _finalize_sample=finalized,
            _source_manifests=Mock(return_value={}),
            _shared_source_display_limits=Mock(return_value={}),
            _write_comparison_plots=Mock(return_value={"panels": [], "recipes": {}}),
            _cleanup_V_work=Mock(),
        ):
            output = experiment.run_experiment(output_dir=temporary)
            report = json.loads(
                (output / "report.json").read_text(encoding="utf-8")
            )

        self.assertEqual(finalized.call_count, len(source_ids) * len(experiment.LABELS))
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


if __name__ == "__main__":
    unittest.main()
