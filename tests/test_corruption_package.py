from __future__ import annotations

import importlib.util
import inspect
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.corruption import (
    CORRUPTION_REPORT_SCHEMA_VERSION,
    write_corruption_reports,
)


class _ExampleCorruption:
    def to_report_dict(self):
        return {"type": "example", "strength": 3.5, "nested": {"enabled": True}}

    def to_report_text(self):
        return "ExampleCorruption(strength=3.5)"


class CorruptionReportingTests(unittest.TestCase):
    def test_generic_writer_uses_object_owned_representations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            json_path, text_path = write_corruption_reports(
                _ExampleCorruption(),
                json_path=root / "corruption.json",
                text_path=root / "corruption.txt",
                context={"sample_id": "0012-399", "seed": 7},
            )
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            rendered = text_path.read_text(encoding="utf-8")

        self.assertEqual(payload["schema_version"], CORRUPTION_REPORT_SCHEMA_VERSION)
        self.assertEqual(payload["context"], {"sample_id": "0012-399", "seed": 7})
        self.assertEqual(payload["configuration"], _ExampleCorruption().to_report_dict())
        self.assertIsNone(payload["solution"])
        self.assertIsNone(payload["metrics"])
        self.assertIn("ExampleCorruption(strength=3.5)", rendered)
        self.assertIn("sample_id: 0012-399", rendered)
        self.assertIn("Corruption metrics: not calculated", rendered)

    def test_new_configuration_type_needs_no_writer_changes(self):
        class OtherCorruption:
            def to_report_dict(self):
                return {"type": "other", "parameter": "value"}

            def to_report_text(self):
                return "OtherCorruption(parameter='value')"

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            json_path, _ = write_corruption_reports(
                OtherCorruption(),
                json_path=root / "other.json",
                text_path=root / "other.txt",
            )
            payload = json.loads(json_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["configuration"]["type"], "other")

    def test_invalid_protocol_and_nonfinite_values_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(TypeError, "must implement"):
                write_corruption_reports(
                    object(),
                    json_path=root / "bad.json",
                    text_path=root / "bad.txt",
                )

            class NonfiniteCorruption(_ExampleCorruption):
                def to_report_dict(self):
                    return {"type": "bad", "strength": math.nan}

            with self.assertRaisesRegex(ValueError, "non-finite"):
                write_corruption_reports(
                    NonfiniteCorruption(),
                    json_path=root / "nan.json",
                    text_path=root / "nan.txt",
                )
            self.assertFalse((root / "nan.json").exists())
            self.assertFalse((root / "nan.txt").exists())

            class NonfiniteMetrics:
                def to_report_dict(self):
                    return {"rho_corr": math.inf}

                def to_report_text(self):
                    return "invalid metrics"

            with self.assertRaisesRegex(ValueError, "non-finite"):
                write_corruption_reports(
                    _ExampleCorruption(),
                    json_path=root / "metrics-nan.json",
                    text_path=root / "metrics-nan.txt",
                    metrics=NonfiniteMetrics(),
                )
            self.assertFalse((root / "metrics-nan.json").exists())
            self.assertFalse((root / "metrics-nan.txt").exists())

    def test_existing_report_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            existing = root / "corruption.json"
            existing.write_text("original\n", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                write_corruption_reports(
                    _ExampleCorruption(),
                    json_path=existing,
                    text_path=root / "corruption.txt",
                )
            self.assertEqual(existing.read_text(encoding="utf-8"), "original\n")
            self.assertFalse((root / "corruption.txt").exists())


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is supplied by CASA")
class CorruptionConfigurationTests(unittest.TestCase):
    def test_old_import_paths_reexport_exact_package_objects(self):
        from scripts import corruption
        from scripts.corrfn import fBM
        from scripts.corrtab_utils import GCOLS, GTabQuery
        from scripts.timegrid import TimeGrid

        self.assertIs(fBM, corruption.fBM)
        self.assertIs(GCOLS, corruption.GCOLS)
        self.assertIs(GTabQuery, corruption.GTabQuery)
        self.assertIs(TimeGrid, corruption.TimeGrid)

    def test_new_metric_api_is_lazily_exported(self):
        from scripts import corruption
        from scripts.corruption.functions import Constant
        from scripts.corruption.metrics import (
            ConstantGainSpec,
            CorruptionMetricDefinition,
            CorruptionMetrics,
        )

        self.assertIs(Constant, corruption.Constant)
        self.assertIs(ConstantGainSpec, corruption.ConstantGainSpec)
        self.assertIs(
            CorruptionMetricDefinition,
            corruption.CorruptionMetricDefinition,
        )
        self.assertIs(CorruptionMetrics, corruption.CorruptionMetrics)

    def test_current_constructor_and_method_interface_is_preserved(self):
        from scripts.corruption import AntennaGainCorruption

        constructor = inspect.signature(AntennaGainCorruption).parameters
        self.assertEqual(tuple(constructor), ("timegrid", "amp_fn", "phase_fn", "query"))
        self.assertIsNone(constructor["amp_fn"].default)
        self.assertIsNone(constructor["phase_fn"].default)
        self.assertIsNone(constructor["query"].default)

        build = inspect.signature(AntennaGainCorruption.build_corrtable).parameters
        apply = inspect.signature(AntennaGainCorruption.apply_corrtable).parameters
        self.assertEqual(
            tuple(build), ("self", "ms", "corrtab", "seed", "diagnostic_plot")
        )
        self.assertEqual(build["seed"].default, 0)
        self.assertEqual(
            build["diagnostic_plot"].default,
            "images/corruption_function.png",
        )
        self.assertEqual(tuple(apply), ("self", "ms", "corrtab", "seed"))
        self.assertEqual(apply["seed"].default, 0)

    def test_nested_objects_own_the_antenna_gain_configuration(self):
        from scripts.corruption import AntennaGainCorruption, GCOLS, GTabQuery, TimeGrid

        class Constant:
            def __init__(self, value):
                self.value = value

            def to_report_dict(self):
                return {"type": "constant", "value": self.value}

            def __repr__(self):
                return f"Constant(value={self.value!r})"

        query = GTabQuery().where_eq(GCOLS.ANTENNA1, 3).group_by([GCOLS.ANTENNA1])
        corruption = AntennaGainCorruption(
            TimeGrid(solint="10m", interp="linear"),
            amp_fn=None,
            phase_fn=Constant(0.25),
            query=query,
        )
        payload = corruption.to_report_dict()

        self.assertEqual(payload["type"], "antenna_gain")
        self.assertEqual(payload["time_grid"]["solint"], "10m")
        self.assertEqual(
            payload["selection"]["filters"],
            [{"operation": "eq", "column": "ANTENNA1", "value": 3}],
        )
        self.assertIsNone(payload["amplitude"])
        self.assertEqual(payload["phase"]["value"], 0.25)
        self.assertEqual(payload["phase"]["unit"], "radian")
        self.assertIn("Constant(value=0.25)", corruption.to_report_text())

    def test_fbm_report_excludes_sampled_array_state(self):
        from scripts.corruption import fBM

        function = fBM(max_amp=0.2, H=0.6)
        self.assertEqual(
            function.to_report_dict(),
            {"type": "fractional_brownian_motion", "max_amp": 0.2, "H": 0.6},
        )
        self.assertEqual(repr(function), "fBM(max_amp=0.2, H=0.6)")

    def test_build_and_apply_methods_are_explicit_non_fluent_calls(self):
        import numpy as np

        import scripts.corruption.core as core
        from scripts.corruption import AntennaGainCorruption, GTab, TimeGrid

        class FakeTable:
            def __init__(self):
                self.parameters = np.ones((1, 1, 2), dtype=complex)

            def open(self, *args, **kwargs):
                return None

            def getcol(self, name):
                self.assert_name = name
                return self.parameters

            def putcol(self, name, values):
                self.assert_name = name
                self.parameters = values

            def flush(self):
                return None

            def close(self):
                return None

        class FakeSimulator:
            def openfromms(self, ms):
                self.ms = ms

            def setseed(self, seed):
                self.seed = seed

            def setapply(self, **parameters):
                self.parameters = parameters

            def corrupt(self):
                self.corrupted = True

            def done(self):
                self.finished = True

        gtab = GTab(
            ROWID=np.array([0, 1]),
            TIME=np.array([0.0, 1.0]),
            FIELD_ID=np.array([0, 0]),
            SPECTRAL_WINDOW_ID=np.array([0, 0]),
            ANTENNA1=np.array([0, 0]),
            ANTENNA2=np.array([-1, -1]),
            INTERVAL=np.array([1.0, 1.0]),
            SCAN_NUMBER=np.array([1, 1]),
            OBSERVATION_ID=np.array([0, 0]),
        )
        corruption = AntennaGainCorruption(TimeGrid(solint="int"))
        fake_table = FakeTable()
        with (
            patch.object(core, "make_template_gain_corrtab"),
            patch("casatools.table", return_value=fake_table),
            patch.object(core.GTab, "from_casa_table", return_value=gtab),
            patch.object(core, "corrfun_plot_start", return_value=(object(), object(), object())),
            patch.object(core, "corrfun_plot_add"),
            patch.object(core, "corrfun_plot_finish"),
        ):
            self.assertIsNone(
                corruption.build_corrtable("input.ms", "gain.G", seed=4)
            )

        fake_simulator = FakeSimulator()
        with patch("casatools.simulator", return_value=fake_simulator):
            self.assertIsNone(
                corruption.apply_corrtable("input.ms", "gain.G", seed=4)
            )
        self.assertEqual(fake_simulator.parameters["table"], "gain.G")
        self.assertFalse(fake_simulator.parameters["calwt"])


if __name__ == "__main__":
    unittest.main()
