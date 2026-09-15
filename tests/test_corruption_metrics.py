from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is supplied by CASA")
class CorruptionMetricTests(unittest.TestCase):
    def setUp(self):
        import numpy as np

        self.np = np
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.ms = self.root / "synthetic.ms"
        self.ms.mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def _tables(self, *, data_value=2.0, weight=1.0):
        np = self.np
        # Axes are correlation, channel, row. Parallel hands are 0 and 3.
        data = np.full((4, 2, 4), complex(data_value), dtype=np.complex128)
        data[1:3, :, :] = 1000.0 + 0.0j  # excluded cross-hands
        data[:, :, 2:] = 1000.0 + 0.0j  # excluded auto and FLAG_ROW
        flag = np.zeros(data.shape, dtype=bool)
        flag[0, 0, 0] = True
        return {
            "MAIN": {
                "DATA": data,
                "FLAG": flag,
                "FLAG_ROW": np.array([False, False, False, True]),
                "ANTENNA1": np.array([0, 1, 0, 2]),
                "ANTENNA2": np.array([1, 2, 0, 0]),
                "DATA_DESC_ID": np.zeros(4, dtype=int),
                "WEIGHT": np.full((4, 4), weight, dtype=float),
                "WEIGHT_SPECTRUM": np.full(data.shape, weight, dtype=float),
            },
            "DATA_DESCRIPTION": {"POLARIZATION_ID": np.array([0], dtype=int)},
            "POLARIZATION": {"CORR_TYPE": [np.array([5, 6, 7, 8], dtype=int)]},
            "ANTENNA": {"NAME": np.array(["A0", "A1", "A2"])},
        }

    def _table_factory(self, tables):
        np = self.np

        class FakeTable:
            def __init__(self):
                self.name = None

            def open(self, path, **kwargs):
                del kwargs
                path = Path(path)
                self.name = "MAIN" if path.suffix == ".ms" else path.name

            def close(self):
                pass

            def colnames(self):
                return list(tables[self.name])

            def nrows(self):
                if self.name == "ANTENNA":
                    return len(tables[self.name]["NAME"])
                if self.name == "POLARIZATION":
                    return len(tables[self.name]["CORR_TYPE"])
                first = next(iter(tables[self.name].values()))
                return first.shape[-1]

            def getcell(self, column, row):
                return tables[self.name][column][row]

            def getcol(self, column, startrow=0, nrow=-1):
                value = tables[self.name][column]
                if isinstance(value, list):
                    return np.asarray(value)
                if nrow == -1:
                    return value
                return value[..., startrow : startrow + nrow]

        return FakeTable

    def _parameters(self, **overrides):
        from scripts.corruption import ConstantGainParameters

        values = {
            "ms": self.ms,
            "antenna_id": 2,
            "corruption_type": "amp",
            "target_rho_corr": math.sqrt(8.0),
            "thermal_noise_jy": 1.0,
            "sign": 1,
        }
        values.update(overrides)
        return ConstantGainParameters(**values)

    def test_hand_calculated_whole_ms_metrics_and_exclusions(self):
        from scripts.corruption import Constant
        import scripts.corruption.metrics as metric_module

        tables = self._tables()
        with patch.object(metric_module, "_new_table", side_effect=self._table_factory(tables)):
            constant = Constant.from_detectability(self._parameters())

        result = constant.metrics
        self.assertIsNotNone(result)
        # Seven valid parallel-hand values at |DATA|^2=4, each debiased by 2.
        self.assertEqual(result.valid_sample_count, 7)
        self.assertEqual(result.affected_sample_count, 4)
        self.assertAlmostEqual(result.total_signal_power, 14.0)
        self.assertAlmostEqual(result.affected_signal_power, 8.0)
        self.assertAlmostEqual(result.weighted_affected_signal_power, 8.0)
        self.assertAlmostEqual(result.gain_error_magnitude, 1.0)
        self.assertAlmostEqual(constant.value, 2.0)
        self.assertAlmostEqual(result.epsilon_vis, math.sqrt(8.0 / 14.0))
        self.assertAlmostEqual(result.rho_corr, result.target_rho_corr)

    def test_amplitude_and_phase_solutions_share_gain_displacement(self):
        from scripts.corruption import Constant
        import scripts.corruption.metrics as metric_module

        tables = self._tables()
        factory = self._table_factory(tables)
        target = math.sqrt(8.0) / 2.0
        with patch.object(metric_module, "_new_table", side_effect=factory):
            amplitude = Constant.from_detectability(
                self._parameters(target_rho_corr=target, sign=-1)
            )
        with patch.object(metric_module, "_new_table", side_effect=factory):
            phase = Constant.from_detectability(
                self._parameters(
                    target_rho_corr=target,
                    corruption_type="phase",
                    sign=-1,
                )
            )

        self.assertAlmostEqual(amplitude.metrics.gain_error_magnitude, 0.5)
        self.assertAlmostEqual(amplitude.value, 0.5)
        self.assertAlmostEqual(phase.metrics.gain_error_magnitude, 0.5)
        self.assertAlmostEqual(phase.value, -2.0 * math.asin(0.25))
        self.assertAlmostEqual(amplitude.metrics.epsilon_vis, phase.metrics.epsilon_vis)
        self.assertAlmostEqual(amplitude.metrics.rho_corr, phase.metrics.rho_corr)

    def test_standard_targets_round_trip_and_chunking_is_invariant(self):
        from scripts.corruption import Constant
        import scripts.corruption.metrics as metric_module

        tables = self._tables()
        factory = self._table_factory(tables)
        for target in (1.0, 2.0, 5.0, 10.0):
            with patch.object(metric_module, "_new_table", side_effect=factory):
                constant = Constant.from_detectability(
                    self._parameters(target_rho_corr=target)
                )
            self.assertAlmostEqual(constant.metrics.rho_corr, target, places=12)

        parameters = self._parameters()
        with patch.object(metric_module, "_new_table", side_effect=factory):
            one_row = metric_module._measure_visibility_powers(
                parameters, chunk_rows=1
            )
        with patch.object(metric_module, "_new_table", side_effect=factory):
            all_rows = metric_module._measure_visibility_powers(
                parameters, chunk_rows=4
            )
        self.assertEqual(one_row, all_rows)

        del tables["MAIN"]["WEIGHT_SPECTRUM"]
        with patch.object(metric_module, "_new_table", side_effect=factory):
            broadcast_weight = metric_module._measure_visibility_powers(
                parameters, chunk_rows=2
            )
        self.assertEqual(all_rows, broadcast_weight)

    def test_weight_spectrum_is_preferred_and_invalid_elements_fall_back(self):
        import scripts.corruption.metrics as metric_module

        tables = self._tables()
        factory = self._table_factory(tables)
        tables["MAIN"]["WEIGHT"] = self.np.array([999.0])
        with patch.object(metric_module, "_new_table", side_effect=factory):
            spectrum_only = metric_module._measure_visibility_powers(
                self._parameters()
            )
        self.assertEqual(spectrum_only.weighted_affected_signal_power, 8.0)

        tables = self._tables()
        factory = self._table_factory(tables)
        tables["MAIN"]["WEIGHT_SPECTRUM"][3, 1, 1] = self.np.nan
        with patch.object(metric_module, "_new_table", side_effect=factory):
            element_fallback = metric_module._measure_visibility_powers(
                self._parameters()
            )
        self.assertEqual(spectrum_only, element_fallback)

    def test_antenna_constructor_and_reports_include_one_metrics_object(self):
        from scripts.corruption import (
            AntennaGainCorruption,
            Constant,
            TimeGrid,
            write_corruption_reports,
        )
        import scripts.corruption.metrics as metric_module

        tables = self._tables()
        with patch.object(
            metric_module, "_new_table", side_effect=self._table_factory(tables)
        ):
            corruption = AntennaGainCorruption.from_detectability(
                TimeGrid(solint="10m"),
                self._parameters(target_rho_corr=math.sqrt(8.0) / 2.0),
            )

        self.assertIsInstance(corruption.amp_fn, Constant)
        self.assertIsNone(corruption.phase_fn)
        self.assertIs(corruption.metrics, corruption.amp_fn.metrics)
        self.assertEqual(
            corruption.to_report_dict()["selection"]["filters"],
            [{"operation": "eq", "column": "ANTENNA1", "value": 2}],
        )
        paths = write_corruption_reports(
            corruption,
            json_path=self.root / "corruption.json",
            text_path=self.root / "corruption.txt",
        )
        payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
        text = paths.text_path.read_text(encoding="utf-8")
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["metrics"], corruption.metrics.to_report_dict())
        self.assertNotIn("metrics", payload["configuration"]["amplitude"])
        self.assertIn("epsilon_g=|g-1|", text)
        self.assertIn("A_k=sqrt(P_A,w)", text)
        self.assertIn("epsilon_vis=epsilon_g*sqrt(P_A/P_all)", text)
        self.assertIn("rho_corr=epsilon_g*sqrt(P_A,w)", text)
        self.assertEqual(
            payload["metrics"]["metric_definitions"][2]["latex"],
            r"\rho_{\mathrm{corr}}=\epsilon_g\sqrt{P_{A,w}}",
        )
        self.assertIn("Interpretation limitation", text)

    def test_fixed_constant_has_no_metrics_and_report_says_not_calculated(self):
        from scripts.corruption import AntennaGainCorruption, Constant, TimeGrid
        from scripts.corruption.reporting import write_corruption_reports

        constant = Constant(1.25)
        self.assertIsNone(constant.metrics)
        self.assertEqual(constant.to_report_dict(), {"type": "constant", "value": 1.25})
        self.assertTrue((constant.eval(self.np.array([0.0, 1.0])) == 1.25).all())
        corruption = AntennaGainCorruption(TimeGrid(solint="1s"), amp_fn=constant)
        paths = write_corruption_reports(
            corruption,
            json_path=self.root / "fixed.json",
            text_path=self.root / "fixed.txt",
        )
        self.assertIsNone(json.loads(paths.json_path.read_text())["metrics"])
        self.assertIn(
            "Detectability metrics: not calculated",
            paths.text_path.read_text(encoding="utf-8"),
        )

    def test_report_adapter_and_input_validation(self):
        from scripts.corruption import ConstantGainParameters

        report = self.root / "simulation.json"
        report.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "output_ms": str(self.ms),
                    "noise": {"simplenoise_jy": 0.25},
                }
            ),
            encoding="utf-8",
        )
        parameters = ConstantGainParameters.from_simulation_report(
            report,
            antenna_id=1,
            corruption_type="phase",
            target_rho_corr=5,
            sign=-1,
        )
        self.assertEqual(parameters.ms, self.ms.resolve())
        self.assertEqual(parameters.thermal_noise_jy, 0.25)
        with self.assertRaisesRegex(ValueError, "corruption_type"):
            self._parameters(corruption_type="amplitude")
        with self.assertRaisesRegex(ValueError, "sign"):
            self._parameters(sign=0)
        with self.assertRaisesRegex(ValueError, "target_rho_corr"):
            self._parameters(target_rho_corr=0)

    def test_weight_mismatch_and_nonpositive_debiased_power_fail(self):
        from scripts.corruption import Constant
        import scripts.corruption.metrics as metric_module

        mismatched = self._tables(weight=2.0)
        with (
            patch.object(
                metric_module, "_new_table", side_effect=self._table_factory(mismatched)
            ),
            self.assertRaisesRegex(ValueError, "weights are inconsistent"),
        ):
            Constant.from_detectability(self._parameters())

        noise_only = self._tables(data_value=1.0)
        with (
            patch.object(
                metric_module, "_new_table", side_effect=self._table_factory(noise_only)
            ),
            self.assertRaisesRegex(ValueError, "after aggregate thermal-noise debiasing"),
        ):
            Constant.from_detectability(self._parameters())

    def test_amplitude_and_phase_domain_failures_do_not_clip(self):
        from scripts.corruption import Constant
        import scripts.corruption.metrics as metric_module

        tables = self._tables()
        factory = self._table_factory(tables)
        with (
            patch.object(metric_module, "_new_table", side_effect=factory),
            self.assertRaisesRegex(ValueError, "non-positive"),
        ):
            Constant.from_detectability(
                self._parameters(target_rho_corr=math.sqrt(8.0), sign=-1)
            )
        with (
            patch.object(metric_module, "_new_table", side_effect=factory),
            self.assertRaisesRegex(ValueError, "<= 2"),
        ):
            Constant.from_detectability(
                self._parameters(
                    target_rho_corr=3.0 * math.sqrt(8.0),
                    corruption_type="phase",
                )
            )


@unittest.skipUnless(
    os.environ.get("RUN_CASA_CORRUPTION_INTEGRATION") == "1",
    "set RUN_CASA_CORRUPTION_INTEGRATION=1",
)
class CorruptionMetricCasaIntegrationTests(unittest.TestCase):
    SOURCE_MS = ROOT / "collect/extracted/0012-399/0012-399/0012-399.ms"

    def test_real_simulation_metrics_and_gain_tables(self):
        import numpy as np
        from casatools import table

        from scripts.corruption import (
            AntennaGainCorruption,
            ConstantGainParameters,
            TimeGrid,
            write_corruption_reports,
        )
        import scripts.corruption.core as core
        from scripts.simulation import phase_center_point_source, simulate_ms

        self.assertTrue(self.SOURCE_MS.is_dir())
        noise_parameters = {
            "sefd_jy": 236.7,
            "eta_c": 0.93,
            "band": "C",
            "sampler": "8bit",
        }
        with tempfile.TemporaryDirectory(prefix="corruption-metrics-") as temporary:
            root = Path(temporary)
            component = phase_center_point_source(self.SOURCE_MS, flux_jy=1.0)
            simulation = simulate_ms(
                self.SOURCE_MS,
                [component],
                root / "source_noise.ms",
                noise_model="vla-thermal",
                noise_parameters=noise_parameters,
                seed=314159,
            )
            self.assertIsNotNone(simulation.simplenoise_jy)

            for corruption_type in ("amp", "phase"):
                parameters = ConstantGainParameters(
                    ms=simulation.ms_path,
                    antenna_id=0,
                    corruption_type=corruption_type,
                    target_rho_corr=5.0,
                    thermal_noise_jy=simulation.simplenoise_jy,
                )
                corruption = AntennaGainCorruption.from_detectability(
                    TimeGrid(solint="10m"), parameters
                )
                self.assertAlmostEqual(corruption.metrics.rho_corr, 5.0, places=12)
                gain_table = root / f"{corruption_type}.G"
                with (
                    patch.object(
                        core,
                        "corrfun_plot_start",
                        return_value=(object(), object(), object()),
                    ),
                    patch.object(core, "corrfun_plot_add"),
                    patch.object(core, "corrfun_plot_finish"),
                ):
                    corruption.build_corrtable(
                        str(simulation.ms_path), str(gain_table), seed=2718
                    )

                tb = table()
                tb.open(str(gain_table))
                try:
                    antenna1 = np.asarray(tb.getcol("ANTENNA1"), dtype=int)
                    gains = np.asarray(tb.getcol("CPARAM"))
                finally:
                    tb.close()
                selected = gains[..., antenna1 == 0]
                expected = (
                    corruption.metrics.amplitude_gain
                    if corruption_type == "amp"
                    else np.exp(1j * corruption.metrics.phase_offset_rad)
                )
                self.assertTrue(np.allclose(selected, expected, rtol=1e-6, atol=1e-7))

                reports = write_corruption_reports(
                    corruption,
                    json_path=root / f"{corruption_type}.corruption.json",
                    text_path=root / f"{corruption_type}.corruption.txt",
                )
                payload = json.loads(reports.json_path.read_text(encoding="utf-8"))
                self.assertAlmostEqual(payload["metrics"]["rho_corr"], 5.0)
                self.assertIn(
                    "rho_corr=epsilon_g*sqrt(P_A,w)",
                    reports.text_path.read_text(encoding="utf-8"),
                )


if __name__ == "__main__":
    unittest.main()
