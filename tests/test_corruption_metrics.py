from __future__ import annotations

import importlib.util
import json
import math
import os
import shutil
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
        self.V_ms = self.root / "V.ms"
        self.V_corr_ms = self.root / "V_corr.ms"
        self.V_ms.mkdir()
        self.V_corr_ms.mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def _main(self, *, affected_factor=1.0):
        np = self.np
        data = np.full((4, 2, 4), 2.0 + 0.0j, dtype=np.complex128)
        data[1:3, :, :] = 1000.0 + 0.0j
        data[:, :, 2:] = 1000.0 + 0.0j
        if affected_factor != 1.0:
            data[:, :, 1] *= affected_factor
        flag = np.zeros(data.shape, dtype=bool)
        flag[0, 0, 0] = True
        return {
            "DATA": data,
            "FLAG": flag,
            "FLAG_ROW": np.array([False, False, False, True]),
            "ANTENNA1": np.array([0, 1, 0, 2]),
            "ANTENNA2": np.array([1, 2, 0, 0]),
            "DATA_DESC_ID": np.zeros(4, dtype=int),
        }

    def _tables(self, *, affected_factor=1.0):
        return {
            "V": self._main(),
            "V_corr": self._main(affected_factor=affected_factor),
            "DATA_DESCRIPTION": {
                "POLARIZATION_ID": self.np.array([0], dtype=int)
            },
            "POLARIZATION": {
                "CORR_TYPE": [self.np.array([5, 6, 7, 8], dtype=int)]
            },
            "ANTENNA": {"NAME": self.np.array(["A0", "A1", "A2"])},
        }

    def _table_factory(self, tables):
        np = self.np

        class FakeTable:
            def __init__(self):
                self.name = None

            def open(self, path, **kwargs):
                del kwargs
                path = Path(path)
                self.name = path.stem if path.suffix == ".ms" else path.name

            def close(self):
                pass

            def colnames(self):
                return list(tables[self.name])

            def nrows(self):
                if self.name == "ANTENNA":
                    return len(tables[self.name]["NAME"])
                if self.name == "POLARIZATION":
                    return len(tables[self.name]["CORR_TYPE"])
                return next(iter(tables[self.name].values())).shape[-1]

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

    def _spec(self, **overrides):
        from scripts.corruption import ConstantGainSpec

        values = {
            "V_ms": self.V_ms,
            "antenna_id": 2,
            "corruption_type": "amp",
            "SNR_corr_target": 2.0,
            "sigma": 2.0,
            "sign": 1,
        }
        values.update(overrides)
        return ConstantGainSpec(**values)

    def test_exact_noiseless_norms_use_only_valid_parallel_hands(self):
        import scripts.corruption.metrics as metric_module
        from scripts.corruption import measure_constant_gain_norms

        factory = self._table_factory(self._tables())
        with patch.object(metric_module, "_new_table", side_effect=factory):
            norms = measure_constant_gain_norms(self._spec(), chunk_rows=1)
        self.assertEqual(norms.valid_sample_count, 7)
        self.assertEqual(norms.affected_sample_count, 4)
        self.assertAlmostEqual(norms.V_L2, math.sqrt(28.0))
        self.assertAlmostEqual(norms.V_Ak_L2, 4.0)
        self.assertAlmostEqual(norms.V_Ak_over_sigma_L2, 2.0)
        with patch.object(metric_module, "_new_table", side_effect=factory):
            whole = measure_constant_gain_norms(self._spec(), chunk_rows=4)
        self.assertEqual(norms, whole)

    def test_amplitude_and_phase_solve_the_equations(self):
        import scripts.corruption.metrics as metric_module
        from scripts.corruption import measure_constant_gain_norms, solve_constant_gain

        factory = self._table_factory(self._tables())
        with patch.object(metric_module, "_new_table", side_effect=factory):
            norms = measure_constant_gain_norms(self._spec())
        target = 1.0
        amplitude = solve_constant_gain(
            self._spec(SNR_corr_target=target, sign=-1), norms
        )
        phase = solve_constant_gain(
            self._spec(
                SNR_corr_target=target,
                corruption_type="phase",
                sign=-1,
            ),
            norms,
        )
        self.assertAlmostEqual(amplitude.eps_g, 0.5)
        self.assertAlmostEqual(amplitude.g_amp, 0.5)
        self.assertAlmostEqual(phase.eps_g, 0.5)
        self.assertAlmostEqual(phase.phi_rad, -2.0 * math.asin(0.25))
        self.assertAlmostEqual(phase.phi_deg, math.degrees(phase.phi_rad))
        self.assertAlmostEqual(amplitude.SNR_corr_expected, target)
        self.assertAlmostEqual(amplitude.eps_vis_expected, 2.0 / math.sqrt(28.0))

    def test_measured_Delta_V_metrics_match_hand_calculation(self):
        import scripts.corruption.metrics as metric_module
        from scripts.corruption import measure_corruption_metrics

        factory = self._table_factory(self._tables(affected_factor=1.5))
        with patch.object(metric_module, "_new_table", side_effect=factory):
            metrics = measure_corruption_metrics(
                self.V_ms, self.V_corr_ms, sigma=2.0, chunk_rows=1
            )
        self.assertEqual(metrics.valid_sample_count, 7)
        self.assertAlmostEqual(metrics.V_L2, math.sqrt(28.0))
        self.assertAlmostEqual(metrics.Delta_V_L2, 2.0)
        self.assertAlmostEqual(metrics.Delta_V_over_sigma_L2, 1.0)
        self.assertAlmostEqual(metrics.eps_vis, 2.0 / math.sqrt(28.0))
        self.assertAlmostEqual(metrics.SNR_corr, 1.0)

    def test_constructor_and_reports_keep_solution_and_measurement_explicit(self):
        import scripts.corruption.metrics as metric_module
        from scripts.corruption import (
            AntennaGainCorruption,
            Constant,
            TimeGrid,
            measure_corruption_metrics,
            solve_constant_gain,
            write_corruption_reports,
        )

        factory = self._table_factory(self._tables(affected_factor=1.5))
        spec = self._spec(SNR_corr_target=1.0)
        with patch.object(metric_module, "_new_table", side_effect=factory):
            solution = solve_constant_gain(spec)
        with patch.object(metric_module, "_new_table", side_effect=factory):
            metrics = measure_corruption_metrics(self.V_ms, self.V_corr_ms, 2.0)
        corruption = AntennaGainCorruption.from_constant_gain_solution(
            TimeGrid(solint="10m"), solution
        )
        self.assertIsInstance(corruption.amp_fn, Constant)
        self.assertFalse(hasattr(corruption, "metrics"))
        json_path, text_path = write_corruption_reports(
            corruption,
            json_path=self.root / "corruption.json",
            text_path=self.root / "corruption.txt",
            solution=solution,
            metrics=metrics,
        )
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        text = text_path.read_text(encoding="utf-8")
        self.assertEqual(payload["schema_version"], 2)
        self.assertEqual(payload["solution"]["SNR_corr_target"], 1.0)
        self.assertEqual(payload["metrics"]["SNR_corr"], 1.0)
        self.assertIn("Delta_V=V_corr-V", text)
        self.assertIn("eps_vis=||Delta_V||_2/||V||_2", text)
        self.assertIn("SNR_corr=||Delta_V/sigma||_2", text)

    def test_validation_does_not_clip_invalid_solutions(self):
        import scripts.corruption.metrics as metric_module
        from scripts.corruption import measure_constant_gain_norms, solve_constant_gain

        factory = self._table_factory(self._tables())
        with patch.object(metric_module, "_new_table", side_effect=factory):
            norms = measure_constant_gain_norms(self._spec())
        with self.assertRaisesRegex(ValueError, "non-positive"):
            solve_constant_gain(self._spec(sign=-1), norms)
        with self.assertRaisesRegex(ValueError, "eps_g <= 2"):
            solve_constant_gain(
                self._spec(corruption_type="phase", SNR_corr_target=5.0), norms
            )
        with self.assertRaisesRegex(ValueError, "SNR_corr_target"):
            self._spec(SNR_corr_target=0)


@unittest.skipUnless(
    os.environ.get("RUN_CASA_CORRUPTION_INTEGRATION") == "1",
    "set RUN_CASA_CORRUPTION_INTEGRATION=1",
)
class CorruptionMetricCasaIntegrationTests(unittest.TestCase):
    SOURCE_MS = ROOT / "collect/extracted/0012-399/0012-399/0012-399.ms"

    def test_real_V_corruption_measurement_and_shared_noise(self):
        import numpy as np
        from casatools import table
        from scripts.corruption import (
            AntennaGainCorruption,
            ConstantGainSpec,
            TimeGrid,
            measure_corruption_metrics,
            solve_constant_gain,
        )
        from scripts.simulation import (
            add_thermal_noise_inplace,
            phase_center_point_source,
            simulate_ms,
        )

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
            V_result = simulate_ms(
                self.SOURCE_MS, [component], root / "V.ms", noise_model=None
            )
            sigma = 0.001
            solution = solve_constant_gain(
                ConstantGainSpec(V_result.ms_path, 0, "amp", 5.0, sigma)
            )
            V_corr = root / "V_corr.ms"
            shutil.copytree(V_result.ms_path, V_corr)
            corruption = AntennaGainCorruption.from_constant_gain_solution(
                TimeGrid(solint="10m"), solution
            )
            gain_table = root / "amp.G"
            corruption.build_corrtable(str(V_corr), str(gain_table), seed=2718)
            corruption.apply_corrtable(str(V_corr), str(gain_table), seed=2718)
            metrics = measure_corruption_metrics(V_result.ms_path, V_corr, sigma)
            self.assertTrue(math.isclose(metrics.SNR_corr, 5.0, rel_tol=0.01))

            baseline = root / "baseline.ms"
            variant = root / "variant.ms"
            shutil.copytree(V_result.ms_path, baseline)
            shutil.copytree(V_corr, variant)
            request = dict(
                noise_model="vla-thermal",
                noise_parameters=noise_parameters,
                seed=314159,
            )
            add_thermal_noise_inplace(baseline, **request)
            add_thermal_noise_inplace(variant, **request)
            arrays = []
            for path, source in ((baseline, V_result.ms_path), (variant, V_corr)):
                tb = table()
                tb.open(str(path))
                observed = np.asarray(tb.getcol("DATA"))
                tb.close()
                tb.open(str(source))
                noiseless = np.asarray(tb.getcol("DATA"))
                tb.close()
                arrays.append(observed - noiseless)
            max_difference = float(np.max(np.abs(arrays[0] - arrays[1])))
            self.assertLessEqual(
                max_difference,
                2.0e-6,
                f"recovered shared-noise arrays differ by {max_difference}",
            )


if __name__ == "__main__":
    unittest.main()
