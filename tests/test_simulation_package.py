from __future__ import annotations

import inspect
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.simulation as simulation
from scripts.simulation import components as component_module
from scripts.simulation import noise as noise_module


class FakeTable:
    def __init__(self, tables):
        self.tables = tables
        self.current = None

    def open(self, path, nomodify=True):
        name = Path(path).name
        self.current = self.tables.get(name, self.tables.get("MAIN"))
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
        return self.current["columns"][column]

    def getcolkeywords(self, column):
        return self.current.get("keywords", {}).get(column, {})


def component_tables(*, num_poly=0, ephemeris=-1, direction_code=0):
    return {
        "FIELD": {
            "nrows": 1,
            "columns": {
                "PHASE_DIR": [[[0.25], [-0.5]]],
                "NUM_POLY": [num_poly],
                "EPHEMERIS_ID": [ephemeris],
                "PhaseDir_Ref": [direction_code],
            },
            "keywords": {
                "PHASE_DIR": {
                    "QuantumUnits": ["rad", "rad"],
                    "MEASINFO": {
                        "VarRefCol": "PhaseDir_Ref",
                        "TabRefCodes": [0, 21],
                        "TabRefTypes": ["J2000", "ITRF"],
                    },
                }
            },
        },
        "SPECTRAL_WINDOW": {
            "nrows": 1,
            "columns": {
                "REF_FREQUENCY": [7.228],
                "MEAS_FREQ_REF": [5],
            },
            "keywords": {
                "REF_FREQUENCY": {
                    "QuantumUnits": ["GHz"],
                    "MEASINFO": {
                        "VarRefCol": "MEAS_FREQ_REF",
                        "TabRefCodes": [1, 5],
                        "TabRefTypes": ["LSRK", "TOPO"],
                    },
                }
            },
        },
    }


def noise_tables(*, exposures=None, antenna2=None, data_description_ids=None, spw_ids=None):
    exposures = [5.0, 5.0] if exposures is None else exposures
    antenna2 = [1, 2] if antenna2 is None else antenna2
    data_description_ids = [0, 0] if data_description_ids is None else data_description_ids
    spw_ids = [0] if spw_ids is None else spw_ids
    return {
        "MAIN": {
            "nrows": len(exposures),
            "columns": {
                "DATA_DESC_ID": data_description_ids,
                "EXPOSURE": exposures,
                "ANTENNA1": [0] * len(exposures),
                "ANTENNA2": antenna2,
            },
            "keywords": {"EXPOSURE": {"QuantumUnits": ["s"]}},
        },
        "DATA_DESCRIPTION": {
            "nrows": len(spw_ids),
            "columns": {"SPECTRAL_WINDOW_ID": spw_ids},
        },
        "SPECTRAL_WINDOW": {
            "nrows": 2,
            "columns": {
                "EFFECTIVE_BW": [[2e6, 2e6], [1e6, 1e6]],
                "CHAN_FREQ": [[7.2e9, 7.3e9], [3e9, 3.1e9]],
            },
            "keywords": {
                "EFFECTIVE_BW": {"QuantumUnits": ["Hz"]},
                "CHAN_FREQ": {"QuantumUnits": ["Hz"]},
            },
        },
    }


class FakeSimulator:
    def __init__(self):
        self.calls = []

    def setseed(self, seed):
        self.calls.append(("setseed", seed))
        return True

    def setnoise(self, **kwargs):
        self.calls.append(("setnoise", kwargs))
        return True

    def corrupt(self):
        self.calls.append(("corrupt",))
        return True


class ComponentTests(unittest.TestCase):
    def factory(self, tables):
        return lambda: FakeTable(tables)

    def test_metadata_uses_variable_reference_mappings_and_units(self):
        tables = component_tables()
        with patch.object(component_module, "_new_table", self.factory(tables)):
            self.assertEqual(
                simulation.get_phase_center("example.ms"),
                "J2000 0.25rad -0.5rad",
            )
            self.assertEqual(
                simulation.get_reference_frequency("example.ms"),
                "TOPO 7228000000Hz",
            )

    def test_source_records_have_one_consistent_family(self):
        tables = component_tables()
        with patch.object(component_module, "_new_table", self.factory(tables)):
            direct = simulation.phase_center_point_source("example.ms", flux_jy=0.01)
            from_snr = simulation.phase_center_point_source_from_snr(
                "example.ms", 100.0, 1e-4
            )
        self.assertEqual(direct, from_snr)
        self.assertEqual(
            direct,
            {
                "flux": [0.01, 0.0, 0.0, 0.0],
                "fluxunit": "Jy",
                "polarization": "Stokes",
                "dir": "J2000 0.25rad -0.5rad",
                "shape": "point",
                "freq": "TOPO 7228000000Hz",
                "spectrumtype": "constant",
            },
        )

    def test_moving_and_polynomial_fields_are_rejected(self):
        for tables, message in (
            (component_tables(num_poly=1), "Polynomial"),
            (component_tables(ephemeris=0), "Ephemeris"),
        ):
            with self.subTest(message=message), patch.object(
                component_module, "_new_table", self.factory(tables)
            ):
                with self.assertRaisesRegex(ValueError, message):
                    simulation.get_phase_center("example.ms")

    def test_invalid_source_scalars_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "snr"):
            simulation.phase_center_point_source_from_snr("x.ms", 0, 1.0)
        with self.assertRaisesRegex(ValueError, "flux_jy"):
            simulation.phase_center_point_source("x.ms", float("nan"))


class NoiseCalculationTests(unittest.TestCase):
    def table_factory(self, tables):
        return lambda: FakeTable(tables)

    def test_option_a2_matches_verified_0012_399_value(self):
        tables = noise_tables()
        with patch.object(noise_module, "_new_table", self.table_factory(tables)):
            actual = simulation.theoretical_vla_simplenoise(
                "example.ms", sefd_jy=310.0, eta_c=0.93
            )
        self.assertAlmostEqual(actual, 0.074535599249993, places=14)

    def test_ect_equivalent_0012_399_override(self):
        tables = noise_tables()
        with patch.object(noise_module, "_new_table", self.table_factory(tables)):
            actual = simulation.theoretical_vla_simplenoise(
                "example.ms", sefd_jy=236.7, eta_c=0.93
            )
        self.assertAlmostEqual(actual, 0.05691153658862367, places=14)
        self.assertAlmostEqual(actual / math.sqrt(5_738_686), 2.3757135808443826e-5)

    def test_image_rms_inverse_relationship(self):
        actual = simulation.simplenoise_from_image_rms(
            2.29388305498282e-5,
            nchan=64,
            npol=2,
            nbaselines=351,
            nintegrations=235,
        )
        self.assertAlmostEqual(actual, 0.074535599249993, places=14)

    def test_sampling_rejections_are_explicit(self):
        cases = (
            (noise_tables(exposures=[5.0, 6.0]), "EXPOSURE is not homogeneous"),
            (noise_tables(antenna2=[0, 2]), "Autocorrelation"),
            (
                noise_tables(data_description_ids=[0, 1], spw_ids=[0, 1]),
                "exactly one used spectral window",
            ),
        )
        for tables, message in cases:
            with self.subTest(message=message), patch.object(
                noise_module, "_new_table", self.table_factory(tables)
            ):
                with self.assertRaisesRegex(ValueError, message):
                    simulation.theoretical_vla_simplenoise(
                        "example.ms", sefd_jy=310.0, eta_c=0.93
                    )

    def test_invalid_counts_and_physical_values_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "nchan"):
            simulation.simplenoise_from_image_rms(
                1.0, nchan=0, npol=2, nbaselines=1, nintegrations=1
            )
        with self.assertRaisesRegex(ValueError, "sefd_jy"):
            simulation.theoretical_vla_simplenoise(
                "unused.ms", sefd_jy=-1, eta_c=0.93
            )


class NoiseSelectorTests(unittest.TestCase):
    def test_vla_selector_uses_snapshot_and_exact_casa_calls(self):
        simulator = FakeSimulator()
        sampling = noise_module._HomogeneousSampling(0, 2e6, 5.0, (7.2e9, 7.3e9))
        with patch.object(noise_module, "_inspect_homogeneous_sampling", return_value=sampling):
            metadata = noise_module._apply_noise(
                simulator,
                "example.ms",
                noise_model="vla-thermal",
                noise_parameters={"band": "C", "sampler": "8-bit"},
                seed=12345,
            )
        self.assertEqual(simulator.calls[0], ("setseed", 12345))
        self.assertEqual(simulator.calls[1][0], "setnoise")
        self.assertEqual(simulator.calls[1][1]["mode"], "simplenoise")
        self.assertEqual(simulator.calls[2], ("corrupt",))
        self.assertAlmostEqual(metadata["simplenoise_jy"], 0.074535599249993)
        self.assertEqual(metadata["resolved_parameters"]["sefd_jy"], 310.0)
        self.assertEqual(metadata["resolved_parameters"]["eta_c"], 0.93)

    def test_vla_overrides_and_three_bit_requirement(self):
        simulator = FakeSimulator()
        sampling = noise_module._HomogeneousSampling(0, 2e6, 5.0, (7.2e9,))
        with patch.object(noise_module, "_inspect_homogeneous_sampling", return_value=sampling):
            with self.assertRaisesRegex(ValueError, "eta_c is required"):
                noise_module._apply_noise(
                    simulator,
                    "example.ms",
                    noise_model="vla-thermal",
                    noise_parameters={"band": "C", "sampler": "3bit"},
                    seed=1,
                )
            metadata = noise_module._apply_noise(
                simulator,
                "example.ms",
                noise_model="vla-thermal",
                noise_parameters={
                    "band": "C",
                    "sampler": "3bit",
                    "sefd_jy": 236.7,
                    "eta_c": 0.8,
                },
                seed=2,
            )
        self.assertEqual(metadata["resolved_parameters"]["sefd_source"], "override")
        self.assertEqual(metadata["resolved_parameters"]["eta_c_source"], "override")
        self.assertAlmostEqual(metadata["resolved_parameters"]["sefd_jy"], 236.7)

    def test_direct_simplenoise_is_normalized_before_casa(self):
        simulator = FakeSimulator()
        with patch.object(noise_module, "_quantity_in_jy", return_value=0.125):
            metadata = noise_module._apply_noise(
                simulator,
                "example.ms",
                noise_model="simplenoise",
                noise_parameters={"simplenoise": "125mJy"},
                seed=7,
            )
        self.assertEqual(
            simulator.calls,
            [
                ("setseed", 7),
                ("setnoise", {"mode": "simplenoise", "simplenoise": "0.125Jy"}),
                ("corrupt",),
            ],
        )
        self.assertEqual(metadata["simplenoise_jy"], 0.125)

    def test_reserved_unknown_and_conflicting_inputs_fail_before_casa(self):
        for model in ("tsys-atm", "tsys-manual"):
            with self.subTest(model=model), self.assertRaisesRegex(
                NotImplementedError, "varying-weight"
            ):
                noise_module._normalized_noise_request(model, {})
        with self.assertRaisesRegex(ValueError, "Unknown noise_model"):
            noise_module._normalized_noise_request("other", {})
        with self.assertRaisesRegex(ValueError, "cannot contain"):
            noise_module._normalized_noise_request(
                "simplenoise", {"simplenoise": "1Jy", "mode": "simplenoise"}
            )
        with self.assertRaisesRegex(ValueError, "Unknown parameters"):
            noise_module._normalized_noise_request(
                "vla-thermal", {"band": "C", "sampler": "8bit", "pwv": "1mm"}
            )

    def test_wrong_band_label_is_rejected(self):
        simulator = FakeSimulator()
        sampling = noise_module._HomogeneousSampling(0, 2e6, 5.0, (7.2e9, 7.3e9))
        with patch.object(
            noise_module,
            "_inspect_homogeneous_sampling",
            return_value=sampling,
        ):
            with self.assertRaisesRegex(ValueError, "do not all fall"):
                noise_module._apply_noise(
                    simulator,
                    "example.ms",
                    noise_model="vla-thermal",
                    noise_parameters={"band": "X", "sampler": "8bit"},
                    seed=1,
                )
        self.assertEqual(simulator.calls, [])

    def test_two_channels_at_band_edge_are_tolerated(self):
        simulator = FakeSimulator()
        sampling = noise_module._HomogeneousSampling(
            0,
            2e6,
            5.0,
            tuple(1.878e9 + index * 2e6 for index in range(64)),
        )
        with patch.object(noise_module, "_inspect_homogeneous_sampling", return_value=sampling):
            noise_module._apply_noise(
                simulator,
                "example.ms",
                noise_model="vla-thermal",
                noise_parameters={"band": "L", "sampler": "8bit"},
                seed=1,
            )
        self.assertEqual(simulator.calls[-1], ("corrupt",))


class PublicContractTests(unittest.TestCase):
    def test_versioned_reports_write_as_a_pair_and_read_schema_one(self):
        report = {
            "schema_version": simulation.SIMULATION_REPORT_SCHEMA_VERSION,
            "sample_id": "sample",
            "components": [],
            "noise": None,
            "operations": ["copy", "weights"],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            json_path, text_path = simulation.write_simulation_reports(
                report, root / "simulation.json", root / "simulation.txt"
            )
            self.assertEqual(simulation.load_simulation_report(json_path), report)
            self.assertIn("Operation stages", text_path.read_text(encoding="utf-8"))
            legacy = root / "legacy.json"
            legacy.write_text('{"schema_version": 1}\n', encoding="utf-8")
            self.assertEqual(simulation.load_simulation_report(legacy)["schema_version"], 1)

    def test_nonfinite_simulation_report_leaves_no_pair(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "strict JSON"):
                simulation.write_simulation_reports(
                    {
                        "schema_version": simulation.SIMULATION_REPORT_SCHEMA_VERSION,
                        "bad": float("nan"),
                    },
                    root / "simulation.json",
                    root / "simulation.txt",
                )
            self.assertFalse((root / "simulation.json").exists())
            self.assertFalse((root / "simulation.txt").exists())

    def test_public_exports_are_exact(self):
        self.assertEqual(
            set(simulation.__all__),
            {
                "SimulationResult",
                "SIMULATION_REPORT_SCHEMA_VERSION",
                "SUPPORTED_SIMULATION_REPORT_SCHEMAS",
                "add_thermal_noise_inplace",
                "get_phase_center",
                "get_reference_frequency",
                "natural_image_rms_from_simplenoise",
                "load_simulation_report",
                "phase_center_point_source",
                "phase_center_point_source_from_snr",
                "simulate_ms",
                "render_simulation_text",
                "simplenoise_from_image_rms",
                "theoretical_vla_simplenoise",
                "write_simulation_reports",
            },
        )
        parameters = inspect.signature(simulation.simulate_ms).parameters
        self.assertEqual(parameters["noise_model"].default, None)
        self.assertEqual(parameters["noise_parameters"].default, None)
        self.assertEqual(parameters["seed"].default, 185349251)

    def test_result_is_an_immutable_small_record(self):
        result = simulation.SimulationResult(
            Path("result.ms"), None, Path("result.simulation.json"), None, None
        )
        with self.assertRaises(Exception):
            result.seed = 1


if __name__ == "__main__":
    unittest.main()
