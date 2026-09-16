"""Public corruption configuration, gain-table, and reporting API.

Heavy scientific and CASA dependencies are imported only when their public
objects are requested, so the reporting module remains usable in plain Python.
"""

from __future__ import annotations

from importlib import import_module


_PUBLIC_IMPORTS = {
    "AntennaGainCorruption": (".core", "AntennaGainCorruption"),
    "Corruption": (".core", "Corruption"),
    "GainCorruption": (".core", "GainCorruption"),
    "CorrFn": (".functions", "CorrFn"),
    "CorrFnRealization": (".functions", "CorrFnRealization"),
    "Constant": (".functions", "Constant"),
    "MagnitudeSpec": (".functions", "MagnitudeSpec"),
    "MaxLinearDrift": (".functions", "MaxLinearDrift"),
    "MaxSineWave": (".functions", "MaxSineWave"),
    "RandomPhaseMaxSineWave": (".functions", "RandomPhaseMaxSineWave"),
    "fBM": (".functions", "fBM"),
    "ConstantGainSpec": (".metrics", "ConstantGainSpec"),
    "ConstantGainNorms": (".metrics", "ConstantGainNorms"),
    "ConstantGainSolution": (".metrics", "ConstantGainSolution"),
    "CORRUPTION_METRIC_DEFINITIONS": (
        ".metrics",
        "CORRUPTION_METRIC_DEFINITIONS",
    ),
    "CorruptionMetricDefinition": (
        ".metrics",
        "CorruptionMetricDefinition",
    ),
    "CorruptionMetrics": (".metrics", "CorruptionMetrics"),
    "corruption_metric_definitions": (
        ".metrics",
        "corruption_metric_definitions",
    ),
    "measure_constant_gain_norms": (".metrics", "measure_constant_gain_norms"),
    "measure_corruption_metrics": (".metrics", "measure_corruption_metrics"),
    "solve_constant_gain": (".metrics", "solve_constant_gain"),
    "CORRUPTION_REPORT_SCHEMA_VERSION": (
        ".reporting",
        "CORRUPTION_REPORT_SCHEMA_VERSION",
    ),
    "ReportableCorruption": (".reporting", "ReportableCorruption"),
    "write_corruption_reports": (".reporting", "write_corruption_reports"),
    "FilterSpec": (".tables", "FilterSpec"),
    "GCOLS": (".tables", "GCOLS"),
    "GTab": (".tables", "GTab"),
    "GTabQuery": (".tables", "GTabQuery"),
    "get_unflagged_antennas": (".tables", "get_unflagged_antennas"),
    "make_corrtab_identity": (".tables", "make_corrtab_identity"),
    "make_template_gain_corrtab": (".tables", "make_template_gain_corrtab"),
    "verify_corrtab_is_identity": (".tables", "verify_corrtab_is_identity"),
    "TimeGrid": (".timegrid", "TimeGrid"),
}

__all__ = list(_PUBLIC_IMPORTS)


def __getattr__(name: str):
    target = _PUBLIC_IMPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
