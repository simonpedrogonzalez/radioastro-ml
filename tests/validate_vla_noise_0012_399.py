"""Run the fixed 0012-399 VLA thermal-noise validation and HTML report.

Run from the repository root with:

    casa --nogui --nologger -c tests/validate_vla_noise_0012_399.py
"""

from __future__ import annotations

import json
import math
import shutil
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SOURCE_MS = ROOT / "collect/extracted/0012-399/0012-399/0012-399.ms"
OUTPUT_ROOT = ROOT / "experiments/vla_noise_validation_0012_399"
REPORT_TEMPLATE = ROOT / "tests/vla_noise_validation_report.qmd"
BAND = "C"
SAMPLER = "8bit"
SEFD_JY = 236.7
ETA_C = 0.93
SEED = 20260907
IMSIZE = [256, 256]
CELL = "0.15arcsec"
ECT_REPORTED_BANDWIDTH_MHZ = 69.7525
ECT_REPORTED_RMS_MICROJY_PER_BEAM = 23.7264
ECT_APPROXIMATE_BEAM_ARCSEC = 1.318
OSS_SOURCE = (
    "https://science.nrao.edu/facilities/vla/docs/manuals/"
    "oss2026a/performance/sensitivity"
)


def _flat(value):
    return value.ravel().tolist() if hasattr(value, "ravel") else list(value)


def _unit(tb, column: str) -> str:
    units = _flat((tb.getcolkeywords(column) or {}).get("QuantumUnits", []))
    if len(units) != 1:
        raise RuntimeError(f"{column} must have exactly one unit, found {units!r}")
    return str(units[0]).lower()


def _frequency_scale(unit: str) -> float:
    try:
        return {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9}[unit]
    except KeyError as exc:
        raise RuntimeError(f"Unsupported frequency unit {unit!r}") from exc


def _time_scale(unit: str) -> float:
    try:
        return {"s": 1.0, "sec": 1.0, "ms": 1e-3, "min": 60.0}[unit]
    except KeyError as exc:
        raise RuntimeError(f"Unsupported time unit {unit!r}") from exc


def _measurement_summary(ms: Path) -> dict[str, object]:
    """Independently count the image-noise samples retained by MS flags."""
    import numpy as np
    from casatools import table

    tb = table()
    tb.open(str(ms / "ANTENNA"), nomodify=True)
    try:
        nantennas = int(tb.nrows())
    finally:
        tb.close()

    tb = table()
    tb.open(str(ms / "DATA_DESCRIPTION"), nomodify=True)
    try:
        if int(tb.nrows()) != 1:
            raise ValueError("The fixed validation requires exactly one data description")
        spw_id = int(tb.getcell("SPECTRAL_WINDOW_ID", 0))
        polarization_id = int(tb.getcell("POLARIZATION_ID", 0))
    finally:
        tb.close()

    tb = table()
    tb.open(str(ms / "POLARIZATION"), nomodify=True)
    try:
        correlation_types = [int(item) for item in _flat(tb.getcell("CORR_TYPE", polarization_id))]
    finally:
        tb.close()
    parallel_indices = [
        index for index, correlation in enumerate(correlation_types) if correlation in {5, 8}
    ]
    if len(parallel_indices) != 2:
        raise ValueError(
            "The fixed Stokes-I validation requires RR and LL; "
            f"found correlation codes {correlation_types}"
        )

    tb = table()
    tb.open(str(ms / "SPECTRAL_WINDOW"), nomodify=True)
    try:
        frequency_scale = _frequency_scale(_unit(tb, "CHAN_FREQ"))
        bandwidth_scale = _frequency_scale(_unit(tb, "EFFECTIVE_BW"))
        frequencies_hz = np.asarray(tb.getcell("CHAN_FREQ", spw_id), dtype=float) * frequency_scale
        bandwidths_hz = (
            np.asarray(tb.getcell("EFFECTIVE_BW", spw_id), dtype=float) * bandwidth_scale
        )
    finally:
        tb.close()
    if frequencies_hz.size == 0 or bandwidths_hz.size != frequencies_hz.size:
        raise RuntimeError("Invalid fixed-case spectral-window metadata")

    baselines: set[tuple[int, int]] = set()
    times: set[float] = set()
    exposures: set[float] = set()
    usable_samples = 0
    total_rows = 0
    tb = table()
    tb.open(str(ms), nomodify=True)
    try:
        columns = set(tb.colnames())
        exposure_scale = _time_scale(_unit(tb, "EXPOSURE"))
        for start in range(0, int(tb.nrows()), 2048):
            count = min(2048, int(tb.nrows()) - start)
            antenna1 = np.asarray(tb.getcol("ANTENNA1", startrow=start, nrow=count), dtype=int)
            antenna2 = np.asarray(tb.getcol("ANTENNA2", startrow=start, nrow=count), dtype=int)
            data_description = np.asarray(
                tb.getcol("DATA_DESC_ID", startrow=start, nrow=count), dtype=int
            )
            if np.any(data_description != 0):
                raise ValueError("The fixed validation requires DATA_DESC_ID=0 only")
            if np.any(antenna1 == antenna2):
                raise ValueError("The fixed validation does not support autocorrelations")
            baselines.update(
                (min(int(left), int(right)), max(int(left), int(right)))
                for left, right in zip(antenna1, antenna2)
            )
            times.update(
                float(item)
                for item in np.asarray(
                    tb.getcol("TIME", startrow=start, nrow=count), dtype=float
                )
            )
            exposures.update(
                float(item) * exposure_scale
                for item in np.asarray(
                    tb.getcol("EXPOSURE", startrow=start, nrow=count), dtype=float
                )
            )
            flags = np.asarray(tb.getcol("FLAG", startrow=start, nrow=count), dtype=bool)
            usable = ~flags[parallel_indices, :, :]
            if "FLAG_ROW" in columns:
                flag_rows = np.asarray(
                    tb.getcol("FLAG_ROW", startrow=start, nrow=count), dtype=bool
                )
                usable &= ~flag_rows.reshape(1, 1, -1)
            usable_samples += int(np.count_nonzero(usable))
            total_rows += count
    finally:
        tb.close()

    if len(exposures) != 1:
        raise ValueError(f"Expected one exposure in the fixed case, found {sorted(exposures)}")
    exposure_s = next(iter(exposures))
    nbaselines = len(baselines)
    nintegrations = len(times)
    if total_rows != nbaselines * nintegrations:
        raise ValueError("The fixed case is not a complete baseline-by-integration grid")
    nominal_samples = frequencies_hz.size * 2 * nbaselines * nintegrations
    nominal_bandwidth_hz = float(np.sum(bandwidths_hz))
    return {
        "representative_frequency_hz": float(np.mean(frequencies_hz)),
        "nantennas": nantennas,
        "nbaselines": nbaselines,
        "nintegrations": nintegrations,
        "exposure_s": exposure_s,
        "on_source_s": nintegrations * exposure_s,
        "nchannels": int(frequencies_hz.size),
        "npol_stokes_i": 2,
        "nominal_bandwidth_hz": nominal_bandwidth_hz,
        "usable_samples": usable_samples,
        "nominal_samples": nominal_samples,
        "usable_bandwidth_hz": nominal_bandwidth_hz * usable_samples / nominal_samples,
    }


def _image_values(path: Path):
    import numpy as np
    from casatools import image

    ia = image()
    ia.open(str(path))
    try:
        values = np.squeeze(np.asarray(ia.getchunk(), dtype=float))
        mask = np.squeeze(np.asarray(ia.getchunk(getmask=True), dtype=bool))
    finally:
        ia.close()
    while values.ndim > 2:
        values = values[..., 0]
        mask = mask[..., 0]
    if values.ndim != 2:
        raise RuntimeError(f"Unexpected image shape for {path}: {values.shape}")
    values[~mask] = np.nan
    return values


def _write_noise_plot(run_dir: Path, image_path: Path) -> None:
    from scripts.imaging.plot_utils import casa_image_to_png

    casa_image_to_png(
        image_path,
        run_dir / "dirty.png",
        title="0012-399 simulated VLA thermal noise: dirty image",
        draw_beam=True,
    )


def _report_text(results: dict[str, object]) -> str:
    central = results["measurements"]["central"]
    annulus = results["measurements"].get("annulus")
    predictions = results["predictions"]
    metadata = results["measurement_set"]

    def number(value, digits=10):
        return "unavailable" if value is None else f"{float(value):.{digits}g}"

    failures = results["scientific_failures"]
    replacements = {
        "__RUN_TIMESTAMP__": str(results["run_timestamp"]),
        "__SOURCE_MS__": str(results["source_ms"]),
        "__REPRESENTATIVE_FREQUENCY_GHZ__": number(metadata["representative_frequency_hz"] / 1e9),
        "__NANTENNAS__": str(metadata["nantennas"]),
        "__NBASELINES__": str(metadata["nbaselines"]),
        "__NINTEGRATIONS__": str(metadata["nintegrations"]),
        "__EXPOSURE_S__": number(metadata["exposure_s"]),
        "__ON_SOURCE_S__": number(metadata["on_source_s"]),
        "__NCHANNELS__": str(metadata["nchannels"]),
        "__NOMINAL_BANDWIDTH_MHZ__": number(metadata["nominal_bandwidth_hz"] / 1e6),
        "__USABLE_BANDWIDTH_MHZ__": number(metadata["usable_bandwidth_hz"] / 1e6),
        "__OSS_SOURCE__": OSS_SOURCE,
        "__SEFD_JY__": number(results["noise_model"]["sefd_jy"]),
        "__ECT_EXPECTED_RMS_MICROJY__": number(
            results["exposure_calculator"]["corrected_expected_rms_jy_per_beam"] * 1e6,
            6,
        ),
        "__ECT_EXPECTED_TB_K__": number(
            results["exposure_calculator"]["corrected_expected_rms_brightness_k"], 4
        ),
        "__ECT_EXPECTED_LINE_WIDTH_KMS__": number(
            results["exposure_calculator"]["corrected_expected_line_width_km_s"], 6
        ),
        "__SIMPLENOISE_JY__": number(predictions["simplenoise_jy"]),
        "__FLAGGED_RMS_JY__": number(predictions["flag_adjusted_rms_jy_per_beam"]),
        "__CENTRAL_RMS_JY__": number(central["rms_jy_per_beam"]),
        "__CENTRAL_MAD_JY__": number(central["scaled_mad_jy_per_beam"]),
        "__STATUS__": "PASS" if not failures else "FAIL",
        "__FAILURES__": (
            "All scientific acceptance checks passed."
            if not failures
            else "Scientific failures:\n\n" + "\n".join(f"- {item}" for item in failures)
        ),
    }
    text = REPORT_TEMPLATE.read_text(encoding="utf-8")
    for marker, value in replacements.items():
        text = text.replace(marker, value)
    return text


def main() -> Path:
    import numpy as np
    from casatasks import tclean

    from scripts.reporting import QuartoReporter
    from scripts.imaging import measure_pb_region
    from scripts.simulation import simulate_ms, theoretical_vla_simplenoise

    if not SOURCE_MS.is_dir():
        raise FileNotFoundError(f"Fixed validation MS is missing: {SOURCE_MS}")
    timestamp = datetime.now().astimezone()
    run_dir = OUTPUT_ROOT / timestamp.strftime("%Y%m%dT%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"Validation products: {run_dir}")

    summary = _measurement_summary(SOURCE_MS)
    sigma = theoretical_vla_simplenoise(SOURCE_MS, sefd_jy=SEFD_JY, eta_c=ETA_C)
    nominal_rms = sigma / math.sqrt(summary["nominal_samples"])
    flagged_rms = sigma / math.sqrt(summary["usable_samples"])
    corrected_ect_rms = ECT_REPORTED_RMS_MICROJY_PER_BEAM * 1e-6 * math.sqrt(
        ECT_REPORTED_BANDWIDTH_MHZ * 1e6 / summary["usable_bandwidth_hz"]
    )
    corrected_ect_line_width = (
        299_792.458
        * summary["usable_bandwidth_hz"]
        / summary["representative_frequency_hz"]
    )
    corrected_ect_brightness = (
        1.222e6
        * corrected_ect_rms
        / (
            (summary["representative_frequency_hz"] / 1e9) ** 2
            * ECT_APPROXIMATE_BEAM_ARCSEC**2
        )
    )

    print(f"Representative frequency: {summary['representative_frequency_hz'] / 1e9:.9g} GHz")
    print(f"Nominal / flag-adjusted usable bandwidth: {summary['nominal_bandwidth_hz'] / 1e6:.6g} / {summary['usable_bandwidth_hz'] / 1e6:.6g} MHz")
    print(f"Antennas / cross baselines / integrations: {summary['nantennas']} / {summary['nbaselines']} / {summary['nintegrations']}")
    print(f"Exposure / nominal time: {summary['exposure_s']:.6g} / {summary['on_source_s']:.6g} s")
    print("Imaging: Stokes I, dual polarization (RR/LL), natural weighting, no taper, niter=0")
    print(
        f"Noise assumptions: C band, 8-bit, ECT-equivalent SEFD={SEFD_JY:g} Jy, "
        f"eta_c={ETA_C:g}, seed={SEED}"
    )
    print(f"Reference: {OSS_SOURCE}")
    print(
        "ECT comparison settings: B configuration, 27 antennas, Zenith, Winter, "
        "and the calculator's frequency-dependent sensitivity at 7.291 GHz."
    )
    print(f"Calculated simplenoise: {sigma:.15g} Jy")
    print(f"Nominal / flag-adjusted image RMS: {nominal_rms:.15g} / {flagged_rms:.15g} Jy/beam")
    print(f"Corrected manual ECT target: {corrected_ect_rms * 1e6:.6g} microJy/beam")

    simulation = simulate_ms(
        SOURCE_MS,
        [],
        run_dir / "noise.ms",
        noise_model="vla-thermal",
        noise_parameters={
            "band": BAND,
            "sampler": SAMPLER,
            "sefd_jy": SEFD_JY,
            "eta_c": ETA_C,
        },
        seed=SEED,
    )
    image_base = run_dir / "noise_dirty"
    tclean(
        vis=str(simulation.ms_path),
        imagename=str(image_base),
        datacolumn="data",
        specmode="mfs",
        gridder="standard",
        stokes="I",
        deconvolver="hogbom",
        weighting="natural",
        imsize=IMSIZE,
        cell=CELL,
        niter=0,
        interactive=False,
        parallel=False,
    )
    image_path = Path(f"{image_base}.image")
    pb_path = Path(f"{image_base}.pb")
    if not image_path.is_dir() or not pb_path.is_dir():
        raise RuntimeError("tclean did not create the expected image and PB products")

    central = measure_pb_region(image_path, pb_path, pb_min=0.5)
    pb_values = _image_values(pb_path)
    has_annulus = bool(np.any(np.isfinite(pb_values) & (pb_values >= 0.2) & (pb_values <= 0.3)))
    annulus = measure_pb_region(image_path, pb_path, pb_min=0.2, pb_max=0.3) if has_annulus else None

    failures: list[str] = []

    if not math.isclose(flagged_rms, corrected_ect_rms, rel_tol=1e-4, abs_tol=0.0):
        failures.append(
            "The 236.7 Jy package prediction does not reproduce the corrected manual "
            "ECT target within 0.01%"
        )

    def check_close(
        label: str,
        actual: float,
        expected: float,
        *,
        reference_label: str = "flag-adjusted prediction",
    ) -> None:
        ratio = actual / expected
        print(f"{label}: {actual:.15g} Jy/beam; ratio to {reference_label} = {ratio:.6g}")
        if not math.isclose(actual, expected, rel_tol=0.1, abs_tol=0.0):
            failures.append(f"{label} ratio {ratio:.6g} is outside the 10% tolerance")

    if not math.isclose(float(simulation.simplenoise_jy), sigma, rel_tol=1e-12):
        failures.append("simulate_ms returned a simplenoise inconsistent with the public formula")
    check_close("Central RMS", central.rms_jy_per_beam, flagged_rms)
    check_close("Central scaled MAD", central.scaled_mad_jy_per_beam, flagged_rms)
    check_close(
        "Central RMS",
        central.rms_jy_per_beam,
        corrected_ect_rms,
        reference_label="corrected ECT target",
    )
    if not math.isclose(
        central.rms_jy_per_beam, central.scaled_mad_jy_per_beam, rel_tol=0.1, abs_tol=0.0
    ):
        failures.append("Central RMS and scaled MAD differ by more than 10%")
    if annulus is None:
        print("PB 0.2--0.3 is not covered by this image; annulus metrics are unavailable.")
    else:
        check_close("PB 0.2--0.3 RMS", annulus.rms_jy_per_beam, flagged_rms)
        check_close("PB 0.2--0.3 scaled MAD", annulus.scaled_mad_jy_per_beam, flagged_rms)
        if not math.isclose(central.rms_jy_per_beam, annulus.rms_jy_per_beam, rel_tol=0.1):
            failures.append("Central and annulus RMS differ by more than 10%")

    def metric_payload(metric):
        if metric is None:
            return None
        return {
            "rms_jy_per_beam": metric.rms_jy_per_beam,
            "scaled_mad_jy_per_beam": metric.scaled_mad_jy_per_beam,
            "n_pixels": metric.n_pixels,
            "rms_to_flag_adjusted": metric.rms_jy_per_beam / flagged_rms,
            "scaled_mad_to_flag_adjusted": metric.scaled_mad_jy_per_beam / flagged_rms,
            "rms_to_corrected_ect": metric.rms_jy_per_beam / corrected_ect_rms,
        }

    results = {
        "schema_version": 1,
        "run_timestamp": timestamp.isoformat(),
        "source_ms": str(SOURCE_MS),
        "output_directory": str(run_dir),
        "measurement_set": summary,
        "noise_model": {
            "selector": "vla-thermal",
            "band": BAND,
            "sampler": SAMPLER,
            "sefd_jy": SEFD_JY,
            "eta_c": ETA_C,
            "seed": SEED,
            "reference": OSS_SOURCE,
            "sefd_source": (
                "Effective value inferred from the manual ECT result 23.7264 microJy/beam "
                "at the mistakenly entered 69.7525 MHz; rounded to 236.7 Jy"
            ),
            "assumptions": (
                "ECT-matched test scenario using B configuration, 27 antennas, dual "
                "polarization, natural weighting, 8-bit sampling, Zenith, and Winter. "
                "These are declared validation inputs, not recovered observing conditions."
            ),
        },
        "exposure_calculator": {
            "url": "https://obs.vla.nrao.edu/ect",
            "purpose": "Noise Simulation Check",
            "array_configuration": "B",
            "number_of_antennas": 27,
            "polarization": "Dual",
            "number_of_sources": 1,
            "number_of_frequencies": 1,
            "session_includes_pointing": False,
            "image_weighting": "Natural",
            "representative_frequency_mhz": summary["representative_frequency_hz"] / 1e6,
            "receiver_band": "C",
            "approximate_beam_arcsec": ECT_APPROXIMATE_BEAM_ARCSEC,
            "sampler": "8 bit",
            "elevation": "Zenith (90 degrees)",
            "weather": "Winter",
            "calculation_type": "Noise/Tb",
            "time_on_source_s": summary["on_source_s"],
            "frequency_bandwidth_mhz": summary["usable_bandwidth_hz"] / 1e6,
            "corrected_expected_line_width_km_s": corrected_ect_line_width,
            "corrected_expected_rms_jy_per_beam": corrected_ect_rms,
            "corrected_expected_rms_brightness_k": corrected_ect_brightness,
            "source_manual_result": {
                "entered_bandwidth_mhz": ECT_REPORTED_BANDWIDTH_MHZ,
                "reported_rms_microjy_per_beam": ECT_REPORTED_RMS_MICROJY_PER_BEAM,
            },
        },
        "imaging": {
            "datacolumn": "data",
            "specmode": "mfs",
            "gridder": "standard",
            "stokes": "I",
            "deconvolver": "hogbom",
            "weighting": "natural",
            "uvtaper": None,
            "imsize": IMSIZE,
            "cell": CELL,
            "niter": 0,
        },
        "predictions": {
            "simplenoise_jy": sigma,
            "nominal_rms_jy_per_beam": nominal_rms,
            "flag_adjusted_rms_jy_per_beam": flagged_rms,
        },
        "measurements": {
            "central": metric_payload(central),
            "annulus": metric_payload(annulus),
            "annulus_unavailable_reason": None if annulus else "PB 0.2--0.3 is outside this image",
        },
        "simulation_metadata_json": str(simulation.metadata_json),
        "products": {
            "dirty_image": str(image_path),
            "pb": str(pb_path),
            "dirty_png": "dirty.png",
        },
        "scientific_failures": failures,
        "status": "pass" if not failures else "fail",
    }
    _write_noise_plot(run_dir, image_path)
    (run_dir / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    report_path = run_dir / "report.qmd"
    shutil.copy2(REPORT_TEMPLATE, report_path)
    report_path.write_text(_report_text(results), encoding="utf-8")
    reporter = QuartoReporter(report_path, every=1)
    reporter.sample_completed()
    reporter.finish()
    html_path = report_path.with_suffix(".html")
    if not html_path.is_file():
        raise RuntimeError(f"Quarto did not create the report; inspect retained products in {run_dir}")

    print("\nExposure Calculator comparison instructions:")
    print("Open https://obs.vla.nrao.edu/ect")
    print("Enter 7291 MHz first and press Tab; enter 69.5725 MHz bandwidth and press Tab.")
    print("Then use B, 27 antennas, Dual, Natural, 8 bit, Zenith, Winter, Noise/Tb,")
    print("one source, one frequency, no pointing, and 1175s time on source.")
    print(f"Expected calculator RMS: approximately {corrected_ect_rms * 1e6:.4f} microJy/beam.")
    print("Compare it with the non-PB-corrected dirty-image RMS, not the real Pipeline RMS.")
    print(f"HTML report: {html_path}")
    if failures:
        raise AssertionError("Scientific validation failed:\n" + "\n".join(failures))
    return run_dir


if __name__ == "__main__":
    main()
