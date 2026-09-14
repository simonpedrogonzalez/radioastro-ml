"""Print the MeasurementSet metadata that defines visibility sampling.

Run inside CASA from the repository root:

    casa --nogui --nologger -c scripts/inspect_visibility_metadata.py

Analyze one canonical MS from every directory under collect/extracted:

    casa --nogui --nologger -c scripts/inspect_visibility_metadata.py --all-extracted

Change MS_PATH below to inspect a different extracted MeasurementSet.
The script is read-only: it does not modify the dataset.
"""

from __future__ import annotations

import io
import os
import re
import sys
from contextlib import contextmanager, nullcontext, redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

try:
    from casatools import table
except ModuleNotFoundError as exc:
    raise SystemExit(
        "casatools is unavailable. Run this script with CASA, for example:\n"
        "  casa --nogui --nologger -c scripts/inspect_visibility_metadata.py"
    ) from exc


# Pick one extracted visibility dataset. Change only this value for another MS.
MS_PATH = Path(
    "/Users/u1528314/Documents/radioastro-ml/collect/extracted/"
    "0059+001/0059+001/0059+001.ms"
)

# Add paths here if pipeline products live somewhere other than the automatic
# locations checked beside the MS and under collect/downloads/<sample-name>.
PIPELINE_SEARCH_ROOTS = []

C_M_S = 299_792_458.0
MJD_ZERO = datetime(1858, 11, 17, tzinfo=timezone.utc)
CORRELATION_NAMES = {
    1: "I",
    2: "Q",
    3: "U",
    4: "V",
    5: "RR",
    6: "RL",
    7: "LR",
    8: "LL",
    9: "XX",
    10: "XY",
    11: "YX",
    12: "YY",
}


@contextmanager
def open_table(path: Path):
    """Open a CASA table read-only and always close it."""
    tb = table()
    tb.open(str(path), nomodify=True)
    try:
        yield tb
    finally:
        tb.close()


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def values(tb, column: str, dtype=None):
    if column not in tb.colnames():
        return None
    result = np.asarray(tb.getcol(column))
    return result.astype(dtype) if dtype is not None else result


def fmt_number(value: float, unit: str = "", digits: int = 3) -> str:
    return f"{value:,.{digits}f}{(' ' + unit) if unit else ''}"


def fmt_range(array, unit: str = "", digits: int = 3) -> str:
    array = np.asarray(array, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return "n/a"
    return (
        f"{fmt_number(float(np.min(array)), unit, digits)} .. "
        f"{fmt_number(float(np.max(array)), unit, digits)}"
    )


def mjd_seconds_to_utc(value: float) -> str:
    return (MJD_ZERO + timedelta(seconds=float(value))).isoformat().replace(
        "+00:00", "Z"
    )


def radians_to_hms(value: float) -> str:
    hours = (np.degrees(value) / 15.0) % 24.0
    h = int(hours)
    minutes = (hours - h) * 60.0
    m = int(minutes)
    s = (minutes - m) * 60.0
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def radians_to_dms(value: float) -> str:
    degrees = float(np.degrees(value))
    sign = "+" if degrees >= 0 else "-"
    degrees = abs(degrees)
    d = int(degrees)
    minutes = (degrees - d) * 60.0
    m = int(minutes)
    s = (minutes - m) * 60.0
    return f"{sign}{d:02d}:{m:02d}:{s:05.2f}"


def report_observation(ms_path: Path) -> None:
    section("1. Observation: who, where, and when")
    with open_table(ms_path / "OBSERVATION") as tb:
        telescope = values(tb, "TELESCOPE_NAME")
        observer = values(tb, "OBSERVER")
        project = values(tb, "PROJECT")
        time_ranges = values(tb, "TIME_RANGE", float)

    print(f"MeasurementSet : {ms_path}")
    print(f"Telescope      : {', '.join(map(str, telescope)) if telescope is not None else 'n/a'}")
    print(f"Project        : {', '.join(map(str, project)) if project is not None else 'n/a'}")
    print(f"Observer       : {', '.join(map(str, observer)) if observer is not None else 'n/a'}")
    if time_ranges is not None and time_ranges.size:
        print(f"UTC range      : {mjd_seconds_to_utc(np.min(time_ranges))}")
        print(f"                 {mjd_seconds_to_utc(np.max(time_ranges))}")
        print(
            "Duration       : "
            + fmt_number((np.max(time_ranges) - np.min(time_ranges)) / 3600.0, "h")
        )


def report_antennas(ms_path: Path):
    section("2. Array and baseline geometry")
    with open_table(ms_path / "ANTENNA") as tb:
        names = values(tb, "NAME")
        stations = values(tb, "STATION")
        mounts = values(tb, "MOUNT")
        diameters = values(tb, "DISH_DIAMETER", float)
        positions = values(tb, "POSITION", float)

    if names is None or positions is None:
        print("ANTENNA table lacks NAME or POSITION.")
        return np.array([]), None

    # POSITION is (x, y, z) in the table's terrestrial reference frame, in metres.
    xyz = positions.T if positions.shape[0] == 3 else positions
    centre = np.mean(xyz, axis=0)
    separations = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
    nonzero = separations[separations > 0]

    print(f"Antenna count  : {len(names)}")
    if diameters is not None:
        print(f"Dish diameters : {fmt_range(diameters, 'm', 1)}")
    if mounts is not None:
        print(f"Mount types    : {', '.join(sorted(set(map(str, mounts))))}")
    print(
        "Array centre   : ITRF XYZ = "
        + ", ".join(fmt_number(v, "m", 1) for v in centre)
    )
    if nonzero.size:
        print(f"Possible physical baseline lengths: {fmt_range(nonzero, 'm', 1)}")
    print("Antennas       :")
    for idx, name in enumerate(names):
        station = str(stations[idx]) if stations is not None else "?"
        xyz_text = ", ".join(f"{v:.1f}" for v in xyz[idx])
        print(f"  {idx:2d}  {str(name):<10} station={station:<10} XYZ_m=[{xyz_text}]")
    return names, xyz


def report_fields(ms_path: Path) -> None:
    section("3. Fields and sky directions")
    with open_table(ms_path / "FIELD") as tb:
        names = values(tb, "NAME")
        source_ids = values(tb, "SOURCE_ID", int)
        nrows = tb.nrows()
        phase_dirs = [np.asarray(tb.getcell("PHASE_DIR", row)) for row in range(nrows)]

    for field_id, direction in enumerate(phase_dirs):
        flat = direction.reshape(2, -1)
        ra, dec = float(flat[0, 0]), float(flat[1, 0])
        name = str(names[field_id]) if names is not None else "?"
        source_id = int(source_ids[field_id]) if source_ids is not None else -1
        print(
            f"Field {field_id:2d}: {name:<18} source_id={source_id:<3d} "
            f"RA={radians_to_hms(ra)} Dec={radians_to_dms(dec)} (J2000-like radians)"
        )


def report_spectral_and_polarization(ms_path: Path):
    section("4. Frequency and polarization sampling")
    spw_info = {}
    with open_table(ms_path / "SPECTRAL_WINDOW") as tb:
        names = values(tb, "NAME")
        ref_freqs = values(tb, "REF_FREQUENCY", float)
        total_bandwidths = values(tb, "TOTAL_BANDWIDTH", float)
        for spw_id in range(tb.nrows()):
            chan_freq = np.asarray(tb.getcell("CHAN_FREQ", spw_id), dtype=float).ravel()
            chan_width = np.asarray(tb.getcell("CHAN_WIDTH", spw_id), dtype=float).ravel()
            spw_info[spw_id] = {
                "chan_freq": chan_freq,
                "ref_freq": float(ref_freqs[spw_id]),
            }
            name = str(names[spw_id]) if names is not None else "?"
            bandwidth = (
                float(total_bandwidths[spw_id])
                if total_bandwidths is not None
                else float(np.sum(np.abs(chan_width)))
            )
            print(
                f"SPW {spw_id:2d}: {name}\n"
                f"  channels={chan_freq.size}, frequency={fmt_range(chan_freq / 1e9, 'GHz', 6)}, "
                f"channel_width={fmt_range(np.abs(chan_width) / 1e3, 'kHz', 3)}, "
                f"total_bandwidth={bandwidth / 1e6:.3f} MHz"
            )

    with open_table(ms_path / "POLARIZATION") as tb:
        corr_products = []
        for pol_id in range(tb.nrows()):
            corr_type = np.asarray(tb.getcell("CORR_TYPE", pol_id), dtype=int).ravel()
            products = [CORRELATION_NAMES.get(int(code), str(int(code))) for code in corr_type]
            corr_products.append(products)
            print(f"Polarization setup {pol_id}: {', '.join(products)}")

    with open_table(ms_path / "DATA_DESCRIPTION") as tb:
        dd_spw = values(tb, "SPECTRAL_WINDOW_ID", int)
        dd_pol = values(tb, "POLARIZATION_ID", int)
    if dd_spw is not None and dd_pol is not None:
        print("DATA_DESC_ID mapping (used by each MAIN-table row):")
        for ddid, (spw_id, pol_id) in enumerate(zip(dd_spw, dd_pol)):
            print(f"  DDID {ddid}: SPW {int(spw_id)}, polarization setup {int(pol_id)}")
    return spw_info, dd_spw


def report_sampling(ms_path: Path, antenna_names, spw_info, dd_spw) -> None:
    section("5. Visibility rows: time, antenna pair, UVW, scan, and data shape")
    with open_table(ms_path) as tb:
        nrows = tb.nrows()
        columns = set(tb.colnames())
        times = values(tb, "TIME", float)
        intervals = values(tb, "INTERVAL", float)
        exposures = values(tb, "EXPOSURE", float)
        ant1 = values(tb, "ANTENNA1", int)
        ant2 = values(tb, "ANTENNA2", int)
        uvw = values(tb, "UVW", float)
        scans = values(tb, "SCAN_NUMBER", int)
        fields = values(tb, "FIELD_ID", int)
        ddids = values(tb, "DATA_DESC_ID", int)
        observations = values(tb, "OBSERVATION_ID", int)
        arrays = values(tb, "ARRAY_ID", int)
        flag_rows = values(tb, "FLAG_ROW", bool)
        data_column = next(
            (name for name in ("DATA", "CORRECTED_DATA", "MODEL_DATA") if name in columns),
            None,
        )
        first_data_shape = (
            np.asarray(tb.getcell(data_column, 0)).shape
            if data_column is not None and nrows > 0
            else None
        )

    print(f"MAIN rows     : {nrows:,}")
    print(f"Data columns  : {', '.join(c for c in ('DATA', 'CORRECTED_DATA', 'MODEL_DATA') if c in columns)}")
    if first_data_shape is not None:
        print(
            f"One {data_column} cell: shape={first_data_shape} = "
            "(polarization correlations, frequency channels)"
        )

    if times is not None and times.size:
        unique_times = np.unique(times)
        print(f"Time samples  : {unique_times.size:,}")
        print(f"UTC first/last: {mjd_seconds_to_utc(unique_times[0])}")
        print(f"                {mjd_seconds_to_utc(unique_times[-1])}")
        if unique_times.size > 1:
            print(f"Timestamp step: {fmt_range(np.diff(unique_times), 's', 3)}")
    if intervals is not None:
        print(f"Integration   : {fmt_range(intervals, 's', 3)}")
    if exposures is not None:
        print(f"Exposure      : {fmt_range(exposures, 's', 3)}")

    if ant1 is not None and ant2 is not None:
        pairs = np.column_stack((ant1, ant2))
        unique_pairs = np.unique(pairs, axis=0)
        cross_pairs = unique_pairs[unique_pairs[:, 0] != unique_pairs[:, 1]]
        autocorr_rows = int(np.count_nonzero(ant1 == ant2))
        print(f"Observed antenna pairs: {len(unique_pairs)} ({len(cross_pairs)} cross-correlations)")
        print(f"Autocorrelation rows  : {autocorr_rows:,}")
        if len(unique_pairs) <= 40 and len(antenna_names):
            pair_names = [
                f"{antenna_names[a]}-{antenna_names[b]}" for a, b in unique_pairs
            ]
            print("Baseline pairs: " + ", ".join(pair_names))

    if uvw is not None and uvw.size:
        uvw_rows = uvw.T if uvw.shape[0] == 3 else uvw
        uv_radius_m = np.hypot(uvw_rows[:, 0], uvw_rows[:, 1])
        projected_m = np.linalg.norm(uvw_rows, axis=1)
        print(f"u coordinate  : {fmt_range(uvw_rows[:, 0], 'm', 1)}")
        print(f"v coordinate  : {fmt_range(uvw_rows[:, 1], 'm', 1)}")
        print(f"w coordinate  : {fmt_range(uvw_rows[:, 2], 'm', 1)}")
        print(f"sqrt(u^2+v^2) : {fmt_range(uv_radius_m, 'm', 1)}")
        print(f"UVW length    : {fmt_range(projected_m, 'm', 1)}")

        if ddids is not None and dd_spw is not None:
            wavelength_samples = []
            for ddid in np.unique(ddids):
                spw_id = int(dd_spw[int(ddid)])
                ref_freq = spw_info[spw_id]["ref_freq"]
                wavelength_samples.append(uv_radius_m[ddids == ddid] * ref_freq / C_M_S)
            if wavelength_samples:
                uv_lambda = np.concatenate(wavelength_samples)
                print(f"UV distance   : {fmt_range(uv_lambda / 1e3, 'kλ', 3)} at SPW reference frequencies")

    for label, array in (
        ("Scan numbers", scans),
        ("Field IDs", fields),
        ("Data-desc IDs", ddids),
        ("Observation IDs", observations),
        ("Array IDs", arrays),
    ):
        if array is not None:
            print(f"{label:<15}: {', '.join(map(str, np.unique(array)))}")
    if flag_rows is not None:
        print(
            f"FLAG_ROW      : {np.count_nonzero(flag_rows):,}/{flag_rows.size:,} "
            "rows fully flagged"
        )


def record_has_content(value) -> bool:
    """Best-effort test for a CASA record or array-valued table cell."""
    if value is None:
        return False
    if isinstance(value, dict):
        return bool(value)
    try:
        return np.asarray(value).size > 0
    except Exception:
        return True


def report_model_data(ms_path: Path) -> None:
    section("6. Model visibility availability")

    physical_present = False
    physical_defined = False
    physical_shape = None
    sampled_magnitudes = []
    main_model_keywords = []

    with open_table(ms_path) as tb:
        physical_present = "MODEL_DATA" in tb.colnames()
        main_model_keywords = [
            name
            for name in tb.keywordnames()
            if name.lower().startswith("model_") or "definedmodel" in name.lower()
        ]

        if physical_present and tb.nrows() > 0:
            # Sample across the table instead of loading a potentially huge column.
            sample_rows = np.unique(
                np.linspace(0, tb.nrows() - 1, min(32, tb.nrows()), dtype=int)
            )
            for row in sample_rows:
                try:
                    is_defined = tb.iscelldefined("MODEL_DATA", int(row))
                except Exception:
                    is_defined = True
                if not is_defined:
                    continue
                physical_defined = True
                cell = np.asarray(tb.getcell("MODEL_DATA", int(row)))
                if physical_shape is None:
                    physical_shape = cell.shape
                finite = np.abs(cell[np.isfinite(cell)])
                if finite.size:
                    sampled_magnitudes.append(finite)

    source_model_rows = []
    source_model_keywords = []
    source_path = ms_path / "SOURCE"
    if source_path.is_dir():
        with open_table(source_path) as tb:
            source_model_keywords = [
                name
                for name in tb.keywordnames()
                if name.lower().startswith("model_") or "definedmodel" in name.lower()
            ]
            if "SOURCE_MODEL" in tb.colnames():
                for row in range(tb.nrows()):
                    try:
                        is_defined = tb.iscelldefined("SOURCE_MODEL", row)
                    except Exception:
                        is_defined = True
                    if not is_defined:
                        continue
                    try:
                        has_content = record_has_content(tb.getcell("SOURCE_MODEL", row))
                    except Exception:
                        has_content = False
                    if has_content:
                        source_model_rows.append(row)

    virtual_present = bool(
        main_model_keywords or source_model_keywords or source_model_rows
    )

    if not physical_present:
        print("MODEL_DATA column : NO")
    elif not physical_defined:
        print("MODEL_DATA column : present, but sampled cells are undefined")
    else:
        print(f"MODEL_DATA column : YES; sampled cell shape={physical_shape}")
        if sampled_magnitudes:
            magnitudes = np.concatenate(sampled_magnitudes)
            nonzero = int(np.count_nonzero(magnitudes))
            print(
                "Sampled |model|  : "
                f"{fmt_range(magnitudes, '', 6)}; "
                f"{nonzero:,}/{magnitudes.size:,} sampled values nonzero"
            )

    print(f"Virtual CASA model: {'YES' if virtual_present else 'NO'}")
    if source_model_rows:
        print("  SOURCE_MODEL rows with content: " + ", ".join(map(str, source_model_rows)))
    if main_model_keywords:
        print("  MAIN model keywords: " + ", ".join(main_model_keywords))
    if source_model_keywords:
        print("  SOURCE model keywords: " + ", ".join(source_model_keywords))

    if virtual_present and physical_defined:
        print("Effective model    : virtual model (CASA gives it precedence)")
    elif virtual_present:
        print("Effective model    : virtual/on-the-fly model")
    elif physical_defined:
        print("Effective model    : MODEL_DATA scratch column")
    else:
        print("Effective model    : NONE stored in this MeasurementSet")


def compact_text_line(line: str, limit: int = 240) -> str:
    line = re.sub(r"<[^>]+>", " ", line)
    line = " ".join(line.split())
    return line if len(line) <= limit else line[: limit - 3] + "..."


def pipeline_search_roots(ms_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    sample_root = ms_path.parents[1]
    sample_name = sample_root.name
    candidates = [
        sample_root,
        repo_root / "collect" / "downloads" / sample_name,
        *[Path(path).expanduser() for path in PIPELINE_SEARCH_ROOTS],
    ]
    roots = []
    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_dir() and candidate not in seen:
            roots.append(candidate)
            seen.add(candidate)
    return roots, sample_name, repo_root / "collect" / "downloads" / sample_name


def candidate_pipeline_files(root: Path):
    """Yield small text artifacts while pruning CASA data/table directories."""
    skip_suffixes = (
        ".ms",
        ".image",
        ".model",
        ".mask",
        ".pb",
        ".psf",
        ".residual",
        ".sumwt",
        ".tbl",
        ".g",
        ".k",
        ".b",
    )
    allowed_suffixes = {".log", ".txt", ".html", ".htm", ".xml", ".json", ".csv", ".py"}
    important_names = {"casa_commands.log", "casa_pipescript.py", "flux.csv"}

    for directory, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            name
            for name in dirnames
            if not name.lower().endswith(skip_suffixes)
            and name not in {".git", ".venv", "__pycache__"}
        ]
        for name in filenames:
            path = Path(directory) / name
            lower_name = name.lower()
            if lower_name not in important_names and path.suffix.lower() not in allowed_suffixes:
                continue
            try:
                if path.stat().st_size > 25 * 1024 * 1024:
                    continue
            except OSError:
                continue
            yield path


def find_history_fluxboot_evidence(ms_path: Path):
    history_path = ms_path / "HISTORY"
    if not history_path.is_dir():
        return []

    pattern = re.compile(r"hifv_fluxboot|fluxboot|flux\s*density\s*bootstrap|fluxscale", re.I)
    findings = []
    with open_table(history_path) as tb:
        for column in ("APPLICATION", "ORIGIN", "MESSAGE", "CLI_COMMAND"):
            if column not in tb.colnames():
                continue
            try:
                column_values = np.asarray(tb.getcol(column), dtype=object).ravel()
            except Exception:
                continue
            for row, value in enumerate(column_values):
                if isinstance(value, np.ndarray):
                    text = " ".join(map(str, value.ravel()))
                else:
                    text = str(value)
                if pattern.search(text):
                    findings.append((column, row, compact_text_line(text)))
                    if len(findings) >= 12:
                        return findings
    return findings


def find_file_fluxboot_evidence(roots):
    generic = re.compile(r"hifv_fluxboot|fluxboot|flux\s*density\s*bootstrap|fluxscale\s*\(", re.I)
    result_pattern = re.compile(
        r"fitted spectral index|fluxdensity\s*=|flux density bootstrapping finished|"
        r"fluxboot summary|writing solutions to table|spix\s*=",
        re.I,
    )
    artifacts = []
    for root in roots:
        for path in candidate_pipeline_files(root):
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            lines = text.splitlines()
            if not any(generic.search(line) for line in lines):
                continue
            result_lines = [compact_text_line(line) for line in lines if result_pattern.search(line)]
            if not result_lines:
                result_lines = [compact_text_line(line) for line in lines if generic.search(line)][:3]
            artifacts.append((root, path, result_lines[:12]))
            if len(artifacts) >= 20:
                return artifacts
    return artifacts


def quantitative_flux_lines(file_findings):
    """Return deduplicated lines containing fitted or assigned flux values."""
    quantitative = re.compile(
        r"fitted spectral index|fluxdensity\s*=\s*\[(?!\s*-1(?:\.0*)?\s*[,\]])|"
        r"spix\s*=\s*[-+]?\d",
        re.I,
    )
    results = []
    seen = set()
    for _, _, lines in file_findings:
        for line in lines:
            if quantitative.search(line) and line not in seen:
                results.append(line)
                seen.add(line)
    return results


def report_fluxboot_provenance(ms_path: Path):
    """Print provenance and return quantitative flux information or False."""
    section("7. Pipeline hifv_fluxboot provenance")
    roots, sample_name, expected_download_root = pipeline_search_roots(ms_path)
    print(f"Sample identity : {sample_name}")
    print("Search roots    :")
    for root in roots:
        print(f"  {root}")
    if not expected_download_root.is_dir():
        print(f"  MISSING expected original products: {expected_download_root}")

    history_findings = find_history_fluxboot_evidence(ms_path)
    file_findings = find_file_fluxboot_evidence(roots)
    flux_results = quantitative_flux_lines(file_findings)

    if history_findings:
        print("MS HISTORY evidence:")
        for column, row, text in history_findings:
            print(f"  row={row} column={column}: {text}")
    else:
        print("MS HISTORY evidence: none")

    if file_findings:
        print("Pipeline/weblog artifacts:")
        for root, path, lines in file_findings:
            try:
                display_path = path.relative_to(root)
            except ValueError:
                display_path = path
            print(f"  FILE {display_path}")
            for line in lines:
                print(f"    {line}")
    else:
        print("Pipeline/weblog artifacts: none found")

    if flux_results:
        print("Quantitative flux information:")
        for line in flux_results:
            print(f"  {line}")

    if not history_findings and not file_findings:
        print(
            "Conclusion       : no retained hifv_fluxboot evidence was found. "
            "This does NOT prove the stage was not run; extracted MS products often "
            "omit the original pipeline weblog and logs."
        )
    else:
        print("Conclusion       : retained hifv_fluxboot/fluxscale evidence was found above.")

    if not flux_results:
        return False
    return {
        "sample": sample_name,
        "measurement_set": str(ms_path),
        "results": flux_results,
        "history_evidence": [
            {"column": column, "row": row, "text": text}
            for column, row, text in history_findings
        ],
        "artifact_paths": [str(path) for _, path, _ in file_findings],
    }


def report_structure(ms_path: Path) -> None:
    section("8. MeasurementSet structure")
    with open_table(ms_path) as tb:
        print("MAIN columns (one row is one baseline/time/DDID sample):")
        print("  " + ", ".join(tb.colnames()))
        subtable_names = sorted(
            name for name in tb.keywordnames() if (ms_path / name).is_dir()
        )
    print("Subtables:")
    print("  " + ", ".join(subtable_names))
    print(
        "\nConceptually, one visibility is selected by:\n"
        "  TIME + INTERVAL/EXPOSURE + ANTENNA1/ANTENNA2 + UVW + FIELD_ID +\n"
        "  DATA_DESC_ID -> (spectral window/channels + polarization products),\n"
        "with SCAN/STATE/OBSERVATION/ARRAY IDs adding observing context. The DATA\n"
        "cell then holds one complex value per polarization product and channel."
    )


def analyze_metadata(ms_path=MS_PATH, logging: bool = True):
    """Analyze one MS and return its flux information, or False if unavailable.

    Set logging=False to suppress the detailed per-MS console report.
    """
    ms_path = Path(ms_path).expanduser().resolve()
    if not ms_path.is_dir():
        raise FileNotFoundError(f"MeasurementSet does not exist: {ms_path}")
    if not (ms_path / "ANTENNA").is_dir():
        raise ValueError(f"Path does not look like a MeasurementSet: {ms_path}")

    output_context = nullcontext() if logging else redirect_stdout(io.StringIO())
    with output_context:
        report_observation(ms_path)
        antenna_names, _ = report_antennas(ms_path)
        report_fields(ms_path)
        spw_info, dd_spw = report_spectral_and_polarization(ms_path)
        report_sampling(ms_path, antenna_names, spw_info, dd_spw)
        report_model_data(ms_path)
        flux_information = report_fluxboot_provenance(ms_path)
        report_structure(ms_path)
    return flux_information


def find_extracted_ms_paths(extracted_root: Path):
    """Choose one primary MeasurementSet from each extracted sample directory."""
    paths = []
    for sample_dir in sorted(extracted_root.iterdir()):
        if not sample_dir.is_dir():
            continue
        sample = sample_dir.name
        expected = sample_dir / sample / f"{sample}.ms"
        if expected.is_dir():
            paths.append(expected)
            continue

        candidates = [
            path
            for path in sample_dir.rglob("*.ms")
            if path.is_dir()
            and "selfcal" not in path.parts
            and "pipeline" not in "/".join(path.parts).lower()
            and not path.name.endswith("_imgprep.ms")
        ]
        if candidates:
            paths.append(min(candidates, key=lambda path: (len(path.parts), str(path))))
    return paths


def print_progress(completed: int, total: int, label: str) -> None:
    width = 32
    filled = width if total == 0 else int(width * completed / total)
    bar = "#" * filled + "-" * (width - filled)
    short_label = label if len(label) <= 28 else label[:25] + "..."
    print(
        f"\r[{bar}] {completed:>{len(str(total))}}/{total} {short_label:<28}",
        end="\n" if completed == total else "",
        flush=True,
    )


def analyze_all_extracted_sets(
    extracted_root=None,
    logging: bool = False,
):
    """Analyze every extracted sample and summarize fluxboot availability."""
    if extracted_root is None:
        extracted_root = Path(__file__).resolve().parents[1] / "collect" / "extracted"
    extracted_root = Path(extracted_root).expanduser().resolve()
    if not extracted_root.is_dir():
        raise FileNotFoundError(f"Extracted dataset directory does not exist: {extracted_root}")

    ms_paths = find_extracted_ms_paths(extracted_root)
    with_flux = []
    without_flux = []
    failures = []

    print(f"Analyzing {len(ms_paths)} extracted MeasurementSets under {extracted_root}")
    print_progress(0, len(ms_paths), "starting")
    for index, ms_path in enumerate(ms_paths, start=1):
        try:
            flux_information = analyze_metadata(ms_path, logging=logging)
            if flux_information is False:
                without_flux.append(str(ms_path))
            else:
                with_flux.append(flux_information)
        except Exception as exc:
            failures.append({"measurement_set": str(ms_path), "error": str(exc)})
        print_progress(index, len(ms_paths), ms_path.parents[1].name)

    print("\nFlux information summary")
    print(f"  MeasurementSets analyzed : {len(ms_paths)}")
    print(f"  With flux information    : {len(with_flux)}")
    print(f"  Without flux information : {len(without_flux)}")
    print(f"  Analysis failures        : {len(failures)}")
    if failures:
        print("  Failed MeasurementSets:")
        for failure in failures:
            print(f"    {failure['measurement_set']}: {failure['error']}")

    return {
        "total": len(ms_paths),
        "with_flux": with_flux,
        "without_flux": without_flux,
        "failures": failures,
    }


def main() -> None:
    if "--all-extracted" in sys.argv:
        analyze_all_extracted_sets(logging=False)
    else:
        analyze_metadata(MS_PATH, logging=True)


if __name__ == "__main__":
    main()
