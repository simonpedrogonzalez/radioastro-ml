from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import log1p, sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from casatools import table


DEFAULT_CONFIG = {
    "uvdist_nbins": 24,
    "channel_bin_size": 8,
    "time_nbins": 64,
    "min_bin_count": 50,
    "mad_floor": 1e-6,
    "strong_z": 8.0,
    "moderate_z": 5.0,
    "min_bad_samples": 20,
    "min_group_bad_fraction": 0.05,
    "min_enrichment": 5.0,
    "min_coverage": 0.25,
    "max_data_loss_for_recommendation": 0.25,
    "plot_prefix": "clean_corrected",
    "max_top_candidates": 20,
    "max_plot_points": 12000,
    "rows_per_chunk": 4096,
    "chunk_target_mb": 96,
    "normalization_sample_limit": 1024,
    "metric_sample_limit": 50000,
    "candidate_metric_sample_limit": 12000,
    "source_structure_uv_corr_threshold": 0.75,
    "source_structure_max_group_coverage": 0.2,
    "adjacent_bin_relative_threshold": 0.5,
}

MJD_EPOCH = datetime(1858, 11, 17, tzinfo=timezone.utc)
LIGHT_SPEED_M_S = 299792458.0


@dataclass(frozen=True)
class Candidate:
    type: str
    key: object
    id: str
    n_total: int
    n_bad: int
    bad_fraction: float
    coverage: float
    enrichment: float
    data_loss: float
    score: float


def _merged_config(config: dict | None) -> dict:
    merged = dict(DEFAULT_CONFIG)
    if config:
        merged.update(config)
    return merged


def _finite_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _plain_number(value: object) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, (bool, int, str)):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    if number.is_integer():
        return int(number)
    return number


def _plain_dict(dct: dict) -> dict:
    return {str(k): _plain_value(v) for k, v in dct.items()}


def _plain_value(value: object) -> object:
    if isinstance(value, dict):
        return _plain_dict(value)
    if isinstance(value, (list, tuple)):
        return [_plain_value(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return _plain_number(value) if not isinstance(value, str) else value


def _target_label(ms_path: Path, output_dir: Path) -> str:
    ms_name = ms_path.stem.replace(".ms", "")
    return output_dir.name or ms_name or str(ms_path)


def _object_list(items) -> list[object]:
    return list(items)


def _safe_ratio(num: float, den: float, default: float = 0.0) -> float:
    if den == 0 or not np.isfinite(den):
        return default
    out = num / den
    return float(out) if np.isfinite(out) else default


def _robust_loc_scale(values: np.ndarray, mad_floor: float) -> tuple[float, float, int]:
    if values.size == 0:
        return 0.0, mad_floor, 0
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    sigma = max(1.4826 * mad, mad_floor)
    return median, sigma, int(values.size)


def _parse_spw_selection(spw: str) -> set[int] | None:
    text = str(spw).strip()
    if not text:
        return None

    selected: set[int] = set()
    for part in text.split(","):
        token = part.strip()
        if not token:
            continue
        token = token.split(":")[0].strip()
        if "~" in token:
            lo_str, hi_str = token.split("~", 1)
            try:
                lo = int(lo_str)
                hi = int(hi_str)
            except ValueError:
                continue
            for val in range(min(lo, hi), max(lo, hi) + 1):
                selected.add(val)
            continue
        try:
            selected.add(int(token))
        except ValueError:
            continue
    return selected or None


def _parse_uvrange_limits(uvrange: str) -> tuple[float | None, float | None]:
    text = str(uvrange).strip().lower().replace(" ", "")
    if not text:
        return None, None

    def parse_klambda(value: str) -> float | None:
        if value.endswith("klambda"):
            value = value[:-7]
        try:
            return float(value)
        except ValueError:
            return None

    if text.startswith(">"):
        return parse_klambda(text[1:]), None
    if text.startswith("<"):
        return None, parse_klambda(text[1:])
    if "~" in text:
        lo_str, hi_str = text.split("~", 1)
        return parse_klambda(lo_str), parse_klambda(hi_str)
    return None, None


def _select_data_column(colnames: list[str], preference: str) -> str:
    pref = str(preference).strip().lower()
    cols = set(colnames)
    if pref == "corrected" and "CORRECTED_DATA" in cols:
        return "CORRECTED_DATA"
    if "DATA" in cols:
        return "DATA"
    if "CORRECTED_DATA" in cols:
        return "CORRECTED_DATA"
    raise RuntimeError("MS has neither DATA nor CORRECTED_DATA")


def _read_antenna_names(ms_path: Path) -> np.ndarray:
    tb = table()
    tb.open(str(ms_path / "ANTENNA"))
    try:
        names = np.asarray(tb.getcol("NAME"))
    finally:
        tb.close()
    return names.astype(str)


def _read_ddid_to_spw(ms_path: Path) -> np.ndarray:
    tb = table()
    tb.open(str(ms_path / "DATA_DESCRIPTION"))
    try:
        out = np.asarray(tb.getcol("SPECTRAL_WINDOW_ID"), dtype=int)
    finally:
        tb.close()
    return out


def _read_spw_freq_lookup(ms_path: Path) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    tb = table()
    tb.open(str(ms_path / "SPECTRAL_WINDOW"))
    try:
        nspw = tb.nrows()
        freqs_by_spw: dict[int, np.ndarray] = {}
        mean_freq_hz = np.full(nspw, np.nan, dtype=float)
        for spw_id in range(nspw):
            freqs = np.asarray(tb.getcell("CHAN_FREQ", spw_id), dtype=float)
            freqs_by_spw[spw_id] = freqs
            if freqs.size:
                mean_freq_hz[spw_id] = float(np.nanmean(freqs))
    finally:
        tb.close()
    return mean_freq_hz, freqs_by_spw


def _subsample_indices(size: int, limit: int | None) -> np.ndarray:
    if limit is None or size <= limit or limit <= 0:
        return np.arange(size, dtype=int)
    return np.linspace(0, size - 1, num=limit, dtype=int)


def _digitize_equal_width(values: np.ndarray, nbins: int) -> tuple[np.ndarray, np.ndarray]:
    if values.size == 0:
        return np.zeros(0, dtype=int), np.array([0.0, 1.0], dtype=float)
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        edges = np.array([lo if np.isfinite(lo) else 0.0, (lo if np.isfinite(lo) else 0.0) + 1.0], dtype=float)
        return np.zeros(values.size, dtype=int), edges
    edges = np.linspace(lo, hi, nbins + 1, dtype=float)
    bins = np.digitize(values, edges[1:-1], right=False).astype(int)
    bins = np.clip(bins, 0, nbins - 1)
    return bins, edges


def _time_bin_label(time_edges: np.ndarray, bin_idx: int) -> str:
    start = time_edges[bin_idx]
    end = time_edges[min(bin_idx + 1, len(time_edges) - 1)]
    return f"bin {bin_idx} ({(end - start) / 60.0:.1f} min span)"


def _timerange_string(start_sec: float, end_sec: float) -> str:
    dt0 = MJD_EPOCH + timedelta(seconds=float(start_sec))
    dt1 = MJD_EPOCH + timedelta(seconds=float(end_sec))
    return f"{dt0.isoformat()} ~ {dt1.isoformat()}"


def _channel_range_label(start_chan: int, end_chan: int) -> str:
    return f"chan {start_chan}" if start_chan == end_chan else f"chan {start_chan}-{end_chan}"


def _candidate_id(
    candidate_type: str,
    key: object,
    *,
    antenna_names: np.ndarray,
    time_edges: np.ndarray,
    channel_bin_size: int,
) -> str:
    if candidate_type == "antenna":
        return str(key)
    if candidate_type == "baseline":
        a1, a2 = key
        return f"{a1}&{a2}"
    if candidate_type == "time":
        return _time_bin_label(time_edges, int(key))
    if candidate_type == "channel":
        start = int(key) * channel_bin_size
        end = start + channel_bin_size - 1
        return _channel_range_label(start, end)
    if candidate_type == "spw":
        return f"SPW {int(key)}"
    if candidate_type == "antenna_time":
        ant, time_bin = key
        return f"{ant} @ {_time_bin_label(time_edges, int(time_bin))}"
    if candidate_type == "baseline_time":
        baseline, time_bin = key
        return f"{baseline[0]}&{baseline[1]} @ {_time_bin_label(time_edges, int(time_bin))}"
    if candidate_type == "antenna_channel":
        ant, chan_bin = key
        start = int(chan_bin) * channel_bin_size
        end = start + channel_bin_size - 1
        return f"{ant} @ {_channel_range_label(start, end)}"
    if candidate_type == "baseline_channel":
        baseline, chan_bin = key
        start = int(chan_bin) * channel_bin_size
        end = start + channel_bin_size - 1
        return f"{baseline[0]}&{baseline[1]} @ {_channel_range_label(start, end)}"
    if candidate_type == "spw_channel":
        spw_id, chan_bin = key
        start = int(chan_bin) * channel_bin_size
        end = start + channel_bin_size - 1
        return f"SPW {int(spw_id)} {_channel_range_label(start, end)}"
    return str(key)


def _expand_group_stats(
    candidates: list[Candidate],
    indices_by_type: dict[str, dict[object, np.ndarray]],
    values_by_type: dict[str, np.ndarray],
) -> list[dict]:
    out: list[dict] = []
    for candidate in candidates:
        indices = indices_by_type.get(candidate.type, {}).get(candidate.key, np.zeros(0, dtype=int))
        values = values_by_type.get(candidate.type, np.zeros(0, dtype=float))
        group_abs_z = values[indices] if indices.size else np.zeros(0, dtype=float)
        if group_abs_z.size:
            median_abs_z = float(np.median(group_abs_z))
            p95_abs_z = float(np.percentile(group_abs_z, 95))
            max_abs_z = float(np.max(group_abs_z))
        else:
            median_abs_z = 0.0
            p95_abs_z = 0.0
            max_abs_z = 0.0
        out.append(
            {
                "type": candidate.type,
                "id": candidate.id,
                "n_total": candidate.n_total,
                "n_bad": candidate.n_bad,
                "bad_fraction": candidate.bad_fraction,
                "coverage": candidate.coverage,
                "enrichment": candidate.enrichment,
                "data_loss": candidate.data_loss,
                "median_abs_z": median_abs_z,
                "p95_abs_z": p95_abs_z,
                "max_abs_z": max_abs_z,
                "score": candidate.score,
            }
        )
    return out


def _build_group_index_map(keys) -> dict[object, np.ndarray]:
    buckets: dict[object, list[int]] = {}
    for idx, key in enumerate(keys):
        buckets.setdefault(key, []).append(idx)
    return {key: np.asarray(indices, dtype=int) for key, indices in buckets.items()}


def _build_candidates(
    candidate_type: str,
    keys,
    strong_bad: np.ndarray,
    total_used: int,
    total_bad: int,
    *,
    channel_bin_size: int,
    antenna_names: np.ndarray,
    time_edges: np.ndarray,
) -> tuple[list[Candidate], dict[object, np.ndarray]]:
    group_indices = _build_group_index_map(keys)
    global_bad_fraction = _safe_ratio(total_bad, total_used, default=0.0)
    eps = 1e-12
    candidates: list[Candidate] = []
    for key, indices in group_indices.items():
        n_total = int(indices.size)
        n_bad = int(np.count_nonzero(strong_bad[indices]))
        bad_fraction = _safe_ratio(n_bad, n_total, default=0.0)
        coverage = _safe_ratio(n_bad, total_bad, default=0.0)
        enrichment = bad_fraction / max(global_bad_fraction, eps)
        data_loss = _safe_ratio(n_total, total_used, default=0.0)
        score = coverage * log1p(max(enrichment, 0.0)) * bad_fraction / sqrt(max(data_loss, eps))
        candidates.append(
            Candidate(
                type=candidate_type,
                key=key,
                id=_candidate_id(
                    candidate_type,
                    key,
                    antenna_names=antenna_names,
                    time_edges=time_edges,
                    channel_bin_size=channel_bin_size,
                ),
                n_total=n_total,
                n_bad=n_bad,
                bad_fraction=bad_fraction,
                coverage=coverage,
                enrichment=enrichment,
                data_loss=data_loss,
                score=float(score),
            )
        )
    candidates.sort(key=lambda item: (-item.score, -item.coverage, -item.bad_fraction, item.id))
    return candidates, group_indices


def _adjacent_bins_from_candidate(
    best_bin: int,
    candidates: list[Candidate],
    *,
    relative_threshold: float,
    key_fn=None,
) -> list[int]:
    if not candidates:
        return [best_bin]
    if key_fn is None:
        key_fn = lambda candidate: candidate.key
    by_bin = {}
    for candidate in candidates:
        try:
            key_value = int(key_fn(candidate))
        except (TypeError, ValueError):
            continue
        by_bin[key_value] = candidate
    best = by_bin.get(int(best_bin))
    if best is None:
        return [best_bin]
    threshold = relative_threshold * best.bad_fraction
    selected = {int(best_bin)}

    cursor = int(best_bin) - 1
    while cursor in by_bin and by_bin[cursor].n_bad > 0 and by_bin[cursor].bad_fraction >= threshold:
        selected.add(cursor)
        cursor -= 1

    cursor = int(best_bin) + 1
    while cursor in by_bin and by_bin[cursor].n_bad > 0 and by_bin[cursor].bad_fraction >= threshold:
        selected.add(cursor)
        cursor += 1

    return sorted(selected)


def _contiguous_timerange(time_edges: np.ndarray, bins: list[int]) -> str | None:
    if not bins:
        return None
    start = time_edges[min(bins)]
    end = time_edges[min(max(bins) + 1, len(time_edges) - 1)]
    return _timerange_string(start, end)


def _channel_selection(spw_id: int | None, bins: list[int], channel_bin_size: int) -> str | None:
    if not bins:
        return None
    chan_start = min(bins) * channel_bin_size
    chan_end = (max(bins) + 1) * channel_bin_size - 1
    prefix = "" if spw_id is None else f"{int(spw_id)}:"
    if chan_start == chan_end:
        return f"{prefix}{chan_start}"
    return f"{prefix}{chan_start}~{chan_end}"


def _qualifies(candidate: Candidate, cfg: dict) -> bool:
    return (
        candidate.n_bad >= int(cfg["min_bad_samples"])
        and candidate.coverage >= float(cfg["min_coverage"])
        and candidate.enrichment >= float(cfg["min_enrichment"])
        and candidate.bad_fraction >= float(cfg["min_group_bad_fraction"])
        and candidate.data_loss <= float(cfg["max_data_loss_for_recommendation"])
    )


def _touching_vs_non_touching_enrichment(
    antenna_name: str,
    ant1_name: np.ndarray,
    ant2_name: np.ndarray,
    strong_bad: np.ndarray,
) -> tuple[float, float, float]:
    touching = (ant1_name == antenna_name) | (ant2_name == antenna_name)
    non_touching = ~touching
    touching_bad_fraction = _safe_ratio(np.count_nonzero(strong_bad[touching]), np.count_nonzero(touching), default=0.0)
    non_touching_bad_fraction = _safe_ratio(np.count_nonzero(strong_bad[non_touching]), np.count_nonzero(non_touching), default=1e-12)
    enrichment = touching_bad_fraction / max(non_touching_bad_fraction, 1e-12)
    return touching_bad_fraction, non_touching_bad_fraction, enrichment


def _possible_source_structure_or_model_mismatch(
    strong_bad: np.ndarray,
    uvdist_bins: np.ndarray,
    candidates: list[Candidate],
    cfg: dict,
) -> bool:
    if strong_bad.size == 0 or np.count_nonzero(strong_bad) == 0:
        return False
    best_coverage = candidates[0].coverage if candidates else 0.0
    if best_coverage > float(cfg["source_structure_max_group_coverage"]):
        return False

    fractions = []
    centers = []
    for uv_bin in sorted(set(int(x) for x in uvdist_bins.tolist())):
        mask = uvdist_bins == uv_bin
        if np.count_nonzero(mask) < 10:
            continue
        fractions.append(_safe_ratio(np.count_nonzero(strong_bad[mask]), np.count_nonzero(mask), default=0.0))
        centers.append(float(uv_bin))
    if len(fractions) < 4:
        return False
    corr = np.corrcoef(np.asarray(centers, dtype=float), np.asarray(fractions, dtype=float))[0, 1]
    return bool(np.isfinite(corr) and abs(corr) >= float(cfg["source_structure_uv_corr_threshold"]))


def _make_summary_text_block(
    ax,
    *,
    title: str,
    summary: dict,
    metrics: dict,
    top_candidates: list[dict],
) -> None:
    lines = [
        title,
        "",
        f"status: {summary.get('status', '?')}",
        f"recommendation: {summary.get('recommendation', 'none')}",
        f"culprit: {summary.get('culprit_type') or 'none'} {summary.get('culprit_id') or ''}".strip(),
        f"strong bad frac: {_finite_float(metrics.get('strong_bad_fraction')):.4f}",
        f"p99 |z|: {_finite_float(metrics.get('p99_abs_z')):.2f}",
        f"max |z|: {_finite_float(metrics.get('max_abs_z')):.2f}",
        "",
        summary.get("conclusion", "No visibility anomaly analysis available"),
        "",
        "Recommended inspection target:",
        f"antenna={summary.get('recommended_selection', {}).get('antenna') or 'none'}",
        f"baseline={summary.get('recommended_selection', {}).get('baseline') or 'none'}",
        f"timerange={summary.get('recommended_selection', {}).get('timerange') or 'none'}",
        f"spw={summary.get('recommended_selection', {}).get('spw') or 'none'}",
        "",
        "No automatic flagging applied",
    ]
    if top_candidates:
        best = top_candidates[0]
        lines.extend(
            [
                "",
                "Top candidate:",
                f"{best.get('type')} | {best.get('id')}",
                f"coverage={_finite_float(best.get('coverage')):.2f}",
                f"enrichment={_finite_float(best.get('enrichment')):.2f}",
                f"bad_fraction={_finite_float(best.get('bad_fraction')):.2f}",
                f"data_loss={_finite_float(best.get('data_loss')):.2f}",
            ]
        )
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=10, family="monospace")
    ax.axis("off")


def _scatter_subset(x: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if x.size <= max_points:
        return x, y
    idx = np.linspace(0, x.size - 1, num=max_points, dtype=int)
    return x[idx], y[idx]


def _log_visibility_qa_result(target: str, result: dict) -> None:
    summary = result.get("summary", {}) or {}
    metrics = result.get("metrics", {}) or {}
    culprit_type = summary.get("culprit_type") or "none"
    culprit_id = summary.get("culprit_id") or ""
    culprit_text = f"{culprit_type} {culprit_id}".strip()
    print(
        f"[VISQA] {target} | status={result.get('status', '?')} | "
        f"recommendation={summary.get('recommendation', 'none')} | "
        f"culprit={culprit_text or 'none'} | "
        f"used={int(_finite_float(metrics.get('n_samples_used'), 0.0))} | "
        f"strong_bad={_finite_float(metrics.get('strong_bad_fraction')):.4f} | "
        f"p99|z|={_finite_float(metrics.get('p99_abs_z')):.2f}"
    )


def _iter_row_chunks(nrows: int, rows_per_chunk: int):
    for start in range(0, nrows, rows_per_chunk):
        yield start, min(rows_per_chunk, nrows - start)


def _effective_rows_per_chunk(
    configured_rows: int,
    *,
    probe_dtype_itemsize: int,
    n_corr_used: int,
    nchan_full: int,
    chunk_target_mb: float,
) -> int:
    rows = max(int(configured_rows), 1)
    if probe_dtype_itemsize <= 0 or n_corr_used <= 0 or nchan_full <= 0 or chunk_target_mb <= 0:
        return rows
    target_bytes = float(chunk_target_mb) * 1024.0 * 1024.0
    per_row_data_bytes = float(n_corr_used * nchan_full * (probe_dtype_itemsize + 1))
    if per_row_data_bytes <= 0:
        return rows
    adaptive_rows = max(64, int(target_bytes // per_row_data_bytes))
    return max(64, min(rows, adaptive_rows))


def _equal_width_edges(lo: float, hi: float, nbins: int) -> np.ndarray:
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        base = lo if np.isfinite(lo) else 0.0
        return np.array([base, base + 1.0], dtype=float)
    return np.linspace(lo, hi, nbins + 1, dtype=float)


def _digitize_with_edges(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros(0, dtype=int)
    nbins = max(len(edges) - 1, 1)
    bins = np.digitize(values, edges[1:-1], right=False).astype(int)
    return np.clip(bins, 0, nbins - 1)


def _build_selected_ordinals(total_selected_rows: int, max_rows: int | None) -> np.ndarray | None:
    if max_rows is None or max_rows <= 0 or total_selected_rows <= max_rows:
        return None
    return np.linspace(0, total_selected_rows - 1, num=max_rows, dtype=np.int64)


def _apply_row_subsampling(
    row_mask: np.ndarray,
    selected_ordinals: np.ndarray | None,
    seen_selected_rows: int,
) -> tuple[np.ndarray, int]:
    selected_in_chunk = int(np.count_nonzero(row_mask))
    if selected_ordinals is None or selected_in_chunk == 0:
        return row_mask, seen_selected_rows + selected_in_chunk

    selected_idx = np.flatnonzero(row_mask)
    ordinal_range = np.arange(
        seen_selected_rows,
        seen_selected_rows + selected_in_chunk,
        dtype=np.int64,
    )
    keep_local = np.isin(ordinal_range, selected_ordinals, assume_unique=False)
    new_mask = np.zeros_like(row_mask, dtype=bool)
    new_mask[selected_idx[keep_local]] = True
    return new_mask, seen_selected_rows + selected_in_chunk


def _count_map_to_candidates(
    candidate_type: str,
    count_map: dict[int, list[int]],
    total_used: int,
    total_bad: int,
    *,
    decode_key,
    channel_bin_size: int,
    antenna_names: np.ndarray,
    time_edges: np.ndarray,
) -> list[Candidate]:
    global_bad_fraction = _safe_ratio(total_bad, total_used, default=0.0)
    eps = 1e-12
    candidates: list[Candidate] = []
    for numeric_key, (n_total, n_bad) in count_map.items():
        semantic_key = decode_key(int(numeric_key))
        bad_fraction = _safe_ratio(n_bad, n_total, default=0.0)
        coverage = _safe_ratio(n_bad, total_bad, default=0.0)
        enrichment = bad_fraction / max(global_bad_fraction, eps)
        data_loss = _safe_ratio(n_total, total_used, default=0.0)
        score = coverage * log1p(max(enrichment, 0.0)) * bad_fraction / sqrt(max(data_loss, eps))
        candidates.append(
            Candidate(
                type=candidate_type,
                key=semantic_key,
                id=_candidate_id(
                    candidate_type,
                    semantic_key,
                    antenna_names=antenna_names,
                    time_edges=time_edges,
                    channel_bin_size=channel_bin_size,
                ),
                n_total=int(n_total),
                n_bad=int(n_bad),
                bad_fraction=bad_fraction,
                coverage=coverage,
                enrichment=enrichment,
                data_loss=data_loss,
                score=float(score),
            )
        )
    candidates.sort(key=lambda item: (-item.score, -item.coverage, -item.bad_fraction, item.id))
    return candidates


def _encode_candidate_key(
    candidate_type: str,
    key: object,
    *,
    antenna_index_by_name: dict[str, int],
    n_antennas: int,
    time_nbins: int,
    n_channel_bins: int,
) -> int:
    if candidate_type == "antenna":
        return int(antenna_index_by_name[str(key)])
    if candidate_type == "baseline":
        a1_idx = antenna_index_by_name[str(key[0])]
        a2_idx = antenna_index_by_name[str(key[1])]
        lo = min(a1_idx, a2_idx)
        hi = max(a1_idx, a2_idx)
        return int(lo * n_antennas + hi)
    if candidate_type in {"time", "channel", "spw"}:
        return int(key)
    if candidate_type == "antenna_time":
        ant_idx = antenna_index_by_name[str(key[0])]
        return int(ant_idx * time_nbins + int(key[1]))
    if candidate_type == "baseline_time":
        a1_idx = antenna_index_by_name[str(key[0][0])]
        a2_idx = antenna_index_by_name[str(key[0][1])]
        lo = min(a1_idx, a2_idx)
        hi = max(a1_idx, a2_idx)
        return int((lo * n_antennas + hi) * time_nbins + int(key[1]))
    if candidate_type == "antenna_channel":
        ant_idx = antenna_index_by_name[str(key[0])]
        return int(ant_idx * n_channel_bins + int(key[1]))
    if candidate_type == "baseline_channel":
        a1_idx = antenna_index_by_name[str(key[0][0])]
        a2_idx = antenna_index_by_name[str(key[0][1])]
        lo = min(a1_idx, a2_idx)
        hi = max(a1_idx, a2_idx)
        return int((lo * n_antennas + hi) * n_channel_bins + int(key[1]))
    if candidate_type == "spw_channel":
        return int(int(key[0]) * n_channel_bins + int(key[1]))
    raise ValueError(f"Unsupported candidate type: {candidate_type}")


def _accumulate_numeric_counts(store: dict[int, list[int]], keys: np.ndarray, strong_bad: np.ndarray) -> None:
    if keys.size == 0:
        return
    unique_keys, inverse = np.unique(keys.astype(np.int64, copy=False), return_inverse=True)
    totals = np.bincount(inverse)
    bads = np.bincount(inverse, weights=strong_bad.astype(np.int64))
    for key, total, bad in zip(unique_keys.tolist(), totals.tolist(), bads.tolist()):
        bucket = store.setdefault(int(key), [0, 0])
        bucket[0] += int(total)
        bucket[1] += int(round(bad))


def _downsample_evenly(values: np.ndarray, limit: int) -> np.ndarray:
    arr = np.asarray(values)
    if limit <= 0 or arr.size <= limit:
        return arr.astype(np.float32, copy=False)
    idx = np.linspace(0, arr.size - 1, num=limit, dtype=int)
    return arr[idx].astype(np.float32, copy=False)


def _merge_capped_values(existing: np.ndarray | None, values: np.ndarray, limit: int) -> np.ndarray:
    new_values = np.asarray(values, dtype=np.float32)
    if new_values.size == 0:
        return (
            np.zeros(0, dtype=np.float32)
            if existing is None
            else np.asarray(existing, dtype=np.float32, copy=False)
        )
    if existing is None or np.asarray(existing).size == 0:
        return _downsample_evenly(new_values, limit)
    merged = np.concatenate(
        [np.asarray(existing, dtype=np.float32, copy=False), new_values.astype(np.float32, copy=False)]
    )
    return _downsample_evenly(merged, limit)


def _merge_capped_store(
    store: dict[object, np.ndarray],
    key: object,
    values: np.ndarray,
    limit: int,
) -> None:
    store[key] = _merge_capped_values(store.get(key), values, limit)


def _accumulate_value_samples(
    sample_store: dict[object, np.ndarray],
    count_store: dict[object, int],
    key: object,
    values: np.ndarray,
    limit: int,
) -> None:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return
    count_store[key] = int(count_store.get(key, 0)) + int(arr.size)
    _merge_capped_store(sample_store, key, arr, limit)


def _stats_from_sample_store(
    sample_store: dict[object, np.ndarray],
    count_store: dict[object, int],
    mad_floor: float,
) -> dict[object, tuple[float, float, int]]:
    out: dict[object, tuple[float, float, int]] = {}
    for key, count in count_store.items():
        sample = np.asarray(sample_store.get(key, np.zeros(0, dtype=np.float32)), dtype=np.float32)
        median, sigma, _ = _robust_loc_scale(sample, mad_floor)
        out[key] = (median, sigma, int(count))
    return out


def _select_normalization_stats(
    *,
    key1: object,
    key2: object,
    key3: object,
    key4: object,
    stats_level1: dict,
    stats_level2: dict,
    stats_level3: dict,
    stats_level4: dict,
    stats_level5: dict,
    min_bin_count: int,
) -> tuple[float, float, int]:
    for level, key, stats_map in (
        (0, key1, stats_level1),
        (1, key2, stats_level2),
        (2, key3, stats_level3),
        (3, key4, stats_level4),
        (4, "global", stats_level5),
    ):
        median, sigma, count = stats_map[key]
        if count >= min_bin_count or level == 4:
            return float(median), float(sigma), int(level)
    return 0.0, 1.0, 4


def _append_plot_samples(plot_store: dict[str, list[np.ndarray]], sample_block: dict, max_points: int) -> None:
    current = sum(arr.size for arr in plot_store.get("abs_z", []))
    remaining = max(max_points * 3 - current, 0)
    if remaining <= 0:
        return
    take = min(int(sample_block["abs_z"].size), max(remaining, max_points // 4))
    if take <= 0:
        return
    idx = np.linspace(0, sample_block["abs_z"].size - 1, num=take, dtype=int)
    for key, values in sample_block.items():
        plot_store.setdefault(key, []).append(np.asarray(values)[idx])


def _finalize_plot_samples(plot_store: dict[str, list[np.ndarray]], max_points: int) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, chunks in plot_store.items():
        if not chunks:
            out[key] = np.zeros(0, dtype=float if key in {"uvdist_kl", "abs_z", "time_minutes"} else bool)
            continue
        merged = np.concatenate(chunks)
        if merged.size > max_points:
            idx = np.linspace(0, merged.size - 1, num=max_points, dtype=int)
            merged = merged[idx]
        out[key] = merged
    return out


def _iter_valid_group_samples(
    *,
    data_chunk: np.ndarray,
    flags_chunk: np.ndarray,
    corr_indices: list[int],
    row_spw: np.ndarray,
    row_uvdist_kl: np.ndarray,
    row_uvdist_bin: np.ndarray,
    row_time: np.ndarray,
    row_time_bin: np.ndarray,
    row_ant1: np.ndarray,
    row_ant2: np.ndarray,
    freqs_by_spw: dict[int, np.ndarray],
    channel_bin_size: int,
) -> dict:
    nchan_total = int(data_chunk.shape[1])
    if nchan_total <= 0 or row_spw.size == 0:
        return

    channel_bins_all = (np.arange(nchan_total, dtype=int) // int(channel_bin_size)).astype(int)
    for spw_id in sorted(set(int(value) for value in row_spw.tolist())):
        row_mask_spw = row_spw == spw_id
        if not np.any(row_mask_spw):
            continue
        freqs = freqs_by_spw.get(int(spw_id), np.zeros(0, dtype=float))
        valid_nchan = min(nchan_total, int(freqs.size))
        if valid_nchan <= 0:
            continue

        chan_bins_valid = channel_bins_all[:valid_nchan]
        spw_uvdist_kl = row_uvdist_kl[row_mask_spw]
        spw_uvbin = row_uvdist_bin[row_mask_spw]
        spw_time = row_time[row_mask_spw]
        spw_time_bin = row_time_bin[row_mask_spw]
        spw_ant1 = row_ant1[row_mask_spw]
        spw_ant2 = row_ant2[row_mask_spw]
        unique_uvbins = np.unique(spw_uvbin)

        for corr_local, corr_idx in enumerate(corr_indices):
            amp_spw = np.abs(data_chunk[corr_local, :valid_nchan, :][:, row_mask_spw]).astype(np.float32, copy=False)
            valid_spw = (~flags_chunk[corr_local, :valid_nchan, :][:, row_mask_spw]) & np.isfinite(amp_spw) & (amp_spw > 0)
            if not np.any(valid_spw):
                continue

            for uvbin in unique_uvbins.tolist():
                row_mask_uv = spw_uvbin == int(uvbin)
                if not np.any(row_mask_uv):
                    continue
                amp_uv = amp_spw[:, row_mask_uv]
                valid_uv = valid_spw[:, row_mask_uv]
                if not np.any(valid_uv):
                    continue

                uv_rows = spw_uvdist_kl[row_mask_uv]
                time_rows = spw_time[row_mask_uv]
                time_bin_rows = spw_time_bin[row_mask_uv]
                ant1_rows = spw_ant1[row_mask_uv]
                ant2_rows = spw_ant2[row_mask_uv]

                for chan_bin in range(int(chan_bins_valid.max()) + 1):
                    chan_mask = chan_bins_valid == chan_bin
                    if not np.any(chan_mask):
                        continue
                    valid_block = valid_uv[chan_mask, :]
                    if not np.any(valid_block):
                        continue
                    amp_block = amp_uv[chan_mask, :]
                    chan_offsets, row_offsets = np.nonzero(valid_block)
                    if row_offsets.size == 0:
                        continue
                    local_chan_indices = np.flatnonzero(chan_mask)[chan_offsets]
                    yield {
                        "spw": int(spw_id),
                        "corr": int(corr_idx),
                        "uvbin": int(uvbin),
                        "chanbin": int(chan_bin),
                        "channel_index": local_chan_indices.astype(np.int32, copy=False),
                        "log_amp": np.log10(amp_block[valid_block]).astype(np.float32, copy=False),
                        "uvdist_kl": uv_rows[row_offsets].astype(np.float32, copy=False),
                        "time": time_rows[row_offsets].astype(np.float64, copy=False),
                        "time_bin": time_bin_rows[row_offsets].astype(np.int32, copy=False),
                        "ant1": ant1_rows[row_offsets].astype(np.int32, copy=False),
                        "ant2": ant2_rows[row_offsets].astype(np.int32, copy=False),
                    }


def _plot_no_culprit(main_ax, *, metrics: dict, summary: dict, top_candidates: list[dict], uvdist_kl: np.ndarray, abs_z: np.ndarray, strong_bad: np.ndarray) -> None:
    if abs_z.size:
        x, y = _scatter_subset(uvdist_kl, abs_z, DEFAULT_CONFIG["max_plot_points"])
        main_ax.scatter(x, y, s=6, alpha=0.25, color="#4C78A8", linewidths=0, rasterized=True)
        if np.count_nonzero(strong_bad):
            xb, yb = _scatter_subset(uvdist_kl[strong_bad], abs_z[strong_bad], DEFAULT_CONFIG["max_plot_points"] // 2)
            main_ax.scatter(xb, yb, s=8, alpha=0.55, color="#D62728", linewidths=0, rasterized=True)
        main_ax.set_xlabel("uv distance (kλ)")
        main_ax.set_ylabel("|z|")
        main_ax.set_title("Visibility anomaly QA overview")
        main_ax.grid(alpha=0.2)
    else:
        main_ax.text(0.5, 0.5, "No usable visibilities", ha="center", va="center")
        main_ax.axis("off")


def _plot_bad_antenna(main_ax, *, culprit: str, metrics: dict, summary: dict, top_candidates: list[dict], ant1_name: np.ndarray, ant2_name: np.ndarray, uvdist_kl: np.ndarray, abs_z: np.ndarray) -> None:
    touching = (ant1_name == culprit) | (ant2_name == culprit)
    x_touch, y_touch = _scatter_subset(uvdist_kl[touching], abs_z[touching], DEFAULT_CONFIG["max_plot_points"] // 2)
    x_other, y_other = _scatter_subset(uvdist_kl[~touching], abs_z[~touching], DEFAULT_CONFIG["max_plot_points"] // 2)
    main_ax.scatter(x_other, y_other, s=6, alpha=0.18, color="#9E9E9E", linewidths=0, label="other baselines", rasterized=True)
    main_ax.scatter(x_touch, y_touch, s=7, alpha=0.45, color="#D62728", linewidths=0, label=f"touching {culprit}", rasterized=True)
    main_ax.set_xlabel("uv distance (kλ)")
    main_ax.set_ylabel("|z|")
    main_ax.set_title(f"Baselines touching antenna {culprit}")
    main_ax.legend(loc="upper right", fontsize=9)
    main_ax.grid(alpha=0.2)


def _plot_bad_baseline(main_ax, *, culprit: tuple[str, str], metrics: dict, summary: dict, top_candidates: list[dict], baseline_name: np.ndarray, time_minutes: np.ndarray, abs_z: np.ndarray) -> None:
    baseline_id = f"{culprit[0]}&{culprit[1]}"
    match = baseline_name == baseline_id
    x_match, y_match = _scatter_subset(time_minutes[match], abs_z[match], DEFAULT_CONFIG["max_plot_points"] // 2)
    x_other, y_other = _scatter_subset(time_minutes[~match], abs_z[~match], DEFAULT_CONFIG["max_plot_points"] // 2)
    main_ax.scatter(x_other, y_other, s=6, alpha=0.15, color="#9E9E9E", linewidths=0, label="other baselines", rasterized=True)
    main_ax.scatter(x_match, y_match, s=8, alpha=0.55, color="#E45756", linewidths=0, label=baseline_id, rasterized=True)
    main_ax.set_xlabel("time offset (min)")
    main_ax.set_ylabel("|z|")
    main_ax.set_title(f"Baseline {baseline_id}")
    main_ax.legend(loc="upper right", fontsize=9)
    main_ax.grid(alpha=0.2)


def _plot_bad_time(main_ax, *, candidate_list: list[Candidate], highlighted_bins: list[int], metrics: dict, summary: dict, top_candidates: list[dict], time_edges: np.ndarray) -> None:
    ordered = sorted(candidate_list, key=lambda candidate: int(candidate.key))
    x = np.array([int(candidate.key) for candidate in ordered], dtype=int)
    y = np.array([candidate.bad_fraction for candidate in ordered], dtype=float)
    main_ax.plot(x, y, color="#4C78A8", lw=1.5)
    for bin_idx in highlighted_bins:
        main_ax.axvspan(bin_idx - 0.5, bin_idx + 0.5, color="#E45756", alpha=0.25)
    main_ax.set_xlabel("time bin")
    main_ax.set_ylabel("strong bad fraction")
    main_ax.set_title("Anomaly concentration in time")
    main_ax.grid(alpha=0.2)


def _plot_bad_channel(main_ax, *, candidate_list: list[Candidate], highlighted_bins: list[int], metrics: dict, summary: dict, top_candidates: list[dict]) -> None:
    ordered = sorted(
        candidate_list,
        key=lambda candidate: int(candidate.key[1] if isinstance(candidate.key, tuple) else candidate.key),
    )
    x = np.array([int(candidate.key[1] if isinstance(candidate.key, tuple) else candidate.key) for candidate in ordered], dtype=int)
    y = np.array([candidate.bad_fraction for candidate in ordered], dtype=float)
    main_ax.plot(x, y, color="#4C78A8", lw=1.5)
    for bin_idx in highlighted_bins:
        main_ax.axvspan(bin_idx - 0.5, bin_idx + 0.5, color="#E45756", alpha=0.25)
    main_ax.set_xlabel("channel bin")
    main_ax.set_ylabel("strong bad fraction")
    main_ax.set_title("Anomaly concentration in channel bins")
    main_ax.grid(alpha=0.2)


def _plot_heatmap(fig, main_ax, *, rows: list[str], cols: np.ndarray, values: np.ndarray, title: str, xlabel: str, ylabel: str, summary: dict, metrics: dict, top_candidates: list[dict], highlight_row: int | None = None, highlight_col: int | None = None) -> None:
    image = main_ax.imshow(values, aspect="auto", origin="lower", cmap="magma")
    main_ax.set_title(title)
    main_ax.set_xlabel(xlabel)
    main_ax.set_ylabel(ylabel)
    main_ax.set_yticks(np.arange(len(rows)))
    main_ax.set_yticklabels(rows, fontsize=8)
    if cols.size <= 16:
        main_ax.set_xticks(np.arange(cols.size))
        main_ax.set_xticklabels([str(int(c)) for c in cols], fontsize=8, rotation=45)
    if highlight_row is not None and highlight_col is not None:
        main_ax.add_patch(
            plt.Rectangle(
                (highlight_col - 0.5, highlight_row - 0.5),
                1.0,
                1.0,
                fill=False,
                edgecolor="cyan",
                linewidth=2.0,
            )
        )
    fig.colorbar(image, ax=main_ax, fraction=0.046, pad=0.04, label="strong bad fraction")


def _create_confirmation_plot(
    *,
    png_path: Path,
    summary: dict,
    metrics: dict,
    top_candidates: list[dict],
    used: dict,
    plot_context: dict,
    cfg: dict,
) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig, main_ax = plt.subplots(1, 1, figsize=(9, 6))

    culprit_type = summary.get("culprit_type")
    culprit_key = plot_context.get("culprit_key")
    if culprit_type == "bad_antenna" and isinstance(culprit_key, str):
        _plot_bad_antenna(
            main_ax,
            culprit=culprit_key,
            metrics=metrics,
            summary=summary,
            top_candidates=top_candidates,
            ant1_name=used["ant1_name"],
            ant2_name=used["ant2_name"],
            uvdist_kl=used["uvdist_kl"],
            abs_z=used["abs_z"],
        )
    elif culprit_type == "bad_baseline" and isinstance(culprit_key, tuple):
        _plot_bad_baseline(
            main_ax,
            culprit=culprit_key,
            metrics=metrics,
            summary=summary,
            top_candidates=top_candidates,
            baseline_name=used["baseline_name"],
            time_minutes=used["time_minutes"],
            abs_z=used["abs_z"],
        )
    elif culprit_type == "bad_time_range":
        _plot_bad_time(
            main_ax,
            candidate_list=plot_context.get("time_candidates", []),
            highlighted_bins=plot_context.get("highlight_bins", []),
            metrics=metrics,
            summary=summary,
            top_candidates=top_candidates,
            time_edges=plot_context["time_edges"],
        )
    elif culprit_type == "bad_channel_range":
        _plot_bad_channel(
            main_ax,
            candidate_list=plot_context.get("channel_candidates", []),
            highlighted_bins=plot_context.get("highlight_bins", []),
            metrics=metrics,
            summary=summary,
            top_candidates=top_candidates,
        )
    elif culprit_type in {"bad_antenna_time", "bad_baseline_time"}:
        values = plot_context.get("heatmap_values", np.zeros((1, 1), dtype=float))
        labels = plot_context.get("heatmap_labels", ["?"])
        cols = plot_context.get("heatmap_cols", np.arange(values.shape[1], dtype=int))
        _plot_heatmap(
            fig,
            main_ax,
            rows=labels,
            cols=cols,
            values=values,
            title="Antenna/baseline vs time anomaly map",
            xlabel="time bin",
            ylabel="candidate",
            summary=summary,
            metrics=metrics,
            top_candidates=top_candidates,
            highlight_row=plot_context.get("highlight_row"),
            highlight_col=plot_context.get("highlight_col"),
        )
    elif culprit_type in {"bad_antenna_channel", "bad_baseline_channel"}:
        values = plot_context.get("heatmap_values", np.zeros((1, 1), dtype=float))
        labels = plot_context.get("heatmap_labels", ["?"])
        cols = plot_context.get("heatmap_cols", np.arange(values.shape[1], dtype=int))
        _plot_heatmap(
            fig,
            main_ax,
            rows=labels,
            cols=cols,
            values=values,
            title="Antenna/baseline vs channel anomaly map",
            xlabel="channel bin",
            ylabel="candidate",
            summary=summary,
            metrics=metrics,
            top_candidates=top_candidates,
            highlight_row=plot_context.get("highlight_row"),
            highlight_col=plot_context.get("highlight_col"),
        )
    else:
        _plot_no_culprit(
            main_ax,
            metrics=metrics,
            summary=summary,
            top_candidates=top_candidates,
            uvdist_kl=used["uvdist_kl"],
            abs_z=used["abs_z"],
            strong_bad=used["strong_bad"],
        )

    fig.tight_layout()
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def analyze_visibility_anomalies(
    ms_path: Path,
    output_dir: Path,
    *,
    spw: str = "",
    uvrange: str = "",
    datacolumn_preference: str = "corrected",
    max_rows: int | None = None,
    config: dict | None = None,
) -> dict:
    cfg = _merged_config(config)
    confirmation_png = output_dir / f"{cfg['plot_prefix']}_visibility_anomaly_confirmation.png"
    target = _target_label(ms_path, output_dir)
    configured_rows_per_chunk = int(cfg.get("rows_per_chunk", 4096))
    print(
        f"[VISQA] {target} | start | ms={ms_path.name} | "
        f"spw={spw or 'all'} | uvrange={uvrange or 'all'} | "
        f"preference={datacolumn_preference}"
    )

    try:
        antenna_names = _read_antenna_names(ms_path)
        ddid_to_spw = _read_ddid_to_spw(ms_path)
        mean_freq_hz_by_spw, freqs_by_spw = _read_spw_freq_lookup(ms_path)
        antenna_index_by_name = {str(name): idx for idx, name in enumerate(antenna_names.tolist())}

        tb = table()
        tb.open(str(ms_path))
        try:
            data_column_used = _select_data_column(list(tb.colnames()), datacolumn_preference)
            nrows = int(tb.nrows())
            probe = np.asarray(tb.getcol(data_column_used, startrow=0, nrow=min(1, nrows)))
        finally:
            tb.close()

        if probe.size == 0:
            ncorr_full = 0
            nchan_full = 0
            probe_itemsize = 0
        else:
            ncorr_full = int(probe.shape[0])
            nchan_full = int(probe.shape[1])
            probe_itemsize = int(np.asarray(probe).dtype.itemsize)
        n_samples_total = int(nrows * ncorr_full * nchan_full)
        corr_indices = [0] if ncorr_full <= 1 else [0, ncorr_full - 1]
        rows_per_chunk = _effective_rows_per_chunk(
            configured_rows_per_chunk,
            probe_dtype_itemsize=probe_itemsize,
            n_corr_used=len(corr_indices),
            nchan_full=nchan_full,
            chunk_target_mb=float(cfg.get("chunk_target_mb", 96)),
        )
        selected_spws = _parse_spw_selection(spw)
        uvmin_kl, uvmax_kl = _parse_uvrange_limits(uvrange)
        total_selected_rows = 0
        selected_time_min = float("inf")
        selected_time_max = float("-inf")
        selected_uv_min = float("inf")
        selected_uv_max = float("-inf")

        print(f"[VISQA] {target} | pass=metadata | rows_per_chunk={rows_per_chunk}")
        tb = table()
        tb.open(str(ms_path))
        try:
            has_flag_row = "FLAG_ROW" in tb.colnames()
            for start, nrow_chunk in _iter_row_chunks(nrows, rows_per_chunk):
                uvw = np.asarray(tb.getcol("UVW", startrow=start, nrow=nrow_chunk), dtype=float)
                antenna1 = np.asarray(tb.getcol("ANTENNA1", startrow=start, nrow=nrow_chunk), dtype=int)
                antenna2 = np.asarray(tb.getcol("ANTENNA2", startrow=start, nrow=nrow_chunk), dtype=int)
                times = np.asarray(tb.getcol("TIME", startrow=start, nrow=nrow_chunk), dtype=float)
                ddid = np.asarray(tb.getcol("DATA_DESC_ID", startrow=start, nrow=nrow_chunk), dtype=int)
                flag_row = (
                    np.asarray(tb.getcol("FLAG_ROW", startrow=start, nrow=nrow_chunk), dtype=bool)
                    if has_flag_row
                    else np.zeros(nrow_chunk, dtype=bool)
                )

                row_spw = ddid_to_spw[ddid]
                row_mask = (~flag_row) & (antenna1 != antenna2)
                if selected_spws is not None:
                    row_mask &= np.isin(row_spw, sorted(selected_spws))

                uvdist_m = np.sqrt(uvw[0] ** 2 + uvw[1] ** 2)
                row_mean_freq_hz = mean_freq_hz_by_spw[row_spw]
                row_uvdist_kl = uvdist_m * row_mean_freq_hz / LIGHT_SPEED_M_S / 1e3
                if uvmin_kl is not None:
                    row_mask &= row_uvdist_kl >= uvmin_kl
                if uvmax_kl is not None:
                    row_mask &= row_uvdist_kl <= uvmax_kl

                if not np.any(row_mask):
                    continue

                total_selected_rows += int(np.count_nonzero(row_mask))
                selected_time_min = min(selected_time_min, float(np.min(times[row_mask])))
                selected_time_max = max(selected_time_max, float(np.max(times[row_mask])))
                selected_uv_min = min(selected_uv_min, float(np.min(row_uvdist_kl[row_mask])))
                selected_uv_max = max(selected_uv_max, float(np.max(row_uvdist_kl[row_mask])))
        finally:
            tb.close()

        selected_ordinals = _build_selected_ordinals(total_selected_rows, max_rows)
        effective_selected_rows = total_selected_rows if selected_ordinals is None else int(selected_ordinals.size)

        if effective_selected_rows == 0:
            summary = {
                "status": "insufficient_data",
                "conclusion": "Visibility anomaly QA: insufficient unflagged data after selection.",
                "recommendation": "none",
                "culprit_type": None,
                "culprit_id": None,
                "recommended_selection": {
                    "antenna": None,
                    "baseline": None,
                    "timerange": None,
                    "spw": None,
                    "reason": "No automatic flagging applied",
                },
            }
            metrics = {
                "data_column_used": data_column_used,
                "n_samples_total": n_samples_total,
                "n_samples_used": 0,
                "n_strong_bad": 0,
                "n_moderate_bad": 0,
                "strong_bad_fraction": 0.0,
                "moderate_bad_fraction": 0.0,
                "median_abs_z": 0.0,
                "p95_abs_z": 0.0,
                "p99_abs_z": 0.0,
                "max_abs_z": 0.0,
            }
            result = {
                "status": "insufficient_data",
                "summary": summary,
                "metrics": metrics,
                "top_candidates": [],
                "thresholds": {k: _plain_value(v) for k, v in cfg.items() if k != "plot_prefix"},
                "normalization": {
                    "method": "local_log_amp_median_mad",
                    "uvdist_nbins": int(cfg["uvdist_nbins"]),
                    "channel_bin_size": int(cfg["channel_bin_size"]),
                    "time_nbins": int(cfg["time_nbins"]),
                    "min_bin_count": int(cfg["min_bin_count"]),
                    "fallback_fraction": 0.0,
                },
                "artifacts": {
                    "visibility_anomaly_confirmation_png": str(confirmation_png)
                },
            }
            _create_confirmation_plot(
                png_path=confirmation_png,
                summary=summary,
                metrics=metrics,
                top_candidates=[],
                used={
                    "uvdist_kl": np.zeros(0, dtype=float),
                    "abs_z": np.zeros(0, dtype=float),
                    "strong_bad": np.zeros(0, dtype=bool),
                },
                plot_context={},
                cfg=cfg,
            )
            plain_result = _plain_dict(result)
            _log_visibility_qa_result(target, plain_result)
            return plain_result

        if selected_ordinals is not None:
            print(f"[VISQA] {target} | row_subsample={effective_selected_rows}/{total_selected_rows}")
            selected_time_min = float("inf")
            selected_time_max = float("-inf")
            selected_uv_min = float("inf")
            selected_uv_max = float("-inf")
            seen_selected_rows = 0
            tb = table()
            tb.open(str(ms_path))
            try:
                has_flag_row = "FLAG_ROW" in tb.colnames()
                for start, nrow_chunk in _iter_row_chunks(nrows, rows_per_chunk):
                    uvw = np.asarray(tb.getcol("UVW", startrow=start, nrow=nrow_chunk), dtype=float)
                    antenna1 = np.asarray(tb.getcol("ANTENNA1", startrow=start, nrow=nrow_chunk), dtype=int)
                    antenna2 = np.asarray(tb.getcol("ANTENNA2", startrow=start, nrow=nrow_chunk), dtype=int)
                    times = np.asarray(tb.getcol("TIME", startrow=start, nrow=nrow_chunk), dtype=float)
                    ddid = np.asarray(tb.getcol("DATA_DESC_ID", startrow=start, nrow=nrow_chunk), dtype=int)
                    flag_row = (
                        np.asarray(tb.getcol("FLAG_ROW", startrow=start, nrow=nrow_chunk), dtype=bool)
                        if has_flag_row
                        else np.zeros(nrow_chunk, dtype=bool)
                    )
                    row_spw = ddid_to_spw[ddid]
                    row_mask = (~flag_row) & (antenna1 != antenna2)
                    if selected_spws is not None:
                        row_mask &= np.isin(row_spw, sorted(selected_spws))
                    uvdist_m = np.sqrt(uvw[0] ** 2 + uvw[1] ** 2)
                    row_mean_freq_hz = mean_freq_hz_by_spw[row_spw]
                    row_uvdist_kl = uvdist_m * row_mean_freq_hz / LIGHT_SPEED_M_S / 1e3
                    if uvmin_kl is not None:
                        row_mask &= row_uvdist_kl >= uvmin_kl
                    if uvmax_kl is not None:
                        row_mask &= row_uvdist_kl <= uvmax_kl
                    row_mask, seen_selected_rows = _apply_row_subsampling(row_mask, selected_ordinals, seen_selected_rows)
                    if not np.any(row_mask):
                        continue
                    selected_time_min = min(selected_time_min, float(np.min(times[row_mask])))
                    selected_time_max = max(selected_time_max, float(np.max(times[row_mask])))
                    selected_uv_min = min(selected_uv_min, float(np.min(row_uvdist_kl[row_mask])))
                    selected_uv_max = max(selected_uv_max, float(np.max(row_uvdist_kl[row_mask])))
            finally:
                tb.close()

        uv_edges = _equal_width_edges(selected_uv_min, selected_uv_max, int(cfg["uvdist_nbins"]))
        time_edges = _equal_width_edges(selected_time_min, selected_time_max, int(cfg["time_nbins"]))
        channel_bin_size = int(cfg["channel_bin_size"])
        n_channel_bins = max(1, (nchan_full + channel_bin_size - 1) // channel_bin_size)
        n_antennas = int(len(antenna_names))
        print(
            f"[VISQA] {target} | selected_rows={effective_selected_rows} | "
            f"nchan={nchan_full} | ncorr_used={len(corr_indices)} | "
            f"uvbins={int(cfg['uvdist_nbins'])} | timebins={int(cfg['time_nbins'])}"
        )

        print(f"[VISQA] {target} | pass=normalization")
        normalization_sample_limit = int(cfg.get("normalization_sample_limit", 1024))
        level1_samples: dict[object, np.ndarray] = {}
        level2_samples: dict[object, np.ndarray] = {}
        level3_samples: dict[object, np.ndarray] = {}
        level4_samples: dict[object, np.ndarray] = {}
        level5_samples: dict[object, np.ndarray] = {}
        level1_counts: dict[object, int] = {}
        level2_counts: dict[object, int] = {}
        level3_counts: dict[object, int] = {}
        level4_counts: dict[object, int] = {}
        level5_counts: dict[object, int] = {}
        seen_selected_rows = 0
        tb = table()
        tb.open(str(ms_path))
        try:
            has_flag_row = "FLAG_ROW" in tb.colnames()
            for start, nrow_chunk in _iter_row_chunks(nrows, rows_per_chunk):
                uvw = np.asarray(tb.getcol("UVW", startrow=start, nrow=nrow_chunk), dtype=float)
                antenna1 = np.asarray(tb.getcol("ANTENNA1", startrow=start, nrow=nrow_chunk), dtype=int)
                antenna2 = np.asarray(tb.getcol("ANTENNA2", startrow=start, nrow=nrow_chunk), dtype=int)
                times = np.asarray(tb.getcol("TIME", startrow=start, nrow=nrow_chunk), dtype=float)
                ddid = np.asarray(tb.getcol("DATA_DESC_ID", startrow=start, nrow=nrow_chunk), dtype=int)
                flag_row = (
                    np.asarray(tb.getcol("FLAG_ROW", startrow=start, nrow=nrow_chunk), dtype=bool)
                    if has_flag_row
                    else np.zeros(nrow_chunk, dtype=bool)
                )
                row_spw = ddid_to_spw[ddid]
                row_mask = (~flag_row) & (antenna1 != antenna2)
                if selected_spws is not None:
                    row_mask &= np.isin(row_spw, sorted(selected_spws))
                uvdist_m = np.sqrt(uvw[0] ** 2 + uvw[1] ** 2)
                row_mean_freq_hz = mean_freq_hz_by_spw[row_spw]
                row_uvdist_kl = uvdist_m * row_mean_freq_hz / LIGHT_SPEED_M_S / 1e3
                if uvmin_kl is not None:
                    row_mask &= row_uvdist_kl >= uvmin_kl
                if uvmax_kl is not None:
                    row_mask &= row_uvdist_kl <= uvmax_kl
                row_mask, seen_selected_rows = _apply_row_subsampling(row_mask, selected_ordinals, seen_selected_rows)
                if not np.any(row_mask):
                    continue

                data_chunk = np.asarray(tb.getcol(data_column_used, startrow=start, nrow=nrow_chunk))[corr_indices, :, :]
                flags_chunk = np.asarray(tb.getcol("FLAG", startrow=start, nrow=nrow_chunk), dtype=bool)[corr_indices, :, :]
                row_uvbin = _digitize_with_edges(row_uvdist_kl[row_mask], uv_edges)
                row_time_bin = _digitize_with_edges(times[row_mask], time_edges)
                for sample_group in _iter_valid_group_samples(
                    data_chunk=data_chunk[:, :, row_mask],
                    flags_chunk=flags_chunk[:, :, row_mask],
                    corr_indices=corr_indices,
                    row_spw=row_spw[row_mask],
                    row_uvdist_kl=row_uvdist_kl[row_mask],
                    row_uvdist_bin=row_uvbin,
                    row_time=times[row_mask],
                    row_time_bin=row_time_bin,
                    row_ant1=antenna1[row_mask],
                    row_ant2=antenna2[row_mask],
                    freqs_by_spw=freqs_by_spw,
                    channel_bin_size=channel_bin_size,
                ):
                    level1_key = (
                        int(sample_group["spw"]),
                        int(sample_group["corr"]),
                        int(sample_group["uvbin"]),
                        int(sample_group["chanbin"]),
                    )
                    level2_key = (
                        int(sample_group["spw"]),
                        int(sample_group["corr"]),
                        int(sample_group["uvbin"]),
                    )
                    level3_key = (
                        int(sample_group["spw"]),
                        int(sample_group["corr"]),
                    )
                    level4_key = int(sample_group["corr"])
                    log_amp = sample_group["log_amp"]
                    _accumulate_value_samples(level1_samples, level1_counts, level1_key, log_amp, normalization_sample_limit)
                    _accumulate_value_samples(level2_samples, level2_counts, level2_key, log_amp, normalization_sample_limit)
                    _accumulate_value_samples(level3_samples, level3_counts, level3_key, log_amp, normalization_sample_limit)
                    _accumulate_value_samples(level4_samples, level4_counts, level4_key, log_amp, normalization_sample_limit)
                    _accumulate_value_samples(level5_samples, level5_counts, "global", log_amp, max(normalization_sample_limit, 4 * normalization_sample_limit))
        finally:
            tb.close()

        if not level1_counts:
            summary = {
                "status": "insufficient_data",
                "conclusion": "Visibility anomaly QA: no valid unflagged amplitudes remained after filtering.",
                "recommendation": "none",
                "culprit_type": None,
                "culprit_id": None,
                "recommended_selection": {
                    "antenna": None,
                    "baseline": None,
                    "timerange": None,
                    "spw": None,
                    "reason": "No automatic flagging applied",
                },
            }
            metrics = {
                "data_column_used": data_column_used,
                "n_samples_total": n_samples_total,
                "n_samples_used": 0,
                "n_strong_bad": 0,
                "n_moderate_bad": 0,
                "strong_bad_fraction": 0.0,
                "moderate_bad_fraction": 0.0,
                "median_abs_z": 0.0,
                "p95_abs_z": 0.0,
                "p99_abs_z": 0.0,
                "max_abs_z": 0.0,
            }
            result = {
                "status": "insufficient_data",
                "summary": summary,
                "metrics": metrics,
                "top_candidates": [],
                "thresholds": {k: _plain_value(v) for k, v in cfg.items() if k != "plot_prefix"},
                "normalization": {
                    "method": "local_log_amp_median_mad",
                    "uvdist_nbins": int(cfg["uvdist_nbins"]),
                    "channel_bin_size": int(cfg["channel_bin_size"]),
                    "time_nbins": int(cfg["time_nbins"]),
                    "min_bin_count": int(cfg["min_bin_count"]),
                    "fallback_fraction": 0.0,
                },
                "artifacts": {
                    "visibility_anomaly_confirmation_png": str(confirmation_png)
                },
            }
            _create_confirmation_plot(
                png_path=confirmation_png,
                summary=summary,
                metrics=metrics,
                top_candidates=[],
                used={
                    "uvdist_kl": np.zeros(0, dtype=float),
                    "abs_z": np.zeros(0, dtype=float),
                    "strong_bad": np.zeros(0, dtype=bool),
                },
                plot_context={},
                cfg=cfg,
            )
            plain_result = _plain_dict(result)
            _log_visibility_qa_result(target, plain_result)
            return plain_result
        mad_floor = float(cfg["mad_floor"])
        min_bin_count = int(cfg["min_bin_count"])

        stats_level1 = _stats_from_sample_store(level1_samples, level1_counts, mad_floor)
        stats_level2 = _stats_from_sample_store(level2_samples, level2_counts, mad_floor)
        stats_level3 = _stats_from_sample_store(level3_samples, level3_counts, mad_floor)
        stats_level4 = _stats_from_sample_store(level4_samples, level4_counts, mad_floor)
        stats_level5 = _stats_from_sample_store(level5_samples, level5_counts, mad_floor)
        del level1_samples
        del level2_samples
        del level3_samples
        del level4_samples
        del level5_samples

        print(f"[VISQA] {target} | pass=scoring")
        count_maps = {
            "antenna": {},
            "baseline": {},
            "time": {},
            "channel": {},
            "spw": {},
            "antenna_time": {},
            "baseline_time": {},
            "antenna_channel": {},
            "baseline_channel": {},
            "spw_channel": {},
        }
        uvbin_count_map: dict[int, list[int]] = {}
        metric_sample_limit = int(cfg.get("metric_sample_limit", 50000))
        global_abs_z_sample = np.zeros(0, dtype=np.float32)
        global_abs_z_max = 0.0
        plot_store: dict[str, list[np.ndarray]] = {}
        n_samples_used = 0
        n_strong_bad = 0
        n_moderate_bad = 0
        fallback_sample_count = 0
        seen_selected_rows = 0

        tb = table()
        tb.open(str(ms_path))
        try:
            has_flag_row = "FLAG_ROW" in tb.colnames()
            for start, nrow_chunk in _iter_row_chunks(nrows, rows_per_chunk):
                uvw = np.asarray(tb.getcol("UVW", startrow=start, nrow=nrow_chunk), dtype=float)
                antenna1 = np.asarray(tb.getcol("ANTENNA1", startrow=start, nrow=nrow_chunk), dtype=int)
                antenna2 = np.asarray(tb.getcol("ANTENNA2", startrow=start, nrow=nrow_chunk), dtype=int)
                times = np.asarray(tb.getcol("TIME", startrow=start, nrow=nrow_chunk), dtype=float)
                ddid = np.asarray(tb.getcol("DATA_DESC_ID", startrow=start, nrow=nrow_chunk), dtype=int)
                flag_row = (
                    np.asarray(tb.getcol("FLAG_ROW", startrow=start, nrow=nrow_chunk), dtype=bool)
                    if has_flag_row
                    else np.zeros(nrow_chunk, dtype=bool)
                )
                row_spw = ddid_to_spw[ddid]
                row_mask = (~flag_row) & (antenna1 != antenna2)
                if selected_spws is not None:
                    row_mask &= np.isin(row_spw, sorted(selected_spws))
                uvdist_m = np.sqrt(uvw[0] ** 2 + uvw[1] ** 2)
                row_mean_freq_hz = mean_freq_hz_by_spw[row_spw]
                row_uvdist_kl = uvdist_m * row_mean_freq_hz / LIGHT_SPEED_M_S / 1e3
                if uvmin_kl is not None:
                    row_mask &= row_uvdist_kl >= uvmin_kl
                if uvmax_kl is not None:
                    row_mask &= row_uvdist_kl <= uvmax_kl
                row_mask, seen_selected_rows = _apply_row_subsampling(row_mask, selected_ordinals, seen_selected_rows)
                if not np.any(row_mask):
                    continue

                data_chunk = np.asarray(tb.getcol(data_column_used, startrow=start, nrow=nrow_chunk))[corr_indices, :, :]
                flags_chunk = np.asarray(tb.getcol("FLAG", startrow=start, nrow=nrow_chunk), dtype=bool)[corr_indices, :, :]
                row_uvbin = _digitize_with_edges(row_uvdist_kl[row_mask], uv_edges)
                row_time_bin = _digitize_with_edges(times[row_mask], time_edges)

                for sample_group in _iter_valid_group_samples(
                    data_chunk=data_chunk[:, :, row_mask],
                    flags_chunk=flags_chunk[:, :, row_mask],
                    corr_indices=corr_indices,
                    row_spw=row_spw[row_mask],
                    row_uvdist_kl=row_uvdist_kl[row_mask],
                    row_uvdist_bin=row_uvbin,
                    row_time=times[row_mask],
                    row_time_bin=row_time_bin,
                    row_ant1=antenna1[row_mask],
                    row_ant2=antenna2[row_mask],
                    freqs_by_spw=freqs_by_spw,
                    channel_bin_size=channel_bin_size,
                ):
                    key1 = (sample_group["spw"], sample_group["corr"], sample_group["uvbin"], sample_group["chanbin"])
                    key2 = (sample_group["spw"], sample_group["corr"], sample_group["uvbin"])
                    key3 = (sample_group["spw"], sample_group["corr"])
                    key4 = sample_group["corr"]
                    median, sigma, fallback_level = _select_normalization_stats(
                        key1=key1,
                        key2=key2,
                        key3=key3,
                        key4=key4,
                        stats_level1=stats_level1,
                        stats_level2=stats_level2,
                        stats_level3=stats_level3,
                        stats_level4=stats_level4,
                        stats_level5=stats_level5,
                        min_bin_count=min_bin_count,
                    )
                    z = np.abs((sample_group["log_amp"] - median) / sigma).astype(np.float32, copy=False)
                    if fallback_level > 0:
                        fallback_sample_count += int(z.size)

                    strong_bad = z >= float(cfg["strong_z"])
                    moderate_bad = z >= float(cfg["moderate_z"])
                    n_samples_used += int(z.size)
                    n_strong_bad += int(np.count_nonzero(strong_bad))
                    n_moderate_bad += int(np.count_nonzero(moderate_bad))
                    global_abs_z_sample = _merge_capped_values(global_abs_z_sample, z, metric_sample_limit)
                    if z.size:
                        global_abs_z_max = max(global_abs_z_max, float(np.max(z)))

                    ant1_vals = sample_group["ant1"].astype(np.int64, copy=False)
                    ant2_vals = sample_group["ant2"].astype(np.int64, copy=False)
                    lo = np.minimum(ant1_vals, ant2_vals)
                    hi = np.maximum(ant1_vals, ant2_vals)
                    baseline_codes = lo * n_antennas + hi
                    time_bins = sample_group["time_bin"].astype(np.int64, copy=False)
                    chan_bins = np.full(z.size, int(sample_group["chanbin"]), dtype=np.int64)
                    spw_codes = np.full(z.size, int(sample_group["spw"]), dtype=np.int64)

                    _accumulate_numeric_counts(count_maps["antenna"], np.concatenate([ant1_vals, ant2_vals]), np.concatenate([strong_bad, strong_bad]))
                    _accumulate_numeric_counts(count_maps["baseline"], baseline_codes, strong_bad)
                    _accumulate_numeric_counts(count_maps["time"], time_bins, strong_bad)
                    _accumulate_numeric_counts(count_maps["channel"], chan_bins, strong_bad)
                    _accumulate_numeric_counts(count_maps["spw"], spw_codes, strong_bad)
                    _accumulate_numeric_counts(uvbin_count_map, np.full(z.size, int(sample_group["uvbin"]), dtype=np.int64), strong_bad)

                    antenna_time_codes = np.concatenate(
                        [
                            ant1_vals * int(cfg["time_nbins"]) + time_bins,
                            ant2_vals * int(cfg["time_nbins"]) + time_bins,
                        ]
                    )
                    antenna_channel_codes = np.concatenate(
                        [
                            ant1_vals * n_channel_bins + chan_bins,
                            ant2_vals * n_channel_bins + chan_bins,
                        ]
                    )
                    _accumulate_numeric_counts(count_maps["antenna_time"], antenna_time_codes, np.concatenate([strong_bad, strong_bad]))
                    _accumulate_numeric_counts(
                        count_maps["baseline_time"],
                        baseline_codes * int(cfg["time_nbins"]) + time_bins,
                        strong_bad,
                    )
                    _accumulate_numeric_counts(count_maps["antenna_channel"], antenna_channel_codes, np.concatenate([strong_bad, strong_bad]))
                    _accumulate_numeric_counts(
                        count_maps["baseline_channel"],
                        baseline_codes * n_channel_bins + chan_bins,
                        strong_bad,
                    )
                    _accumulate_numeric_counts(
                        count_maps["spw_channel"],
                        spw_codes * n_channel_bins + chan_bins,
                        strong_bad,
                    )

                    _append_plot_samples(
                        plot_store,
                        {
                            "uvdist_kl": sample_group["uvdist_kl"],
                            "abs_z": z,
                            "strong_bad": strong_bad.astype(bool),
                            "ant1": ant1_vals.astype(np.int32, copy=False),
                            "ant2": ant2_vals.astype(np.int32, copy=False),
                            "time_minutes": ((sample_group["time"] - selected_time_min) / 60.0).astype(np.float32, copy=False),
                        },
                        int(cfg["max_plot_points"]),
                    )
        finally:
            tb.close()

        strong_bad_fraction = _safe_ratio(n_strong_bad, n_samples_used, default=0.0)
        moderate_bad_fraction = _safe_ratio(n_moderate_bad, n_samples_used, default=0.0)

        decode_baseline = lambda code: tuple(
            sorted(
                (
                    str(antenna_names[int(code) // n_antennas]),
                    str(antenna_names[int(code) % n_antennas]),
                )
            )
        )
        decode_antenna = lambda code: str(antenna_names[int(code)])
        decode_time = lambda code: int(code)
        decode_channel = lambda code: int(code)
        decode_spw = lambda code: int(code)
        decode_antenna_time = lambda code: (
            str(antenna_names[int(code) // int(cfg["time_nbins"])]),
            int(code) % int(cfg["time_nbins"]),
        )
        decode_baseline_time = lambda code: (
            decode_baseline(int(code) // int(cfg["time_nbins"])),
            int(code) % int(cfg["time_nbins"]),
        )
        decode_antenna_channel = lambda code: (
            str(antenna_names[int(code) // n_channel_bins]),
            int(code) % n_channel_bins,
        )
        decode_baseline_channel = lambda code: (
            decode_baseline(int(code) // n_channel_bins),
            int(code) % n_channel_bins,
        )
        decode_spw_channel = lambda code: (
            int(code) // n_channel_bins,
            int(code) % n_channel_bins,
        )

        candidates_by_type = {
            "antenna": _count_map_to_candidates("antenna", count_maps["antenna"], n_samples_used, n_strong_bad, decode_key=decode_antenna, channel_bin_size=channel_bin_size, antenna_names=antenna_names, time_edges=time_edges),
            "baseline": _count_map_to_candidates("baseline", count_maps["baseline"], n_samples_used, n_strong_bad, decode_key=decode_baseline, channel_bin_size=channel_bin_size, antenna_names=antenna_names, time_edges=time_edges),
            "time": _count_map_to_candidates("time", count_maps["time"], n_samples_used, n_strong_bad, decode_key=decode_time, channel_bin_size=channel_bin_size, antenna_names=antenna_names, time_edges=time_edges),
            "channel": _count_map_to_candidates("channel", count_maps["channel"], n_samples_used, n_strong_bad, decode_key=decode_channel, channel_bin_size=channel_bin_size, antenna_names=antenna_names, time_edges=time_edges),
            "spw": _count_map_to_candidates("spw", count_maps["spw"], n_samples_used, n_strong_bad, decode_key=decode_spw, channel_bin_size=channel_bin_size, antenna_names=antenna_names, time_edges=time_edges),
            "antenna_time": _count_map_to_candidates("antenna_time", count_maps["antenna_time"], n_samples_used, n_strong_bad, decode_key=decode_antenna_time, channel_bin_size=channel_bin_size, antenna_names=antenna_names, time_edges=time_edges),
            "baseline_time": _count_map_to_candidates("baseline_time", count_maps["baseline_time"], n_samples_used, n_strong_bad, decode_key=decode_baseline_time, channel_bin_size=channel_bin_size, antenna_names=antenna_names, time_edges=time_edges),
            "antenna_channel": _count_map_to_candidates("antenna_channel", count_maps["antenna_channel"], n_samples_used, n_strong_bad, decode_key=decode_antenna_channel, channel_bin_size=channel_bin_size, antenna_names=antenna_names, time_edges=time_edges),
            "baseline_channel": _count_map_to_candidates("baseline_channel", count_maps["baseline_channel"], n_samples_used, n_strong_bad, decode_key=decode_baseline_channel, channel_bin_size=channel_bin_size, antenna_names=antenna_names, time_edges=time_edges),
            "spw_channel": _count_map_to_candidates("spw_channel", count_maps["spw_channel"], n_samples_used, n_strong_bad, decode_key=decode_spw_channel, channel_bin_size=channel_bin_size, antenna_names=antenna_names, time_edges=time_edges),
        }
        all_candidates = [candidate for candidates in candidates_by_type.values() for candidate in candidates]
        all_candidates.sort(key=lambda item: (-item.score, -item.coverage, -item.bad_fraction, item.id))

        top_candidate_objects = all_candidates[: int(cfg["max_top_candidates"])]
        candidate_metric_sample_limit = int(cfg.get("candidate_metric_sample_limit", 12000))
        top_abs_z_store: dict[tuple[str, object], np.ndarray] = {}

        if top_candidate_objects:
            print(f"[VISQA] {target} | pass=top_candidate_metrics | n={len(top_candidate_objects)}")
            top_by_type: dict[str, list[Candidate]] = defaultdict(list)
            for candidate in top_candidate_objects:
                top_by_type[candidate.type].append(candidate)
            top_codes_by_type: dict[str, np.ndarray] = {}
            code_to_key_by_type: dict[str, dict[int, object]] = {}
            for candidate_type, candidates in top_by_type.items():
                encoded = [
                    _encode_candidate_key(
                        candidate_type,
                        candidate.key,
                        antenna_index_by_name=antenna_index_by_name,
                        n_antennas=n_antennas,
                        time_nbins=int(cfg["time_nbins"]),
                        n_channel_bins=n_channel_bins,
                    )
                    for candidate in candidates
                ]
                top_codes_by_type[candidate_type] = np.asarray(encoded, dtype=np.int64)
                code_to_key_by_type[candidate_type] = {
                    int(code): candidate.key
                    for code, candidate in zip(encoded, candidates)
                }

            seen_selected_rows = 0
            tb = table()
            tb.open(str(ms_path))
            try:
                has_flag_row = "FLAG_ROW" in tb.colnames()
                for start, nrow_chunk in _iter_row_chunks(nrows, rows_per_chunk):
                    uvw = np.asarray(tb.getcol("UVW", startrow=start, nrow=nrow_chunk), dtype=float)
                    antenna1 = np.asarray(tb.getcol("ANTENNA1", startrow=start, nrow=nrow_chunk), dtype=int)
                    antenna2 = np.asarray(tb.getcol("ANTENNA2", startrow=start, nrow=nrow_chunk), dtype=int)
                    times = np.asarray(tb.getcol("TIME", startrow=start, nrow=nrow_chunk), dtype=float)
                    ddid = np.asarray(tb.getcol("DATA_DESC_ID", startrow=start, nrow=nrow_chunk), dtype=int)
                    flag_row = (
                        np.asarray(tb.getcol("FLAG_ROW", startrow=start, nrow=nrow_chunk), dtype=bool)
                        if has_flag_row
                        else np.zeros(nrow_chunk, dtype=bool)
                    )
                    row_spw = ddid_to_spw[ddid]
                    row_mask = (~flag_row) & (antenna1 != antenna2)
                    if selected_spws is not None:
                        row_mask &= np.isin(row_spw, sorted(selected_spws))
                    uvdist_m = np.sqrt(uvw[0] ** 2 + uvw[1] ** 2)
                    row_mean_freq_hz = mean_freq_hz_by_spw[row_spw]
                    row_uvdist_kl = uvdist_m * row_mean_freq_hz / LIGHT_SPEED_M_S / 1e3
                    if uvmin_kl is not None:
                        row_mask &= row_uvdist_kl >= uvmin_kl
                    if uvmax_kl is not None:
                        row_mask &= row_uvdist_kl <= uvmax_kl
                    row_mask, seen_selected_rows = _apply_row_subsampling(row_mask, selected_ordinals, seen_selected_rows)
                    if not np.any(row_mask):
                        continue

                    data_chunk = np.asarray(tb.getcol(data_column_used, startrow=start, nrow=nrow_chunk))[corr_indices, :, :]
                    flags_chunk = np.asarray(tb.getcol("FLAG", startrow=start, nrow=nrow_chunk), dtype=bool)[corr_indices, :, :]
                    row_uvbin = _digitize_with_edges(row_uvdist_kl[row_mask], uv_edges)
                    row_time_bin = _digitize_with_edges(times[row_mask], time_edges)

                    for sample_group in _iter_valid_group_samples(
                        data_chunk=data_chunk[:, :, row_mask],
                        flags_chunk=flags_chunk[:, :, row_mask],
                        corr_indices=corr_indices,
                        row_spw=row_spw[row_mask],
                        row_uvdist_kl=row_uvdist_kl[row_mask],
                        row_uvdist_bin=row_uvbin,
                        row_time=times[row_mask],
                        row_time_bin=row_time_bin,
                        row_ant1=antenna1[row_mask],
                        row_ant2=antenna2[row_mask],
                        freqs_by_spw=freqs_by_spw,
                        channel_bin_size=channel_bin_size,
                    ):
                        key1 = (sample_group["spw"], sample_group["corr"], sample_group["uvbin"], sample_group["chanbin"])
                        key2 = (sample_group["spw"], sample_group["corr"], sample_group["uvbin"])
                        key3 = (sample_group["spw"], sample_group["corr"])
                        key4 = sample_group["corr"]
                        median, sigma, _fallback_level = _select_normalization_stats(
                            key1=key1,
                            key2=key2,
                            key3=key3,
                            key4=key4,
                            stats_level1=stats_level1,
                            stats_level2=stats_level2,
                            stats_level3=stats_level3,
                            stats_level4=stats_level4,
                            stats_level5=stats_level5,
                            min_bin_count=min_bin_count,
                        )
                        z = np.abs((sample_group["log_amp"] - median) / sigma).astype(np.float32, copy=False)

                        ant1_vals = sample_group["ant1"].astype(np.int64, copy=False)
                        ant2_vals = sample_group["ant2"].astype(np.int64, copy=False)
                        lo = np.minimum(ant1_vals, ant2_vals)
                        hi = np.maximum(ant1_vals, ant2_vals)
                        baseline_codes = lo * n_antennas + hi
                        time_bins = sample_group["time_bin"].astype(np.int64, copy=False)
                        chan_bins = np.full(z.size, int(sample_group["chanbin"]), dtype=np.int64)
                        spw_codes = np.full(z.size, int(sample_group["spw"]), dtype=np.int64)

                        type_arrays = {
                            "antenna": np.concatenate([ant1_vals, ant2_vals]),
                            "baseline": baseline_codes,
                            "time": time_bins,
                            "channel": chan_bins,
                            "spw": spw_codes,
                            "antenna_time": np.concatenate(
                                [
                                    ant1_vals * int(cfg["time_nbins"]) + time_bins,
                                    ant2_vals * int(cfg["time_nbins"]) + time_bins,
                                ]
                            ),
                            "baseline_time": baseline_codes * int(cfg["time_nbins"]) + time_bins,
                            "antenna_channel": np.concatenate(
                                [
                                    ant1_vals * n_channel_bins + chan_bins,
                                    ant2_vals * n_channel_bins + chan_bins,
                                ]
                            ),
                            "baseline_channel": baseline_codes * n_channel_bins + chan_bins,
                            "spw_channel": spw_codes * n_channel_bins + chan_bins,
                        }
                        z_arrays = {
                            "antenna": np.concatenate([z, z]),
                            "baseline": z,
                            "time": z,
                            "channel": z,
                            "spw": z,
                            "antenna_time": np.concatenate([z, z]),
                            "baseline_time": z,
                            "antenna_channel": np.concatenate([z, z]),
                            "baseline_channel": z,
                            "spw_channel": z,
                        }
                        for candidate_type, candidates in top_by_type.items():
                            if not candidates:
                                continue
                            code_array = type_arrays[candidate_type]
                            z_array = z_arrays[candidate_type]
                            selected_codes_arr = top_codes_by_type.get(candidate_type, np.zeros(0, dtype=np.int64))
                            if selected_codes_arr.size == 0:
                                continue
                            mask = np.isin(code_array, selected_codes_arr)
                            if not np.any(mask):
                                continue
                            matched_codes = code_array[mask]
                            matched_z = z_array[mask]
                            unique_codes = np.unique(matched_codes)
                            code_to_key = code_to_key_by_type.get(candidate_type, {})
                            for code in unique_codes.tolist():
                                semantic_key = code_to_key.get(int(code))
                                if semantic_key is not None:
                                    _merge_capped_store(
                                        top_abs_z_store,
                                        (candidate_type, semantic_key),
                                        matched_z[matched_codes == code].astype(np.float32, copy=False),
                                        candidate_metric_sample_limit,
                                    )
            finally:
                tb.close()

        top_candidates = []
        for candidate in top_candidate_objects:
            group_abs_z = np.asarray(
                top_abs_z_store.get((candidate.type, candidate.key), np.zeros(0, dtype=np.float32)),
                dtype=np.float32,
            )
            if group_abs_z.size:
                median_abs_z = float(np.median(group_abs_z))
                p95_abs_z = float(np.percentile(group_abs_z, 95))
                max_abs_z = float(np.max(group_abs_z))
            else:
                median_abs_z = 0.0
                p95_abs_z = 0.0
                max_abs_z = 0.0
            top_candidates.append(
                {
                    "type": candidate.type,
                    "id": candidate.id,
                    "n_total": candidate.n_total,
                    "n_bad": candidate.n_bad,
                    "bad_fraction": candidate.bad_fraction,
                    "coverage": candidate.coverage,
                    "enrichment": candidate.enrichment,
                    "data_loss": candidate.data_loss,
                    "median_abs_z": median_abs_z,
                    "p95_abs_z": p95_abs_z,
                    "max_abs_z": max_abs_z,
                    "score": candidate.score,
                }
            )

        metrics = {
            "data_column_used": data_column_used,
            "n_samples_total": n_samples_total,
            "n_samples_used": n_samples_used,
            "n_strong_bad": n_strong_bad,
            "n_moderate_bad": n_moderate_bad,
            "strong_bad_fraction": strong_bad_fraction,
            "moderate_bad_fraction": moderate_bad_fraction,
            "median_abs_z": float(np.median(global_abs_z_sample)) if global_abs_z_sample.size else 0.0,
            "p95_abs_z": float(np.percentile(global_abs_z_sample, 95)) if global_abs_z_sample.size else 0.0,
            "p99_abs_z": float(np.percentile(global_abs_z_sample, 99)) if global_abs_z_sample.size else 0.0,
            "max_abs_z": float(global_abs_z_max),
        }

        normalization = {
            "method": "local_log_amp_median_mad",
            "uvdist_nbins": int(cfg["uvdist_nbins"]),
            "channel_bin_size": int(cfg["channel_bin_size"]),
            "time_nbins": int(cfg["time_nbins"]),
            "min_bin_count": int(cfg["min_bin_count"]),
            "fallback_fraction": _safe_ratio(fallback_sample_count, n_samples_used, default=0.0),
        }

        summary = {
            "status": "ok",
            "conclusion": "Visibility anomaly QA: OK / no strong culprit",
            "recommendation": "none",
            "culprit_type": None,
            "culprit_id": None,
            "recommended_selection": {
                "antenna": None,
                "baseline": None,
                "timerange": None,
                "spw": None,
                "reason": "No automatic flagging applied",
            },
        }
        plot_context = {
            "time_edges": time_edges,
            "time_candidates": candidates_by_type.get("time", []),
            "channel_candidates": candidates_by_type.get("spw_channel", []),
        }

        best_baseline = next((candidate for candidate in candidates_by_type.get("baseline", []) if _qualifies(candidate, cfg)), None)
        best_antenna = next((candidate for candidate in candidates_by_type.get("antenna", []) if _qualifies(candidate, cfg)), None)
        best_time = next((candidate for candidate in candidates_by_type.get("time", []) if _qualifies(candidate, cfg)), None)
        best_spw_channel = next((candidate for candidate in candidates_by_type.get("spw_channel", []) if _qualifies(candidate, cfg)), None)
        best_antenna_time = next((candidate for candidate in candidates_by_type.get("antenna_time", []) if _qualifies(candidate, cfg)), None)
        best_baseline_time = next((candidate for candidate in candidates_by_type.get("baseline_time", []) if _qualifies(candidate, cfg)), None)
        best_antenna_channel = next((candidate for candidate in candidates_by_type.get("antenna_channel", []) if _qualifies(candidate, cfg)), None)
        best_baseline_channel = next((candidate for candidate in candidates_by_type.get("baseline_channel", []) if _qualifies(candidate, cfg)), None)

        shared_baselines = 0
        antenna_touching_enrichment = 0.0
        if best_antenna is not None:
            _, _, antenna_touching_enrichment = _touching_vs_non_touching_enrichment(
                str(best_antenna.key),
                ant1_name,
                ant2_name,
                strong_bad,
            )
            top_bad_baselines = candidates_by_type.get("baseline", [])[:6]
            shared_baselines = sum(
                1
                for candidate in top_bad_baselines
                if isinstance(candidate.key, tuple) and str(best_antenna.key) in candidate.key
            )
        prefer_antenna = (
            best_antenna is not None
            and (shared_baselines >= 2 or antenna_touching_enrichment >= max(best_antenna.enrichment, 2.0))
        )

        if best_baseline is not None and not prefer_antenna:
            summary = {
                "status": "suspect",
                "conclusion": (
                    f"Likely bad baseline {best_baseline.key[0]}&{best_baseline.key[1]}: explains "
                    f"{100.0 * best_baseline.coverage:.0f}% of strong outliers with "
                    f"{best_baseline.enrichment:.1f}x enrichment."
                ),
                "recommendation": "flag_candidate",
                "culprit_type": "bad_baseline",
                "culprit_id": f"{best_baseline.key[0]}&{best_baseline.key[1]}",
                "recommended_selection": {
                    "antenna": None,
                    "baseline": f"{best_baseline.key[0]}&{best_baseline.key[1]}",
                    "timerange": None,
                    "spw": None,
                    "reason": (
                        f"Suggested flag candidate. coverage={best_baseline.coverage:.2f}, "
                        f"enrichment={best_baseline.enrichment:.1f}, data_loss={best_baseline.data_loss:.2f}. "
                        "No automatic flagging applied."
                    ),
                },
            }
            plot_context["culprit_key"] = tuple(best_baseline.key)

        if summary["culprit_type"] is None and best_antenna is not None and (prefer_antenna or best_baseline is None):
            summary = {
                "status": "suspect",
                "conclusion": (
                    f"Likely bad antenna {best_antenna.key}: baselines touching it have "
                    f"{antenna_touching_enrichment:.1f}x higher anomaly rate."
                ),
                "recommendation": "flag_candidate",
                "culprit_type": "bad_antenna",
                "culprit_id": str(best_antenna.key),
                "recommended_selection": {
                    "antenna": str(best_antenna.key),
                    "baseline": None,
                    "timerange": None,
                    "spw": None,
                    "reason": (
                        f"Recommended inspection target. coverage={best_antenna.coverage:.2f}, "
                        f"enrichment={best_antenna.enrichment:.1f}, touching/non-touching={antenna_touching_enrichment:.1f}x. "
                        "No automatic flagging applied."
                    ),
                },
            }
            plot_context["culprit_key"] = str(best_antenna.key)

        if summary["culprit_type"] is None and best_time is not None:
            highlight_bins = _adjacent_bins_from_candidate(
                int(best_time.key),
                candidates_by_type.get("time", []),
                relative_threshold=float(cfg["adjacent_bin_relative_threshold"]),
            )
            summary = {
                "status": "suspect",
                "conclusion": (
                    f"Likely bad time range: anomaly is concentrated in {len(highlight_bins)} adjacent time bins "
                    "and affects many baselines."
                ),
                "recommendation": "flag_candidate",
                "culprit_type": "bad_time_range",
                "culprit_id": ", ".join(str(x) for x in highlight_bins),
                "recommended_selection": {
                    "antenna": None,
                    "baseline": None,
                    "timerange": _contiguous_timerange(time_edges, highlight_bins),
                    "spw": None,
                    "reason": (
                        f"Suggested flag candidate. coverage={best_time.coverage:.2f}, "
                        f"enrichment={best_time.enrichment:.1f}. No automatic flagging applied."
                    ),
                },
            }
            plot_context["highlight_bins"] = highlight_bins

        if summary["culprit_type"] is None and best_spw_channel is not None:
            spw_id, chan_bin = best_spw_channel.key
            same_spw = [candidate for candidate in candidates_by_type.get("spw_channel", []) if isinstance(candidate.key, tuple) and int(candidate.key[0]) == int(spw_id)]
            highlight_bins = _adjacent_bins_from_candidate(
                int(chan_bin),
                same_spw,
                relative_threshold=float(cfg["adjacent_bin_relative_threshold"]),
                key_fn=lambda candidate: candidate.key[1],
            )
            summary = {
                "status": "suspect",
                "conclusion": (
                    f"Likely bad channel range: anomaly is concentrated in SPW {spw_id} "
                    f"channels {min(highlight_bins) * int(cfg['channel_bin_size'])}-"
                    f"{(max(highlight_bins) + 1) * int(cfg['channel_bin_size']) - 1}."
                ),
                "recommendation": "flag_candidate",
                "culprit_type": "bad_channel_range",
                "culprit_id": f"SPW {spw_id}",
                "recommended_selection": {
                    "antenna": None,
                    "baseline": None,
                    "timerange": None,
                    "spw": _channel_selection(int(spw_id), highlight_bins, int(cfg["channel_bin_size"])),
                    "reason": (
                        f"Suggested flag candidate. coverage={best_spw_channel.coverage:.2f}, "
                        f"enrichment={best_spw_channel.enrichment:.1f}. No automatic flagging applied."
                    ),
                },
            }
            plot_context["highlight_bins"] = highlight_bins
            plot_context["channel_candidates"] = same_spw

        if summary["culprit_type"] is None and best_antenna_time is not None:
            antenna_name, time_bin = best_antenna_time.key
            top_rows = [candidate for candidate in candidates_by_type.get("antenna_time", []) if isinstance(candidate.key, tuple) and candidate.key[1] == time_bin][:12]
            row_names = [str(candidate.key[0]) for candidate in top_rows] or [str(antenna_name)]
            values = np.zeros((len(row_names), int(cfg["time_nbins"])), dtype=float)
            for row_idx, row_name in enumerate(row_names):
                for candidate in candidates_by_type.get("antenna_time", []):
                    if isinstance(candidate.key, tuple) and candidate.key[0] == row_name:
                        values[row_idx, int(candidate.key[1])] = candidate.bad_fraction
            summary = {
                "status": "suspect",
                "conclusion": (
                    f"Likely antenna-time issue: antenna {antenna_name} is strongly anomalous "
                    f"during time bin {time_bin}."
                ),
                "recommendation": "flag_candidate",
                "culprit_type": "bad_antenna_time",
                "culprit_id": f"{antenna_name} @ time bin {time_bin}",
                "recommended_selection": {
                    "antenna": str(antenna_name),
                    "baseline": None,
                    "timerange": _contiguous_timerange(time_edges, [int(time_bin)]),
                    "spw": None,
                    "reason": (
                        f"Recommended inspection target. coverage={best_antenna_time.coverage:.2f}, "
                        f"enrichment={best_antenna_time.enrichment:.1f}. No automatic flagging applied."
                    ),
                },
            }
            plot_context.update(
                {
                    "heatmap_values": values,
                    "heatmap_labels": row_names,
                    "heatmap_cols": np.arange(values.shape[1], dtype=int),
                    "highlight_row": row_names.index(str(antenna_name)) if str(antenna_name) in row_names else 0,
                    "highlight_col": int(time_bin),
                }
            )

        if summary["culprit_type"] is None and best_baseline_time is not None:
            baseline_pair, time_bin = best_baseline_time.key
            top_rows = [candidate for candidate in candidates_by_type.get("baseline_time", []) if isinstance(candidate.key, tuple) and candidate.key[1] == time_bin][:12]
            row_names = [f"{candidate.key[0][0]}&{candidate.key[0][1]}" for candidate in top_rows] or [f"{baseline_pair[0]}&{baseline_pair[1]}"]
            values = np.zeros((len(row_names), int(cfg["time_nbins"])), dtype=float)
            for row_idx, row_name in enumerate(row_names):
                for candidate in candidates_by_type.get("baseline_time", []):
                    if isinstance(candidate.key, tuple):
                        name = f"{candidate.key[0][0]}&{candidate.key[0][1]}"
                        if name == row_name:
                            values[row_idx, int(candidate.key[1])] = candidate.bad_fraction
            summary = {
                "status": "suspect",
                "conclusion": (
                    f"Likely baseline-time issue: baseline {baseline_pair[0]}&{baseline_pair[1]} "
                    f"is strongly anomalous during time bin {time_bin}."
                ),
                "recommendation": "flag_candidate",
                "culprit_type": "bad_baseline_time",
                "culprit_id": f"{baseline_pair[0]}&{baseline_pair[1]} @ time bin {time_bin}",
                "recommended_selection": {
                    "antenna": None,
                    "baseline": f"{baseline_pair[0]}&{baseline_pair[1]}",
                    "timerange": _contiguous_timerange(time_edges, [int(time_bin)]),
                    "spw": None,
                    "reason": (
                        f"Recommended inspection target. coverage={best_baseline_time.coverage:.2f}, "
                        f"enrichment={best_baseline_time.enrichment:.1f}. No automatic flagging applied."
                    ),
                },
            }
            plot_context.update(
                {
                    "heatmap_values": values,
                    "heatmap_labels": row_names,
                    "heatmap_cols": np.arange(values.shape[1], dtype=int),
                    "highlight_row": row_names.index(f"{baseline_pair[0]}&{baseline_pair[1]}") if f"{baseline_pair[0]}&{baseline_pair[1]}" in row_names else 0,
                    "highlight_col": int(time_bin),
                }
            )

        if summary["culprit_type"] is None and best_antenna_channel is not None:
            antenna_name, chan_bin = best_antenna_channel.key
            top_rows = [candidate for candidate in candidates_by_type.get("antenna_channel", []) if isinstance(candidate.key, tuple) and candidate.key[1] == chan_bin][:12]
            row_names = [str(candidate.key[0]) for candidate in top_rows] or [str(antenna_name)]
            values = np.zeros((len(row_names), n_channel_bins), dtype=float)
            for row_idx, row_name in enumerate(row_names):
                for candidate in candidates_by_type.get("antenna_channel", []):
                    if isinstance(candidate.key, tuple) and candidate.key[0] == row_name:
                        values[row_idx, int(candidate.key[1])] = candidate.bad_fraction
            summary = {
                "status": "suspect",
                "conclusion": (
                    f"Likely antenna-channel issue: antenna {antenna_name} is strongly anomalous "
                    f"in channel bin {chan_bin}."
                ),
                "recommendation": "flag_candidate",
                "culprit_type": "bad_antenna_channel",
                "culprit_id": f"{antenna_name} @ channel bin {chan_bin}",
                "recommended_selection": {
                    "antenna": str(antenna_name),
                    "baseline": None,
                    "timerange": None,
                    "spw": _channel_selection(None, [int(chan_bin)], int(cfg["channel_bin_size"])),
                    "reason": (
                        f"Recommended inspection target. coverage={best_antenna_channel.coverage:.2f}, "
                        f"enrichment={best_antenna_channel.enrichment:.1f}. No automatic flagging applied."
                    ),
                },
            }
            plot_context.update(
                {
                    "heatmap_values": values,
                    "heatmap_labels": row_names,
                    "heatmap_cols": np.arange(values.shape[1], dtype=int),
                    "highlight_row": row_names.index(str(antenna_name)) if str(antenna_name) in row_names else 0,
                    "highlight_col": int(chan_bin),
                }
            )

        if summary["culprit_type"] is None and best_baseline_channel is not None:
            baseline_pair, chan_bin = best_baseline_channel.key
            top_rows = [candidate for candidate in candidates_by_type.get("baseline_channel", []) if isinstance(candidate.key, tuple) and candidate.key[1] == chan_bin][:12]
            row_names = [f"{candidate.key[0][0]}&{candidate.key[0][1]}" for candidate in top_rows] or [f"{baseline_pair[0]}&{baseline_pair[1]}"]
            values = np.zeros((len(row_names), n_channel_bins), dtype=float)
            for row_idx, row_name in enumerate(row_names):
                for candidate in candidates_by_type.get("baseline_channel", []):
                    if isinstance(candidate.key, tuple):
                        name = f"{candidate.key[0][0]}&{candidate.key[0][1]}"
                        if name == row_name:
                            values[row_idx, int(candidate.key[1])] = candidate.bad_fraction
            summary = {
                "status": "suspect",
                "conclusion": (
                    f"Likely baseline-channel issue: baseline {baseline_pair[0]}&{baseline_pair[1]} "
                    f"is strongly anomalous in channel bin {chan_bin}."
                ),
                "recommendation": "flag_candidate",
                "culprit_type": "bad_baseline_channel",
                "culprit_id": f"{baseline_pair[0]}&{baseline_pair[1]} @ channel bin {chan_bin}",
                "recommended_selection": {
                    "antenna": None,
                    "baseline": f"{baseline_pair[0]}&{baseline_pair[1]}",
                    "timerange": None,
                    "spw": _channel_selection(None, [int(chan_bin)], int(cfg["channel_bin_size"])),
                    "reason": (
                        f"Recommended inspection target. coverage={best_baseline_channel.coverage:.2f}, "
                        f"enrichment={best_baseline_channel.enrichment:.1f}. No automatic flagging applied."
                    ),
                },
            }
            plot_context.update(
                {
                    "heatmap_values": values,
                    "heatmap_labels": row_names,
                    "heatmap_cols": np.arange(values.shape[1], dtype=int),
                    "highlight_row": row_names.index(f"{baseline_pair[0]}&{baseline_pair[1]}") if f"{baseline_pair[0]}&{baseline_pair[1]}" in row_names else 0,
                    "highlight_col": int(chan_bin),
                }
            )

        uv_fractions = []
        uv_centers = []
        best_coverage = all_candidates[0].coverage if all_candidates else 0.0
        if best_coverage <= float(cfg["source_structure_max_group_coverage"]):
            for uv_bin, (n_total, n_bad) in sorted(uvbin_count_map.items()):
                if int(n_total) < 10:
                    continue
                uv_centers.append(float(uv_bin))
                uv_fractions.append(_safe_ratio(n_bad, n_total, default=0.0))
        possible_source_structure = False
        if len(uv_fractions) >= 4:
            corr = np.corrcoef(np.asarray(uv_centers, dtype=float), np.asarray(uv_fractions, dtype=float))[0, 1]
            possible_source_structure = bool(
                np.isfinite(corr) and abs(corr) >= float(cfg["source_structure_uv_corr_threshold"])
            )

        if summary["culprit_type"] is None and possible_source_structure:
            summary = {
                "status": "suspect",
                "conclusion": (
                    "Possible source structure or model mismatch: anomaly strength varies smoothly with "
                    "uvdistance and is not concentrated by antenna, baseline, time, or channel."
                ),
                "recommendation": "inspect",
                "culprit_type": "possible_source_structure_or_model_mismatch",
                "culprit_id": None,
                "recommended_selection": {
                    "antenna": None,
                    "baseline": None,
                    "timerange": None,
                    "spw": None,
                    "reason": "Inspect imaging/model assumptions. No automatic flagging applied.",
                },
            }

        if summary["culprit_type"] is None and (strong_bad_fraction > 0.01 or metrics["p99_abs_z"] >= float(cfg["strong_z"])):
            summary["status"] = "suspect"
            summary["conclusion"] = (
                "Visibility anomaly QA found elevated outlier rates, but no single antenna/baseline/time/channel "
                "candidate passed the recommendation thresholds."
            )
            summary["recommendation"] = "inspect"
            summary["recommended_selection"]["reason"] = "Recommended inspection target is unclear. No automatic flagging applied."

        plot_arrays = _finalize_plot_samples(plot_store, int(cfg["max_plot_points"]))
        ant1_plot = plot_arrays.get("ant1", np.zeros(0, dtype=np.int32)).astype(np.int32, copy=False)
        ant2_plot = plot_arrays.get("ant2", np.zeros(0, dtype=np.int32)).astype(np.int32, copy=False)
        ant1_name = antenna_names[ant1_plot] if ant1_plot.size else np.zeros(0, dtype=object)
        ant2_name = antenna_names[ant2_plot] if ant2_plot.size else np.zeros(0, dtype=object)
        baseline_name = (
            np.array(
                [
                    f"{min(str(a1), str(a2))}&{max(str(a1), str(a2))}"
                    for a1, a2 in zip(ant1_name.tolist(), ant2_name.tolist())
                ],
                dtype=object,
            )
            if ant1_plot.size
            else np.zeros(0, dtype=object)
        )
        used = {
            "uvdist_kl": plot_arrays.get("uvdist_kl", np.zeros(0, dtype=np.float32)),
            "abs_z": plot_arrays.get("abs_z", np.zeros(0, dtype=np.float32)),
            "strong_bad": plot_arrays.get("strong_bad", np.zeros(0, dtype=bool)).astype(bool, copy=False),
            "ant1_name": ant1_name,
            "ant2_name": ant2_name,
            "baseline_name": baseline_name,
            "time_minutes": plot_arrays.get("time_minutes", np.zeros(0, dtype=np.float32)),
        }
        _create_confirmation_plot(
            png_path=confirmation_png,
            summary=summary,
            metrics=metrics,
            top_candidates=top_candidates,
            used=used,
            plot_context=plot_context,
            cfg=cfg,
        )

        result = {
            "status": summary["status"],
            "summary": summary,
            "metrics": metrics,
            "top_candidates": top_candidates,
            "thresholds": {k: _plain_value(v) for k, v in cfg.items() if k != "plot_prefix"},
            "normalization": normalization,
            "artifacts": {
                "visibility_anomaly_confirmation_png": str(confirmation_png),
            },
        }
        plain_result = _plain_dict(result)
        _log_visibility_qa_result(target, plain_result)
        return plain_result
    except Exception as exc:
        summary = {
            "status": "error",
            "conclusion": f"Visibility anomaly QA failed: {type(exc).__name__}: {exc}",
            "recommendation": "inspect",
            "culprit_type": None,
            "culprit_id": None,
            "recommended_selection": {
                "antenna": None,
                "baseline": None,
                "timerange": None,
                "spw": None,
                "reason": "Inspect the QA failure. No automatic flagging applied.",
            },
        }
        metrics = {
            "data_column_used": None,
            "n_samples_total": 0,
            "n_samples_used": 0,
            "n_strong_bad": 0,
            "n_moderate_bad": 0,
            "strong_bad_fraction": 0.0,
            "moderate_bad_fraction": 0.0,
            "median_abs_z": 0.0,
            "p95_abs_z": 0.0,
            "p99_abs_z": 0.0,
            "max_abs_z": 0.0,
        }
        _create_confirmation_plot(
            png_path=confirmation_png,
            summary=summary,
            metrics=metrics,
            top_candidates=[],
            used={
                "uvdist_kl": np.zeros(0, dtype=float),
                "abs_z": np.zeros(0, dtype=float),
                "strong_bad": np.zeros(0, dtype=bool),
            },
            plot_context={},
            cfg=cfg,
        )
        result = {
            "status": "error",
            "summary": summary,
            "metrics": metrics,
            "top_candidates": [],
            "thresholds": {k: _plain_value(v) for k, v in cfg.items() if k != "plot_prefix"},
            "normalization": {
                "method": "local_log_amp_median_mad",
                "uvdist_nbins": int(cfg["uvdist_nbins"]),
                "channel_bin_size": int(cfg["channel_bin_size"]),
                "time_nbins": int(cfg["time_nbins"]),
                "min_bin_count": int(cfg["min_bin_count"]),
                "fallback_fraction": 0.0,
            },
            "artifacts": {
                "visibility_anomaly_confirmation_png": str(confirmation_png),
            },
        }
        plain_result = _plain_dict(result)
        _log_visibility_qa_result(target, plain_result)
        return plain_result
