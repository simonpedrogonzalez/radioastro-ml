# process_projects.py
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd

# CASA
from casatasks import listobs, split
from casatools import table

# your helpers
from scripts.img_utils import make_clean, casa_image_to_png


# -------------------------
# Config
# -------------------------
PROJECT_LIST = "/Users/u1528314/repos/radioastro-ml/collect/small_subset/small_selection.csv"
DOWNLOAD_DIR = Path("/Users/u1528314/repos/radioastro-ml/collect/downloads")
EXTRACTED_DIR = Path("/Users/u1528314/repos/radioastro-ml/collect/extracted")

TCLEAN_CFG = dict(
    specmode="mfs",
    imsize=256,
    cell="0.5arcsec",
    weighting="briggs",
    robust=0.5,
    stokes="I",
    deconvolver="hogbom",
    gridder="standard",
    interactive=False,
)

# how close the actual download size must be to expected size
# example: 0.75 means accept if actual >= 75% of expected
MIN_SIZE_FRAC = 0.75

# minimum nontrivial folder size to consider "something got downloaded"
MIN_PRESENT_GB = 0.01  # ~10 MB


# -------------------------
# CSV schema helpers
# -------------------------
STATE_COL = "status"
ERR_COL = "wget_error"

STRING_COLS = [
    "name",
    "folder",
    "wget_command",
    STATE_COL,
    ERR_COL,
    "downloaded_folder",
    "extracted_ms",
    "spw_selected",
    "gain_calibrator_name",
    "band_guess",
    "band_code",
    "gain_array_config",
]

NUMERIC_COLS = [
    "size",   # expected size from original table
    "new_size",
    "extracted_gain_onsource_min",
    "caltables_found",
    "corrected_used",
]


def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    for c in STRING_COLS:
        if c not in df.columns:
            df[c] = pd.NA
    for c in NUMERIC_COLS:
        if c not in df.columns:
            df[c] = np.nan

    for c in STRING_COLS:
        df[c] = df[c].astype("string")

    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def load_projects(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return _ensure_cols(df)


def save_projects(df: pd.DataFrame, csv_path: str | Path) -> None:
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)


def set_row(df: pd.DataFrame, idx: int, **updates: Any) -> None:
    for k, v in updates.items():
        if k not in df.columns:
            df[k] = pd.NA

        if isinstance(v, str):
            if pd.api.types.is_numeric_dtype(df[k].dtype):
                df[k] = df[k].astype("string")
            else:
                df[k] = df[k].astype("string")

        df.loc[idx, k] = v


# -------------------------
# Filesystem helpers
# -------------------------
def dir_size_gb(path: str | Path) -> float:
    p = Path(path)
    if not p.exists():
        return float("nan")
    total = 0
    for root, _, files in os.walk(p):
        for f in files:
            fp = Path(root) / f
            try:
                total += fp.stat().st_size
            except OSError:
                pass
    return total / (1024**3)


def find_first_ms(root: str | Path) -> Optional[Path]:
    root = Path(root)
    if not root.exists():
        return None
    for p in sorted(root.glob("*.ms")):
        if p.is_dir():
            return p
    for p in sorted(root.rglob("*.ms")):
        if p.is_dir():
            return p
    return None



def find_caltables(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []

    hits = []
    for p in root.rglob("*"):
        name = p.name.lower()

        if (
            name.endswith((".tbl", ".g", ".b", ".k")) or
            name.endswith(".caltables.tgz")
        ):
            hits.append(p)

    return hits[:50]


def download_folder_ok(download_root: Path, expected_size_gb: float | None) -> tuple[bool, float, str]:
    """
    Decide if the download looks acceptable even if wget returned nonzero.
    """
    if not download_root.exists():
        return False, float("nan"), "download folder does not exist"

    actual = dir_size_gb(download_root)

    if not np.isfinite(actual) or actual < MIN_PRESENT_GB:
        return False, actual, f"downloaded folder too small ({actual:.3f} GB)"

    if expected_size_gb is None or not np.isfinite(expected_size_gb) or expected_size_gb <= 0:
        return True, actual, f"folder present, size={actual:.3f} GB, no expected size to compare"

    ratio = actual / expected_size_gb
    if ratio >= MIN_SIZE_FRAC:
        return True, actual, (
            f"folder present, size looks acceptable: actual={actual:.3f} GB "
            f"expected={expected_size_gb:.3f} GB ratio={ratio:.3f}"
        )

    return False, actual, (
        f"downloaded folder too small: actual={actual:.3f} GB "
        f"expected={expected_size_gb:.3f} GB ratio={ratio:.3f}"
    )


# -------------------------
# WGET
# -------------------------
def run_wget(cmd: str, download_dir: str | Path = DOWNLOAD_DIR) -> Tuple[int, str]:
    if not isinstance(cmd, str) or not cmd.strip():
        return 2, "empty wget_command"

    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("[WGET] Preparing command")
    print("=" * 80)

    s = cmd.strip()

    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")

    while s and s[-1] in ("'", '"'):
        if s.count(s[-1]) % 2 == 1:
            s = s[:-1].rstrip()
        else:
            break

    s = re.sub(r"--reject\s+index\.html\*\s*\"", r"--reject 'index.html*'", s)
    s = re.sub(r"-P\s+\S+", "", s)

    urls = re.findall(r"https?://[^\s\"']+", s)
    if not urls:
        return 2, "wget_command has no URL"

    url = urls[0]
    s = s.replace(url, "").strip()

    if not s.startswith("wget"):
        s = "wget " + s

    if "dl-dsoc.nrao.edu" in url and "--no-check-certificate" not in s:
        s = s.replace("wget ", "wget --no-check-certificate ", 1)

    if "--progress" not in s:
        s += " --progress=dot:giga"

    s_final = f'{s} -P "{download_dir}" "{url}"'

    print("[WGET] Download directory:")
    print(download_dir)
    print()
    print("[WGET] Final command:")
    print(s_final)
    print("=" * 80)
    print("[WGET] Starting download...\n")

    proc = subprocess.Popen(
        s_final,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    lines = []
    assert proc.stdout is not None

    for line in proc.stdout:
        print(line.rstrip())
        lines.append(line)

    proc.wait()

    rc = proc.returncode
    msg = "".join(lines)

    print("\n" + "=" * 80)
    print(f"[WGET] Finished with return code: {rc}")
    print("=" * 80)

    return rc, msg


# -------------------------
# MS checks / selection
# -------------------------
def ms_has_corrected(ms_path: str | Path) -> bool:
    tb = table()
    tb.open(str(ms_path))
    cols = set(tb.colnames())
    tb.close()
    return "CORRECTED_DATA" in cols


def field_id_for_name(ms_path: str | Path, field_name: str) -> Optional[int]:
    tb = table()
    tb.open(str(Path(ms_path) / "FIELD"))
    names = tb.getcol("NAME")
    tb.close()
    for i, n in enumerate(names):
        if str(n) == str(field_name):
            return int(i)
    return None


from pathlib import Path
from typing import Optional
import numpy as np
from casatools import table


def pick_one_spw_by_band(ms_path: str | Path, band_guess: Optional[str]) -> str:
    ms_path = Path(ms_path)

    band_ranges = {
        "L": (1e9, 2e9),
        "S": (2e9, 4e9),
        "C": (4e9, 8e9),
        "X": (8e9, 12e9),
        "KU": (12e9, 18e9),
        "K": (18e9, 26.5e9),
        "KA": (26.5e9, 40e9),
        "Q": (40e9, 50e9),
    }

    band = None
    if isinstance(band_guess, str) and band_guess.strip():
        band = band_guess.strip().upper()
        if band not in band_ranges:
            band = None

    # -------------------------
    # Read SPW metadata
    # -------------------------
    tb = table()
    tb.open(str(ms_path / "SPECTRAL_WINDOW"))
    nspw = tb.nrows()

    spw_info = []

    for spw in range(nspw):
        try:
            freqs = np.array(tb.getcell("CHAN_FREQ", spw), dtype=float)
            if freqs.size == 0:
                continue

            f_min = float(np.min(freqs))
            f_max = float(np.max(freqs))
            f_c = float(np.median(freqs))
            bw = float(f_max - f_min)

            spw_info.append({
                "spw": spw,
                "f_min": f_min,
                "f_max": f_max,
                "f_center": f_c,
                "bandwidth": bw,
            })
        except Exception as e:
            print(f"[SPW] failed reading SPW {spw}: {e}")

    tb.close()

    if not spw_info:
        print("[SPW PICK] no readable SPWs found, falling back to 0")
        return "0", "No SPWs found, falling back to 0", None

    # -------------------------
    # Compute unflagged fraction from main MS
    # -------------------------
    tb = table()
    tb.open(str(ms_path))

    try:
        data_desc_ids = np.array(tb.getcol("DATA_DESC_ID"))
        flags = np.array(tb.getcol("FLAG"))  # shape usually (nchan, npol, nrows)
    finally:
        tb.close()

    # map DATA_DESC_ID -> SPW_ID
    tb = table()
    tb.open(str(ms_path / "DATA_DESCRIPTION"))
    try:
        dd_to_spw = np.array(tb.getcol("SPECTRAL_WINDOW_ID"), dtype=int)
    finally:
        tb.close()

    for info in spw_info:
        spw = info["spw"]

        ddids_for_spw = np.where(dd_to_spw == spw)[0]
        if ddids_for_spw.size == 0:
            info["unflagged_frac"] = np.nan
            info["n_rows"] = 0
            continue

        row_mask = np.isin(data_desc_ids, ddids_for_spw)
        n_rows = int(np.sum(row_mask))
        info["n_rows"] = n_rows

        if n_rows == 0:
            info["unflagged_frac"] = np.nan
            continue

        spw_flags = flags[:, :, row_mask]
        total = spw_flags.size
        flagged = int(np.count_nonzero(spw_flags))
        info["unflagged_frac"] = 1.0 - (flagged / total if total > 0 else np.nan)

    # -------------------------
    # Band matching
    # -------------------------
    if band is not None:
        f_lo, f_hi = band_ranges[band]
        for info in spw_info:
            f_c = info["f_center"]
            info["matches_band"] = (f_lo <= f_c <= f_hi)
    else:
        for info in spw_info:
            info["matches_band"] = False

    # -------------------------
    # Pretty printing helpers
    # -------------------------
    def fmt(info: dict) -> str:
        uf = info["unflagged_frac"]
        uf_s = "nan" if not np.isfinite(uf) else f"{uf:.3f}"
        return (
            f"spw={info['spw']:>2d} | "
            f"center={info['f_center']/1e9:6.3f} GHz | "
            f"bw={info['bandwidth']/1e6:7.2f} MHz | "
            f"unflagged={uf_s} | "
            f"rows={info['n_rows']:>7d}"
        )

    print("\n[SPW PICK] ----------------------------------------")
    print(f"[SPW PICK] MS: {ms_path}")
    print(f"[SPW PICK] guessed band: {band_guess!r}")

    top_bw = sorted(spw_info, key=lambda x: x["bandwidth"], reverse=True)[:3]
    print("\n[SPW PICK] top 3 by bandwidth")
    for x in top_bw:
        print("  " + fmt(x))

    top_unflagged = sorted(
        spw_info,
        key=lambda x: (-np.inf if not np.isfinite(x["unflagged_frac"]) else x["unflagged_frac"]),
        reverse=True,
    )[:3]
    print("\n[SPW PICK] top 3 by unflagged fraction")
    for x in top_unflagged:
        print("  " + fmt(x))

    if band is not None:
        matches = [x for x in spw_info if x["matches_band"]]
        print(f"\n[SPW PICK] candidates matching guessed band {band}")
        if matches:
            for x in sorted(matches, key=lambda x: (x["bandwidth"], x["unflagged_frac"]), reverse=True):
                print("  " + fmt(x))
        else:
            print("  none")
    else:
        matches = []
        print("\n[SPW PICK] no valid guessed band provided")

    # -------------------------
    # Selection rule
    # pick top candidate matching guessed band
    # ranking: bandwidth first, then unflagged fraction
    # -------------------------
    def rank_key(x: dict):
        uf = x["unflagged_frac"]
        uf = -1.0 if not np.isfinite(uf) else uf
        return (x["bandwidth"], uf)

    def validate_spw_candidate(info: dict) -> str | None:
        MIN_UNFLAGGED = 0.5
        MIN_BW_HZ = 10e6  # 10 MHz

        uf = info.get("unflagged_frac", np.nan)
        bw = info.get("bandwidth", np.nan)

        if (not np.isfinite(uf)) or uf < MIN_UNFLAGGED:
            return f"SPW {info['spw']} rejected: unflagged fraction too low ({uf:.3f} < {MIN_UNFLAGGED})"
        if (not np.isfinite(bw)) or bw < MIN_BW_HZ:
            return f"SPW {info['spw']} rejected: bandwidth too narrow ({bw/1e6:.2f} MHz < {MIN_BW_HZ/1e6:.2f} MHz)"
        return None
        
    if matches:
        best = max(matches, key=rank_key)
        err = validate_spw_candidate(best)
        print(f"\n[SPW PICK] selected guessed-band candidate: {fmt(best)}")
        return str(best["spw"]), err, best

    best_overall = max(spw_info, key=rank_key)
    err = validate_spw_candidate(best_overall)
    print(f"\n[SPW PICK] WARNING: no SPW matched guessed band, falling back to best overall")
    print(f"[SPW PICK] selected fallback: {fmt(best_overall)}")
    return str(best_overall["spw"]), err, best_overall


def compute_time_on_source_min(ms_path: str | Path, field_name: str) -> float:
    fid = field_id_for_name(ms_path, field_name)
    if fid is None:
        return float("nan")

    tb = table()
    tb.open(str(ms_path))
    field_ids = tb.getcol("FIELD_ID")
    times = tb.getcol("TIME")
    intervals = tb.getcol("INTERVAL")
    tb.close()

    m = (field_ids == fid)
    if not np.any(m):
        return float("nan")

    t = times[m]
    dt = intervals[m]

    t_round = np.round(t).astype(np.int64)
    _, first_idx = np.unique(t_round, return_index=True)
    total_s = float(np.sum(dt[first_idx]))
    return total_s / 60.0


# -------------------------
# Extract / verify / image
# -------------------------
def ensure_out_paths(extracted_dir: Path, folder_name: str) -> Tuple[Path, Path]:
    out_top = extracted_dir / folder_name
    out_leaf = out_top / folder_name
    out_top.mkdir(parents=True, exist_ok=True)
    out_leaf.mkdir(parents=True, exist_ok=True)
    return out_top, out_leaf


def extract_ms(src_ms: Path, out_ms: Path, *, field: str, spw: str, prefer_corrected: bool) -> None:
    datacol = "corrected" if prefer_corrected else "data"
    print(f"[EXTRACT] split field={field} spw={spw} datacolumn={datacol}")

    if out_ms.exists():
        print(f"[EXTRACT] removing existing {out_ms}")
        shutil.rmtree(out_ms)

    split(
        vis=str(src_ms),
        outputvis=str(out_ms),
        field=str(field),
        spw=str(spw),
        datacolumn=datacol,
        keepflags=True,
    )


def write_listobs(ms_path: Path, out_txt: Path) -> None:
    print(f"[VERIFY] listobs -> {out_txt}")
    listobs(vis=str(ms_path), listfile=str(out_txt), overwrite=True)


def image_ms(ms_path: Path, out_top: Path, *, field: str, spw: str, prefer_corrected: bool, suffix=None, niter:int=100) -> None:
    datacol = "corrected" if prefer_corrected else "data"
    imbase = str(out_top / "clean")
    if suffix is not None:
        imbase += suffix

    cfg = dict(TCLEAN_CFG)
    cfg["field"] = field
    cfg["spw"] = spw
    cfg["datacolumn"] = datacol

    make_clean(str(ms_path), imbase, cfg, niter=niter)
    casa_image_to_png(imbase + ".image", str(out_top / f"{imbase}.png"))


# -------------------------
# Download stage
# -------------------------
def ensure_download_present(df: pd.DataFrame, idx: int, csv_path: str | Path) -> Optional[Path]:
    row = df.loc[idx].to_dict()

    name = str(row.get("name") or "").strip()
    wget_cmd = str(row.get("wget_command") or "").strip()
    folder_name = str(row.get("folder") or "").strip()
    expected_size = row.get("size")

    if not folder_name:
        folder_name = "".join(ch if (ch.isalnum() or ch in "_-.+") else "_" for ch in name)
        set_row(df, idx, folder=folder_name)
        save_projects(df, csv_path)

    download_root = DOWNLOAD_DIR / folder_name
    download_root.mkdir(parents=True, exist_ok=True)

    rc, msg = run_wget(wget_cmd, download_root)

    ok, actual_size, note = download_folder_ok(download_root, expected_size)
    print(f"[WGET CHECK] ok={ok} | {note}")

    if not ok:
        set_row(
            df,
            idx,
            status="error",
            wget_error=f"wget rc={rc}; {note}\n{msg[:3000]}",
            downloaded_folder=str(download_root),
        )
        save_projects(df, csv_path)
        return None

    warning_text = "" if rc == 0 else f"wget rc={rc}; accepted because {note}"

    set_row(
        df,
        idx,
        status="downloaded",
        wget_error=warning_text,
        downloaded_folder=str(download_root),
    )
    save_projects(df, csv_path)

    return download_root

def store_spw_info(df: pd.DataFrame, idx: int, spw: str, info: dict, csv_path: str | Path) -> None:
    set_row(
        df,
        idx,
        spw_selected=str(spw),
        spw_center_ghz=float(info["f_center"] / 1e9),
        spw_bandwidth_mhz=float(info["bandwidth"] / 1e6),
        spw_unflagged_frac=float(info["unflagged_frac"]) if np.isfinite(info["unflagged_frac"]) else np.nan,
        spw_n_rows=float(info["n_rows"]),
        spw_matches_band=str(bool(info.get("matches_band", False))),
    )
    save_projects(df, csv_path)

# -------------------------
# Continue after download
# -------------------------
def continue_after_download(df: pd.DataFrame, idx: int, csv_path: str | Path) -> None:
    row = df.loc[idx].to_dict()

    name = str(row.get("name") or "").strip()
    band_guess = row.get("band_guess")
    gain_field = str(row.get("gain_calibrator_name") or "").strip()
    folder_name = str(row.get("folder") or "").strip()

    if not gain_field:
        print(f"[MISS] {name}: missing gain_calibrator_name in CSV")
        set_row(df, idx, status="error", wget_error="missing gain_calibrator_name")
        save_projects(df, csv_path)
        return

    if not folder_name:
        folder_name = "".join(ch if (ch.isalnum() or ch in "_-.+") else "_" for ch in name)
        set_row(df, idx, folder=folder_name)
        save_projects(df, csv_path)

    download_root = DOWNLOAD_DIR / folder_name
    if not download_root.exists():
        set_row(df, idx, status="error", wget_error="downloaded status but download folder does not exist")
        save_projects(df, csv_path)
        return

    ms_src = find_first_ms(download_root)
    if ms_src is None:
        set_row(
            df,
            idx,
            status="error",
            wget_error="download looks present but no .ms found in download folder",
            downloaded_folder=str(download_root),
        )
        save_projects(df, csv_path)
        return

    prefer_corrected = ms_has_corrected(ms_src)
    caltabs = find_caltables(download_root)

    if not prefer_corrected:
        set_row(
            df,
            idx,
            status="error",
            wget_error="MS doesn't have CORRECTED_DATA column",
            downloaded_folder=str(download_root),
        )
        save_projects(df, csv_path)
        return

    out_top, out_leaf = ensure_out_paths(EXTRACTED_DIR, folder_name)
    ms_out = out_leaf / f"{folder_name}.ms"

    set_row(df, idx, status="extracting", extracted_ms=str(ms_out))
    save_projects(df, csv_path)

    spw, spw_err_msg, info = pick_one_spw_by_band(ms_src, str(band_guess) if band_guess is not None else None)
    
    if spw_err_msg is not None:
        set_row(
            df,
            idx,
            status="error",
            wget_error=f"[SPW] {spw_err_msg}",
            downloaded_folder=str(download_root),
        )
        save_projects(df, csv_path)
        return

    store_spw_info(df, idx, spw, info, csv_path)

    extract_ms(ms_src, ms_out, field=gain_field, spw=spw, prefer_corrected=prefer_corrected)

    new_size = dir_size_gb(ms_out)

    set_row(
        df,
        idx,
        status="verificating",
        new_size=float(new_size),
        spw_selected=str(spw),
        corrected_used=float(1.0 if prefer_corrected else 0.0),
        caltables_found=float(len(caltabs)),
    )
    save_projects(df, csv_path)

    listobs_path = out_top / "listobs.txt"
    write_listobs(ms_out, listobs_path)

    tos = compute_time_on_source_min(ms_out, gain_field)
    set_row(df, idx, extracted_gain_onsource_min=float(tos))
    save_projects(df, csv_path)

    set_row(df, idx, status="imaging")
    save_projects(df, csv_path)

    # This is not meaningful
    # image_ms(ms_src, out_top, field="", spw="", prefer_corrected=prefer_corrected, suffix='src_no_filt')

    # These r all equal
    # image_ms(ms_src, out_top, field=gain_field, spw=spw, prefer_corrected=prefer_corrected, suffix='src_filt')
    # image_ms(ms_out, out_top, field="", spw="", prefer_corrected=prefer_corrected, suffix='ext_no_filt')
    
    image_ms(ms_out, out_top, field="", spw="", prefer_corrected=True, suffix='_corrected')
    image_ms(ms_out, out_top, field="", spw="", prefer_corrected=False, suffix='_data')
    
    image_ms(ms_out, out_top, field="", spw="", prefer_corrected=True, suffix='_corrected_dirty', niter=0)


    

    set_row(df, idx, status="done")
    save_projects(df, csv_path)

    print(f"[DONE] {name} | extracted={ms_out} | new_size={new_size:.3f} GB | spw={spw} | tos~{tos:.2f} min")


# -------------------------
# Pipeline for one row
# -------------------------
def process_one(df: pd.DataFrame, idx: int, csv_path: str | Path) -> None:
    row = df.loc[idx].to_dict()

    name = str(row.get("name") or "").strip()
    wget_cmd = str(row.get("wget_command") or "").strip()
    status = str(row.get(STATE_COL) or "").strip().lower()

    if not name or not wget_cmd:
        print(f"[SKIP] idx={idx} missing name/wget_command")
        return

    # case 1: already downloaded -> continue from there
    if status == "downloaded":
        print(f"[RESUME] continuing downloaded row: {name}")
        continue_after_download(df, idx, csv_path)
        return

    # case 2: empty status -> start new download, then continue
    if status == "":
        print(f"[START] new row: {name}")
        set_row(df, idx, status="downloading")
        save_projects(df, csv_path)

        download_root = ensure_download_present(df, idx, csv_path)
        if download_root is None:
            return

        continue_after_download(df, idx, csv_path)
        return

    print(f"[SKIP] idx={idx} name={name} status={status}")


# -------------------------
# Main loop
# -------------------------
def main(project_list: str | Path):
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

    df = load_projects(project_list)

    wgetc = df["wget_command"].astype("string").fillna("").str.strip()
    status = df[STATE_COL].astype("string").fillna("").str.strip().str.lower()

    # prioritize rows already downloaded
    todo_downloaded = df.index[
        wgetc.ne("") & status.eq("downloaded")
    ].tolist()

    # then brand new rows
    todo_empty = df.index[
        wgetc.ne("") & status.eq("")
    ].tolist()

    todo = todo_downloaded + todo_empty

    print(f"[INFO] total rows={len(df)}")
    print(f"[INFO] resume downloaded={len(todo_downloaded)}")
    print(f"[INFO] start fresh={len(todo_empty)}")
    print(f"[INFO] total todo={len(todo)}")

    if not todo:
        return

    for idx in todo:
        print("\n" + "=" * 80)
        print(f"[PROCESS] idx={idx} name={df.at[idx, 'name']}")
        print("=" * 80)
        try:
            process_one(df, idx, project_list)
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            print(f"[ERROR] idx={idx} {msg}")
            set_row(df, idx, status="error", wget_error=msg[:4000])
            save_projects(df, project_list)


def run():
    main(PROJECT_LIST)


# if __name__ == "__main__":
#     main(PROJECT_LIST)