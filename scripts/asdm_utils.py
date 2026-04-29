from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET
import shutil

try:
    from casatasks import importasdm
    from casatools import table
    CASA_AVAILABLE = True
except Exception:
    importasdm = None
    table = None
    CASA_AVAILABLE = False


ASDM_XML_MARKERS = {
    "ASDM.xml",
    "Main.xml",
    "Antenna.xml",
    "Station.xml",
    "SpectralWindow.xml",
    "Scan.xml",
    "ConfigDescription.xml",
    "DataDescription.xml",
    "Field.xml",
    "Source.xml",
    "CalData.xml",
    "CalReduction.xml",
    "CalAntennaSolutions.xml",
}


@dataclass
class ASDMInspection:
    root: Path
    is_asdm_like: bool
    marker_files_found: list[str]
    cal_xml_files_found: list[str]
    has_asdm_xml: bool
    has_main_xml: bool
    has_calibration_xml: bool
    calibrated: Optional[bool]
    exec_block_id: Optional[str]
    message: str


def _xml_names_direct(folder: Path) -> set[str]:
    if not folder.exists() or not folder.is_dir():
        return set()
    return {p.name for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".xml"}


def looks_like_asdm_dir(folder: str | Path, *, min_marker_hits: int = 3) -> bool:
    """
    True only if THIS folder itself contains ASDM marker XML files.
    """
    folder = Path(folder)
    names = _xml_names_direct(folder)
    hits = sum(1 for name in ASDM_XML_MARKERS if name in names)
    return hits >= min_marker_hits and "ASDM.xml" in names


def find_first_asdm(root: str | Path, *, min_marker_hits: int = 3) -> Optional[Path]:
    """
    Search for the actual ASDM directory, i.e. the directory that directly
    contains ASDM.xml and other marker XML files.
    """
    root = Path(root)
    if not root.exists():
        return None

    candidates = []

    # include root itself plus all descendants
    for d in [root] + [p for p in root.rglob("*") if p.is_dir()]:
        names = _xml_names_direct(d)
        hits = sum(1 for name in ASDM_XML_MARKERS if name in names)
        if hits >= min_marker_hits and "ASDM.xml" in names:
            candidates.append((hits, d))

    if not candidates:
        return None

    # prefer the directory with the most marker hits
    candidates.sort(key=lambda x: (-x[0], len(str(x[1]))))
    return candidates[0][1]


def _safe_text(elem):
    if elem is None or elem.text is None:
        return None
    s = elem.text.strip()
    return s or None


def _find_first_text_anywhere(root: ET.Element, candidate_tags: list[str]) -> Optional[str]:
    candidate_tags = [x.lower() for x in candidate_tags]
    for elem in root.iter():
        tag = elem.tag.split("}")[-1].lower()
        if tag in candidate_tags:
            txt = _safe_text(elem)
            if txt is not None:
                return txt
    return None


def _parse_asdm_xml(asdm_xml: Path) -> tuple[Optional[bool], Optional[str], str]:
    try:
        tree = ET.parse(asdm_xml)
        root = tree.getroot()
    except Exception as e:
        return None, None, f"failed to parse ASDM.xml: {type(e).__name__}: {e}"

    cal_text = _find_first_text_anywhere(
        root,
        ["calStatus", "calibrationStatus", "processingStatus", "status"],
    )

    exec_block_id = _find_first_text_anywhere(
        root,
        ["execBlockId", "execblockid", "uid", "entityId"],
    )

    calibrated = None
    if cal_text is not None:
        ct = cal_text.strip().lower()
        if "calibrated" in ct:
            calibrated = True
        elif "uncalibrated" in ct or "raw" in ct:
            calibrated = False

    return calibrated, exec_block_id, "parsed ASDM.xml"


def inspect_asdm(folder: str | Path) -> ASDMInspection:
    folder = Path(folder)

    if not folder.exists() or not folder.is_dir():
        return ASDMInspection(
            root=folder,
            is_asdm_like=False,
            marker_files_found=[],
            cal_xml_files_found=[],
            has_asdm_xml=False,
            has_main_xml=False,
            has_calibration_xml=False,
            calibrated=None,
            exec_block_id=None,
            message="folder does not exist or is not a directory",
        )

    names = sorted(_xml_names_direct(folder))
    marker_files_found = [x for x in names if x in ASDM_XML_MARKERS]
    cal_xml_files_found = [x for x in names if x.lower().startswith("cal")]

    has_asdm_xml = "ASDM.xml" in names
    has_main_xml = "Main.xml" in names
    has_calibration_xml = len(cal_xml_files_found) > 0
    is_asdm_like = looks_like_asdm_dir(folder)

    calibrated = None
    exec_block_id = None
    message = "marker-based ASDM detection only"

    if has_asdm_xml:
        calibrated, exec_block_id, message = _parse_asdm_xml(folder / "ASDM.xml")

    return ASDMInspection(
        root=folder,
        is_asdm_like=is_asdm_like,
        marker_files_found=marker_files_found,
        cal_xml_files_found=cal_xml_files_found,
        has_asdm_xml=has_asdm_xml,
        has_main_xml=has_main_xml,
        has_calibration_xml=has_calibration_xml,
        calibrated=calibrated,
        exec_block_id=exec_block_id,
        message=message,
    )


def ms_has_corrected(ms_path: str | Path) -> bool:
    if not CASA_AVAILABLE:
        raise RuntimeError("CASA is not available")

    tb = table()
    try:
        tb.open(str(ms_path))
        cols = set(tb.colnames())
    finally:
        tb.close()
    return "CORRECTED_DATA" in cols


def import_asdm_to_ms(
    asdm_dir: str | Path,
    out_ms: str | Path,
    *,
    overwrite: bool = True,
    with_pointing_correction: bool = False,
    ocorr_mode: str = "co",
    asis: str = "",
) -> Path:
    if not CASA_AVAILABLE:
        raise RuntimeError("CASA is not available")

    asdm_dir = Path(asdm_dir)
    out_ms = Path(out_ms)

    if not asdm_dir.exists():
        raise FileNotFoundError(f"ASDM folder not found: {asdm_dir}")

    if not looks_like_asdm_dir(asdm_dir):
        raise ValueError(f"Folder does not look like a direct ASDM directory: {asdm_dir}")

    if out_ms.exists():
        if overwrite:
            shutil.rmtree(out_ms)
        else:
            raise FileExistsError(f"Output MS already exists: {out_ms}")

    importasdm(
        asdm=str(asdm_dir),
        vis=str(out_ms),
        overwrite=overwrite,
        with_pointing_correction=with_pointing_correction,
        ocorr_mode=ocorr_mode,
        asis=asis,
    )

    if not out_ms.exists():
        raise RuntimeError(f"importasdm finished but output MS was not created: {out_ms}")

    return out_ms

def main():
    root = Path("/Users/u1528314/repos/radioastro-ml/collect/downloads/1714-252")

    asdm_dir = find_first_asdm(root)
    print("FOUND ASDM DIR:", asdm_dir)

    info = inspect_asdm(asdm_dir)
    print(info)

    ms_path = import_asdm_to_ms(asdm_dir, "test_import.ms", overwrite=True)
    print("MS:", ms_path)
    print("HAS CORRECTED:", ms_has_corrected(ms_path))
