from astropy.time import Time


def date_to_mjd(date_str: str) -> float:
    """'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS' -> MJD (UTC)."""
    return Time(date_str, format="iso", scale="utc").mjd


def mjd_to_iso(mjd: float) -> str:
    """MJD -> 'YYYY-MM-DD HH:MM:SS' (UTC)."""
    return Time(mjd, format="mjd", scale="utc").iso

def mjd_seconds_to_iso(mdj: float) -> str:
    return Time(mdj / 86400.0, format="mjd", scale="utc").iso