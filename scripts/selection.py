# Maybe for later


@dataclass(frozen=True)
class Selection:
    fields: Optional[Sequence[int]] = None
    spws:   Optional[Sequence[int]] = None
    chans:  Optional[Sequence[int]] = None
    corrs:  Optional[Sequence[int]] = None
    ants:   Optional[Sequence[int]] = None
    scans:  Optional[Sequence[int]] = None
    tmin: Optional[str] = None  # "yyyy-mm-ddThh:mm:ss"
    tmax: Optional[str] = None  # "same"

    def __post_init__(self):
        # Validate / normalize time bounds if present
        if (self.tmin is None) ^ (self.tmax is None):
            raise ValueError("Provide both tmin and tmax, or neither.")

        if self.tmin is not None and self.tmax is not None:
            t0 = _parse_iso_to_casa_seconds(self.tmin)
            t1 = _parse_iso_to_casa_seconds(self.tmax)
            if t0 > t1:
                raise ValueError(f"tmin > tmax: {self.tmin} > {self.tmax}")

        # Validate indices are ints (light check)
        for name in ["fields", "spws", "chans", "corrs", "ants", "scans"]:
            seq = getattr(self, name)
            if seq is not None:
                _ = [int(v) for v in seq]  # raises if not int-like

    def time_bounds_seconds(self) -> tuple[Optional[float], Optional[float]]:
        if self.tmin is None:
            return None, None
        return _parse_iso_to_casa_seconds(self.tmin), _parse_iso_to_casa_seconds(self.tmax)

    def row_mask(self, tb: table) -> np.ndarray:
        """
        Return boolean mask over rows in an *opened* CASA table `tb` (main table),
        selecting rows matching the filters that correspond to columns present.

        Columns are used only if they exist in tb.colnames().
        """
        colnames = set(tb.colnames())

        # Determine number of rows
        nrow = tb.nrows()
        m = np.ones(nrow, dtype=bool)

        def apply_in(col: str, allowed: Optional[Sequence[int]]):
            nonlocal m
            if allowed is None:
                return
            if col not in colnames:
                return
            vals = np.asarray(tb.getcol(col), dtype=int)
            m &= np.isin(vals, list(map(int, allowed)))

        def apply_time(col: str = "TIME"):
            nonlocal m
            if self.tmin is None or self.tmax is None:
                return
            if col not in colnames:
                return
            t = np.asarray(tb.getcol(col), dtype=float)
            t0, t1 = self.time_bounds_seconds()
            m &= (t >= t0) & (t <= t1)

        # These column names match typical gain caltables
        apply_in("FIELD_ID", self.fields)
        apply_in("SPECTRAL_WINDOW_ID", self.spws)
        apply_in("ANTENNA1", self.ants)
        apply_in("SCAN_NUMBER", self.scans)
        apply_time("TIME")

        # NOTE: chans/corrs are not row-level in caltables; they index CPARAM dimensions,
        # so we don't apply them here.
        return m

    def row_indices(self, tb: table) -> np.ndarray:
        """Convenience: return np.ndarray of row indices that pass row_mask()."""
        mask = self.row_mask(tb)
        return np.nonzero(mask)[0]