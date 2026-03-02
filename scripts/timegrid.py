from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence, Literal
import numpy as np
from .time_utils import mjd_to_iso

@dataclass
class TimeGrid:
    solint: str | int = "int"  # can become int (seconds)
    interp: Literal["linear"] = "linear" # idk if there could be another one that makes sense

    def __post_init__(self):
        if self.solint == "int":
            self.dt = "int"
            return

        if isinstance(self.solint, int): #assume seconds
            self.dt = float(self.solint)
            return

        value = self.solint.strip().lower()

        if value.endswith("s"):
            seconds = int(value[:-1])
        elif value.endswith("m"):
            seconds = int(value[:-1]) * 60
        else:
            raise ValueError(
                f"Invalid solint '{self.solint}'. Use 'int', '#s', or '#m'."
            )

        self.dt = float(seconds)
    
    def get_times(self, times: np.ndarray, *, t0: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        times : (N,) array
            Time stamps (seconds)
        t0 : float
            Reference time for binning (usually min(times))

        Returns
        -------
        bin_id_per_time : (N,) int array
            Bin index for each time sample
        unique_bin_centers : (nBins,) float array
            Representative time for each bin
        """

        times = np.asarray(times, dtype=float)

        # No binning → one bin per integration
        if self.dt == "int":
            n = times.shape[0]
            bin_ids = np.arange(n, dtype=np.int64)
            centers = times.copy()
            return bin_ids, centers

        dt = float(self.dt)

        if dt <= 0:
            raise ValueError(f"dt must be > 0, got {dt}")

        # Compute bin index for each time
        bin_ids = np.floor((times - t0) / dt).astype(np.int64)

        # Unique bins (sorted)
        unique_bins = np.unique(bin_ids)

        # Bin centers
        # centers = t0 + (unique_bins.astype(float) + 0.5) * dt

        # LEFT BOUND
        centers = t0 + unique_bins.astype(float) * dt   # left edge / bin start
        
        # add one last bin so we make sure to cover last observartion
        last = unique_bins.max()
        centers = np.concatenate([centers, [t0 + (last + 1.0) * dt]])
        
        # print(f"CENTERS: {mjd_to_iso(centers/86400.0)}")
        # print(f"START:{mjd_to_iso(t0/86400.0)}")

        return bin_ids, centers

    def full_grid(self, t0: float, tf: float) -> np.ndarray:
        """
        Full uniform grid from t0 up to tf, PLUS one extra knot at tf+dt
        (aligned to t0), so the last interval always exists for interpolation.
        """
        if self.dt == "int":
            raise ValueError("full_grid requires numeric solint (e.g. '5s'), not solint='int'.")

        dt = float(self.dt)
        t0 = float(t0)
        tf = float(tf)

        if tf < t0:
            raise ValueError(f"tf must be >= t0, got {t0}..{tf}")

        # bins covering up to tf
        n = int(np.floor((tf - t0) / dt))

        # include 0..n plus one extra (n+1)
        grid = t0 + np.arange(n + 2, dtype=np.int64) * dt
        return grid