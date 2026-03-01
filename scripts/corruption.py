from __future__ import annotations
import numpy as np
from casatools import table, simulator
from .corrfn import CorrFn
from .corrtab_utils import make_template_gain_corrtab, GTab, GTabQuery, GCOLS
import matplotlib.pyplot as plt
from .time_utils import mjd_seconds_to_iso
from .plot_utils import corrfun_plot_add, corrfun_plot_finish, corrfun_plot_start

class Corruption:
    def build_corrtable(self, ms: str, corrtab: str, *, seed: int = 0):
        raise NotImplementedError

class GainCorruption(Corruption):
    pass


class AntennaGainCorruption(GainCorruption):
    def __init__(
        self,
        timegrid,
        amp_fn: CorrFn | None = None,
        phase_fn: CorrFn | None = None,
        query: GTabQuery | None = None,
    ):
        self.tg = timegrid
        self.amp_fn = amp_fn
        self.phase_fn = phase_fn
        self.query = query

    def build_corrtable(self, ms: str, corrtab: str, *, seed: int = 0):
        make_template_gain_corrtab(ms, corrtab, seed=seed)

        tb = table()
        tb.open(corrtab, nomodify=False)
        try:
            # Load table once
            gtab0 = GTab.from_casa_table(tb)

            t0_global = float(gtab0.TIME.min())

            CP = np.asarray(tb.getcol("CPARAM"))  # (nCorr, nChan, nRow)
            CP_new = CP.copy()

            q = (self.query or GTabQuery()).sort_by([GCOLS.TIME])

            result = q.apply(gtab0)

            if isinstance(result, dict):
                groups = list(result.items())  # (key, GTab)
            else:
                groups = [(None, result)]      # single group


            fig, ax0, ax1 = corrfun_plot_start()

            for group_key, gtab in groups:

                print(f"Grup: {group_key}")

                if gtab.nrow == 0:
                    continue

                # per-group RNG: stable but different across groups
                # (hash() is salted per process, so don't use it!)
                key_bytes = repr(group_key).encode("utf-8")
                key_mix = int(np.frombuffer(key_bytes, dtype=np.uint8).sum())  # simple deterministic mix
                rng = np.random.default_rng(seed + key_mix)

                # Times for this group (already sorted if query sorted)
                t = gtab.TIME
                rowids = gtab.ROWID  # indices into CP_new last dimension

                # Build time grid for this group
                bin_ids, centers = self.tg.get_times(t, t0=t0_global)
                unique_bins = np.unique(bin_ids)

                # Evaluate at centers
                if self.amp_fn is None:
                    amp_c = np.ones_like(centers, dtype=float)
                else:
                    amp_c = self.amp_fn.sample(rng).eval(centers)

                if self.phase_fn is None:
                    phase_c = np.zeros_like(centers, dtype=float)
                else:
                    phase_c = self.phase_fn.sample(rng).eval(centers)

                if amp_c.shape != centers.shape or phase_c.shape != centers.shape:
                    raise ValueError(
                        f"amp/phase must be shape {centers.shape}, got {amp_c.shape}, {phase_c.shape}"
                    )

                gain_centers = amp_c * np.exp(1j * phase_c)

                # Map each row time -> its bin center
                # Map each row time -> gain according to TimeGrid interp
                if self.tg.dt == "int":
                    # no binning: centers are the row times, so just use 1:1
                    gain = gain_centers
                else:
                    unique_bins = np.unique(bin_ids)

                    if self.tg.interp == "nearest":
                        pos = np.searchsorted(unique_bins, bin_ids)
                        gain = gain_centers[pos]  # piecewise constant

                    elif self.tg.interp == "linear":
                        amp_interp = np.interp(t, centers, amp_c)
                        phase_interp = np.interp(t, centers, np.unwrap(phase_c))
                        gain = amp_interp * np.exp(1j * phase_interp)

                    else:
                        raise ValueError(f"Unsupported TimeGrid.interp='{self.tg.interp}'. Use 'linear' or 'nearest'.")

                # Write back into global CPARAM
                CP_new[:, :, rowids] = gain[None, None, :]

                label = str(group_key) if group_key is not None else "all"
                corrfun_plot_add(
                    ax0, ax1,
                    amp_fn=self.amp_fn,
                    phase_fn=self.phase_fn,
                    centers=centers,
                    amp_eff=amp_c,
                    phase_eff=phase_c,
                    t=t,
                    gain=gain,
                    label=label,
                )
            
            corrfun_plot_finish(fig, ax0, ax1, "images/corruption_function.png")
            
            tb.putcol("CPARAM", CP_new)
            tb.flush()
        finally:
            tb.close()

        return self

    
    def apply_corrtable(self, ms: str, corrtab: str, seed: int = 0):
        sm = simulator()
        sm.openfromms(ms)
        sm.setseed(seed)

        sm.setapply(
            table=corrtab,
            type="G",
            # field=GAINCAL_FIELD,
            interp="linear",
            calwt=False,
        )

        sm.corrupt()
        sm.done()

        return self
