"""Current antenna-gain corruption interface and implementation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .diagnostics import corrfun_plot_add, corrfun_plot_finish, corrfun_plot_start
from .tables import GCOLS, GTab, GTabQuery, make_template_gain_corrtab

if TYPE_CHECKING:
    from .functions import CorrFn
    from .metrics import ConstantGainParameters
else:
    CorrFn = Any


def _child_report(value: object | None, *, unit: str) -> dict[str, object] | None:
    if value is None:
        return None
    method = getattr(value, "to_report_dict", None)
    if not callable(method):
        raise TypeError(
            f"{type(value).__name__} must implement to_report_dict() to be reported"
        )
    report = method()
    if not isinstance(report, dict):
        raise TypeError(
            f"{type(value).__name__}.to_report_dict() must return a dictionary"
        )
    result = dict(report)
    result.setdefault("unit", unit)
    return result


class Corruption:
    def build_corrtable(
        self,
        ms: str,
        corrtab: str,
        *,
        seed: int = 0,
        diagnostic_plot: str | Path = "images/corruption_function.png",
    ):
        raise NotImplementedError

    def to_report_dict(self) -> dict[str, object]:
        raise NotImplementedError

    def to_report_text(self) -> str:
        return repr(self)


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
        self.metrics = None

    @classmethod
    def from_detectability(
        cls,
        timegrid,
        parameters: "ConstantGainParameters",
    ) -> "AntennaGainCorruption":
        """Construct a one-antenna constant gain at a requested detectability."""
        from .functions import Constant
        from .metrics import ConstantGainParameters

        if not isinstance(parameters, ConstantGainParameters):
            raise TypeError("parameters must be ConstantGainParameters")
        constant = Constant.from_detectability(parameters)
        query = GTabQuery().where_eq(
            GCOLS.ANTENNA1, parameters.antenna_id
        ).group_by([GCOLS.ANTENNA1])
        if parameters.corruption_type == "amp":
            result = cls(timegrid, amp_fn=constant, query=query)
        else:
            result = cls(timegrid, phase_fn=constant, query=query)
        result.metrics = constant.metrics
        return result

    def __repr__(self) -> str:
        return (
            "AntennaGainCorruption("
            f"timegrid={self.tg!r}, amp_fn={self.amp_fn!r}, "
            f"phase_fn={self.phase_fn!r}, query={self.query!r})"
        )

    def to_report_dict(self) -> dict[str, object]:
        time_grid_method = getattr(self.tg, "to_report_dict", None)
        if not callable(time_grid_method):
            raise TypeError(
                f"{type(self.tg).__name__} must implement to_report_dict() to be reported"
            )
        time_grid = time_grid_method()
        if not isinstance(time_grid, dict):
            raise TypeError(
                f"{type(self.tg).__name__}.to_report_dict() must return a dictionary"
            )

        if self.query is None:
            selection = None
        else:
            selection_method = getattr(self.query, "to_report_dict", None)
            if not callable(selection_method):
                raise TypeError(
                    f"{type(self.query).__name__} must implement to_report_dict() to be reported"
                )
            selection = selection_method()
            if not isinstance(selection, dict):
                raise TypeError(
                    f"{type(self.query).__name__}.to_report_dict() must return a dictionary"
                )

        return {
            "type": "antenna_gain",
            "time_grid": dict(time_grid),
            "selection": None if selection is None else dict(selection),
            "amplitude": _child_report(
                self.amp_fn, unit="dimensionless_gain_magnitude"
            ),
            "phase": _child_report(self.phase_fn, unit="radian"),
        }

    def to_report_text(self) -> str:
        return (
            "AntennaGainCorruption(\n"
            f"  timegrid={self.tg!r},\n"
            f"  amp_fn={self.amp_fn!r},\n"
            f"  phase_fn={self.phase_fn!r},\n"
            f"  query={self.query!r}\n"
            ")"
        )

    def build_corrtable(
        self,
        ms: str,
        corrtab: str,
        *,
        seed: int = 0,
        diagnostic_plot: str | Path = "images/corruption_function.png",
    ):
        try:
            from casatools import table
        except ImportError as exc:
            raise RuntimeError("CASA casatools is required to build corruption tables") from exc

        make_template_gain_corrtab(ms, corrtab, seed=seed)
        tb = table()
        tb.open(corrtab, nomodify=False)
        try:
            gtab0 = GTab.from_casa_table(tb)
            t0_global = float(gtab0.TIME.min())
            tf_global = float(gtab0.TIME.max())
            centers = self.tg.full_grid(t0_global, tf_global)
            current_parameters = np.asarray(tb.getcol("CPARAM"))
            new_parameters = current_parameters.copy()
            query = (self.query or GTabQuery()).sort_by([GCOLS.TIME])
            selected = query.apply(gtab0)
            groups = list(selected.items()) if isinstance(selected, dict) else [(None, selected)]

            figure, phase_axis, amplitude_axis = corrfun_plot_start()
            for group_key, gtab in groups:
                print(f"Grup: {group_key}")
                if gtab.nrow == 0:
                    continue

                key_bytes = repr(group_key).encode("utf-8")
                key_mix = int(np.frombuffer(key_bytes, dtype=np.uint8).sum())
                rng = np.random.default_rng(seed + key_mix)
                times = gtab.TIME
                row_ids = gtab.ROWID

                if self.amp_fn is None:
                    amplitude_centers = np.ones_like(centers, dtype=float)
                else:
                    amplitude_centers = self.amp_fn.sample(
                        rng, times=centers
                    ).eval(centers)
                if self.phase_fn is None:
                    phase_centers = np.zeros_like(centers, dtype=float)
                else:
                    phase_centers = self.phase_fn.sample(
                        rng, times=centers
                    ).eval(centers)

                if amplitude_centers.shape != centers.shape or phase_centers.shape != centers.shape:
                    raise ValueError(
                        f"amp/phase must be shape {centers.shape}, got "
                        f"{amplitude_centers.shape}, {phase_centers.shape}"
                    )
                gain_centers = amplitude_centers * np.exp(1j * phase_centers)

                if self.tg.dt == "int":
                    if self.amp_fn is None:
                        amplitude_rows = np.ones_like(times, dtype=float)
                    else:
                        amplitude_rows = self.amp_fn.sample(rng, times=times).eval(times)
                    if self.phase_fn is None:
                        phase_rows = np.zeros_like(times, dtype=float)
                    else:
                        phase_rows = self.phase_fn.sample(rng, times=times).eval(times)
                    gain = amplitude_rows * np.exp(1j * phase_rows)
                elif self.tg.interp == "linear":
                    amplitude_rows = np.interp(times, centers, amplitude_centers)
                    phase_rows = np.interp(times, centers, np.unwrap(phase_centers))
                    gain = amplitude_rows * np.exp(1j * phase_rows)
                elif self.tg.interp == "nearest":
                    indices = np.searchsorted(centers, times, side="left")
                    indices = np.clip(indices, 0, len(centers) - 1)
                    left = np.maximum(indices - 1, 0)
                    use_left = (times - centers[left]) <= (centers[indices] - times)
                    indices = np.where(use_left, left, indices)
                    gain = gain_centers[indices]
                else:
                    raise ValueError(
                        f"Unsupported TimeGrid.interp='{self.tg.interp}'. "
                        "Use 'linear' or 'nearest'."
                    )

                new_parameters[:, :, row_ids] = gain[None, None, :]
                label = str(group_key) if group_key is not None else "all"
                corrfun_plot_add(
                    phase_axis,
                    amplitude_axis,
                    amp_fn=self.amp_fn,
                    phase_fn=self.phase_fn,
                    centers=centers,
                    amp_eff=amplitude_centers,
                    phase_eff=phase_centers,
                    t=times,
                    gain=gain,
                    label=label,
                )

            corrfun_plot_finish(
                figure,
                phase_axis,
                amplitude_axis,
                diagnostic_plot,
            )
            tb.putcol("CPARAM", new_parameters)
            tb.flush()
        finally:
            tb.close()
        return self

    def apply_corrtable(self, ms: str, corrtab: str, seed: int = 0):
        try:
            from casatools import simulator
        except ImportError as exc:
            raise RuntimeError("CASA casatools is required to apply corruption tables") from exc

        sm = simulator()
        sm.openfromms(ms)
        sm.setseed(seed)
        sm.setapply(table=corrtab, type="G", interp="linear", calwt=False)
        sm.corrupt()
        sm.done()
        return self


__all__ = ["AntennaGainCorruption", "Corruption", "GainCorruption"]
