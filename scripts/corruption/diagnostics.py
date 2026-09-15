"""Diagnostic plots for configured and applied corruption functions."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def corrfun_plot_start():
    import matplotlib.pyplot as plt

    figure, (phase_axis, amplitude_axis) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, constrained_layout=True
    )
    phase_axis.set_ylabel("Phase (deg)")
    amplitude_axis.set_ylabel("Amplitude")
    amplitude_axis.set_xlabel("TIME (CASA seconds)")
    return figure, phase_axis, amplitude_axis


def corrfun_plot_add(
    ax0,
    ax1,
    *,
    amp_fn,
    phase_fn,
    centers,
    amp_eff,
    phase_eff,
    t,
    gain=None,
    label,
):
    t = np.asarray(t, dtype=float)
    centers = np.asarray(centers, dtype=float)
    theoretical_times = np.linspace(float(t.min()), float(t.max()), 300)
    phase_theoretical = (
        np.zeros_like(theoretical_times, dtype=float)
        if phase_fn is None
        else phase_fn.eval(theoretical_times)
    )
    amplitude_theoretical = (
        np.ones_like(theoretical_times, dtype=float)
        if amp_fn is None
        else amp_fn.eval(theoretical_times)
    )
    color = ax0._get_lines.get_next_color()

    ax0.plot(
        centers,
        np.rad2deg(phase_eff),
        linestyle="-",
        marker="o",
        markersize=3,
        linewidth=1.2,
        color=color,
        label=f"{label} sampled",
    )
    ax0.plot(
        theoretical_times,
        np.rad2deg(phase_theoretical),
        linestyle="--",
        linewidth=1.0,
        color=color,
        label=f"{label} function",
    )
    ax1.plot(
        centers,
        amp_eff,
        linestyle="-",
        marker="o",
        markersize=3,
        linewidth=1.2,
        color=color,
        label=f"{label} sampled",
    )
    ax1.plot(
        theoretical_times,
        amplitude_theoretical,
        linestyle="--",
        linewidth=1.0,
        color=color,
        label=f"{label} function",
    )

    if gain is not None:
        gain = np.asarray(gain)
        if gain.shape[0] != t.shape[0]:
            raise ValueError(
                f"gain must have same length as t. got gain={gain.shape}, t={t.shape}"
            )
        ax0.plot(
            t,
            np.rad2deg(np.angle(gain)),
            linestyle="None",
            marker="x",
            markersize=4,
            alpha=0.9,
            color=color,
            label=f"{label} corruption",
        )
        ax1.plot(
            t,
            np.abs(gain),
            linestyle="None",
            marker="x",
            markersize=4,
            alpha=0.9,
            color=color,
            label=f"{label} corruption",
        )


def corrfun_plot_finish(fig, ax0, ax1, path="images/corruption_function.png"):
    import matplotlib.pyplot as plt

    ax0.legend(fontsize=8, ncol=2)
    ax1.legend(fontsize=8, ncol=2)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=200)
    plt.close(fig)


__all__ = ["corrfun_plot_add", "corrfun_plot_finish", "corrfun_plot_start"]
