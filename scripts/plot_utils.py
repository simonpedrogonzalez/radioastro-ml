import numpy as np
import matplotlib.pyplot as plt

def corrfun_plot_start():
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    ax0.set_ylabel("Phase (deg)")
    ax1.set_ylabel("Amplitude")
    ax1.set_xlabel("TIME (CASA seconds)")
    return fig, ax0, ax1


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
    tt = np.linspace(float(t.min()), float(t.max()), 300)

    # Theoretical (continuous)
    phase_th = np.zeros_like(tt, dtype=float) if phase_fn is None else phase_fn.eval(tt)
    amp_th   = np.ones_like(tt, dtype=float)  if amp_fn   is None else amp_fn.eval(tt)

    # Choose one color per group explicitly (avoid cycler surprises)
    color = ax0._get_lines.get_next_color()

    # Phase: sampled points + function curve
    ax0.plot(
        centers, np.rad2deg(phase_eff),
        linestyle="-", marker="o", markersize=3, linewidth=1.2,
        color=color, label=f"{label} sampled"
    )
    ax0.plot(
        tt, np.rad2deg(phase_th),
        linestyle="--", linewidth=1.0,
        color=color, label=f"{label} function"
    )

    # Amp: sampled points + function curve
    ax1.plot(
        centers, amp_eff,
        linestyle="-", marker="o", markersize=3, linewidth=1.2,
        color=color, label=f"{label} sampled"
    )
    ax1.plot(
        tt, amp_th,
        linestyle="--", linewidth=1.0,
        color=color, label=f"{label} function"
    )

    # Row-level applied corruption (x at actual updated row times)
    if gain is not None:
        gain = np.asarray(gain)
        if gain.shape[0] != t.shape[0]:
            raise ValueError(f"gain must have same length as t. got gain={gain.shape}, t={t.shape}")

        phase_row = np.rad2deg(np.angle(gain))
        amp_row = np.abs(gain)

        ax0.plot(
            t, phase_row,
            linestyle="None", marker="x", markersize=4, alpha=0.9,
            color=color, label=f"{label} corruption"
        )
        ax1.plot(
            t, amp_row,
            linestyle="None", marker="x", markersize=4, alpha=0.9,
            color=color, label=f"{label} corruption"
        )

def corrfun_plot_finish(fig, ax0, ax1, path="images/corruption_function.png"):
    ax0.legend(fontsize=8, ncol=2)
    ax1.legend(fontsize=8, ncol=2)
    fig.savefig(path, dpi=200)
    plt.close(fig)
