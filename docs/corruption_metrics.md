# Constant one-antenna corruption metrics

This document records the implemented metric contract. All corruption metrics
are evaluated before thermal noise is added.

## Visibility definitions

- `V` = exact noiseless, uncorrupted sky visibility predicted into the MS.
- `V_corr=C(V)` = visibility after applying the gain corruption.
- `Delta_V=V_corr-V`.
- `sigma` = requested per-visibility thermal-noise standard deviation in Jy.

The retained baseline is `V+n`; a retained corrupted variant is
`V_corr+n`. Every variant of one source uses the same noise request and the same
source-level random seed.

The L2 vectors include finite, unflagged, cross-correlation samples from the
parallel hands (RR, LL, XX, or YY). Autocorrelations, flagged samples, and
cross-hands are excluded.

## Metrics

\[
\epsilon_g=|g-1|
\]

`eps_g=|g-1|` is the magnitude of the antenna-gain displacement.

\[
\epsilon_{\mathrm{vis}}
=\frac{\lVert\Delta\mathbf V\rVert_2}{\lVert\mathbf V\rVert_2}
\]

`eps_vis=||Delta_V||_2/||V||_2` compares the corruption with the exact
noiseless sky signal. Thermal noise is not included in either norm.

\[
\mathrm{SNR}_{\mathrm{corr}}
=\left\lVert\frac{\Delta\mathbf V}{\boldsymbol\sigma}\right\rVert_2
\]

`SNR_corr=||Delta_V/sigma||_2` compares the corruption with the requested
thermal-noise scale. It is an aggregate visibility-space quantity, not an
image-domain artifact significance.

## Constant one-antenna solution

For antenna `k`, let `V_Ak` contain the valid visibility samples on baselines
involving that antenna. The implemented solver calculates:

\[
V_{L2}=\lVert\mathbf V\rVert_2,\qquad
V_{Ak,L2}=\lVert\mathbf V_{Ak}\rVert_2,\qquad
V_{Ak/\sigma,L2}=\left\lVert\frac{\mathbf V_{Ak}}{\boldsymbol\sigma}\right\rVert_2.
\]

Then:

\[
\epsilon_g
=\frac{\mathrm{SNR}_{\mathrm{corr,target}}}{V_{Ak/\sigma,L2}}.
\]

For an amplitude corruption with sign `s`:

\[
g_{\mathrm{amp}}=1+s\epsilon_g.
\]

For a phase corruption:

\[
\phi_{\mathrm{rad}}=s\,2\arcsin(\epsilon_g/2).
\]

The driver measures `Delta_V` from the actual post-CASA `V_corr` and verifies
that measured `SNR_corr` and `eps_vis` match the solved values before adding
noise.

## Package API

- `ConstantGainSpec` holds `V_ms`, `antenna_id`, `corruption_type`,
  `SNR_corr_target`, `sigma`, and `sign`.
- `measure_constant_gain_norms(spec)` measures `V_L2`, `V_Ak_L2`, and
  `V_Ak_over_sigma_L2` from `V`.
- `solve_constant_gain(spec, norms)` returns a `ConstantGainSolution`.
- `AntennaGainCorruption.from_constant_gain_solution(...)` creates the generic
  corruption configuration without storing scientific results on it.
- `measure_corruption_metrics(V_ms, V_corr_ms, sigma)` returns the actual
  `CorruptionMetrics`.
- `write_corruption_reports(..., solution=..., metrics=...)` writes the explicit
  configuration, solution, and measurement as separate report sections.

The simulation package exposes `add_thermal_noise_inplace(...)` so prediction,
corruption, measurement, and noise addition remain distinct stages.
