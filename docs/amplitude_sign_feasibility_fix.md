# Dataset v1 amplitude-sign feasibility fix

## Goal

Recover the five failed `amp_snr_100` variants without changing their antenna,
target corruption SNR, or amplitude-only corruption model:

- `0449+113_amp_snr_100`
- `1221+282_amp_snr_100`
- `1327+221_amp_snr_100`
- `1416+347_amp_snr_100`
- `1430+107_amp_snr_100`

## Minimal change

Limit the change to `create_dataset_v1.py`; keep the general corruption solver
strict. Resolve the seeded random sign as usual. For an amplitude variant only,
if the result is negative and `1 - eps_g <= 0`, replace it with the positive
sign before calling `solve_constant_gain`.

Conceptually:

```python
sign = seeded_random_sign
if variant.family == "amp" and sign == -1 and target / norm >= 1:
    sign = 1
```

Pass the resolved sign explicitly to the solver and retain the original seed in
the sample provenance. Record that the sign policy is feasibility-conditioned.
All feasible random signs, all phase variants, and all imaging and simulation
behavior remain unchanged.

Do not clip `eps_g` or the gain: the positive branch exactly preserves the
requested corruption SNR.

## Verification

- Unit-test the boundary: infeasible negative amplitude becomes positive;
  feasible negative amplitude and phase signs are unchanged.
- Confirm the five variants solve with positive gains and achieve
  `SNR_corr_expected == SNR_corr_target` within numerical tolerance.
- After the current run finishes, resume the dataset command so completed
  samples are reused and only missing variants are retried.

This does not address the two `0449+113` phase failures; their requested SNR is
outside the one-antenna, phase-only model's attainable range.
