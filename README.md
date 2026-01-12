# radioastro-ml
ML for Radoastronomy calibration debugging

# Notes on corruption functions of CASA simulator

Tests on J1822-0938 (gaincal, point source, low observation time ~15min)

## [setnoise](https://casadocs.readthedocs.io/en/stable/api/tt/casatools.simulator.html#casatools.simulator.simulator.setnoise)

Random additive noise to visibilities.

$$
V_{\text{obs}} = V_{\text{true}} + n_r + i\,n_i
$$
where:
- $ n_r, n_i \sim \mathcal{N}(0,\sigma^2) $
- $ \sigma = $  `simplenoise` (e.g. `"0.1Jy"`)

**Fixed sigma**  
$$
\sigma_{\text{image}} \approx
\frac{\sigma}{
\sqrt{n_{\text{pol}}\,n_{\text{baselines}}\,n_{\text{integrations}}\,n_{\text{chan}}}
}
$$
with:
- $ n_{\text{pol}} $ : number of polarizations (usually 2)  
- $ n_{\text{baselines}} = N_{\text{ant}}(N_{\text{ant}}-1)/2 $
- $ n_{\text{integrations}} \approx \text{num of correlation integration times in the MS} $


### Brownian

$$
\Delta S = \frac{4\sqrt{2}\,\left(T_{\text{rx}} e^{-\tau_{\text{atm}}} + T_{\text{atm}}\left(e^{\tau_{\text{atm}}}-\epsilon_l\right) + T_{\text{cmb}}\right)}{\epsilon_q\,\epsilon_a\,\pi D^2\,\sqrt{\Delta\nu\,\Delta t}}
$$

where:
- $ T_{\text{rx}} $ : receiver temperature [K]  
- $ T_{\text{atm}} $ : atmospheric temperature [K]  
- $ T_{\text{cmb}} $ : CMB temperature [K]  
- $ \tau_{\text{atm}} $ : zenith atmospheric opacity  
- $ \epsilon_a $ : antenna efficiency  
- $ \epsilon_q $ : correlator efficiency  
- $ \epsilon_l $ : forward spillover efficiency  
- $ D $ : dish diameter  
- $ \Delta\nu $ : channel bandwidth  
- $ \Delta t $ : integration time  

- `tsys-atm`: $ T_{\text{atm}} $  computed from an atmospheric model using PWV  
- `tsys-manual`: $ T_{\text{atm}} $  user specified  
- Noise increases with airmass if $ \tau_{\text{atm}} > 0 $

### Example (extreme noise)

**Before**
![before](images/gaincal_before.png)

**After (very high noise, ~100 Jy)**
![after](images/gaincal_noise_100Jy.png)

**Difference (after − before)**
![diff](images/gaincal_noise_100Jy_diff.png)

## [setgain](https://casadocs.readthedocs.io/en/stable/api/tt/casatools.simulator.html#casatools.simulator.simulator.setgain)(mode='fbm', ...)

Time variable antenna gains (complex, drift),  as fractional Brownian (random wandering) motion with an rms amplitude scale.

**Before**
![before](images/gaincal_before.png)

**After**
![after](images/gaincal_antenna_gain_drift_0.2.png)

**Difference (after − before)**
![diff](images/gaincal_antenna_gain_drift_0.2_diff.png)



## [setleakage](https://casadocs.readthedocs.io/en/stable/api/tt/casatools.simulator.html#casatools.simulator.simulator.setleakage)

Constant polarization leakage between feeds (D-matrix), currently no time dependent available (constant).

## setpointingerror

Per antenna pointing offset, mis-pointing error.

## [settrop](https://casadocs.readthedocs.io/en/stable/api/tt/casatools.simulator.html#casatools.simulator.simulator.settrop)

Atmospheric effects. T-matrix defined in terms of precipitable wate vapor in mm & windspeed.


**Before**
![before](images/gaincal_trop_before.png)

**After**
![after](images/gaincal_trop_after_hurricane.png)

**Difference (after − before)**
![diff](images/gaincal_trop_hurricane_diff.png)

## setlimits(shadowlimit=..., elevationlimit=...)

Flags data when below an elevation limit or shadowing fraction exceeds a threshold, impose some observational constraint, not a calibration corruption per se.

## setauto(autocorrwt=...)

Sets the weight of autocorrelations. Not a physics based corruption as far as I am aware, but an analysis choice.

## setapply(...)

Apply some existing calibration tables as a corruption. We can build the caltables and apply them.

## setbandpass(...)

Bandpass errors with normal distributions, not implemented.