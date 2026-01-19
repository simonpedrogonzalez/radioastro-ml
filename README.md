# radioastro-ml
ML for Radoastronomy calibration debugging

# Notes on corruption functions of CASA simulator

Tests on J1822-0938 (gaincal, point source, low observation time ~15min)

## [setnoise](https://casadocs.readthedocs.io/en/stable/api/tt/casatools.simulator.html#casatools.simulator.simulator.setnoise)

Random additive noise to visibilities.

$$
V_{\text{obs}} = V_{\text{true}} + n_r + in_i
$$

where:
- $n_r, n_i \sim \mathcal{N}(0,\sigma^2)$
- $\sigma =$  `simplenoise` (e.g. `"0.1Jy"`)

**Fixed sigma**

$$
\sigma_{\text{image}} \approx
\frac{\sigma}{
\sqrt{n_{\text{pol}}n_{\text{baselines}}n_{\text{integrations}}n_{\text{chan}}}
}
$$

with:
- $n_{\text{pol}}$ : number of polarizations (usually 2)  
- $n_{\text{baselines}} = N_{\text{ant}}(N_{\text{ant}}-1)/2$
- $n_{\text{integrations}} \approx \text{num of correlation integration times in the MS}$



There is a pattern here, it shouldnt have any pattern probably, worth investigating.

The psf should be the same. Clean the images first. Check where the structure comes from

### Brownian

$$
\Delta S = \frac{4\sqrt{2}\left(T_{\text{rx}} e^{-\tau_{\text{atm}}} + T_{\text{atm}}\left(e^{\tau_{\text{atm}}}-\epsilon_l\right) + T_{\text{cmb}}\right)}{\epsilon_q\epsilon_a\pi D^2\sqrt{\Delta\nu\Delta t}}
$$

where:
- $T_{\text{rx}}$ : receiver temperature [K]  
- $T_{\text{atm}}$ : atmospheric temperature [K]  
- $T_{\text{cmb}}$ : CMB temperature [K]  
- $\tau_{\text{atm}}$ : zenith atmospheric opacity  
- $\epsilon_a$ : antenna efficiency  
- $\epsilon_q$ : correlator efficiency  
- $\epsilon_l$ : forward spillover efficiency  
- $D$ : dish diameter  
- $\Delta\nu$ : channel bandwidth  
- $\Delta t$ : integration time  

- `tsys-atm`: $T_{\text{atm}}$  computed from an atmospheric model using PWV  
- `tsys-manual`: $T_{\text{atm}}$  user specified  
- Noise increases with airmass if $\tau_{\text{atm}} > 0$


### Example (extreme noise)

**Before**
![before](images/gaincal_before.png)

**After (very high noise, ~100 Jy)**
![after](images/gaincal_noise_100Jy.png)

**Difference (after − before)**
![diff](images/gaincal_noise_100Jy_diff.png)

## [setgain](https://casadocs.readthedocs.io/en/stable/api/tt/casatools.simulator.html#casatools.simulator.simulator.setgain)(mode='fbm', ...)

Time variable antenna gains (complex, drift),  as fractional Brownian (random wandering) motion with an rms amplitude scale.

### How does it work

For each antenna and each time $t_i$, the code generates a complex gain

$$
g(t_i) = A(t_i)\,e^{i\phi(t_i)}
$$

Time is divided into discrete slots $t_0, t_1, t_2, \dots, t_N$, the gain is defined only at these times. fBM generates a random but correlated sequence $x(t_0), x(t_1), \dots, x(t_N)$ where each value is random, nearby times are strongly correlated, distant times are less correlated and there's a 'smoothness' parameter $\beta$. All values are generated together to satisfy a global correlation structure. Then, the values are scaled so that $\mathrm{RMS}\big(x_a(t)\big) = \text{scale}$ given by the user ('amplitude' parameter). The amplitude gain is then $A(t_i) = 1 + x_a(t_i)$ (because $g_i(t)=1+0i$ if instruments were perfect / already perfectly calibrated).

A 2nd independent fBM sequence $x_\phi(t_i)$ is generated, scaled to radians $\phi(t_i) = \pi \cdot x_\phi(t_i)$, and then the phase gain is applied multiplicatively as $e^{i\phi(t_i)}$

So the final complex gain for one antenna at time $t_i$ is:

$$
g(t_i) = \bigl(1 + x_a(t_i)\bigr)\; e^{\,i\,\pi\,x_\phi(t_i)}
$$


[NOTES]
Do time plot for gain to check. Do it in fractional residual, as a fraction of the peak of the image. So as to be able to compare. Clean the image also do them. symmetric andtisymmetric in the final residuals. That's why do a clean test.

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

[NOTES]
Plot the phaszers, again. Go from here to the visibilities. It's important to know how the corruption looks like in the visibility domain.

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

[NOTES]
The amplitude adn tropospheric thing should allow to change the data in a noticieable way, make some plots of the visibilities. to check if this would be good for corrupting the data.

[NOTES]
Look at the code of setgain, to see how setnoise actually works. Setgains, see what the code actually does. Chekc if one antenna can be fiddled with (just one antenna).

set how the setgain does aactually work.

Going to write the grant again
make some edits

including the corrections.

due 23rd january at 5pm. to check



TODOS:

0. Fractional residual plot.
1. Noise shouldnt have any pattern, investigate.
2. Make visibility plots for all functions
3. Study how setgains, setrop actually works
4. [LATER] check grant 23rd 5pm
5. Check no-op but performing setgain with values that should make it to no-op.