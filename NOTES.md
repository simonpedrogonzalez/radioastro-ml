## Main

1. Create an abstract for CosmicAI by February 16/23

## Side quests

1. Setnoise shouldnt have any pattern, investigate.
2. The amplitude adn tropospheric thing should allow to change the data in a noticieable way, make some plots of the visibilities. to check if this would be good for corrupting the data.
3. Study how setrop actually works

## Notes

- If its something calibratable then it would prly have being calibrated out (problem in the definition of realistic corruption). Ideally it would be able to detect problems when the calibration failed or is corrupted because of some effect that is hasn't being able to account.
- Data: VLAs sky survey https://data.nrao.edu/portal/#/, 1822-096, 36286
- Irrecoverable corruption regime: if the corruption interval is smaller than the integration interval used in the calibration (solint).

## TODOs

Closure quntities:
- check the closure relations / closure quantities for the baselines: before corruptions, after corruptions and after recalibrating. They should stay the same, otherwise the data doesn't make sense statistically.
- once i have 3 numbers, form that complex quantity, apply twice and check they are equal
- the phase might be distributed among antennas. The closure quantities before after the corruptions and after recovery. If those are all the same, the data is statistically equivalent. if they are not the same, the corruption process is violating closure, so we might need to try a different corruption process that maintians closure, otherwise the data won't satisfy some of the fundamental properties.
- check literature channel for `closure`, there might be some useful assumptions used there ot do the calibration.
- check reference antenna (the phase added to the reference antenna got added to the rest of the antennas), is possible that the closure is the problem.

- check if calibrating before corruption would yield a small improvemnet (more like the model)

First do the corruption.

Amplitude vs phase errors we can test. that could be an starting point.


## Claim 1: We can generat physically possible corrupted datasets



## Claim 2: A CNN can detect calibration errors from image products

1. 


# Data collection

0017+154
0153-410
0253+180


0005+544


GO after 2012 - up until now (unfirm format)
VLA, EVLA (expanded), JVLA (jansky VLA) etc, (what does that mean)
Should have calibrator tables.


listobs will show 

bit of code for alma to pull out info from listobs and tell you which is the gain calibrator

run a split with the field name and the data column, (when in doubt ask for the corrupted column, and then move to the data column)

start with that scripts that reads in the file.


only grab P in all.


noise property over the course fo observation

C band is usually stable
but other errors might not
chunking in time will give you slightly diff noise properties.
Slightly different uv coverage, so when you add corruptions, it will give you different patterns for the antennas so it should be a good idea
STN is pretty large in single integration
Make images for every integration

still should be high STNR even if it's single integration

STN is swqrt of time, but also improves with the dynamic range, some of the weaker corruptions will be visible on high dynamic range, and you will need longer integration times. 1% corruption or so, will be only detectible this way.
STN 50 or 30% we wont see a 1% corruption

you will need to have a larger corruption to see them in the shorter integration.

factor of 5x shorter is gaincal than observations, so isnt direct application use case. but in terms of STN because the noise is lower and also the signal is lower so there is a way to translate that.

The uv coverage will be different. From a proof of principle perspective it's fine. Once we demonstrate this kinda works, we move on to target fields


Peak anything that has a P, if im gonna check the configuration
  some of the data is taken, so sometimes is in between configuration, remove them
  

Int time is usually 3s, Smaller B than A,
There will be more calibrator data in A.

