
ms0 = 'SNR_G55_10s.calib.ms'
ms_cor = 'SNR_G55_10s.calib.corrupted.ms'

import shutil
shutil.copytree(ms0, ms_cor)
shutil.copystat(ms0, ms_cor)

# tclean(vis=ms0, imagename='SNR_G55_10s.wProj.ASP',
#        gridder='wproject', wprojplanes=-1, deconvolver='asp',  
#        fusedthreshold=0.0, largestscale=-1, imsize=1280, cell='8arcsec', pblimit=-0.01, niter=1000, weighting='briggs',
#        stokes='I', robust=0.0, interactive=False, threshold='0.15mJy', savemodel='modelcolumn')

from casatools import simulator
sm = simulator()
sm.openfromms(ms_cor)
sm.setseed(42)

# add some tropospheric effects phase corruption
sm.settrop(mode='screen', pwv=3.0, deltapwv=0.15)
sm.corrupt()
sm.close()

tclean(vis=ms_cor, imagename='SNR_G55_10s_corrupted.wProj.ASP',
       gridder='wproject', wprojplanes=-1, deconvolver='asp',  
       fusedthreshold=0.0, largestscale=-1, imsize=1280, cell='8arcsec', pblimit=-0.01, niter=1000, weighting='briggs',
       stokes='I', robust=0.0, interactive=False, threshold='0.15mJy', savemodel='modelcolumn')



# Use the gain calibrator data (not the target data) because its a point source
# Because we know the brightness (should move in the sky)
# Pick the gain calibrator data from the calibrator tutorial (not the imaging tutorial)
# J1822-0938
# go back to 1st dataset (split the gain calibrator data) and image that alone.
# Then corrupt that data and use that.
# not effect of tropospheric (really crank it up to have visible effects)
# 
# before adding corruption: take a dataset, imaging, generate some tests, to make sure wiring them back out
# uv sampling (plot uv distribution) should look the same, images should look the same, plot the visibilities should be equal
# sanity checks in the loading and writing the data
# check if xx yy match, they are not swapped, the antennas names are the same, the visibilities are identical,
# then check if the images are identical withing computer precision, then make changes
