from casatasks import split

split(
    vis='data/3c391_ctm_mosaic_10s_spw0.ms',     # MS AFTER applycal
    outputvis='data/J1822_spw0.calibrated.ms',   # output MS
    field='J1822-0938',                     # gain calibrator
    spw='0'
)
