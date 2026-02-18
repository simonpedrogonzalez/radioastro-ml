# 3C391 tutorial pipeline script (sorted outputs)
# Assumes working directory is a per-run folder with:
#   ./data  -> repo data folder (symlink)
#   ./caltables/
#   ./images/
#
# Tested style for CASA 6.x

import os

# -------------------------
# Paths / helpers
# -------------------------
RUN = os.getcwd()
DATA = os.path.join(RUN, "data")
CALT = os.path.join(RUN, "caltables")
IMGS = os.path.join(RUN, "images")

for d in (CALT, IMGS):
    os.makedirs(d, exist_ok=True)

def dpath(*p):  # data input path
    return os.path.join(DATA, *p)

def cpath(*p):  # caltable output path
    return os.path.join(CALT, *p)

def ipath(*p):  # image/plot output path
    return os.path.join(IMGS, *p)

def ensure_local_ms(ms_rel):
    """
    OPTIONAL convenience: symlink a data MS into run root so tasks can use ms basename.
    Not required if you prefer using dpath(...).
    """
    src = dpath(ms_rel)
    dst = os.path.join(RUN, os.path.basename(ms_rel))
    if not os.path.exists(dst):
        os.symlink(src, dst)
    return os.path.basename(ms_rel)

# -------------------------
# Inputs (edit if your data layout differs)
# -------------------------
# If your MS sits directly under data/, set ms_rel = "3c391_ctm_mosaic_10s_spw0.ms"
ms_rel = "3c391_ctm_mosaic_10s_spw0.ms"
ms = dpath(ms_rel)

# Model image referenced by setjy; keep it in data/ as well
model_im = dpath("3C286_C.im")

# -------------------------
# 1) Inspect / initial plots
# -------------------------
obs_dict = listobs(vis=ms)

plotants(vis=ms, figfile=ipath("plotants_3c391_antenna_layout.png"))
clearstat()  # remove table lock from plotants in script mode

# -------------------------
# 2) Flagging / data inspection
# -------------------------
flagdata(vis=ms, flagbackup=True, mode="manual", scan="1")
flagdata(vis=ms, flagbackup=True, mode="manual", antenna="ea13,ea15")
flagdata(vis=ms, mode="quack", quackinterval=10.0, quackmode="beg")
clearstat()

plotms(
    vis=ms,
    selectdata=True,
    correlation="RR,LL",
    averagedata=True,
    avgchannel="64",
    coloraxis="field",
    plotfile=ipath("colorbyfield.png"),
    overwrite=True,
    showgui=False,
)

# Figure 3A: amp vs uvdist
plotms(
    vis=ms,
    field="0",
    selectdata=True,
    correlation="RR,LL",
    averagedata=True,
    xaxis="UVdist",
    coloraxis="field",
    plotfile=ipath("amp-vs-uvdist.png"),
    overwrite=True,
    showgui=False,
)

# Figure 4: datastream-style plot
plotms(
    vis=ms,
    field="",
    correlation="RR,LL",
    timerange="",
    antenna="ea01",
    spw="0:31",
    xaxis="time",
    yaxis="antenna2",
    plotrange=[-1, -1, 0, 26],
    coloraxis="field",
    plotfile=ipath("3c391-datastream.png"),
    overwrite=True,
    showgui=False,
)

# -------------------------
# 3) Calibration tables (all under caltables/)
# -------------------------
antpos = cpath("3c391_ctm_mosaic_10s_spw0.antpos")
G0all  = cpath("3c391_ctm_mosaic_10s_spw0.G0all")
G0     = cpath("3c391_ctm_mosaic_10s_spw0.G0")
K0     = cpath("3c391_ctm_mosaic_10s_spw0.K0")
B0     = cpath("3c391_ctm_mosaic_10s_spw0.B0")
G1     = cpath("3c391_ctm_mosaic_10s_spw0.G1")
FS1    = cpath("3c391_ctm_mosaic_10s_spw0.fluxscale1")

gencal(vis=ms, caltable=antpos, caltype="antpos")

setjy(vis=ms, listmodels=True)
setjy(
    vis=ms,
    field="J1331+3030",
    standard="Perley-Butler 2017",
    model=model_im,
    usescratch=True,
    scalebychan=True,
    spw="",
)

# Initial Phase Calibration
gaincal(
    vis=ms,
    caltable=G0all,
    field="0,1,9",
    refant="ea21",
    spw="0:27~36",
    gaintype="G",
    calmode="p",
    solint="int",
    minsnr=5,
    gaintable=[antpos],
)

# Figure 5
plotms(
    vis=G0all,
    xaxis="time",
    yaxis="phase",
    coloraxis="corr",
    iteraxis="antenna",
    plotrange=[-1, -1, -180, 180],
    plotfile=ipath("3c391-polzn.png"),
    overwrite=True,
    showgui=False,
)

flagdata(vis=ms, flagbackup=True, mode="manual", antenna="ea05")

gaincal(
    vis=ms,
    caltable=G0,
    field="J1331+3030",
    refant="ea21",
    spw="0:27~36",
    calmode="p",
    solint="int",
    minsnr=5,
    gaintable=[antpos],
)

# Figure 5a
plotms(
    vis=G0,
    xaxis="time",
    yaxis="phase",
    coloraxis="corr",
    field="J1331+3030",
    iteraxis="antenna",
    plotrange=[-1, -1, -180, 180],
    timerange="08:02:00~08:17:00",
    plotfile=ipath("3C286-G0all-phase-ea05.png"),
    overwrite=True,
    showgui=False,
)

# Delay calibration
gaincal(
    vis=ms,
    caltable=K0,
    field="J1331+3030",
    refant="ea21",
    spw="0:5~58",
    gaintype="K",
    solint="inf",
    combine="scan",
    minsnr=5,
    gaintable=[antpos, G0],
)

# Figure 6
plotms(
    vis=K0,
    xaxis="antenna1",
    yaxis="delay",
    coloraxis="baseline",
    plotfile=ipath("3c391-K0-delay.png"),
    overwrite=True,
    showgui=False,
)

# Bandpass calibration
bandpass(
    vis=ms,
    caltable=B0,
    field="J1331+3030",
    spw="",
    refant="ea21",
    combine="scan",
    solint="inf",
    bandtype="B",
    gaintable=[antpos, G0, K0],
)

# Figure 8A amp
plotms(
    vis=B0,
    field="J1331+3030",
    xaxis="chan",
    yaxis="amp",
    coloraxis="corr",
    iteraxis="antenna",
    gridrows=2,
    gridcols=2,
    plotfile=ipath("3C286-B0-amp.png"),
    overwrite=True,
    showgui=False,
)

# Figure 8B phase
plotms(
    vis=B0,
    field="J1331+3030",
    xaxis="chan",
    yaxis="phase",
    coloraxis="corr",
    plotrange=[-1, -1, -180, 180],
    iteraxis="antenna",
    gridrows=2,
    gridcols=2,
    plotfile=ipath("3C286-B0-phase.png"),
    overwrite=True,
    showgui=False,
)

# Gain calibration
gaincal(
    vis=ms,
    caltable=G1,
    field="J1331+3030",
    spw="0:5~58",
    solint="inf",
    refant="ea21",
    gaintype="G",
    calmode="ap",
    solnorm=False,
    gaintable=[antpos, K0, B0],
    interp=["", "", "nearest"],
)

gaincal(
    vis=ms,
    caltable=G1,
    field="J1822-0938",
    spw="0:5~58",
    solint="inf",
    refant="ea21",
    gaintype="G",
    calmode="ap",
    gaintable=[antpos, K0, B0],
    append=True,
)

# Figure 9A
plotms(
    vis=G1,
    xaxis="time",
    yaxis="phase",
    gridrows=1,
    gridcols=2,
    iteraxis="corr",
    coloraxis="baseline",
    plotrange=[-1, -1, -180, 180],
    plotfile=ipath("plotms_3c391-G1-phase.png"),
    overwrite=True,
    showgui=False,
)

# Figure 9B
plotms(
    vis=G1,
    xaxis="time",
    yaxis="amp",
    gridrows=1,
    gridcols=2,
    iteraxis="corr",
    coloraxis="baseline",
    plotfile=ipath("plotms_3c391-G1-amp.png"),
    overwrite=True,
    showgui=False,
)

# Figure 10
plotms(
    vis=G1,
    xaxis="time",
    yaxis="phase",
    correlation="/",
    coloraxis="baseline",
    plotrange=[-1, -1, -180, 180],
    plotfile=ipath("3c391-G1-RLphasediff.png"),
    overwrite=True,
    showgui=False,
)

# Fluxscale
myscale = fluxscale(
    vis=ms,
    caltable=G1,
    fluxtable=FS1,
    reference="J1331+3030",
    transfer=["J1822-0938"],
    incremental=False,
)

# Figure 11A/B
plotms(
    vis=FS1,
    xaxis="time",
    yaxis="amp",
    correlation="R",
    coloraxis="baseline",
    plotfile=ipath("3c391-fluxscale1-amp-R.png"),
    overwrite=True,
    showgui=False,
)

plotms(
    vis=FS1,
    xaxis="time",
    yaxis="amp",
    correlation="L",
    coloraxis="baseline",
    plotfile=ipath("3c391-fluxscale1-amp-L.png"),
    overwrite=True,
    showgui=False,
)

# Applycal
applycal(
    vis=ms,
    field="J1331+3030",
    gaintable=[antpos, FS1, K0, B0],
    gainfield=["", "J1331+3030", "", ""],
    interp=["", "nearest", "", ""],
    calwt=False,
)

applycal(
    vis=ms,
    field="J1822-0938",
    gaintable=[antpos, FS1, K0, B0],
    gainfield=["", "J1822-0938", "", ""],
    interp=["", "nearest", "", ""],
    calwt=False,
)

applycal(
    vis=ms,
    field="2~8",
    gaintable=[antpos, FS1, K0, B0],
    gainfield=["", "J1822-0938", "", ""],
    interp=["", "linear", "", ""],
    calwt=False,
)

# Figure 12A–D (corrected column)
plotms(
    vis=ms,
    field="0",
    correlation="RR,LL",
    avgtime="60",
    xaxis="channel",
    yaxis="amp",
    ydatacolumn="corrected",
    coloraxis="corr",
    plotfile=ipath("plotms_3c391-fld0-corrected-amp.png"),
    overwrite=True,
    showgui=False,
)

plotms(
    vis=ms,
    field="0",
    correlation="RR,LL",
    avgtime="60",
    xaxis="channel",
    yaxis="phase",
    ydatacolumn="corrected",
    coloraxis="corr",
    plotrange=[-1, -1, -180, 180],
    plotfile=ipath("plotms_3c391-fld0-corrected-phase.png"),
    overwrite=True,
    showgui=False,
)

plotms(
    vis=ms,
    field="1",
    correlation="RR,LL",
    avgtime="60",
    xaxis="channel",
    yaxis="amp",
    ydatacolumn="corrected",
    coloraxis="corr",
    plotfile=ipath("plotms_3c391-fld1-corrected-amp.png"),
    overwrite=True,
    showgui=False,
)

plotms(
    vis=ms,
    field="1",
    correlation="RR,LL",
    avgtime="60",
    xaxis="channel",
    yaxis="phase",
    ydatacolumn="corrected",
    coloraxis="corr",
    plotrange=[-1, -1, -180, 180],
    plotfile=ipath("plotms_3c391-fld1-corrected-phase.png"),
    overwrite=True,
    showgui=False,
)

# -------------------------
# 4) Split + weights (outputs go into run root by design; keep in run root, but named)
# -------------------------
ms_split = os.path.join(RUN, "3c391_ctm_mosaic_spw0.ms")

split(
    vis=ms,
    outputvis=ms_split,
    datacolumn="corrected",
    field="2~8",
    correlation="RR,LL",
)

statwt(vis=ms_split, datacolumn="data")

# Figure 13
plotms(
    vis=ms_split,
    xaxis="uvwave",
    yaxis="amp",
    ydatacolumn="data",
    field="0",
    avgtime="30",
    correlation="RR",
    plotfile=ipath("plotms_3c391-mosaic0-uvwave.png"),
    overwrite=True,
    showgui=False,
)

# -------------------------
# 5) Imaging (put ALL tclean outputs into images/)
# -------------------------
img_base = ipath("3c391_ctm_spw0_multiscale")  # tclean will create multiple products with this basename

tclean(
    vis=ms_split,
    imagename=img_base,
    field="",
    spw="",
    specmode="mfs",
    niter=20000,
    gain=0.1,
    threshold="1.0mJy",
    gridder="mosaic",
    deconvolver="multiscale",
    scales=[0, 5, 15, 45],
    smallscalebias=0.9,
    interactive=False,
    imsize=[480, 480],
    cell=["2.5arcsec", "2.5arcsec"],
    stokes="I",
    weighting="briggs",
    robust=0.5,
    pbcor=False,
    savemodel="modelcolumn",
    mask=ipath("tclean-1.mask"),  # keep masks in images/
)

impbcor(
    imagename=img_base + ".image",
    pbimage=img_base + ".pb",
    outfile=img_base + ".pbcorimage",
)

mystat = imstat(imagename=img_base + ".pbcorimage")

# -------------------------
# 6) Selfcal (keep products in caltables/ and images/)
# -------------------------
delmod(ms_split)

tclean(
    vis=ms_split,
    imagename=ipath("3c391_ctm_spw0_ms_I"),
    field="",
    spw="",
    specmode="mfs",
    niter=500,
    gain=0.1,
    threshold="1mJy",
    gridder="mosaic",
    deconvolver="multiscale",
    scales=[0, 5, 15, 45],
    smallscalebias=0.9,
    interactive=False,
    imsize=[480, 480],
    cell=["2.5arcsec", "2.5arcsec"],
    stokes="I",
    weighting="briggs",
    robust=0.5,
    pbcor=False,
    savemodel="modelcolumn",
    mask=ipath("tclean-1.mask"),
)

selfcal1 = cpath("3c391_ctm_mosaic_spw0.selfcal1")

gaincal(
    vis=ms_split,
    caltable=selfcal1,
    field="",
    spw="",
    selectdata=False,
    solint="30s",
    refant="ea21",
    minblperant=4,
    minsnr=3,
    gaintype="G",
    calmode="p",
)

applycal(
    vis=ms_split,
    field="",
    spw="",
    selectdata=False,
    gaintable=[selfcal1],
    gainfield=[""],
    interp=["nearest"],
    calwt=[False],
    applymode="calflag",
)

tclean(
    vis=ms_split,
    imagename=ipath("3c391_ctm_spw0_multiscale_selfcal1"),
    field="",
    spw="",
    specmode="mfs",
    niter=20000,
    gain=0.1,
    threshold="1mJy",
    gridder="mosaic",
    deconvolver="multiscale",
    scales=[0, 5, 15, 45],
    smallscalebias=0.9,
    interactive=False,
    imsize=[480, 480],
    cell=["2.5arcsec", "2.5arcsec"],
    stokes="I",
    weighting="briggs",
    robust=0.5,
    pbcor=False,
    savemodel="modelcolumn",
    mask=ipath("tclean-1.mask"),
)
