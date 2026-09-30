"""CASA runtime-data location for CHPC launchers."""

import os


measurespath = os.environ["RADIOASTRO_CASA_DATA"]
datapath = [measurespath]
data_auto_update = True
measures_auto_update = True
