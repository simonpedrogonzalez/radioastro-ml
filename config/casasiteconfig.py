"""CASA runtime-data location for CHPC launchers."""

import os


measurespath = os.environ["RADIOASTRO_CASA_DATA"]
datapath = [measurespath]
# Batch jobs set RADIOASTRO_CASA_AUTO_UPDATE=0 so many concurrent CASA processes
# do not all try to update the shared runtime data.
_auto_update = os.environ.get("RADIOASTRO_CASA_AUTO_UPDATE", "1") != "0"
data_auto_update = _auto_update
measures_auto_update = _auto_update
