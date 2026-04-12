import os, sys, importlib, site

# This allows for user packages in the same casa session
sys.path.insert(0, site.getusersitepackages())


# Add scripts/ so you can import modules without "scripts."
sys.path.insert(0, os.getcwd())
from scripts import corruption_gaindrift
from scripts import extraction_pipeline
from scripts import plot_extracted
from scripts import asdm_utils
from scripts import image_extracted
from scripts import image_extracted_in_depth
from scripts import image_extracted_selfcal
from scripts import image_diagnosis
for lib in [corruption_gaindrift, extraction_pipeline, plot_extracted, asdm_utils, image_extracted, image_extracted_in_depth, image_extracted_selfcal, image_diagnosis]:
    importlib.reload(lib)


# corruption_gaindrift.new_corruption()
extraction_pipeline.run()
# image_extracted.main()
# plot_extracted.main()
# asdm_utils.main()
# image_extracted_in_depth.main("0739+016")
# image_extracted_selfcal.main("0739+016")

# image_diagnosis.main("0739+016")
# image_diagnosis.main("0259+077")
# image_diagnosis.main("0025-260") # uv-limit
# image_extracted_selfcal.main("0259+077")
# image_extracted_in_depth.main("0259+077")

# image_diagnosis.main("1719+177")
# image_extracted_selfcal.main("1719+177")
# image_extracted_in_depth.main("1719+177")
