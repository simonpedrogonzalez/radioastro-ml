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
from scripts import check_uv_lim_issues
from scripts import uvlim_recal
from scripts import vla_pipe
from scripts import selfcal_extracted
from scripts import selfcal_compare
from scripts import single_image
from scripts import reproduction
from scripts import reproduction_selfcal_compare
from scripts import compare_image_geometry

for lib in [corruption_gaindrift, extraction_pipeline,
plot_extracted, asdm_utils, image_extracted, image_extracted_in_depth,
image_extracted_selfcal, image_diagnosis, check_uv_lim_issues,
uvlim_recal, vla_pipe, selfcal_extracted, selfcal_compare, single_image, reproduction,
reproduction_selfcal_compare,
compare_image_geometry]:
    importlib.reload(lib)


# check_uv_lim_issues.main()

# corruption_gaindrift.new_corruption()
# extraction_pipeline.run()

# uvlim_recal.main("/Users/u1528314/repos/radioastro-ml/collect/extracted/0205+322/0205+322/0205+322.ms")
reproduction_selfcal_compare.main()
# uvlim_recal.main("/Users/u1528314/repos/radioastro-ml/runs/vla_pipe_test/0205+322_pipeline_input.ms", initial_plot_only=True)
# uvlim_recal.main("/Users/u1528314/repos/radioastro-ml/collect/extracted/0205+322/selfcal/0205+322_selfcal.ms", initial_plot_only=True)
# single_image.main(
#     "/Users/u1528314/repos/radioastro-ml/runs/vla_pipe_test/uvlim_recal_0205+322_pipeline_input_20260418_233730/0205+322_from_data_seed.ms"
#     # "/Users/u1528314/repos/radioastro-ml/runs/vla_pipe_test/uvlim_recal_0205+322_pipeline_input_20260418_233730/0205+322_before_probe.ms",
#     # uvrange=">6.000klambda",
# )
# reproduction.main()
# compare_image_geometry.main()
# vla_pipe.main()
# selfcal_extracted.main()
# selfcal_compare.main()

# image_extracted.main()
# selfcal_compare.main()
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
