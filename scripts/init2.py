import os, sys, importlib, site

# This allows for user packages in the same casa session
sys.path.insert(0, site.getusersitepackages())


# Add scripts/ so you can import modules without "scripts."
sys.path.insert(0, os.getcwd())
from scripts import selfcal_extracted

importlib.reload(selfcal_extracted)


selfcal_extracted.main()
