import os, sys, importlib, site

# This allows for user packages in the same casa session
sys.path.insert(0, site.getusersitepackages())


# Add scripts/ so you can import modules without "scripts."
sys.path.insert(0, os.getcwd())
from scripts import corruption_gaindrift
for lib in [corruption_gaindrift]:
    importlib.reload(lib)


corruption_gaindrift.new_corruption()