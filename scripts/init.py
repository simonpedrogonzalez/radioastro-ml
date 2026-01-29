import os, sys, importlib

# Add scripts/ so you can import modules without "scripts."
sys.path.insert(0, os.getcwd())
from scripts import corruption_gaindrift
for lib in [corruption_gaindrift]:
    importlib.reload(lib)


corruption_gaindrift.main_recoverable_corruption()