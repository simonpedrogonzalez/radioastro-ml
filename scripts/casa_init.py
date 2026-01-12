import os, sys

# Add scripts/ so you can import modules without "scripts."
sys.path.insert(0, os.path.abspath("scripts"))

# Optional convenience variables:
DATA = os.path.abspath("data")
RUN  = os.getcwd()

print("CASA init: added scripts/ to sys.path")
print("DATA =", DATA)
print("RUN  =", RUN)
