import faulthandler
faulthandler.enable()
print('before numpy',flush=True)
import numpy
print('before torch',flush=True)
import torch
print('before MACECalculator',flush=True)
from mace.calculators import MACECalculator
print('before standalone surface',flush=True)
import sys
sys.path.insert(0,'/home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity')
from pamssw.standalone.surface import quench
print('imports completed; no calculator loaded, zero PES',flush=True)
