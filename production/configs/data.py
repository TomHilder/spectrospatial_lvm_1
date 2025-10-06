from glob import glob
from pathlib import Path

LOC = Path("/Users/tomhilder/Documents/PhD/research/lvm_project/lvm_spectral_spatial/data/rosette")
DRP_FILES = glob(str(LOC / "lvm*.fits"))
