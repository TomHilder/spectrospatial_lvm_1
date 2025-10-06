import sys
from pathlib import Path

# Add some stuff to the path
sys.path.append("../..")
import matplotlib.pyplot as plt
import numpy as np
from does_it_make_sense_tho2 import load_vcals

FITS_PATH = Path("../../fits_experimental/fcal_investigation")
CAL_RESULTS_PATH = Path("../../calibration_results")

LINE = "Hβλ4861"

vcals = load_vcals(LINE)

np.save(CAL_RESULTS_PATH / f"{LINE}_C_v_cal.npy", vcals)
