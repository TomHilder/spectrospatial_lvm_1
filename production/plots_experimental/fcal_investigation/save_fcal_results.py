import sys
from pathlib import Path

# Add some stuff to the path
sys.path.append("../..")
import matplotlib.pyplot as plt
import numpy as np
from fcal_per_line_ import load_fcals_raw

plt.style.use("plots_lib.custom")

FITS_PATH = Path("../../fits_experimental/fcal_investigation")
CAL_RESULTS_PATH = Path("../../calibration_results")

LINE = "Hαλ6563"

fcal_raw = load_fcals_raw(LINE)

np.save(CAL_RESULTS_PATH / f"{LINE}_fcal_raw.npy", fcal_raw)
