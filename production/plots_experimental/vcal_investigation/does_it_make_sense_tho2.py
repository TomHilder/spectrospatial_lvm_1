# paper_plots.py
# Generate all the plots for the paper

import sys
from pathlib import Path

# Add some stuff to the path
sys.path.append("../..")
import matplotlib.pyplot as plt
import numpy as np
from configs.config_io import load_config
from configs.data import DRP_FILES
from lvm_lib import DataConfig, FitDataBuilder, LVMTile, LVMTileCollection
from modelling_lib import load_model
from plots_lib import COLORS

plt.style.use("plots_lib.custom")

FITS_PATH = Path("../../fits_experimental/vcal_global")
CONFIG_PATH = Path("../../configs")
# PLOTS_PATH = Path("plots_paper")
#
# blue (b: 3600-5800  ̊A),
# red (r: 5750-7570  ̊A)
# infrared (z: 7520-9800  ̊A)


LINES_BLUE = [
    "Hδλ4102",
    "Hγλ4340",
    # "HeIλ4471",
    "Hβλ4861",
    "[OIII]λ5007",
    # "[NII]λ5755",
]


def make_str_label_safe(label: str) -> str:
    """Replace utf-8 greek letters with their LaTeX equivalents."""
    replacements = {
        "λ": r"\lambda",
        "α": r"\alpha",
        "δ": r"\delta",
        "γ": r"\gamma",
        "β": r"\beta",
        "σ": r"\sigma",
        "μ": r"\mu",
    }
    for old, new in replacements.items():
        label = label.replace(old, new)
    return label


def load_models(line_list):
    model_paths = list(FITS_PATH.glob("**/*.model"))
    models = {}
    for model_path in model_paths:
        line = model_path.stem.split(".")[0]
        if line not in line_list:
            continue
        models[line] = load_model(model_path)

    config_paths = list(CONFIG_PATH.glob("*.json"))
    configs = {}
    for config_path in config_paths:
        line = config_path.stem
        if line not in line_list:
            continue
        configs[line] = DataConfig.from_dict(load_config(config_path))

    return models, configs


def load_vcals(line):
    models, configs = load_models([line])
    # fd = FitDataBuilder(tiles, configs[line]).build()
    line_model = models[line]
    return line_model.line.v_cal.C_v_cal.val


def make_title(name, lines):
    title = r""
    for line in lines:
        title += make_str_label_safe(line) + ", "
    title = title[:-2]  # Remove trailing comma and space
    title = rf"{name}: ${{\rm {title}}}$"
    return title


def plot_calibration_results():
    plt.figure(figsize=[10, 5])

    for i, line in enumerate(LINES_BLUE):
        v = load_vcals(line)
        v1 = v[0]
        v2 = v[1]
        wavelength = float(line.split("λ")[1])

        if i == 0:
            plt.scatter(wavelength, v1, color=COLORS[0], marker="o", s=50, label=r"IFU 0")
            plt.scatter(wavelength, 0.0, color=COLORS[1], marker="o", s=50, label=r"IFU 1")
            plt.scatter(wavelength, v2, color=COLORS[2], marker="o", s=50, label=r"IFU 2")
        else:
            plt.scatter(wavelength, v1, color=COLORS[0], marker="o", s=50)
            plt.scatter(wavelength, 0.0, color=COLORS[1], marker="o", s=50)
            plt.scatter(wavelength, v2, color=COLORS[2], marker="o", s=50)

    plt.xlim(3600, 5800)
    plt.xlabel(r"$\lambda \, \mathrm{[\AA]}$")
    plt.ylabel(r"$\Delta \lambda_{\rm cal} \, [\mathrm{\AA}]$")
    plt.legend(bbox_to_anchor=(0.99, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig("vcal_results2_blue.pdf")
    plt.show()


if __name__ == "__main__":
    # print("Reading data...", end=" ", flush=True)
    # tiles = LVMTileCollection.from_tiles([LVMTile.from_file(Path(f)) for f in DRP_FILES])
    # print("Done.")
    plot_calibration_results()
