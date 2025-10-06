# paper_plots.py
# Generate all the plots for the paper

import sys
from pathlib import Path

# Add some stuff to the path
sys.path.append("../..")
import matplotlib.pyplot as plt
import numpy as np
import spectracles as new_pkg
import spectracles.model as new_model_mod
import spectracles.model.share_module as new_share_mod
from configs.config_io import load_config
from configs.data import DRP_FILES
from lvm_tools import DataConfig, FitDataBuilder, LVMTile, LVMTileCollection
from mpl_drip import COLORS
from spectracles import load_model

sys.modules["modelling_lib"] = new_pkg
sys.modules["modelling_lib.model"] = new_model_mod
sys.modules["modelling_lib.model.share_module"] = new_share_mod

plt.style.use("mpl_drip.custom")

FITS_PATH = Path("../../fits_experimental/fcal_investigation")
CONFIG_PATH = Path("../../configs")
# PLOTS_PATH = Path("plots_paper")
#
# blue (b: 3600-5800  ̊A),
# red (r: 5750-7570  ̊A)
# infrared (z: 7520-9800  ̊A)

LINES = [
    # "Hδλ4102",
    "Hγλ4340",
    "Hβλ4861",
    "[OIII]λ5007",
    # "[NII]λ6548",
    "Hαλ6563",
    # "[NII]λ6583",
    # "[SII]λ6731",
    # "[SIII]λ9069",
    "[SIII]λ9531",
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


def load_fcals_raw(line):
    models, configs = load_models([line])
    line_model = models[line]
    fcals = line_model.line.f_cal_raw.tile_values.val
    return fcals


def load_fcals(line, tiles):
    models, configs = load_models([line])
    fd = FitDataBuilder(tiles, configs[line]).build()
    line_model = models[line]
    fcals = line_model.line.f_cal(fd.αδ_data)
    fcals, idx = np.unique(fcals, return_index=True)
    tile_idx = fd.tile_idx[idx]
    i_sort = np.argsort(tile_idx)
    return fcals[i_sort], tile_idx[i_sort]


def big_ass_plot(the_chosen_lines, tiles):
    fig, ax = plt.subplots(figsize=[8, 6])
    for i, line in enumerate(the_chosen_lines):
        fcal, tile_idx = load_fcals(line, tiles)
        fcal = 100 * (fcal / np.nanmedian(fcal) - 1)
        # fcal = 100 * (fcal / fcal[5] - 1)
        # wavelength = float(line.split("λ")[-1]) * np.ones_like(tile_idx)
        # for j in range(len(tile_idx)):
        # plt.scatter(wavelength[j], fcal[j], c=f"C{j}")
        ax.plot(
            tile_idx + 1,
            fcal,
            label=make_title([line]),
            alpha=0.7,
            marker=".",
            markersize=10,
        )
    ax.set_ylabel(r"$F_\mathrm{cal}$ [\%]")
    ax.set_xlabel(r"Tile/Pointing")
    ax.set_xticks(tile_idx + 1)
    plt.tight_layout()
    ax.legend(loc="best")
    plt.savefig("f_cal.pdf", bbox_inches="tight")
    plt.show()


def make_title(lines):
    title = r""
    for line in lines:
        title += make_str_label_safe(line) + ", "
    title = title[:-2]  # Remove trailing comma and space
    title = rf"${{\rm {title}}}$"
    return title


if __name__ == "__main__":
    print("Reading data...", end=" ", flush=True)
    tiles = LVMTileCollection.from_tiles([LVMTile.from_file(Path(f)) for f in DRP_FILES])
    print("Done.")
    big_ass_plot(LINES, tiles)
