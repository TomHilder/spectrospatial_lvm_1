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
from astropy.constants import c as C_CONST
from configs.config_io import load_config
from configs.data import DRP_FILES
from lvm_tools import DataConfig, FitDataBuilder, LVMTile, LVMTileCollection
from mpl_drip import COLORS
from spectracles import load_model

sys.modules["modelling_lib"] = new_pkg
sys.modules["modelling_lib.model"] = new_model_mod
sys.modules["modelling_lib.model.share_module"] = new_share_mod

plt.style.use("mpl_drip.custom")

FITS_PATH = Path("../../fits_experimental/vcal_investigation")
CONFIG_PATH = Path("../../configs")
# PLOTS_PATH = Path("plots_paper")
#
# blue (b: 3600-5800  ̊A),
# red (r: 5750-7570  ̊A)
# infrared (z: 7520-9800  ̊A)

print("Reading data...", end=" ", flush=True)
tiles = LVMTileCollection.from_tiles([LVMTile.from_file(Path(f)) for f in DRP_FILES])
print("Done.")

LINES_BLUE = [
    "Hδλ4102",
    "Hγλ4340",
    # "HeIλ4471",
    "Hβλ4861",
    "[OIII]λ5007",
    # "[NII]λ5755",
]

LINES_RED = [
    # "[SIII]λ6312",
    "[NII]λ6548",
    "Hαλ6563",
    "[NII]λ6583",
    # "HeIλ6678",
    "[SII]λ6731",
    # "[OII]λ7319",
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


def what_the_fuck(line):
    models, configs = load_models([line])
    fd = FitDataBuilder(tiles, configs[line]).build()
    line_model = models[line]
    v_cal = line_model.line.v_cal.ifu_values.val
    n_tiles = len(np.unique(fd.tile_idx))
    v_cal0 = v_cal[:n_tiles, 0]
    v_cal1 = v_cal[:n_tiles, 1]
    v_cal2 = v_cal[:n_tiles, 2]
    wave = float(line.split("λ")[-1])
    res = (
        (v_cal0 - v_cal1) * wave,
        (v_cal1 - v_cal2) * wave,
        (v_cal2 - v_cal0) * wave,
    )
    return res


def big_ass_plot(the_chosen_lines, title, fname):
    i_tiles = np.arange(1, 20)
    fig, ax = plt.subplots(
        1,
        2,
        figsize=[10, 5],
        gridspec_kw={"width_ratios": [7, 1]},
        sharex=False,
        sharey=True,
        layout="compressed",
    )
    # plt.title(title)
    all_v0 = []
    all_v1 = []
    all_v2 = []

    c_kms = C_CONST.to("km/s").value

    for i, line in enumerate(the_chosen_lines):
        v0, v1, v2 = what_the_fuck(line)

        plot_kwargs = dict(
            alpha=0.7,
            marker=".",
            markersize=10,
        )

        if i == 0:
            ax[0].plot(
                i_tiles,
                v0 / c_kms,
                c=COLORS[0],
                label=r"Spectrographs 1 \& 2",
                **plot_kwargs,
            )
            ax[0].plot(
                i_tiles,
                v1 / c_kms,
                c=COLORS[1],
                label=r"Spectrographs 2 \& 3",
                **plot_kwargs,
            )
            ax[0].plot(
                i_tiles,
                v2 / c_kms,
                c=COLORS[2],
                label=r"Spectrographs 3 \& 1",
                **plot_kwargs,
            )
        else:
            ax[0].plot(i_tiles, v0 / c_kms, c=COLORS[0], **plot_kwargs)
            ax[0].plot(i_tiles, v1 / c_kms, c=COLORS[1], **plot_kwargs)
            ax[0].plot(i_tiles, v2 / c_kms, c=COLORS[2], **plot_kwargs)

        all_v0.append(v0)
        all_v1.append(v1)
        all_v2.append(v2)
    # plt.ylabel(r"$\left( v_{\rm{cal},i} - v_{\rm{cal},j}\right) \times \lambda$")

    all_v0 = np.array(all_v0).flatten()
    all_v1 = np.array(all_v1).flatten()
    all_v2 = np.array(all_v2).flatten()

    bins = np.linspace(-1, 1, 80)
    hist_kwargs = dict(bins=bins, alpha=0.7, orientation="horizontal", density=True)
    h0 = ax[1].hist(all_v0 / c_kms, color="C0", **hist_kwargs)
    h1 = ax[1].hist(all_v1 / c_kms, color="C1", **hist_kwargs)
    h2 = ax[1].hist(all_v2 / c_kms, color="C2", **hist_kwargs)
    ax[1].set_xticks([])

    ax[0].set_ylabel(r"$\Delta \lambda \, \mathrm{[\AA]}$")
    ax[0].set_xlabel(r"Tile/Pointing")
    ax[0].set_ylim(-0.5, 0.5)
    ax[0].set_xticks(i_tiles)
    ax[0].legend(loc=(0.15, 0.7))
    plt.savefig(fname)
    plt.show()


def make_title(name, lines):
    title = r""
    for line in lines:
        title += make_str_label_safe(line) + ", "
    title = title[:-2]  # Remove trailing comma and space
    title = rf"{name}: ${{\rm {title}}}$"
    return title


big_ass_plot(LINES_BLUE, make_title("Blue arm", LINES_BLUE), "vcal_blue.pdf")
big_ass_plot(LINES_RED, make_title("Red arm", LINES_RED), "vcal_red.pdf")
