# write.py
# This writes all the DataConfig objects to files for later use.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from config_io import save_config
from line_data import LINE_CENTRES
from lvm_tools import DataConfig, FitDataBuilder, LVMTile, LVMTileCollection

from data import DRP_FILES

plt.style.use("mpl_drip.custom")


plots_path = Path("../plots/config_plots")


# Global settings
BAD_FLUX_THRESHOLD = -0.1e-13
LINE_WINDOW_HALF = 8.0
LINES_TO_SKIP = [
    # "[OII]λ3727", NOTE keep commented (old name for doublet)
    # "[OII]λ3726",
    "HeIλ3820",
    "CaIIλ3934",
    "HeIλ4026",
    "Hδλ4102",
    "HeIλ4121",
    "CIIλ4267",
    "Hγλ4340",
    "[OIII]λ4363",
    "HeIλ4471",
    # "OIIλ4650", NOTE keep commented (V1 multiplet we don't do this)
    # "OIIλ4660", NOTE keep commented (V1 multiplet we don't do this)
    "HeIIλ4686",
    "Hβλ4861",
    "[OIII]λ5007",
    "HeIλ5016",
    "NIIλ5680",
    "[NII]λ5755",
    "HeIλ5876",
    # "NaIλ5890", NOTE absorbtion? I guess sky
    # "NaIλ5896", NOTE absorbtion? I guess sky
    "[SIII]λ6312",
    "[NII]λ6548",
    "Hαλ6563",
    "CIIλ6578",
    "[NII]λ6583",
    "HeIλ6678",
    "[SII]λ6716",
    "[SII]λ6731",
    "FeIλ6855",
    "HeIλ7065",
    "CIIλ7236",
    "HeIλ7281",
    "[OII]λ7319",
    "OIλ7774",
    # "OIλ8446", # NOTE definitely sky
    "CaIIλ8498",
    "[SIII]λ9069",
    "[SIII]λ9531",
]


def get_λ_range(
    line_centre_λ: float,
    window_lower=LINE_WINDOW_HALF,
    window_upper=LINE_WINDOW_HALF,
) -> tuple[float, float]:
    return (line_centre_λ - window_lower, line_centre_λ + window_upper)


# Per line settings
custom_configs = {
    # "[OII]λ3727": {
    #     "λ_range": get_λ_range(LINE_CENTRES["[OII]λ3727"]),
    #     "normalise_F_scale": 1e-12,
    #     "normalise_F_offset": 0.0,
    #     "F_range": (BAD_FLUX_THRESHOLD, 0.5e-13),
    # },
    "[OII]λ3726": {
        "λ_range": get_λ_range(LINE_CENTRES["[OII]λ3726"], window_upper=11),
        "normalise_F_scale": 1e-12,
        "normalise_F_offset": 0.0,
        "F_range": (BAD_FLUX_THRESHOLD, 0.5e-13),
    },
    "HeIλ3820": {
        "λ_range": get_λ_range(LINE_CENTRES["HeIλ3820"]),
        "normalise_F_scale": 1e-14,
        "normalise_F_offset": 0.0,
        "F_range": (BAD_FLUX_THRESHOLD, 0.5e-13),
    },
    "CaIIλ3934": {
        "λ_range": get_λ_range(LINE_CENTRES["CaIIλ3934"]),
        "normalise_F_scale": 1e-14,
        "normalise_F_offset": 0.0,
        "F_range": (BAD_FLUX_THRESHOLD, 0.5e-13),
    },
    "HeIλ4026": {
        "λ_range": get_λ_range(LINE_CENTRES["HeIλ4026"]),
        "normalise_F_scale": 1e-14,
        "normalise_F_offset": 0.0,
        "F_range": (BAD_FLUX_THRESHOLD, 0.5e-13),
    },
    "Hδλ4102": {
        "λ_range": get_λ_range(LINE_CENTRES["Hδλ4102"]),
        "normalise_F_scale": 2e-13,
        "normalise_F_offset": 0.0,
        "F_range": (BAD_FLUX_THRESHOLD, 0.5e-13),
    },
    "HeIλ4121": {
        "λ_range": get_λ_range(LINE_CENTRES["HeIλ4121"]),
        "normalise_F_scale": 1e-14,
        "normalise_F_offset": 0.0,
        "F_range": (BAD_FLUX_THRESHOLD, 0.5e-13),
    },
    "Hγλ4340": {
        "λ_range": get_λ_range(LINE_CENTRES["Hγλ4340"]),
        "normalise_F_scale": 3e-13,
        "normalise_F_offset": 0.0,
        "F_range": (BAD_FLUX_THRESHOLD, 0.5e-13),
    },
    "CIIλ4267": {
        "λ_range": get_λ_range(LINE_CENTRES["CIIλ4267"]),
        "normalise_F_scale": 1e-14,
        "normalise_F_offset": 0.0,
        "F_range": (BAD_FLUX_THRESHOLD, 0.5e-13),
    },
    "[OIII]λ4363": {
        "λ_range": get_λ_range(LINE_CENTRES["[OIII]λ4363"]),
        "normalise_F_scale": 1e-14,
        "normalise_F_offset": 0.0,
        "F_range": (BAD_FLUX_THRESHOLD, 0.5e-13),
    },
    "HeIλ4471": {
        "λ_range": get_λ_range(LINE_CENTRES["HeIλ4471"]),
        "normalise_F_scale": 5e-14,
        "normalise_F_offset": 0.0,
        "F_range": (BAD_FLUX_THRESHOLD, 1e-14),
    },
    # "OIIλ4650": {
    #     "λ_range": get_λ_range(LINE_CENTRES["OIIλ4650"]),
    #     "normalise_F_scale": 1e-14,
    #     "normalise_F_offset": 0.0,
    #     "F_range": (BAD_FLUX_THRESHOLD, 0.5e-13),
    # },
    # "OIIλ4660": {
    #     "λ_range": get_λ_range(LINE_CENTRES["OIIλ4660"]),
    #     "normalise_F_scale": 1e-14,
    #     "normalise_F_offset": 0.0,
    #     "F_range": (BAD_FLUX_THRESHOLD, 0.5e-13),
    # },
    "HeIIλ4686": {
        "λ_range": get_λ_range(LINE_CENTRES["HeIIλ4686"]),
        "normalise_F_scale": 1e-14,
        "normalise_F_offset": 0.0,
        "F_range": (BAD_FLUX_THRESHOLD, 0.5e-13),
    },
    "Hβλ4861": {
        "λ_range": get_λ_range(LINE_CENTRES["Hβλ4861"]),
        "normalise_F_scale": 8e-13,
        "normalise_F_offset": 0.0,
        "F_range": (BAD_FLUX_THRESHOLD, 0.5e-13),
    },
    "[OIII]λ5007": {
        "λ_range": get_λ_range(LINE_CENTRES["[OIII]λ5007"]),
        "normalise_F_scale": 2e-12,
        "normalise_F_offset": 0.0,
        "F_range": (BAD_FLUX_THRESHOLD, 0.5e-13),
    },
    "HeIλ5016": {
        "λ_range": get_λ_range(LINE_CENTRES["HeIλ5016"], window_lower=5),
        "normalise_F_scale": 2e-14,
        "normalise_F_offset": 0.0,
        "F_range": (BAD_FLUX_THRESHOLD, 0.5e-13),
    },
    "NIIλ5680": {
        "λ_range": get_λ_range(LINE_CENTRES["NIIλ5680"]),
        "normalise_F_scale": 1e-14,
        "normalise_F_offset": 0.0,
        "F_range": (BAD_FLUX_THRESHOLD, 0.5e-13),
    },
    "[NII]λ5755": {
        "λ_range": get_λ_range(LINE_CENTRES["[NII]λ5755"]),
        "normalise_F_scale": 1e-14,
        "normalise_F_offset": 0.0,
        "F_range": (BAD_FLUX_THRESHOLD, 0.5e-13),
    },
    "HeIλ5876": {
        "λ_range": get_λ_range(LINE_CENTRES["HeIλ5876"]),
        "normalise_F_scale": 1e-13,
        "normalise_F_offset": 0.0,
        "F_range": (BAD_FLUX_THRESHOLD, 0.5e-13),
    },
    "NaIλ5890": {
        "λ_range": get_λ_range(LINE_CENTRES["NaIλ5890"]),
        "normalise_F_scale": 1e-14,
        "normalise_F_offset": 0.0,
        "F_range": (BAD_FLUX_THRESHOLD, 0.5e-13),
    },
    "NaIλ5896": {
        "λ_range": get_λ_range(LINE_CENTRES["NaIλ5896"]),
        "normalise_F_scale": 1e-14,
        "normalise_F_offset": 0.0,
        "F_range": (BAD_FLUX_THRESHOLD, 0.5e-13),
    },
    "[SIII]λ6312": {
        "λ_range": get_λ_range(LINE_CENTRES["[SIII]λ6312"]),
        "normalise_F_scale": 1.3e-14,
        "normalise_F_offset": 0.0,
        "F_range": (BAD_FLUX_THRESHOLD, 0.5e-13),
    },
    "[NII]λ6548": {
        "λ_range": get_λ_range(LINE_CENTRES["[NII]λ6548"]),
        "normalise_F_scale": 3e-13,
        "normalise_F_offset": 0.0,
        "F_range": (BAD_FLUX_THRESHOLD, 0.5e-13),
    },
    "Hαλ6563": {
        "λ_range": get_λ_range(LINE_CENTRES["Hαλ6563"]),
        "normalise_F_scale": 5e-12,
        "normalise_F_offset": 0.0,
        "F_range": (BAD_FLUX_THRESHOLD, 0.5e-13),
    },
    "CIIλ6578": {
        "λ_range": get_λ_range(
            LINE_CENTRES["CIIλ6578"],
            window_lower=6,
            window_upper=2.5,
        ),
        "normalise_F_scale": 1e-14,
        "normalise_F_offset": 0.0,
        "F_range": (BAD_FLUX_THRESHOLD, 0.5e-13),
    },
    "[NII]λ6583": {
        "λ_range": get_λ_range(LINE_CENTRES["[NII]λ6583"]),
        "normalise_F_scale": 1e-12,
        "normalise_F_offset": 0.0,
        "F_range": (BAD_FLUX_THRESHOLD, 0.5e-13),
    },
    "HeIλ6678": {
        "λ_range": get_λ_range(LINE_CENTRES["HeIλ6678"]),
        "normalise_F_scale": 5e-14,
        "normalise_F_offset": 0.0,
        "F_range": (BAD_FLUX_THRESHOLD, 1e-14),
    },
    "[SII]λ6716": {
        "λ_range": get_λ_range(LINE_CENTRES["[SII]λ6716"]),
        "normalise_F_scale": 5e-13,
        "normalise_F_offset": 0.0,
        "F_range": (BAD_FLUX_THRESHOLD, 0.5e-13),
    },
    "[SII]λ6731": {
        "λ_range": get_λ_range(LINE_CENTRES["[SII]λ6731"]),
        "normalise_F_scale": 3e-13,
        "normalise_F_offset": 0.0,
        "F_range": (BAD_FLUX_THRESHOLD, 0.5e-13),
    },
    "FeIλ6855": {
        "λ_range": get_λ_range(
            LINE_CENTRES["FeIλ6855"],
            window_lower=3,
            window_upper=5,
        ),
        "normalise_F_scale": 1e-14,
        "normalise_F_offset": 0.0,
        "F_range": (BAD_FLUX_THRESHOLD, 0.5e-13),
    },
    "HeIλ7065": {
        "λ_range": get_λ_range(LINE_CENTRES["HeIλ7065"]),
        "normalise_F_scale": 3e-14,
        "normalise_F_offset": 0.0,
        "F_range": (BAD_FLUX_THRESHOLD, 0.5e-13),
    },
    "CIIλ7236": {
        "λ_range": get_λ_range(LINE_CENTRES["CIIλ7236"]),
        "normalise_F_scale": 1e-14,
        "normalise_F_offset": 0.0,
        "F_range": (BAD_FLUX_THRESHOLD, 0.5e-13),
    },
    "HeIλ7281": {
        "λ_range": get_λ_range(LINE_CENTRES["HeIλ7281"]),
        "normalise_F_scale": 1e-14,
        "normalise_F_offset": 0.0,
        "F_range": (BAD_FLUX_THRESHOLD, 0.5e-13),
    },
    "[OII]λ7319": {
        "λ_range": get_λ_range(LINE_CENTRES["[OII]λ7319"]),
        "normalise_F_scale": 2e-14,
        "normalise_F_offset": 0.0,
        "F_range": (BAD_FLUX_THRESHOLD, 0.5e-13),
    },
    "OIλ7774": {
        "λ_range": get_λ_range(LINE_CENTRES["OIλ7774"]),
        "normalise_F_scale": 1e-13,
        "normalise_F_offset": 0.0,
        "F_range": (BAD_FLUX_THRESHOLD, 0.5e-13),
    },
    "OIλ8446": {
        "λ_range": get_λ_range(LINE_CENTRES["OIλ8446"]),
        "normalise_F_scale": 1e-14,
        "normalise_F_offset": 0.0,
        "F_range": (BAD_FLUX_THRESHOLD, 0.5e-13),
    },
    "CaIIλ8498": {
        "λ_range": get_λ_range(
            LINE_CENTRES["CaIIλ8498"],
            window_lower=3,
            window_upper=3,
        ),
        "normalise_F_scale": 1e-14,
        "normalise_F_offset": 0.0,
        "F_range": (BAD_FLUX_THRESHOLD, 0.5e-13),
    },
    "[SIII]λ9069": {
        "λ_range": get_λ_range(LINE_CENTRES["[SIII]λ9069"]),
        "normalise_F_scale": 5e-13,
        "normalise_F_offset": 0.0,
        "F_range": (BAD_FLUX_THRESHOLD, 0.5e-13),
    },
    "[SIII]λ9531": {
        "λ_range": get_λ_range(LINE_CENTRES["[SIII]λ9531"]),
        "normalise_F_scale": 1e-12,
        "normalise_F_offset": 0.0,
        "F_range": (BAD_FLUX_THRESHOLD, 0.5e-13),
    },
}


def main():
    print("Loading tiles...")

    tiles = LVMTileCollection.from_tiles([LVMTile.from_file(Path(f)) for f in DRP_FILES])

    # Iterate over each line and create a DataConfig
    for line, config in custom_configs.items():
        if line in LINES_TO_SKIP:
            print(f"Skipping {line} as requested.")
            continue

        print(f"Creating DataConfig for {line}...")

        data_config = DataConfig.from_tiles(
            tiles,
            **config,
        )
        # Save DataConfig to yaml file
        config_path = Path(f"{line}.yaml")
        data_config_dict = data_config.to_dict()

        builder = FitDataBuilder(tiles, data_config)
        fd = builder.build()

        # Temporarily disable plotting with latex
        with plt.rc_context({"text.usetex": False, "font.family": "sans-serif"}):
            fig, ax = plt.subplots(figsize=[8, 8], layout="compressed")
            ax.set_title(rf"{line} Max flux per spaxel, $\lambda \in {data_config.λ_range}$")
            cs = ax.scatter(
                fd.predict_α(fd.α),
                fd.predict_δ(fd.δ),
                c=np.nanmax(fd.flux, axis=0),
                s=3,
                vmax=1.0,
            )
            plt.colorbar(cs, ax=ax, label=r"$F_{\rm max}$")
            ax.set_aspect(1)
            ax.set_xlabel(r"$\alpha$ [deg]")
            ax.set_ylabel(r"$\delta$ [deg]")
            ax.set_xlim(data_config.α_range[1], data_config.α_range[0])
            ax.set_ylim(*data_config.δ_range)
            plt.savefig(plots_path / f"{line}_max_flux_spaxel.pdf")
            plt.close(fig)

            fig, ax = plt.subplots(figsize=[8, 4], layout="compressed")
            ax.set_title(rf"{line} Average spectrum, $\lambda \in {data_config.λ_range}$")
            ax.plot(fd.λ, np.nanmean(fd.predict_flux(fd.flux), axis=1))
            ax.set_xlabel(r"$\lambda$ [${\rm \AA}$]")
            plt.savefig(plots_path / f"{line}_avg_spectrum.pdf")
            plt.close(fig)

        # Save the DataConfig as a YAML file
        save_config(data_config_dict, config_path)


if __name__ == "__main__":
    main()
