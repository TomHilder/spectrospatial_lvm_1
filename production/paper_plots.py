# paper_plots.py
# Generate all the plots for the paper

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cmasher as cmr
import jax
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import spectracles as new_pkg
import spectracles.model as new_model_mod
import spectracles.model.share_module as new_share_mod
from configs.config_io import load_config
from configs.data import DRP_FILES
from epsf_stuff.estimate_psf_no_deps import estimate_s, estimate_s_nu
from epsf_stuff.estimate_psf_no_deps import kernel as matern_kernel
from lvm_tools import DataConfig, FitDataBuilder, LVMTile, LVMTileCollection
from lvm_tools.utils.mask import mask_near_points
from matplotlib.ticker import MaxNLocator, MultipleLocator
from mpl_drip import colormaps
from scipy.optimize import minimize
from scipy.stats import binned_statistic
from spectracles import load_model
from spectracles.model.data import SpatialDataGeneric
from spectracles.model.spatial import get_freqs

rng = np.random.default_rng(42)
jax.config.update("jax_enable_x64", True)

# Monkey-patch `modelling_lib` since dill looks for it, and it's now `spectracles`
sys.modules["modelling_lib"] = new_pkg
sys.modules["modelling_lib.model"] = new_model_mod
sys.modules["modelling_lib.model.share_module"] = new_share_mod

plt.style.use("mpl_drip.custom")
jax.config.update("jax_enable_x64", True)

mpl.rcParams.update(
    {
        "text.latex.preamble": r"\usepackage{bm}\newcommand\ion[2]{\textrm{#1\,{\Large\uppercase\expandafter{\romannumeral #2}}}}",
    }
)
# mpl.rcParams.update(
#     {
#         "text.latex.preamble": r"\usepackage{bm}\DeclareRobustCommand{\ion}[2]{\textrm{#1\,\MakeUppercase{\romannumeral #2}}}"
#     }
# )

# Constants
FITS_PATH = Path("fits_before_migration/fits")
FITS_PATH2 = Path("fits")
CONFIG_PATH = Path("configs")
PLOTS_PATH = Path("plots_paper")
LAYOUT_DETAILS = dict(w_pad=0.01, h_pad=0.12, hspace=0, wspace=0)

NINE_PANELS_LINES = [
    "[OII]λ3726",
    "Hγλ4340",
    "Hβλ4861",
    "[OIII]λ5007",
    "Hαλ6563",
    "[NII]λ6583",
    "[SII]λ6716",
    "[SII]λ6731",
    "[SIII]λ9531",
]
NINE_PANELS_LINES_LABELS = [
    r"$\textbf{[\ion{O}{2}]} \, \boldsymbol{\lambda}\boldsymbol{\lambda}\textbf{3726,3729}\, \textbf{\AA}$",
    r"$\textbf{H}\boldsymbol{\gamma} \, \boldsymbol{\lambda}\textbf{4340}\, \textbf{\AA}$",
    r"$\textbf{H}\boldsymbol{\beta} \, \boldsymbol{\lambda}\textbf{4861}\, \textbf{\AA}$",
    r"$\textbf{[\ion{O}{3}]} \, \boldsymbol{\lambda}\textbf{5007}\, \textbf{\AA}$",
    r"$\textbf{H}\boldsymbol{\alpha} \, \boldsymbol{\lambda}\textbf{6563}\, \textbf{\AA}$",
    r"$\textbf{[\ion{N}{2}]} \, \boldsymbol{\lambda}\textbf{6583}\, \textbf{\AA}$",
    r"$\textbf{[\ion{S}{2}]} \, \boldsymbol{\lambda}\textbf{6716}\, \textbf{\AA}$",
    r"$\textbf{[\ion{S}{2}]} \, \boldsymbol{\lambda}\textbf{6731}\, \textbf{\AA}$",
    r"$\textbf{[\ion{S}{3}]} \, \boldsymbol{\lambda}\textbf{9531}\, \textbf{\AA}$",
]
NINE_PANELS_LINES_LABELS = {
    line: label for line, label in zip(NINE_PANELS_LINES, NINE_PANELS_LINES_LABELS)
}
AURORAL_LINES = [
    "[OIII]λ4363",
    "[NII]λ5755",
    "[SIII]λ6312",
    "[OII]λ7319",
]
AURORAL_LINES_LABELS = [
    r"$\textbf{[\ion{O}{3}]} \, \boldsymbol{\lambda}\textbf{4363}\, \textbf{\AA}$",
    r"$\textbf{[\ion{N}{2}]} \, \boldsymbol{\lambda}\textbf{5755}\, \textbf{\AA}$",
    r"$\textbf{[\ion{S}{3}]} \, \boldsymbol{\lambda}\textbf{6312}\, \textbf{\AA}$",
    r"$\textbf{[\ion{O}{2}]} \, \boldsymbol{\lambda}\textbf{7319}\, \textbf{\AA}$",
]
AURORAL_LINES_LABELS = {line: label for line, label in zip(AURORAL_LINES, AURORAL_LINES_LABELS)}
HELIUM_RLS = [
    "HeIλ3820",  # probably
    "HeIλ4026",  # probably
    "HeIλ4471",  # definitely include
    # "HeIλ5016",
    "HeIλ5876",  # definitely include
    "HeIλ6678",  # definitely include
    "HeIλ7065",  # probably
    # "HeIλ7281",
]
HELIUM_RLS_LABELS = [
    r"$\textbf{\ion{He}{1}} \, \boldsymbol{\lambda}\textbf{3820}\, \textbf{\AA}$",
    r"$\textbf{\ion{He}{1}} \, \boldsymbol{\lambda}\textbf{4026}\, \textbf{\AA}$",
    r"$\textbf{\ion{He}{1}} \, \boldsymbol{\lambda}\textbf{4471}\, \textbf{\AA}$",
    r"$\textbf{\ion{He}{1}} \, \boldsymbol{\lambda}\textbf{5876}\, \textbf{\AA}$",
    r"$\textbf{\ion{He}{1}} \, \boldsymbol{\lambda}\textbf{6678}\, \textbf{\AA}$",
    r"$\textbf{\ion{He}{1}} \, \boldsymbol{\lambda}\textbf{7065}\, \textbf{\AA}$",
]
HELIUM_RLS_LABELS = {line: label for line, label in zip(HELIUM_RLS, HELIUM_RLS_LABELS)}
METAL_RLS = [
    "CIIλ4267",
    "NIIλ5680",
    "CIIλ6578",
    "CIIλ7236",
]
METAL_RLS_LABELS = [
    r"$\textbf{\ion{C}{2}} \, \boldsymbol{\lambda}\textbf{4267}\, \textbf{\AA}$",
    r"$\textbf{\ion{N}{2}} \, \boldsymbol{\lambda}\textbf{5680}\, \textbf{\AA}$",
    r"$\textbf{\ion{C}{2}} \, \boldsymbol{\lambda}\textbf{6578}\, \textbf{\AA}$",
    r"$\textbf{\ion{C}{2}} \, \boldsymbol{\lambda}\textbf{7236}\, \textbf{\AA}$",
]
METAL_RLS_LABELS = {line: label for line, label in zip(METAL_RLS, METAL_RLS_LABELS)}

ALL_LABELS = NINE_PANELS_LINES_LABELS | AURORAL_LINES_LABELS | HELIUM_RLS_LABELS | METAL_RLS_LABELS


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


def load_models(line_list: List[str]) -> Tuple[Dict, Dict]:
    """Load models and configs for given lines."""
    model_paths = list(FITS_PATH.glob("*/*6.model"))
    model_paths += list(FITS_PATH2.glob("*/*8.model"))
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


def prepare_spatial_grid(
    tiles, line: str, configs: Dict, n_dense: int = 2000, threshold: float = 0.02
):
    """Prepare spatial coordinate data for plotting."""
    fd = FitDataBuilder(tiles, configs[line]).build()

    dense_shape = (n_dense, n_dense)
    α_dense_1D = np.linspace(fd.α.min(), fd.α.max(), n_dense)
    δ_dense_1D = np.linspace(fd.δ.min(), fd.δ.max(), n_dense)
    α_dense, δ_dense = np.meshgrid(α_dense_1D, δ_dense_1D)

    αδ_dense = SpatialDataGeneric(
        α_dense.flatten(),
        δ_dense.flatten(),
        idx=np.arange(n_dense * n_dense, dtype=int),
    )

    α_plot = fd.predict_α(α_dense)
    δ_plot = fd.predict_δ(δ_dense)

    mask = mask_near_points(α_dense_1D, δ_dense_1D, fd.α, fd.δ, threshold=threshold)

    # Plot boundaries
    x_min = fd.predict_α(fd.α.max())
    x_max = fd.predict_α(fd.α.min())
    y_min = fd.predict_δ(fd.δ.min())
    y_max = fd.predict_δ(fd.δ.max())

    return {
        "dense_shape": dense_shape,
        "αδ_dense": αδ_dense,
        "α_plot": α_plot,
        "δ_plot": δ_plot,
        "mask": mask,
        "bounds": (x_min, x_max, y_min, y_max),
    }


def setup_nine_panel_figure(figsize=(16, 16), dpi=500):
    """Create and setup the 3x3 figure."""
    fig, axes = plt.subplots(
        3,
        3,
        figsize=figsize,
        dpi=dpi,
        sharex=True,
        sharey=True,
        layout="compressed",
    )
    fig.get_layout_engine().set(**LAYOUT_DETAILS)
    return fig, axes


def setup_axis(ax, bounds: Tuple, line: str):
    """Setup individual axis properties."""
    x_min, x_max, y_min, y_max = bounds

    # Add line label
    old_label = make_str_label_safe(line)
    # label_line = str(old_label.split(r"\lambda")[0])
    # label_wave = str(old_label.split(r"\lambda")[1])
    label = ALL_LABELS[line]
    t = ax.text(
        99.0,
        5.66,
        label,
        c="black",
        fontsize="large",
        zorder=1,
    )
    t.set_bbox(dict(facecolor="white", alpha=0.8, edgecolor="white"))

    # Set axis limits and properties
    axis_pad_fac = 0.0
    axis_extent = max(abs(x_max - x_min), abs(y_max - y_min))
    axis_pad = axis_extent * axis_pad_fac

    ax.set_xlim(x_min + axis_pad, x_max - axis_pad)
    ax.set_ylim(y_min - axis_pad, y_max + axis_pad)
    ax.set_aspect(1)

    ax.tick_params(zorder=10)
    ax.tick_params(axis="both", which="both", direction="in")

    # Set tick locators
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax.yaxis.set_minor_locator(MultipleLocator(0.25))


def add_colorbar_if_first(fig, cs, ax, i: int, label: str):
    """Add colorbar only for the first subplot."""
    if i == 0:
        cb = fig.colorbar(cs, ax=ax, pad=0.02, location="top", aspect=30)
        cb.ax.tick_params(which="both", top=True, bottom=False, direction="out")
        cb.locator = MaxNLocator(nbins=5)
        cb.update_ticks()
        cb.set_label(label=label, labelpad=10)
        cb.ax.set_axisbelow(False)


def finalize_nine_panel_plot(fig, axes, filename: str):
    """Add labels and save the plot."""
    for i in range(3):
        axes[-1, i].set_xlabel(r"$\alpha$ [deg]")
        axes[i, 0].set_ylabel(r"$\delta$ [deg]")
    plt.savefig(PLOTS_PATH / filename)
    plt.close(fig)


def set_face_color(axes, color: str):
    for ax in axes.flat:
        ax.set_facecolor(color)


# =============================================================================
# PLOT FUNCTIONS
# =============================================================================


def plot_nine_panels_flux(tiles):
    """Create nine panel flux intensity plot."""
    print("Creating flux plot...", flush=True)

    models, configs = load_models(NINE_PANELS_LINES)
    spatial_grid = prepare_spatial_grid(tiles, NINE_PANELS_LINES[0], configs)
    fig, axes = setup_nine_panel_figure()

    fig.suptitle(r"$\textsf{\textbf{Strong Lines}}$", fontsize="32", c="dimgrey")

    for i, (ax, line) in enumerate(zip(axes.flat, NINE_PANELS_LINES)):
        try:
            line_model = models[line]
            line_config = configs[line]

            # Get line-specific fit data
            fd = FitDataBuilder(tiles, line_config).build()

            # Calculate flux data
            try:
                A_pred = fd.predict_flux(
                    np.where(
                        spatial_grid["mask"],
                        line_model.line.A(spatial_grid["αδ_dense"]).reshape(
                            spatial_grid["dense_shape"]
                        ),
                        np.nan,
                    )
                )
            except AttributeError:
                A_pred = fd.predict_flux(
                    np.where(
                        spatial_grid["mask"],
                        line_model.line.A_1(spatial_grid["αδ_dense"]).reshape(
                            spatial_grid["dense_shape"]
                        )
                        + line_model.line.A_2(spatial_grid["αδ_dense"]).reshape(
                            spatial_grid["dense_shape"]
                        ),
                        np.nan,
                    )
                )
            A_pred /= 1e-12

            # Flux calibration for Halpha
            f_cal = np.nanmedian(line_model.line.f_cal(fd.αδ_data))
            A_pred *= f_cal
        except KeyError:
            print(f"{line} missing")
            A_pred = np.nan * np.ones(spatial_grid["dense_shape"])

        # Calculate vmax for this panel
        n_pix_centre = 600
        l_i = 2000 // 2 - n_pix_centre  # Using default n_dense=2000
        u_i = 2000 // 2 + n_pix_centre
        A_vmax = np.nanmax(A_pred[l_i:u_i, l_i:u_i])

        # Create plot
        cs = ax.pcolormesh(
            spatial_grid["α_plot"],
            spatial_grid["δ_plot"],
            A_pred,
            # cmap=sns.color_palette("viridis", as_cmap=True),
            cmap="cmr.voltage_r",
            vmin=0,
            vmax=A_vmax,
            shading="auto",
            rasterized=True,
            zorder=0,
        )

        cb = fig.colorbar(cs, ax=ax, pad=0.02, location="top", aspect=30)
        cb.ax.tick_params(which="both", top=True, bottom=False, direction="out")
        cb.locator = MaxNLocator(nbins=5)
        cb.update_ticks()
        if i == 0:
            cb.set_label(label=r"Flux [10$^{-12}$ erg s$^{-1}$ cm$^{-2}$]", labelpad=10)
        cb.ax.set_axisbelow(False)

        # Setup axis
        setup_axis(ax, spatial_grid["bounds"], line)

        # Facecolor
        set_face_color(axes, "white")

    finalize_nine_panel_plot(fig, axes, "nine_panels_flux.pdf")
    print("Done.")


def plot_nine_panels_velocity(tiles):
    """Create nine panel radial velocity plot."""
    print("Creating velocity plot...", flush=True)

    models, configs = load_models(NINE_PANELS_LINES)
    spatial_grid = prepare_spatial_grid(tiles, NINE_PANELS_LINES[0], configs)
    fig, axes = setup_nine_panel_figure()

    fig.suptitle(r"$\textsf{\textbf{Strong Lines}}$", fontsize="32", c="dimgrey")

    for i, (ax, line) in enumerate(zip(axes.flat, NINE_PANELS_LINES)):
        try:
            line_model = models[line]
            # Calculate velocity data
            v_pred = np.where(
                spatial_grid["mask"],
                line_model.line.v(spatial_grid["αδ_dense"]).reshape(spatial_grid["dense_shape"]),
                np.nan,
            )
            # print(
            #     f"{line} effective vsyst: {line_model.line.v_syst.val[0] - np.nanmedian(v_pred):.2f} km/s"
            # )
            v_pred -= np.nanmedian(v_pred)
        except KeyError:
            print(f"{line} missing")
            v_pred = np.nan * np.ones(spatial_grid["dense_shape"])

        v_vmax = 14

        # Create plot
        cs = ax.pcolormesh(
            spatial_grid["α_plot"],
            spatial_grid["δ_plot"],
            v_pred,
            # cmap="RdBu_r",
            # cmap="cmr.fusion_r",
            cmap="red_white_blue_r",
            vmin=-v_vmax,
            vmax=v_vmax,
            shading="auto",
            rasterized=True,
            zorder=0,
        )

        # Add colorbar for first plot only
        add_colorbar_if_first(fig, cs, ax, i, r"Radial Velocity [km s$^{-1}$]")

        # Setup axis
        setup_axis(ax, spatial_grid["bounds"], line)

    finalize_nine_panel_plot(fig, axes, "nine_panels_v_radial.pdf")
    print("Done.")


def plot_nine_panels_dispersion(tiles):
    """Create nine panel velocity dispersion plot."""
    print("Creating dispersion plot...", flush=True)

    models, configs = load_models(NINE_PANELS_LINES)
    spatial_grid = prepare_spatial_grid(tiles, NINE_PANELS_LINES[0], configs)
    fig, axes = setup_nine_panel_figure()

    fig.suptitle(r"$\textsf{\textbf{Strong Lines}}$", fontsize="32", c="dimgrey")

    for i, (ax, line) in enumerate(zip(axes.flat, NINE_PANELS_LINES)):
        try:
            line_model = models[line]
            # Calculate dispersion data
            σ_pred = np.abs(
                np.where(
                    spatial_grid["mask"],
                    line_model.line.vσ(spatial_grid["αδ_dense"]).reshape(
                        spatial_grid["dense_shape"]
                    ),
                    np.nan,
                )
            )
        except KeyError:
            print(f"{line} missing")
            σ_pred = np.nan * np.ones(spatial_grid["dense_shape"])

        σ_vmax = 40

        # Create plot
        cs = ax.pcolormesh(
            spatial_grid["α_plot"],
            spatial_grid["δ_plot"],
            σ_pred,
            cmap="cmr.torch_r",
            vmin=0,
            vmax=σ_vmax,
            shading="auto",
            rasterized=True,
            zorder=0,
        )

        # Add colorbar for first plot only
        add_colorbar_if_first(fig, cs, ax, i, r"Velocity Dispersion [km s$^{-1}$]")

        # Setup axis
        setup_axis(ax, spatial_grid["bounds"], line)

    finalize_nine_panel_plot(fig, axes, "nine_panels_dispersion.pdf")
    print("Done.")


def plot_auroral_flux(tiles):
    """Create auroral lines flux intensity plot."""
    print("Creating auroral lines flux plot...", flush=True)

    models, configs = load_models(AURORAL_LINES)
    spatial_grid = prepare_spatial_grid(tiles, AURORAL_LINES[0], configs)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16, 16),
        dpi=500,
        sharex=True,
        sharey=True,
        layout="compressed",
    )
    fig.get_layout_engine().set(**LAYOUT_DETAILS)

    fig.suptitle(r"$\textsf{\textbf{Auroral Lines}}$", fontsize="32", c="dimgrey")

    for i, (ax, line) in enumerate(zip(axes.flat, AURORAL_LINES)):
        try:
            line_model = models[line]
            line_config = configs[line]

            # Get line-specific fit data
            fd = FitDataBuilder(tiles, line_config).build()

            # Calculate flux data
            A_pred = fd.predict_flux(
                np.where(
                    spatial_grid["mask"],
                    line_model.line.A(spatial_grid["αδ_dense"]).reshape(
                        spatial_grid["dense_shape"]
                    ),
                    np.nan,
                )
            )
            A_pred /= 1e-12

            # Flux calibration for Halpha
            f_cal = np.nanmedian(line_model.line.f_cal(fd.αδ_data))
            A_pred *= f_cal
        except KeyError:
            print(f"{line} missing")
            A_pred = np.nan * np.ones(spatial_grid["dense_shape"])

        # Calculate vmax for this panel
        n_pix_centre = 600
        l_i = 2000 // 2 - n_pix_centre  # Using default n_dense=2000
        u_i = 2000 // 2 + n_pix_centre
        A_vmax = np.nanmax(A_pred[l_i:u_i, l_i:u_i])

        # Create plot
        cs = ax.pcolormesh(
            spatial_grid["α_plot"],
            spatial_grid["δ_plot"],
            A_pred,
            # cmap=sns.color_palette("viridis", as_cmap=True),
            cmap="cmr.voltage_r",
            vmin=0,
            vmax=A_vmax,
            shading="auto",
            rasterized=True,
            zorder=0,
        )

        cb = fig.colorbar(cs, ax=ax, pad=0.02, location="top", aspect=30)
        cb.ax.tick_params(which="both", top=True, bottom=False, direction="out")
        cb.locator = MaxNLocator(nbins=5)
        cb.update_ticks()
        if i == 0:
            cb.set_label(label=r"Flux [10$^{-12}$ erg s$^{-1}$ cm$^{-2}$]", labelpad=10)
        cb.ax.set_axisbelow(False)

        # Setup axis
        setup_axis(ax, spatial_grid["bounds"], line)

        # Facecolor
        set_face_color(axes, "white")

    for i in range(2):
        axes[-1, i].set_xlabel(r"$\alpha$ [deg]")
        axes[i, 0].set_ylabel(r"$\delta$ [deg]")
    plt.savefig(PLOTS_PATH / "auroral_lines_flux.pdf")
    plt.close(fig)
    print("Done.")


def plot_helium_RL_flux(tiles):
    """Create helium recombination lines flux intensity plot."""
    print("Creating helium RL flux plot...", flush=True)

    models, configs = load_models(HELIUM_RLS)
    spatial_grid = prepare_spatial_grid(tiles, HELIUM_RLS[0], configs)

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(16, 16),
        dpi=500,
        sharex=True,
        sharey=True,
        layout="compressed",
    )
    fig.get_layout_engine().set(**LAYOUT_DETAILS)

    fig.suptitle(r"$\textsf{\textbf{Helium Recombination Lines}}$", fontsize="32", c="dimgrey")

    for i, (ax, line) in enumerate(zip(axes.flat, HELIUM_RLS)):
        try:
            line_model = models[line]
            line_config = configs[line]

            # Get line-specific fit data
            fd = FitDataBuilder(tiles, line_config).build()

            # Calculate flux data
            A_pred = fd.predict_flux(
                np.where(
                    spatial_grid["mask"],
                    line_model.line.A(spatial_grid["αδ_dense"]).reshape(
                        spatial_grid["dense_shape"]
                    ),
                    np.nan,
                )
            )
            A_pred /= 1e-12

            # Flux calibration for Halpha
            f_cal = np.nanmedian(line_model.line.f_cal(fd.αδ_data))
            A_pred *= f_cal
        except KeyError:
            print(f"{line} missing")
            A_pred = np.nan * np.ones(spatial_grid["dense_shape"])

        # Calculate vmax for this panel
        n_pix_centre = 600
        l_i = 2000 // 2 - n_pix_centre  # Using default n_dense=2000
        u_i = 2000 // 2 + n_pix_centre
        A_vmax = np.nanmax(A_pred[l_i:u_i, l_i:u_i])

        # Create plot
        cs = ax.pcolormesh(
            spatial_grid["α_plot"],
            spatial_grid["δ_plot"],
            A_pred,
            # cmap=sns.color_palette("viridis", as_cmap=True),
            cmap="cmr.voltage_r",
            vmin=0,
            vmax=A_vmax,
            shading="auto",
            rasterized=True,
            zorder=0,
        )

        cb = fig.colorbar(cs, ax=ax, pad=0.02, location="top", aspect=30)
        cb.ax.tick_params(which="both", top=True, bottom=False, direction="out")
        cb.locator = MaxNLocator(nbins=5)
        cb.update_ticks()
        if i == 0:
            cb.set_label(label=r"Flux [10$^{-12}$ erg s$^{-1}$ cm$^{-2}$]", labelpad=10)
        cb.ax.set_axisbelow(False)

        # Setup axis
        setup_axis(ax, spatial_grid["bounds"], line)

        # Facecolor
        set_face_color(axes, "white")

    for i in range(3):
        axes[-1, i].set_xlabel(r"$\alpha$ [deg]")
    for i in range(2):
        axes[i, 0].set_ylabel(r"$\delta$ [deg]")
    plt.savefig(PLOTS_PATH / "helium_recombination_lines_flux.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Done.")


def plot_metal_RL_flux(tiles):
    """Create metal recombination lines flux intensity plot."""
    print("Creating metal RL flux plot...", flush=True)

    models, configs = load_models(METAL_RLS)
    spatial_grid = prepare_spatial_grid(tiles, METAL_RLS[0], configs)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16, 16),
        dpi=500,
        sharex=True,
        sharey=True,
        layout="compressed",
    )
    fig.get_layout_engine().set(**LAYOUT_DETAILS)

    fig.suptitle(r"$\textsf{\textbf{Metal Recombination Lines}}$", fontsize="32", c="dimgrey")

    for i, (ax, line) in enumerate(zip(axes.flat, METAL_RLS)):
        try:
            line_model = models[line]
            line_config = configs[line]

            # Get line-specific fit data
            fd = FitDataBuilder(tiles, line_config).build()

            # Calculate flux data
            A_pred = fd.predict_flux(
                np.where(
                    spatial_grid["mask"],
                    line_model.line.A(spatial_grid["αδ_dense"]).reshape(
                        spatial_grid["dense_shape"]
                    ),
                    np.nan,
                )
            )
            A_pred /= 1e-12

            # Flux calibration for Halpha
            f_cal = np.nanmedian(line_model.line.f_cal(fd.αδ_data))
            A_pred *= f_cal
        except KeyError:
            print(f"{line} missing")
            A_pred = np.nan * np.ones(spatial_grid["dense_shape"])

        # Calculate vmax for this panel
        n_pix_centre = 600
        l_i = 2000 // 2 - n_pix_centre  # Using default n_dense=2000
        u_i = 2000 // 2 + n_pix_centre
        A_vmax = np.nanmax(A_pred[l_i:u_i, l_i:u_i])

        # Create plot
        cs = ax.pcolormesh(
            spatial_grid["α_plot"],
            spatial_grid["δ_plot"],
            A_pred,
            # cmap=sns.color_palette("viridis", as_cmap=True),
            cmap="cmr.voltage_r",
            # vmin=0,
            # vmax=A_vmax,
            shading="auto",
            rasterized=True,
            zorder=0,
        )

        cb = fig.colorbar(cs, ax=ax, pad=0.02, location="top", aspect=30)
        cb.ax.tick_params(which="both", top=True, bottom=False, direction="out")
        cb.locator = MaxNLocator(nbins=5)
        cb.update_ticks()
        if i == 0:
            cb.set_label(label=r"Flux [10$^{-12}$ erg s$^{-1}$ cm$^{-2}$]", labelpad=10)
        cb.ax.set_axisbelow(False)

        # Setup axis
        setup_axis(ax, spatial_grid["bounds"], line)

        # Facecolor
        set_face_color(axes, "white")

    for i in range(2):
        axes[-1, i].set_xlabel(r"$\alpha$ [deg]")
        axes[i, 0].set_ylabel(r"$\delta$ [deg]")
    plt.savefig(PLOTS_PATH / "metal_recombination_lines_flux.pdf")
    plt.close(fig)
    print("Done.")


def plot_integrated_spectra(tiles):
    # line_list = METAL_RLS
    # line_list = AURORAL_LINES
    line_list = NINE_PANELS_LINES
    # line_list = HELIUM_RLS

    if line_list == AURORAL_LINES:
        plot_name = "auroral"
    elif line_list == NINE_PANELS_LINES:
        plot_name = "strong"
    elif line_list == METAL_RLS:
        plot_name = "metal_rl"
    elif line_list == HELIUM_RLS:
        plot_name = "helium_rl"
    else:
        plot_name = "other"

    models, configs = load_models(line_list)

    fig, axes = plt.subplots(len(line_list), 1, figsize=[16, 16], layout="compressed")
    for i, (ax, line) in enumerate(zip(axes.flat, line_list)):
        line_model = models[line]
        line_config = configs[line]
        fd = FitDataBuilder(tiles, line_config).build()

        ax.set_title(rf"$\mathrm{{{make_str_label_safe(line)}}}$")

        n_λ_dense = 200
        λ_dense = np.linspace(fd.λ.min(), fd.λ.max(), n_λ_dense)
        pred_flux = jax.vmap(line_model, in_axes=(0, None))(λ_dense, fd.αδ_data)
        pred_flux = fd.predict_flux(pred_flux)

        integrated_model = np.nanmean(pred_flux, axis=1)
        integrated_data = np.nanmean(fd.predict_flux(fd.flux), axis=1)

        ax.scatter(fd.λ, integrated_data / 1e-12, c="C0", s=50, marker=".")
        ax.plot(λ_dense, integrated_model / 1e-12, c="C1")

    plt.savefig(PLOTS_PATH / f"{plot_name}_integrated_9lines.pdf", bbox_inches="tight")


def get_ePSF_for_model(model, fd, plot=False, verbose=False):
    kernel = matern_kernel

    def get_ePSF(GP_component, freq_max=50, plot=False, use_spectral_CDF=False):
        freqs = GP_component._freqs
        prior_s = GP_component.kernel.length_scale.val[0]
        prior_var = GP_component.kernel.variance.val[0]
        conj_func = GP_component._conj_symmetry
        X = GP_component.kernel.feature_weights(freqs) * GP_component.coefficients.val
        # X_conj = conj_func(X.flatten()).reshape(X.shape)

        X = X.flatten()[freqs.flatten() <= freq_max]
        freqs = freqs.flatten()[freqs.flatten() <= freq_max]

        range_X = [0, 50]
        n_bins = 30
        X_std, f_bin_edges, _ = binned_statistic(
            x=freqs,
            values=X,
            range=range_X,
            statistic="std",
            bins=n_bins,
        )
        f_bin, _, _ = binned_statistic(
            x=freqs,
            values=freqs,
            range=range_X,
            statistic="mean",
            bins=n_bins,
        )

        X /= X_std[0]

        def fit_from_binned(f, X_std):
            def loss(x):
                length, ν = x
                r = X_std - kernel(f, length, ν)
                return np.sum(r**2)
                # return np.sum(np.log(1 + r**2 / Q**2))

            res = minimize(
                loss,
                x0=(0.1, 1.5),
                method="L-BFGS-B",
                bounds=[(1e-2, 1e1), (1.0, 3.5)],
            )
            return res.x

        var_hat = 1  # X_std[0] ** 2
        s_hat, info = estimate_s(
            lambda f, s: kernel(f, s) ** 2,
            freqs,
            X,
            bounds=(1e-2, 1e2),
            var=var_hat,
        )
        # (s_hat, nu_hat), info = estimate_s_nu(
        #     lambda f, s, nu: kernel(f, s, nu) ** 2,
        #     freqs,
        #     X,
        #     bounds=[(1e-2, 1e2), (1, 5)],
        #     var=var_hat,
        # )
        # s_hat, nu_hat = fit_from_binned(f_bin, X_std / X_std[0])
        # s_hat, nu_hat = fit_from_binned(f_bin, X_std)

        if verbose:
            print(f"MLE s: {s_hat:.3f}, Prior s: {prior_s:.3f}, Prior var: {prior_var:.3f}")
            # print(f"MLE ν: {nu_hat:.3f}, Prior ν: {1.5:.3f}")

        if plot:
            freqs_plot = np.linspace(freqs.min(), freqs.max(), 200)
            fw_inferred_plot = np.sqrt(var_hat) * kernel(freqs_plot, length=s_hat)
            # fw_inferred_plot = np.sqrt(var_hat) * kernel(freqs_plot, length=s_hat, nu=nu_hat)
            range_X = [0, 50]
            n_bins = 30
            X_std, f_bin_edges, _ = binned_statistic(
                x=freqs,
                values=X,
                range=range_X,
                statistic="std",
                bins=n_bins,
            )
            f_bin, _, _ = binned_statistic(
                x=freqs,
                values=freqs,
                range=range_X,
                statistic="mean",
                bins=n_bins,
            )
            _, ax = plt.subplots(1, 1, figsize=[8, 6], dpi=150, layout="compressed")
            ax.scatter(freqs, X, s=25, linewidths=0.25, edgecolors="white")
            ax.plot(
                freqs_plot,
                fw_inferred_plot,
                c="C1",
                label="Inferred PSD (MLE)",
                ls="--",
                alpha=1.0,
            )
            ax.plot(f_bin, X_std, c="C4", label="Inferred PSD (binning)")
            ax.set_xlim(-3, 153)
            ax.set_ylim(-1.4, 1.4)
            ax.set_ylabel("Fourier weight")
            ax.set_xlabel("Frequency")
            ax.legend(loc="upper right")
            plt.show()

        return s_hat

    # Model predictions
    try:
        A_s = get_ePSF(model.line.A_raw, plot=plot)
    except AttributeError:
        A_s = get_ePSF(model.line.A_raw_1, plot=plot)
    v_s = get_ePSF(model.line.v, plot=plot)
    vσ_s = get_ePSF(model.line.vσ_raw, plot=plot)

    s_hats = np.array([A_s, v_s, vσ_s])

    def get_length_arcseconds(s):
        x1 = 0.0
        x2 = x1 + s
        x1_α = fd.predict_α(x1)
        x2_α = fd.predict_α(x2)
        return np.abs(x2_α - x1_α) * 3600

    return get_length_arcseconds(s_hats)


def get_mean_snr(model, fd, stat=np.nanmean):
    offs = model.offs.const.spaxel_values.val
    flux = fd._flux - offs
    snr = np.nansum(flux * fd._i_var, axis=0) / np.sqrt(np.nansum(fd._i_var, axis=0))
    return stat(snr)


def plot_ePSFs(tiles, show_each_PSF_fit_plot=False):
    all_lines = NINE_PANELS_LINES + AURORAL_LINES + METAL_RLS + HELIUM_RLS
    models, configs = load_models(all_lines)

    ePSF_arcsec = {}
    snrs_peak = {}
    snrs_mean = {}
    snrs_median = {}
    for line in all_lines:
        fd = FitDataBuilder(tiles, configs[line]).build()
        ePSF_arcsec[line] = get_ePSF_for_model(
            model=models[line],
            fd=fd,
            plot=show_each_PSF_fit_plot,
            verbose=True,
        )
        snrs_peak[line] = get_mean_snr(models[line], fd, stat=np.nanmax)
        snrs_mean[line] = get_mean_snr(models[line], fd, stat=np.nanmean)
        snrs_median[line] = get_mean_snr(models[line], fd, stat=np.nanmedian)

    def ePSF_func(x, a, b, c):
        return a * x**-c + b

    def fit_ePSF_func(snr_vals, ePSF_vals, Q=80, x0=(100, 134, 0.4)):
        def loss(x):
            a, b, c = x
            r = ePSF_vals - ePSF_func(snr_vals, a, b, c)
            return np.sum(r**2)
            # return np.sum(np.log(1 + r**2 / Q**2))

        res = minimize(
            loss,
            x0=x0,
            method="L-BFGS-B",
            bounds=[(30, 5000), (0, 500), (0.1, 0.9)],
        )
        return res.x

    snrs_peak = np.array(list(snrs_peak.values()), dtype=float)
    snrs_mean = np.array(list(snrs_mean.values()), dtype=float)
    snrs_median = np.array(list(snrs_median.values()), dtype=float)
    line_names = list(ePSF_arcsec.keys())
    epsfs = np.array(list(ePSF_arcsec.values()), dtype=float)

    A_epsf = epsfs[:, 0]
    v_epsf = epsfs[:, 1]
    vσ_epsf = epsfs[:, 2]

    snrs = snrs_median

    a_A, b_A, c_A = fit_ePSF_func(snrs, A_epsf, x0=(180, 140, 0.4))
    bm_v = snrs > 1.2
    a_v, b_v, c_v = fit_ePSF_func(snrs[bm_v], v_epsf[bm_v], x0=(1150, 10, 0.22))
    bm_vσ = np.logical_and(snrs > 2, vσ_epsf < 800)
    a_vσ, b_vσ, c_vσ = fit_ePSF_func(snrs[bm_vσ], vσ_epsf[bm_vσ], x0=(1300, 10, 0.15))

    print(a_A, b_A, c_A)
    print(a_v, b_v, c_v)
    print(a_vσ, b_vσ, c_vσ)

    x_dense = np.linspace(1e-2, 2 * snrs.max(), 500)
    epfs_func_A = ePSF_func(x_dense, a_A, b_A, c_A)
    epfs_func_v = ePSF_func(x_dense, a_v, b_v, c_v)
    epfs_func_vσ = ePSF_func(x_dense, a_vσ, b_vσ, c_vσ)

    cmap = cmr.get_sub_cmap(plt.get_cmap("turbo"), 0.15, 0.95)
    cols_sel = cmap(np.linspace(0, 1, len(NINE_PANELS_LINES)))

    # cmaps_aur = cmr.get_sub_cmap(..., 0.05, 0.95)
    cols_aur = cmap(np.linspace(0, 1, len(AURORAL_LINES)))

    # cmaps_hel = cmr.get_sub_cmap(cmr.torch, 0.15, 0.95)
    cols_hel = cmap(np.linspace(0, 1, len(HELIUM_RLS)))

    cols_met = cmap(np.linspace(0, 1, len(METAL_RLS)))

    def get_color(line_name):
        if line_name in NINE_PANELS_LINES:
            idx = NINE_PANELS_LINES.index(line_name)
            return cols_sel[idx]
        elif line_name in AURORAL_LINES:
            idx = AURORAL_LINES.index(line_name)
            return cols_aur[idx]
        elif line_name in HELIUM_RLS:
            idx = HELIUM_RLS.index(line_name)
            return cols_hel[idx]
        elif line_name in METAL_RLS:
            idx = METAL_RLS.index(line_name)
            return cols_met[idx]

    def get_marker(line_name):
        if line_name in NINE_PANELS_LINES:
            return "o"
        elif line_name in AURORAL_LINES:
            return "^"
        elif line_name in HELIUM_RLS:
            return "s"
        elif line_name in METAL_RLS:
            return "D"
        else:
            raise Exception("Unrecognised line no marker style chosen.")

    fig, ax = plt.subplots(
        3,
        1,
        figsize=[13, 13],
        dpi=500,
        sharex=True,
        sharey=False,
        layout="compressed",
    )
    fig.get_layout_engine().set(**LAYOUT_DETAILS)

    # for i in range(3):
    # ax[i].set_xscale("log")
    # ax[i].set_yscale("log")

    handles_sels_scatter = []
    labels_sels_scatter = []
    handles_aur_scatter = []
    labels_aur_scatter = []
    handles_hel_scatter = []
    labels_hel_scatter = []
    handles_met_scatter = []
    labels_met_scatter = []

    for i in range(len(line_names)):
        color = get_color(line_names[i])
        marker = get_marker(line_names[i])
        # label_i = rf"$\mathrm{{{make_str_label_safe(line_names[i])}}}$"
        label_i = ALL_LABELS[line_names[i]]
        scatter_kwargs = dict(
            s=220,
            linewidths=1.2,
            edgecolors="white",
            marker=marker,
            color=color,
            label=label_i,
        )

        snr = snrs[i]
        s1 = ax[0].scatter(snr, A_epsf[i], **scatter_kwargs)
        # if True:
        if line_names[i] in NINE_PANELS_LINES:
            s2 = ax[1].scatter(snr, v_epsf[i], **scatter_kwargs)
            s3 = ax[2].scatter(snr, vσ_epsf[i], **scatter_kwargs)
            handles_sels_scatter.append(s1)
            labels_sels_scatter.append(label_i)
        elif line_names[i] in AURORAL_LINES:
            handles_aur_scatter.append(s1)
            labels_aur_scatter.append(label_i)
            s1.set_zorder(100)
        elif line_names[i] in HELIUM_RLS:
            handles_hel_scatter.append(s1)
            labels_hel_scatter.append(label_i)
        elif line_names[i] in METAL_RLS:
            handles_met_scatter.append(s1)
            labels_met_scatter.append(label_i)

    trend_line_kwargs = dict(c="grey", zorder=-10, alpha=0.7)
    ax[0].plot(x_dense, epfs_func_A, **trend_line_kwargs)
    ax[1].plot(x_dense, epfs_func_v, **trend_line_kwargs)
    ax[2].plot(x_dense, epfs_func_vσ, **trend_line_kwargs)
    # ax[0].plot(x_dense, epfs_func_v, c="grey", zorder=-10)
    # ax[0].plot(x_dense, epfs_func_vσ, c="grey", zorder=-10)

    def get_legend_kwargs(i):
        x0 = 0.98 - 0.22  # just outside the axes
        dx = 0.20
        return dict(loc="upper left", bbox_to_anchor=(x0 - i * dx, 0.98), borderaxespad=0.0)

    hands_labs = [
        (handles_sels_scatter, labels_sels_scatter),
        (handles_aur_scatter, labels_aur_scatter),
        (handles_hel_scatter, labels_hel_scatter),
        (handles_met_scatter, labels_met_scatter),
    ]
    for i, (h, l) in enumerate(hands_labs):
        leg = ax[0].legend(h, l, framealpha=None, **get_legend_kwargs(i))
        if i < len(hands_labs) - 1:
            ax[0].add_artist(leg)

    ax[0].set_xlim(-0.7, 11)

    ax[0].set_ylim(160, 1010)
    ax[1].set_ylim(160, 1010)
    ax[2].set_ylim(160, 1010)

    # Set tick locators
    for a in ax:
        # x
        a.xaxis.set_major_locator(MultipleLocator(2))
        a.xaxis.set_minor_locator(MultipleLocator(0.5))
        # y
        a.yaxis.set_major_locator(MultipleLocator(200))
        a.yaxis.set_minor_locator(MultipleLocator(50))

    # labels = [
    #     rf"$\mathrm{{{make_str_label_safe(line_label)}}}$" for line_label in ePSF_arcsec.keys()
    # ]

    # for epsfs_i, ax_i in zip([A_epsf, v_epsf, vσ_epsf], ax):
    #     texts = [
    #         ax_i.annotate(lab, (x, y), fontsize="medium")
    #         for x, y, lab in zip(snrs, epsfs_i, deepcopy(labels))
    #     ]

    #     adjust_text(
    #         texts,
    #         ax=ax_i,
    #         force_static=5.0,
    #         force_texts=30.0,
    #         force_explode=5.0,
    #         max_move=50,
    #         pull_threshold=15,
    #         only_move={"points": "xy", "text": "xy"},
    #         arrowprops=dict(arrowstyle="-", lw=2, color="grey", shrinkA=10, shrinkB=10),
    #         expand_axes=True,
    #         ensure_inside_axes=True,
    #         lim=10000,
    #         expand=(3, 3),
    #     )

    #     for t in texts:
    #         ap = getattr(t, "arrow_patch", None)
    #         if ap is not None:
    #             ap.set_clip_on(True)

    # plt.plot(x_dense, ePSF_func(x_dense, 140, 134), c="C1")

    ax[-1].set_xlabel("Median integrated SNR")
    component_labels = [
        r"Flux",
        r"Velocity",
        r"Dispersion",
    ]
    for ax_, c_lab in zip(ax.flatten(), component_labels):
        ax_.set_ylabel(rf"ePSF for {c_lab} Map [arcsec]")
    plt.savefig(PLOTS_PATH / "epsf.pdf", bbox_inches="tight")

    # plt.figure()
    # plt.scatter(snrs_mean, snrs_peak)
    # plt.xlabel("Mean integrated SNR")
    # plt.ylabel("Peak integrated SNR")
    # plt.axis("scaled")
    # plt.show()


# =============================================================================
# CLI SETUP
# =============================================================================

# Map of available plot functions
PLOT_FUNCTIONS = {
    "flux": plot_nine_panels_flux,
    "velocity": plot_nine_panels_velocity,
    "dispersion": plot_nine_panels_dispersion,
    "auroral": plot_auroral_flux,
    "helium": plot_helium_RL_flux,
    "metal": plot_metal_RL_flux,
    "epsf": plot_ePSFs,
    "integrated": plot_integrated_spectra,
}


def load_data():
    """Load tile data once."""
    print("Reading data...", end=" ", flush=True)
    tiles = LVMTileCollection.from_tiles([LVMTile.from_file(Path(f)) for f in DRP_FILES])
    print("Done.")
    return tiles


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate plots for the paper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s flux                    # Create flux plot only
  %(prog)s velocity dispersion     # Create velocity and dispersion plots  
  %(prog)s --all                   # Create all plots
  %(prog)s --list                  # List available plot types
        """,
    )

    available_plots = list(PLOT_FUNCTIONS.keys())

    parser.add_argument(
        "plots",
        nargs="*",
        choices=available_plots,
        help=f"Plot types to generate. Available: {', '.join(available_plots)}",
    )

    parser.add_argument("--all", action="store_true", help="Generate all available plots")

    parser.add_argument("--list", action="store_true", help="List available plot types and exit")

    args = parser.parse_args()

    # Handle list option
    if args.list:
        print("Available plot types:")
        for plot_type in available_plots:
            print(f"  - {plot_type}")
        return

    # Determine which plots to create
    if args.all:
        plots_to_create = available_plots
    elif args.plots:
        plots_to_create = args.plots
    else:
        # Default to all if no specific plots requested
        plots_to_create = available_plots

    # Load data once
    tiles = load_data()

    # Create requested plots
    for plot_type in plots_to_create:
        PLOT_FUNCTIONS[plot_type](tiles)


if __name__ == "__main__":
    main()
